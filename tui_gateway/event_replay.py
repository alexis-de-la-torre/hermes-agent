"""Per-session event sequencing + bounded replay for WS reconnects.

Every event frame through :func:`server.write_json` (hence ``_emit``) gets a per-session monotonic
``seq`` and lands in a small ring per session; a reconnecting client calls ``session.events.since``
with its last seen seq and gets everything newer. Invariants: stdio TUI unaffected (``seq`` only on
event frames; Ink ignores unknown keys); one lock guards counters + buffers, and write_json already
serializes per-transport writes so stamping cannot reorder frames; memory bound =
_REPLAY_BUFFER_MAX events AND _REPLAY_BUFFER_BYTES_MAX frozen UTF-8 payload bytes per session,
_REPLAY_PROCESS_BYTES_MAX frozen UTF-8 payload bytes across at most _REPLAY_SESSIONS_MAX sessions,
oldest evicted FIFO (their seq/truncation counters are retained so a revisited session stays
monotonic within the epoch, #100122). Evicted or never-retained (oversized) frames leave a truncation watermark so a
reconnecting client refetches instead of trusting a replay with holes. A torn-down session's ring is freed at
teardown (``forget_session``) and never rebuilt by late frames.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections import OrderedDict, deque

# Seq counters live in-process, so a restart resets them to 1 while clients hold high
# watermarks — events_since(sid, 97) would return [] with truncated=False forever. The
# epoch lets clients detect the restart and reset their watermarks.
_REPLAY_EPOCH = uuid.uuid4().hex

# A long turn emits ~hundreds of token events; 512 covers minutes of streaming plus
# all control events. Desktop users rarely exceed a dozen live chats.
_REPLAY_BUFFER_MAX = 512
_REPLAY_SESSIONS_MAX = 64
# A ring may legitimately hold many bounded 64 KiB tool results (512 of them ≈ 32 MiB per
# session, ×64 sessions before any cap); bound the frozen serialized payload so replay memory
# cannot scale with mutable Python object graphs after admission.
_REPLAY_BUFFER_BYTES_MAX = 4 * 1024 * 1024
_REPLAY_PROCESS_BYTES_MAX = 64 * 1024 * 1024

_replay_lock = threading.Lock()
# sid -> deque of (seq, frozen params JSON bytes without the server-owned seq, retained bytes).
# Bytes detach the replay ring from the live event object: later caller mutation cannot grow or
# rewrite an already-accounted cache entry.
_replay_buffers: OrderedDict[str, deque] = OrderedDict()
_replay_buffer_bytes: dict[str, int] = {}
_replay_evicted_through: dict[str, int] = {}
_replay_total_bytes = 0
_replay_next_seq: dict[str, int] = {}
# Torn-down sids (forget_session). A frame stamped later (a turn thread that outlived the teardown join, a
# compute-host bridge echo) still gets a seq and a watermark but is never retained, so it cannot rebuild a
# ring that nothing would free until FIFO eviction. One str per closed session, like _replay_next_seq.
_replay_forgotten: set[str] = set()

# ── durable turn spool ───────────────────────────────────────────────────────────────
# A bounded in-memory ring cannot serve a reconnect that outlasts it, and the transcript is
# only written to SQLite at turn end, so a client that reconnects mid-turn has nothing to
# refetch. Every event frame is therefore ALSO appended to a per-session spool file and
# `events_since` serves the gap from it. This is the resumable-stream pattern the ecosystem
# uses (record every chunk in a log, resume by offset), with a file as the log.
#
# Nothing here is a source of truth: the spool is a replay tail, bounded per session and
# deleted at teardown. A missing or unreadable spool degrades to today's behaviour.
_SPOOL_DIR_NAME = ("runtime", "event_spool")
_SPOOL_MAX_BYTES = 2 * 1024 * 1024
_SPOOL_KEEP_LINES = 16384
# Trim to a target below the bound, not to the bound. Trimming to "just under" leaves no room for
# the next frame, so a 3 KiB frame stream trips the trigger on nearly every append and rewrites
# the whole file each time — 10 ms per event on the thread that emits it. A quarter of the bound
# as headroom turns that into one trim per few hundred frames.
_SPOOL_TRIM_TO_BYTES = (_SPOOL_MAX_BYTES * 3) // 4
_SPOOL_TRIM_TO_LINES = (_SPOOL_KEEP_LINES * 3) // 4
_spool_bytes: dict[str, int] = {}
_spool_lines: dict[str, int] = {}
_spool_lock = threading.Lock()


def _spool_dir() -> str:
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(home, *_SPOOL_DIR_NAME)


def _spool_path(sid: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", sid or "")[:120] or "session"
    return os.path.join(_spool_dir(), f"{safe}.jsonl")


def _spool_append(sid: str, seq: int, payload: bytes) -> None:
    """Append one frame to the session's spool. Best-effort: replay must never break a turn."""
    if not sid:
        return
    path = _spool_path(sid)
    try:
        with _spool_lock:
            size = _spool_bytes.get(sid)
            lines = _spool_lines.get(sid)
            if size is None or lines is None:
                try:
                    size = os.path.getsize(path)
                except OSError:
                    size = 0
                # The cache was cold (first frame after a restart): count what is on disk so the
                # line bound is enforced from the first append, not after it overflows again.
                try:
                    with open(path, "rb") as existing:
                        lines = sum(1 for _ in existing)
                except OSError:
                    lines = 0
            line = str(seq).encode("ascii") + b"\t" + payload + b"\n"
            try:
                handle = open(path, "ab")
            except FileNotFoundError:
                # The cached size said the file was there; it was removed under us (a test reset,
                # a teardown, an operator). Recreate the directory rather than silently drop the
                # turn's replay tail.
                os.makedirs(os.path.dirname(path), exist_ok=True)
                handle = open(path, "ab")
            with handle:
                handle.write(line)
                size += len(line)
            _spool_bytes[sid] = size
            _spool_lines[sid] = lines = lines + 1
            if size > _SPOOL_MAX_BYTES or lines > _SPOOL_KEEP_LINES:
                _spool_trim(sid, path)
    except OSError:
        return


def _spool_trim(sid: str, path: str) -> None:
    """Drop the oldest lines until the spool is under both bounds, with headroom (caller holds the lock).

    Trimming to a line count alone leaves a file that is still over the byte cap, so the append
    that follows trips the same condition and rewrites the whole file again — on the thread that
    emits events, under a lock every session shares (measured: 0.8 s for 20k appends becomes
    165 s). Dropping from the head until both bounds hold makes the trigger answer true once.
    """
    try:
        with open(path, "rb") as handle:
            lines = handle.readlines()
        total = sum(len(line) for line in lines)
        start = 0
        # Keep the newest line whatever its size: a frame up to the ring's 4 MiB cap must still
        # be servable, even when it alone exceeds the trim target.
        while start < len(lines) - 1 and (total > _SPOOL_TRIM_TO_BYTES or len(lines) - start > _SPOOL_TRIM_TO_LINES):
            total -= len(lines[start])
            start += 1
        kept = lines[start:]
        if start:
            with open(path, "wb") as handle:
                handle.writelines(kept)
        _spool_bytes[sid] = total
        _spool_lines[sid] = len(kept)
    except OSError:
        _spool_bytes.pop(sid, None)
        _spool_lines.pop(sid, None)


def _spool_read(sid: str, last_seen: int) -> list[tuple[int, bytes]]:
    """Frames with seq > last_seen still on disk. Caller is a reconnect, so cost is fine."""
    if not sid:
        return []
    path = _spool_path(sid)
    out: list[tuple[int, bytes]] = []
    try:
        with open(path, "rb") as handle:
            for line in handle:
                head, sep, payload = line.partition(b"\t")
                if not sep:
                    continue
                try:
                    seq = int(head)
                except ValueError:
                    continue
                if seq > last_seen:
                    out.append((seq, payload.rstrip(b"\n")))
    except OSError:
        return []
    return out


def _spool_forget(sid: str) -> None:
    try:
        os.unlink(_spool_path(sid))
    except OSError:
        pass
    with _spool_lock:
        _spool_bytes.pop(sid, None)
        _spool_lines.pop(sid, None)


def replay_epoch() -> str:
    """Opaque token identifying this server process's seq numbering."""
    return _REPLAY_EPOCH


def _freeze_params(params: dict) -> bytes:
    """Serialize one replay snapshot before the caller can mutate its object graph."""
    return json.dumps(params, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8", errors="surrogatepass"
    )


def _thaw_params(payload: bytes, seq: int) -> dict:
    """Rebuild a client event object and restore the server-owned sequence number."""
    event = json.loads(payload.decode("utf-8", errors="surrogatepass"))
    event["seq"] = seq
    return event


def _stamp_event(obj: dict) -> None:
    """Stamp one outgoing event frame (mutates obj in place) and record a frozen snapshot."""
    if obj.get("method") != "event":
        return
    params = obj.get("params")
    if not isinstance(params, dict):
        return
    sid = params.get("session_id") or ""
    if not sid:
        # Session-less global events (skin.changed etc.) are re-fetchable via their own RPCs.
        return
    # Freeze OUTSIDE the lock (same rule as transport.write) so one large payload cannot stall
    # other threads' frames. The server-owned seq is restored when replaying, so the snapshot can
    # be serialized before sequence assignment without retaining the mutable params dict.
    payload = _freeze_params(params)
    size = len(payload)
    with _replay_lock:
        global _replay_total_bytes
        _drain_released_locked()
        seq = _replay_next_seq.get(sid, 0) + 1
        _replay_next_seq[sid] = seq
        params["seq"] = seq
        if sid in _replay_forgotten:
            _replay_evicted_through[sid] = seq
            return
        buf = _replay_buffers.get(sid)
        if buf is None:
            buf = _replay_buffers[sid] = deque()
            _replay_buffer_bytes[sid] = 0
            while len(_replay_buffers) > _REPLAY_SESSIONS_MAX:
                _drop_ring_locked(next(iter(_replay_buffers)))
        if size > _REPLAY_BUFFER_BYTES_MAX or size > _REPLAY_PROCESS_BYTES_MAX:
            # Not spooled either: one frame over the cap is not a replayable unit, and writing it
            # would only push the real turn out of the spool.
            _replay_evicted_through[sid] = seq
            return
        buf.append((seq, payload, size))
        _replay_buffer_bytes[sid] += size
        _replay_total_bytes += size
        while len(buf) > _REPLAY_BUFFER_MAX or _replay_buffer_bytes[sid] > _REPLAY_BUFFER_BYTES_MAX:
            evicted_seq, _event, evicted_size = buf.popleft()
            _replay_buffer_bytes[sid] -= evicted_size
            _replay_total_bytes -= evicted_size
            _replay_evicted_through[sid] = max(_replay_evicted_through.get(sid, 0), evicted_seq)
        while _replay_total_bytes > _REPLAY_PROCESS_BYTES_MAX:
            for evict_sid, evict_buf in _replay_buffers.items():
                if evict_buf:
                    evicted_seq, _event, evicted_size = evict_buf.popleft()
                    _replay_buffer_bytes[evict_sid] -= evicted_size
                    _replay_total_bytes -= evicted_size
                    _replay_evicted_through[evict_sid] = max(
                        _replay_evicted_through.get(evict_sid, 0), evicted_seq
                    )
                    break
    # The ring is the fast path; the spool is what makes a reconnect that outlasts it work.
    _spool_append(sid, seq, payload)


def _drop_ring_locked(sid: str) -> None:
    """Free one session's retained frames (FIFO cap eviction and teardown). Caller holds ``_replay_lock``.

    The ring goes, the session's seq numbering stays (#100122): a revisited session continues from its high
    seq instead of restarting at 1 under clients' still-held watermarks (a reset seq is invisible to
    dispatchIfNewer, so replay AND live frames silently vanish). The truncation watermark is raised to the
    latest stamped seq: a client holding any older watermark has a gap and must refetch history instead of
    trusting the new tail. Both counters are one int per session id seen this process."""
    global _replay_total_bytes
    if _replay_buffers.pop(sid, None) is None:
        return
    _replay_total_bytes -= _replay_buffer_bytes.pop(sid, 0)
    _replay_evicted_through[sid] = max(_replay_evicted_through.get(sid, 0), _replay_next_seq.get(sid, 0))
    # A dropped ring also drops its durable tail. Eviction here means the session is idle and
    # past the session cap (or torn down), so the transcript is its source of truth again.
    _spool_forget(sid)


def forget_session(sid: str) -> None:
    """Release a torn-down session's replay ring. A runtime sid is never reattached after teardown
    (resume mints a new one), so its frames can only be replayed to nobody."""
    if not sid:
        return
    with _replay_lock:
        _replay_forgotten.add(sid)
        _drop_ring_locked(sid)


# Private keys whose owner is gone (release_key from a GC finalizer that could not take the lock),
# purged by the next lock holder. list.append/pop are atomic, so a finalizer never needs the lock.
_released_keys: list[str] = []


def _purge_key_locked(key: str) -> None:
    global _replay_total_bytes
    if _replay_buffers.pop(key, None) is not None:
        _replay_total_bytes -= _replay_buffer_bytes.pop(key, 0)
    _replay_evicted_through.pop(key, None)
    _replay_next_seq.pop(key, None)
    _replay_forgotten.discard(key)


def _drain_released_locked() -> None:
    while _released_keys:
        _purge_key_locked(_released_keys.pop())


def release_key(key: str, *, blocking: bool = True) -> None:
    """Drop everything recorded under a PRIVATE lookup key (ring AND counters) once its session ended.

    Unlike :func:`forget_session` (a public sid that late frames and #100122's monotonic numbering
    still name), a private key is only ever stamped and read through its owner
    (``gateway.session_events.SessionEvents``): that owner re-reads :func:`latest_seq` before every
    publish and starts a new epoch when it finds 0, exactly as after FIFO eviction, so nothing needs
    the old counters. ``blocking=False`` is for GC finalizers, which can run on a thread that already
    holds ``_replay_lock``: the key is queued and the next lock holder purges it."""
    if not key:
        return
    _released_keys.append(key)
    if _replay_lock.acquire(blocking=blocking):
        try:
            _drain_released_locked()
        finally:
            _replay_lock.release()


def _collect(sid: str, last_seen: int) -> tuple[list[tuple[int, bytes]], int, bool]:
    """Ring + spool frames after *last_seen*, and the highest seq AT SNAPSHOT TIME.

    ``latest`` is read under the same lock as the ring slice, so a frame stamped while the spool
    is being read cannot make a complete reply look short: comparing a frame count against a
    later ``latest_seq()`` turned every mid-turn reconnect into ``truncated=true`` for the whole
    window (measured 181 of 200 replies on a complete 4,000-frame stream).

    Returns ``(snapshots, latest, forgotten)``; the caller decodes outside the lock.
    """
    sid = sid or ""
    with _replay_lock:
        latest = _replay_next_seq.get(sid, 0)
        buf = _replay_buffers.get(sid)
        snapshots = [(seq, payload) for seq, payload, _size in buf if seq > last_seen] if buf else []
        evicted = _replay_evicted_through.get(sid, 0)
        forgotten = sid in _replay_forgotten
    if last_seen < evicted and not forgotten:
        seen = {seq for seq, _payload in snapshots}
        for seq, payload in _spool_read(sid, last_seen):
            if seq not in seen:
                snapshots.append((seq, payload))
                seen.add(seq)
        snapshots.sort(key=lambda item: item[0])
    return snapshots, latest, forgotten


def events_since(sid: str, last_seen: int) -> list[dict]:
    """Recorded EVENT OBJECTS (each frame's ``params`` dict) with seq > last_seen for *sid*.

    Returning the full JSON-RPC envelope would make every replayed event fail the
    client's ``event.type`` gate and be silently dropped. Decode after releasing the ring lock so a
    large replay cannot block concurrent event stamping.

    When the ring has evicted past *last_seen* the durable spool fills the gap, so a reconnect
    that outlasts the ring still gets the whole turn instead of a refetch it cannot satisfy.
    """
    snapshots, _latest, _forgotten = _collect(sid, last_seen)
    return [_thaw_params(payload, seq) for seq, payload in snapshots]


def events_and_truncation(sid: str, last_seen: int) -> tuple[list[dict], bool]:
    """One snapshot: the frames to send and whether they cover every seq after *last_seen*.

    The RPC's hot path. Reading ``latest`` and the frames separately let a frame stamped in
    between make a complete replay look truncated, and the client then drops the window it was
    just sent and falls back to a transcript that has no assistant message.
    """
    snapshots, latest, forgotten = _collect(sid, last_seen)
    frames = [_thaw_params(payload, seq) for seq, payload in snapshots]
    if last_seen >= latest:
        return frames, False
    if forgotten:
        return frames, True
    # Count only what the snapshot promised; frames appended during the spool read are extra, not
    # missing. A short count is a hole anywhere in the range, not only at its head.
    served = sum(1 for seq, _payload in snapshots if seq <= latest)
    return frames, served != latest - last_seen


def is_truncated(sid: str, last_seen: int, *, served: int | None = None) -> bool:
    """True when the reply would not carry every seq after *last_seen*, so the client must refetch.

    Ring eviction alone is not a gap any more: the durable spool still holds the turn. The
    question is whether the union of the two sources covers the whole range last_seen+1..latest,
    so count what can be served. A short count is a hole anywhere in the range, not only at its
    head: an oversized frame is never retained and never spooled, and one in the middle of a turn
    used to be reported as a complete replay.

    ``served`` is for a caller that already built the frames (the RPC and
    ``gateway.session_events``): it avoids decoding every frame twice on the reconnect path.
    """
    sid = sid or ""
    if served is None:
        return events_and_truncation(sid, last_seen)[1]
    with _replay_lock:
        latest = _replay_next_seq.get(sid, 0)
    if last_seen >= latest:
        return False
    if sid in _replay_forgotten:
        return True
    return served != latest - last_seen


def latest_seq(sid: str) -> int:
    """Current highest stamped seq for *sid* (0 when unknown)."""
    with _replay_lock:
        return _replay_next_seq.get(sid or "", 0)


def reset_replay_state() -> None:
    """Test hook. Also drops the durable spool, or a later run in the same process inherits
    files whose seq numbering it no longer shares."""
    with _replay_lock:
        global _replay_total_bytes
        for sid in list(_replay_forgotten | set(_replay_buffers) | set(_replay_next_seq)):
            _spool_forget(sid)
        with _spool_lock:
            _spool_bytes.clear()
            _spool_lines.clear()
        _replay_buffers.clear()
        _replay_buffer_bytes.clear()
        _replay_evicted_through.clear()
        _replay_next_seq.clear()
        _replay_forgotten.clear()
        _released_keys.clear()
        _replay_total_bytes = 0


def replay_stats() -> dict:
    """Telemetry: buffer occupancy for the ops/debug surface."""
    with _replay_lock:
        _drain_released_locked()
        return {
            "sessions": len(_replay_buffers),
            "events": sum(len(buffer) for buffer in _replay_buffers.values()),
            "bytes": _replay_total_bytes,
            "max_per_session": _REPLAY_BUFFER_MAX,
            "max_bytes_per_session": _REPLAY_BUFFER_BYTES_MAX,
            "max_bytes_process": _REPLAY_PROCESS_BYTES_MAX,
        }
