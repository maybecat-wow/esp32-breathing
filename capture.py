#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture.py — TCP server that receives the binary CSI stream from the ESP32
and writes it to a .bin file verbatim.

The on-disk file is the same byte stream as the wire (length-prefixed
SESSION_INFO / CSI_FRAME / HEARTBEAT messages), so a recording can be
replayed by feeding the .bin to csi_breathing.load_binary().

Usage:
    python capture.py -s csi_data.bin
    python capture.py -s csi_data.bin -p 3490
    python capture.py -s csi_data.bin -p 3490 -l csi_data_log.txt

stats.json keeps the same shape as before (frames_written, gap_count,
total_gap_seconds, longest_segment_seconds, last_heartbeat_utc, gaps[]),
but the values come from a header-only peek of the binary stream rather
than CSV parsing.
"""

import datetime as _dt
import json
import os
import socket
import sys
import time

import csi_protocol as proto

HOST = "0.0.0.0"

RECV_TIMEOUT_S      = 3.0
FSYNC_INTERVAL_S    = 60.0
STATS_INTERVAL_S    = 60.0


class StallError(Exception):
    pass


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from *conn* or raise StallError on timeout /
    ConnectionError on remote close."""
    out = bytearray()
    while len(out) < n:
        try:
            chunk = conn.recv(n - len(out))
        except socket.timeout as e:
            raise StallError("recv timeout") from e
        if not chunk:
            raise ConnectionError("peer closed")
        out.extend(chunk)
    return bytes(out)


class CaptureSession:
    """Tracks stats across reconnects while appending raw bytes to .bin."""

    def __init__(self, bin_fd, log_fd, store_path):
        self.bin_fd      = bin_fd
        self.log_fd      = log_fd
        self.store_path  = store_path
        self.stats_path  = store_path + ".stats.json"

        self.started_wall    = _dt.datetime.now(_dt.timezone.utc)
        self.frames_written  = 0
        self.last_fsync_mono = time.monotonic()
        self.last_stats_mono = 0.0
        self.last_heartbeat_wall = None

        # Per-session timestamp/wrap state (mirrors loader logic).
        self.session_boot_id = None
        self.prev_raw_us     = None
        self.wrap_offset_us  = 0
        self.prev_logical_us = None

        self.gaps              = []      # bounded at 1000 entries
        self.gap_count         = 0
        self.total_gap_ms      = 0
        self.longest_segment_s = 0.0
        self._segment_start_ns = time.monotonic_ns()

    def write_raw(self, blob: bytes):
        """Append *blob* (already-framed bytes) verbatim to the .bin."""
        self.bin_fd.write(blob)
        self._periodic_maintenance()

    def record_message(self, msg_type: int, payload: bytes):
        """Update stats based on a single parsed message. Does NOT write to
        .bin — write_raw() does that."""
        if msg_type == proto.MSG_SESSION_INFO:
            try:
                info = proto.decode_session_info(payload)
            except Exception:
                return
            hard = (self.session_boot_id is not None
                    and info.boot_id != self.session_boot_id)
            if hard:
                self._note_gap("REBOOT", 0)
            self.session_boot_id = info.boot_id
            self.prev_raw_us = None
            self.wrap_offset_us = 0
            self.prev_logical_us = None

        elif msg_type == proto.MSG_CSI_FRAME:
            try:
                meta, _ = proto.decode_csi_frame(payload)
            except Exception:
                return
            raw = meta.local_timestamp_us
            if self.prev_raw_us is not None and raw < self.prev_raw_us:
                self.wrap_offset_us += proto.U32_WRAP
            logical = raw + self.wrap_offset_us
            self.prev_raw_us = raw

            if (self.prev_logical_us is not None
                    and logical - self.prev_logical_us > proto.GAP_THRESHOLD_US):
                self._note_gap("MONOTONIC",
                               (logical - self.prev_logical_us) // 1000)
            self.prev_logical_us = logical
            self.frames_written += 1

        elif msg_type == proto.MSG_HEARTBEAT:
            self.last_heartbeat_wall = _dt.datetime.now(_dt.timezone.utc)

    def _note_gap(self, reason: str, duration_ms: int):
        self.gap_count += 1
        self.total_gap_ms += max(0, int(duration_ms))
        entry = {
            "t": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "duration_ms": int(duration_ms),
            "reason": reason,
        }
        if len(self.gaps) < 1000:
            self.gaps.append(entry)
        now_ns = time.monotonic_ns()
        seg_s = (now_ns - self._segment_start_ns) / 1e9
        if seg_s > self.longest_segment_s:
            self.longest_segment_s = seg_s
        self._segment_start_ns = now_ns

    def note_connection_gap(self, reason: str, duration_ms: int):
        """Called by the server loop on TCP-level breaks."""
        self._note_gap(reason, duration_ms)

    def _periodic_maintenance(self):
        now = time.monotonic()
        if now - self.last_fsync_mono >= FSYNC_INTERVAL_S:
            try:
                os.fsync(self.bin_fd.fileno())
            except OSError:
                pass
            self.last_fsync_mono = now
        if now - self.last_stats_mono >= STATS_INTERVAL_S:
            self.write_stats()
            self.last_stats_mono = now

    def write_stats(self):
        live_seg = (time.monotonic_ns() - self._segment_start_ns) / 1e9
        longest = max(self.longest_segment_s, live_seg)
        stats = {
            "started_utc": self.started_wall.isoformat(timespec="seconds"),
            "now_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "frames_written": self.frames_written,
            "gap_count": self.gap_count,
            "total_gap_seconds": round(self.total_gap_ms / 1000.0, 3),
            "longest_segment_seconds": round(longest, 3),
            "last_heartbeat_utc": (
                self.last_heartbeat_wall.isoformat(timespec="seconds")
                if self.last_heartbeat_wall else None
            ),
            "gaps": self.gaps,
        }
        tmp = self.stats_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(stats, f, indent=2)
            os.replace(tmp, self.stats_path)
        except OSError as e:
            print(f"Could not write stats file: {e}")


def consume_connection(conn: socket.socket, session: CaptureSession):
    """Drive one TCP connection: read message-by-message, validate the first
    is SESSION_INFO, append raw bytes to .bin, update stats from peek."""
    header = _recv_exact(conn, proto.HEADER.size)
    msg_type, length = proto.HEADER.unpack(header)

    if msg_type != proto.MSG_SESSION_INFO:
        print(f"First message was type 0x{msg_type:02x}, not SESSION_INFO — closing")
        session.log_fd.write(
            f"first-message-not-session-info type=0x{msg_type:02x}\n")
        session.log_fd.flush()
        return
    if length > proto.MAX_PAYLOAD_BYTES:
        print(f"SESSION_INFO length {length} exceeds cap — closing")
        return

    payload = _recv_exact(conn, length)
    session.write_raw(header + payload)
    session.record_message(msg_type, payload)

    while True:
        header = _recv_exact(conn, proto.HEADER.size)
        msg_type, length = proto.HEADER.unpack(header)
        if length > proto.MAX_PAYLOAD_BYTES:
            session.log_fd.write(
                f"oversized length {length} for type 0x{msg_type:02x} — closing\n")
            session.log_fd.flush()
            return
        payload = _recv_exact(conn, length) if length else b""
        session.write_raw(header + payload)
        session.record_message(msg_type, payload)


def run_server(port: int, store_path: str, log_path: str):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    print(f"Listening on {HOST}:{port}  →  writing to {store_path}")

    with open(store_path, "wb") as bin_fd, \
         open(log_path, "w") as log_fd:

        session = CaptureSession(bin_fd, log_fd, store_path)
        session.write_stats()

        pending_reason = None
        pending_start_mono_ns = None

        while True:
            print("Waiting for ESP32 connection …")
            try:
                conn, addr = srv.accept()
            except KeyboardInterrupt:
                print("\nShutting down.")
                session.write_stats()
                return
            conn.settimeout(RECV_TIMEOUT_S)
            print(f"Connected: {addr[0]}:{addr[1]}")

            if pending_reason is not None:
                gap_ms = (time.monotonic_ns() - pending_start_mono_ns) // 1_000_000
                session.note_connection_gap(pending_reason, gap_ms)
            pending_reason = None
            pending_start_mono_ns = None

            reason = "CLOSE"
            try:
                consume_connection(conn, session)
            except StallError:
                print(f"Recv stall > {RECV_TIMEOUT_S}s — closing socket.")
                reason = "TIMEOUT"
            except ConnectionError as e:
                print(f"Connection error: {e}")
                reason = "OSERR"
            except OSError as e:
                print(f"Socket error: {e}")
                reason = "OSERR"
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
                print("Connection closed.")

            pending_reason = reason
            pending_start_mono_ns = time.monotonic_ns()
            session.write_stats()


if __name__ == "__main__":
    if sys.version_info < (3, 6):
        print("Python >= 3.6 required")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(
        description="TCP server: receive binary CSI stream and save to .bin")
    parser.add_argument("-p", "--port", type=int, default=3490,
                        help="TCP port to listen on (default: 3490)")
    parser.add_argument("-s", "--store", dest="store_file",
                        default="./csi_data.bin",
                        help="Output .bin file (default: ./csi_data.bin)")
    parser.add_argument("-l", "--log", dest="log_file",
                        default="./csi_data_log.txt",
                        help="Sidecar log for protocol errors (default: ./csi_data_log.txt)")
    args = parser.parse_args()
    run_server(args.port, args.store_file, args.log_file)
