#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture.py — TCP server that receives CSI data from the ESP32 and writes it
to a CSV file.

The ESP32 runs app_main.c which connects here as a TCP *client* and streams
newline-terminated CSV rows.  This script listens on 0.0.0.0:<PORT>, accepts
one connection at a time, and appends every valid CSI_DATA row to the output
CSV.

Usage:
    python capture.py -s csi_data.csv
    python capture.py -s csi_data.csv -p 3490
    python capture.py -s csi_data.csv -p 3490 -l csi_data_log.txt

The ESP32 sends the CSV header automatically on each new connection, so this
script accepts it as the column-name row and validates subsequent rows against
it.

Overnight / sleep-quality behaviour
-----------------------------------
- `CSI_GAP` marker rows are appended to the CSV every time the stream breaks
  (TCP close, recv stall, ESP32 reboot, timestamp jump). They carry the same
  column count as `CSI_DATA` rows so existing consumers that filter on
  `type == "CSI_DATA"` ignore them cleanly. `csi_breathing.py` interprets
  them as segment boundaries.
- `HEARTBEAT` lines emitted by the ESP32 once per second (when idle) let us
  detect silent link death even if TCP keepalive is eaten by the AP.
- Every written row is flushed; `os.fsync` runs every `FSYNC_INTERVAL_S`
  seconds so a host crash loses at most ~1 minute of overnight data.
- A companion `<store_path>.stats.json` is rewritten every
  `STATS_INTERVAL_S` seconds as a morning-report summary of the night.
"""

import csv
import datetime as _dt
import json
import os
import socket
import sys
import time
from io import StringIO

# ── Default column layouts (kept for column-count validation) ───────────────

# Compact header for ESP32-C5 / C6 / C61 targets.
# These chips expose fft_gain and agc_gain directly but omit the richer
# HT/VHT PHY fields present on older chips.
DATA_COLUMNS_NAMES_C5C6 = [
    "type", "id", "mac", "rssi", "rate",
    "noise_floor", "fft_gain", "agc_gain",
    "channel", "local_timestamp", "sig_len", "rx_state",
    "len", "first_word", "data",
]

# Full header for ESP32 / S3 / C3 targets (legacy HT PHY metadata fields).
DATA_COLUMNS_NAMES = [
    "type", "id", "mac", "rssi", "rate",
    "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding",
    "aggregation", "stbc", "fec_coding", "sgi", "noise_floor",
    "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant",
    "sig_len", "rx_state", "len", "first_word", "data",
]

# Accept either layout; validated against the live header from the ESP32.
VALID_COLUMN_COUNTS = {len(DATA_COLUMNS_NAMES), len(DATA_COLUMNS_NAMES_C5C6)}

HOST = "0.0.0.0"   # listen on all interfaces

# ── Overnight robustness tuning ─────────────────────────────────────────────

# If no bytes arrive in this long, treat the link as stalled and reconnect.
# Matches the ESP32-side ≤1 s heartbeat with headroom for AP jitter.
RECV_TIMEOUT_S      = 3.0

# A jump larger than this in the ESP32's microsecond timer between consecutive
# CSI rows is considered a data gap (and a CSI_GAP row is inserted).
GAP_THRESHOLD_US    = 500_000        # 500 ms

# How often to fsync the CSV to disk (host-crash durability window).
FSYNC_INTERVAL_S    = 60.0

# How often to rewrite the stats.json companion file.
STATS_INTERVAL_S    = 60.0


# ── Core receive loop ────────────────────────────────────────────────────────

class StallError(Exception):
    """Raised when the TCP recv timeout fires — the link went silent."""


def _recv_lines(conn: socket.socket):
    """
    Generator: yield complete newline-terminated lines received from *conn*.
    Handles partial reads and multi-line chunks transparently.
    Raises StallError if no bytes arrive within the socket's recv timeout.
    """
    buf = b""
    while True:
        # TCP is a stream — recv() can return any number of bytes, including
        # multiple rows or a partial row.  We accumulate into buf and slice
        # out complete lines as they arrive.
        try:
            chunk = conn.recv(4096)
        except socket.timeout as e:
            raise StallError("recv timeout") from e
        if not chunk:
            return          # remote closed the connection
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            # errors="replace" avoids crashes on occasional corrupt bytes
            yield line.decode("utf-8", errors="replace").strip()


# ── Capture session (per-CSV tracking) ──────────────────────────────────────

class CaptureSession:
    """
    Tracks state across many TCP reconnections while writing to one CSV.

    Responsible for emitting CSI_GAP rows when the stream breaks, enforcing
    timestamp monotonicity (catching ESP32 reboots and >500 ms gaps),
    periodically fsyncing the CSV, and rewriting the stats.json summary.
    """

    def __init__(self, save_fd, log_fd, store_path):
        self.save_fd     = save_fd
        self.log_fd      = log_fd
        self.store_path  = store_path
        self.stats_path  = store_path + ".stats.json"

        self.csv_writer          = None    # set once header is written
        self.expected_col_count  = 0
        self.header_written      = False

        # Running state across reconnects.
        self.started_wall        = _dt.datetime.now(_dt.timezone.utc)
        self.started_mono_ns     = time.monotonic_ns()
        self.frames_written      = 0
        self.last_local_ts       = None          # ESP32 µs timer of previous CSI row
        self.last_wall_ns        = None          # host monotonic_ns of previous CSI row
        self.last_row_was_gap    = False
        self.last_fsync_mono     = time.monotonic()
        self.last_stats_mono     = 0.0
        self.last_heartbeat_wall = None

        # Per-gap log — kept bounded at 1000 entries so an all-night stream of
        # brief hiccups can't blow up RAM. Full counts are always correct.
        self.gaps               = []
        self.gap_count          = 0
        self.total_gap_ms       = 0
        self.longest_segment_s  = 0.0
        self._segment_start_ns  = time.monotonic_ns()

    # ── header / writer setup ──────────────────────────────────────────────

    def ensure_header(self, cols):
        """Called once per connection with the ESP32's header row.

        Writes it to the CSV on the first connection; on later connections the
        header is consumed and validated but not re-written, so the output
        file stays a single valid CSV across the whole overnight recording.
        """
        self.expected_col_count = len(cols)
        if self.header_written:
            # Reuse the file handle; header row is already at the top.
            self.csv_writer = csv.writer(self.save_fd)
            return
        self.csv_writer = csv.writer(self.save_fd)
        self.csv_writer.writerow(cols)
        self.save_fd.flush()
        self.header_written = True

    # ── CSI_GAP row emission ───────────────────────────────────────────────

    def emit_gap(self, reason: str, duration_ms: int):
        """Write a CSI_GAP sentinel row with the same column count as CSI_DATA.

        * column 0  ("type")            → "CSI_GAP"
        * column 1  ("id")              → -1
        * column 2  ("mac")             → "00:00:00:00:00:00"
        * column 3  ("rssi")            → gap duration in ms (repurposed)
        * column 4  ("rate")            → short reason code (TIMEOUT, CLOSE, …)
        * column "local_timestamp"      → host wall-µs (always monotonic)
        * column "len"                  → 0
        * column "first_word"           → 0
        * column "data"                 → "[]"
        * remaining metadata columns    → 0
        """
        if self.csv_writer is None or not self.header_written:
            # No header yet — nothing to align to.
            return
        if self.last_row_was_gap:
            # Don't spam consecutive gap rows (e.g. stall → accept → stall).
            return

        n = self.expected_col_count
        row = ["0"] * n
        row[0] = "CSI_GAP"
        row[1] = "-1"
        row[2] = "00:00:00:00:00:00"
        row[3] = str(int(duration_ms))
        row[4] = reason

        # local_timestamp is at a different index depending on the chip
        # variant — resolve it via the column list we recorded.
        cols = (
            DATA_COLUMNS_NAMES_C5C6
            if n == len(DATA_COLUMNS_NAMES_C5C6)
            else DATA_COLUMNS_NAMES
        )
        host_us = time.monotonic_ns() // 1_000
        try:
            ts_idx = cols.index("local_timestamp")
            row[ts_idx] = str(host_us)
        except ValueError:
            pass

        # len / first_word / data — last three columns.
        if n >= 3:
            row[-3] = "0"
            row[-2] = "0"
            row[-1] = "[]"

        self.csv_writer.writerow(row)
        self.save_fd.flush()
        self.last_row_was_gap = True
        self._record_gap(reason, duration_ms)
        # A gap ends the current contiguous segment.
        self._close_segment()

    def _record_gap(self, reason, duration_ms):
        self.gap_count += 1
        self.total_gap_ms += max(0, int(duration_ms))
        entry = {
            "t": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "duration_ms": int(duration_ms),
            "reason": reason,
        }
        if len(self.gaps) < 1000:
            self.gaps.append(entry)

    def _close_segment(self):
        now_ns = time.monotonic_ns()
        segment_s = (now_ns - self._segment_start_ns) / 1e9
        if segment_s > self.longest_segment_s:
            self.longest_segment_s = segment_s
        self._segment_start_ns = now_ns

    # ── CSI row ingestion ──────────────────────────────────────────────────

    def ingest_csi_row(self, csi_row):
        """Validate, gap-check, write one CSI_DATA row. Returns True if kept."""
        if len(csi_row) != self.expected_col_count:
            msg = (f"Column count mismatch: got {len(csi_row)}, "
                   f"expected {self.expected_col_count}")
            print(msg)
            self.log_fd.write(msg + "\n")
            self.log_fd.flush()
            return False

        # Complete-JSON + declared-length checks for the trailing data array.
        try:
            csi_raw = json.loads(csi_row[-1])
        except json.JSONDecodeError:
            print("Incomplete CSI data, skipping row")
            self.log_fd.write("data is incomplete\n")
            self.log_fd.flush()
            return False
        try:
            declared_len = int(csi_row[-3])
        except (ValueError, IndexError):
            declared_len = len(csi_raw)
        if declared_len != len(csi_raw):
            msg = (f"CSI length mismatch: header={declared_len}, "
                   f"actual={len(csi_raw)}")
            print(msg)
            self.log_fd.write(msg + "\n")
            self.log_fd.flush()
            return False

        # Pull local_timestamp from the row via the header layout.
        cols = (
            DATA_COLUMNS_NAMES_C5C6
            if self.expected_col_count == len(DATA_COLUMNS_NAMES_C5C6)
            else DATA_COLUMNS_NAMES
        )
        try:
            local_ts = int(csi_row[cols.index("local_timestamp")])
        except (ValueError, IndexError):
            local_ts = None

        # Monotonicity / duplicate / reboot checks.
        if local_ts is not None and self.last_local_ts is not None:
            if local_ts == self.last_local_ts:
                # Exact duplicate — silently drop, no gap marker.
                return False
            if local_ts < self.last_local_ts:
                # ESP32 rebooted mid-stream — timer resets to near zero.
                self.emit_gap("REBOOT", 0)
            else:
                dt_us = local_ts - self.last_local_ts
                if dt_us > GAP_THRESHOLD_US:
                    self.emit_gap("MONOTONIC", dt_us // 1000)

        self.csv_writer.writerow(csi_row)
        self.save_fd.flush()
        self.last_row_was_gap = False
        self.frames_written += 1
        self.last_local_ts = local_ts if local_ts is not None else self.last_local_ts
        self.last_wall_ns  = time.monotonic_ns()
        self._periodic_maintenance()
        return True

    # ── heartbeat / periodic I/O ───────────────────────────────────────────

    def handle_heartbeat(self, line: str):
        """Log HEARTBEAT lines to the sidecar log; never write to the CSV."""
        self.last_heartbeat_wall = _dt.datetime.now(_dt.timezone.utc)
        self.log_fd.write(line + "\n")
        self.log_fd.flush()

    def _periodic_maintenance(self):
        now = time.monotonic()
        if now - self.last_fsync_mono >= FSYNC_INTERVAL_S:
            try:
                os.fsync(self.save_fd.fileno())
            except OSError:
                pass
            self.last_fsync_mono = now
        if now - self.last_stats_mono >= STATS_INTERVAL_S:
            self.write_stats()
            self.last_stats_mono = now

    # ── stats.json morning report ──────────────────────────────────────────

    def write_stats(self):
        # Include the active segment in the "longest so far" figure.
        live_segment_s = (time.monotonic_ns() - self._segment_start_ns) / 1e9
        longest = max(self.longest_segment_s, live_segment_s)
        stats = {
            "started_utc": self.started_wall.isoformat(timespec="seconds"),
            "now_utc": _dt.datetime.now(_dt.timezone.utc)
                          .isoformat(timespec="seconds"),
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


# ── TCP → CSV glue ──────────────────────────────────────────────────────────

def consume_connection(conn: socket.socket, session: CaptureSession):
    """Drive one TCP connection's worth of lines through the session.

    Raises StallError if the link stalls so the server can emit a TIMEOUT gap;
    any other error bubbles up to the server loop.
    """
    lines = _recv_lines(conn)

    # Header first.
    header_line = next(lines, None)
    if header_line is None:
        print("Connection closed before header was received.")
        return
    cols = next(csv.reader(StringIO(header_line)), [])
    if len(cols) not in VALID_COLUMN_COUNTS:
        print(f"Unexpected header ({len(cols)} columns): {header_line!r}")
        session.log_fd.write(f"bad header: {header_line}\n")
        session.log_fd.flush()
        return
    session.ensure_header(cols)

    # Data rows.
    for line in lines:
        if not line:
            continue
        if line.startswith("HEARTBEAT,"):
            session.handle_heartbeat(line)
            continue
        if "CSI_DATA" not in line:
            # ESP_LOG leakage or keep-alive text — log, don't store.
            session.log_fd.write(line + "\n")
            session.log_fd.flush()
            continue
        idx = line.find("CSI_DATA")
        if idx > 0:
            line = line[idx:]
        try:
            csi_row = next(csv.reader(StringIO(line)))
        except StopIteration:
            continue
        session.ingest_csi_row(csi_row)


def run_server(port: int, store_path: str, log_path: str):
    """
    Listen for incoming ESP32 connections and stream CSI rows to *store_path*.
    Accepts one connection at a time; when a connection drops, waits for the
    next one (the ESP32 reconnect task will re-dial automatically) and emits
    a single CSI_GAP row to mark the break.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SO_REUSEADDR lets us restart the server immediately after a crash
    # without waiting for the OS TIME_WAIT timeout (~60 s).
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)   # backlog of 1 — we only handle one ESP32 at a time
    print(f"Listening on {HOST}:{port}  →  writing to {store_path}")

    # newline="" keeps csv.writer from injecting an extra \r on Windows; "w"
    # truncates on start so each invocation is one overnight recording.
    with open(store_path, "w", newline="") as save_fd, \
         open(log_path,   "w")             as log_fd:

        session = CaptureSession(save_fd, log_fd, store_path)
        session.write_stats()   # create the file immediately so a watcher sees it

        # Pending gap: set when a connection drops, emitted on the *next*
        # accept so the CSI_GAP row carries the real outage duration and
        # original reason (TIMEOUT / CLOSE / OSERR).
        pending_reason         = None
        pending_start_mono_ns  = None

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

            # Emit the pending gap row — one per outage, with real duration.
            if pending_reason is not None and session.header_written:
                gap_ms = (time.monotonic_ns() - pending_start_mono_ns) // 1_000_000
                session.emit_gap(pending_reason, gap_ms)
            pending_reason = None
            pending_start_mono_ns = None

            reason = "CLOSE"
            try:
                consume_connection(conn, session)
            except StallError:
                print(f"Recv stall > {RECV_TIMEOUT_S}s — closing socket.")
                reason = "TIMEOUT"
            except OSError as e:
                print(f"Connection error: {e}")
                reason = "OSERR"
            except StopIteration:
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
                print("Connection closed.")

            pending_reason        = reason
            pending_start_mono_ns = time.monotonic_ns()
            session.write_stats()


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.version_info < (3, 6):
        print("Python >= 3.6 required")
        sys.exit(1)

    import argparse

    parser = argparse.ArgumentParser(
        description="TCP server: receive CSI data from ESP32 and save to CSV"
    )
    parser.add_argument(
        "-p", "--port",
        dest="port", type=int, default=3490,
        help="TCP port to listen on (default: 3490)",
    )
    parser.add_argument(
        "-s", "--store",
        dest="store_file", default="./csi_data.csv",
        help="Output CSV file (default: ./csi_data.csv)",
    )
    parser.add_argument(
        "-l", "--log",
        dest="log_file", default="./csi_data_log.txt",
        help="Log file for invalid rows (default: ./csi_data_log.txt)",
    )
    args = parser.parse_args()

    run_server(args.port, args.store_file, args.log_file)
