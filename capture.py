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
"""

import sys
import csv
import json
import socket
import argparse
from io import StringIO

# ── Default column layouts (kept for column-count validation) ───────────────

DATA_COLUMNS_NAMES_C5C6 = [
    "type", "id", "mac", "rssi", "rate",
    "noise_floor", "fft_gain", "agc_gain",
    "channel", "local_timestamp", "sig_len", "rx_state",
    "len", "first_word", "data",
]

DATA_COLUMNS_NAMES = [
    "type", "id", "mac", "rssi", "rate",
    "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding",
    "aggregation", "stbc", "fec_coding", "sgi", "noise_floor",
    "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant",
    "sig_len", "rx_state", "len", "first_word", "data",
]

VALID_COLUMN_COUNTS = {len(DATA_COLUMNS_NAMES), len(DATA_COLUMNS_NAMES_C5C6)}

HOST = "0.0.0.0"   # listen on all interfaces


# ── Core receive loop ────────────────────────────────────────────────────────

def _recv_lines(conn: socket.socket):
    """
    Generator: yield complete newline-terminated lines received from *conn*.
    Handles partial reads and multi-line chunks transparently.
    """
    buf = b""
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            return          # remote closed the connection
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode("utf-8", errors="replace").strip()


def csi_data_read_tcp(conn: socket.socket, csv_writer, log_fd, expected_cols: list):
    """
    Read CSI rows from an accepted TCP connection and write them to *csv_writer*.

    *expected_cols* is the column list we already validated from the header row.
    """
    expected_count = len(expected_cols)

    for line in _recv_lines(conn):
        if not line:
            continue

        # Non-CSI lines go to the log
        if "CSI_DATA" not in line:
            log_fd.write(line + "\n")
            log_fd.flush()
            continue

        # Strip any leading garbage before the first field
        idx = line.find("CSI_DATA")
        if idx > 0:
            line = line[idx:]

        reader = csv.reader(StringIO(line))
        try:
            csi_row = next(reader)
        except StopIteration:
            continue

        if len(csi_row) != expected_count:
            msg = (f"Column count mismatch: got {len(csi_row)}, "
                   f"expected {expected_count}\n")
            print(msg.strip())
            log_fd.write(msg)
            log_fd.write(line + "\n")
            log_fd.flush()
            continue

        # Validate the CSI data array (last field) is complete JSON
        try:
            csi_raw = json.loads(csi_row[-1])
        except json.JSONDecodeError:
            print("Incomplete CSI data, skipping row")
            log_fd.write("data is incomplete\n")
            log_fd.write(line + "\n")
            log_fd.flush()
            continue

        # Validate declared length matches actual element count
        try:
            declared_len = int(csi_row[-3])
        except (ValueError, IndexError):
            declared_len = len(csi_raw)

        if declared_len != len(csi_raw):
            msg = (f"CSI length mismatch: header={declared_len}, "
                   f"actual={len(csi_raw)}\n")
            print(msg.strip())
            log_fd.write(msg)
            log_fd.write(line + "\n")
            log_fd.flush()
            continue

        csv_writer.writerow(csi_row)


def run_server(port: int, store_path: str, log_path: str):
    """
    Listen for incoming ESP32 connections and stream CSI rows to *store_path*.
    Accepts one connection at a time; when a connection drops, waits for the
    next one (the ESP32 reconnect task will re-dial automatically).
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(1)
    print(f"Listening on {HOST}:{port}  →  writing to {store_path}")

    header_written = False

    with open(store_path, "w", newline="") as save_fd, \
         open(log_path,   "w")             as log_fd:

        while True:
            print("Waiting for ESP32 connection …")
            conn, addr = srv.accept()
            print(f"Connected: {addr[0]}:{addr[1]}")

            try:
                # ── Read and validate the CSV header sent by the ESP32 ──
                lines_gen = _recv_lines(conn)
                header_line = next(lines_gen, None)

                if header_line is None:
                    print("Connection closed before header was received.")
                    conn.close()
                    continue

                reader = csv.reader(StringIO(header_line))
                cols = next(reader, [])

                if len(cols) not in VALID_COLUMN_COUNTS:
                    print(f"Unexpected header ({len(cols)} columns): {header_line!r}")
                    log_fd.write(f"bad header: {header_line}\n")
                    conn.close()
                    continue

                # Write the CSV header only once (so we can append across
                # reconnections without duplicating the header row)
                if not header_written:
                    writer = csv.writer(save_fd)
                    writer.writerow(cols)
                    save_fd.flush()
                    header_written = True
                    csv_writer = writer
                else:
                    csv_writer = csv.writer(save_fd)

                # ── Stream data rows until disconnected ──
                # We already consumed the first line via the generator, so
                # feed the remaining lines from the same generator.
                # Rebuild a combined generator from the leftover lines:
                def _rows_after_header(first_gen):
                    yield from first_gen

                csi_data_read_tcp(conn, csv_writer, log_fd, cols)

            except (OSError, StopIteration) as e:
                print(f"Connection error: {e}")
            finally:
                conn.close()
                print("Connection closed.")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.version_info < (3, 6):
        print("Python >= 3.6 required")
        sys.exit(1)

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
