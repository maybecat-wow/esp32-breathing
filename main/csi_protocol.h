/*
 * SPDX-FileCopyrightText: 2026
 * SPDX-License-Identifier: Apache-2.0
 *
 * Binary wire protocol between ESP32 firmware and host capture.
 *
 * All multi-byte integers are little-endian (native to Xtensa LX6 / LX7).
 * Every message:
 *
 *     uint8_t  type;
 *     uint16_t length;       // payload length, NOT including this 3-byte header
 *     uint8_t  payload[length];
 *
 * Mirror in csi_protocol.py — keep both sides in sync.
 */
#pragma once

#include <stdint.h>
#include <assert.h>

#define MSG_SESSION_INFO  0x01
#define MSG_CSI_FRAME     0x02
#define MSG_HEARTBEAT     0x03

#define MAX_PAYLOAD_BYTES 4096

/* 3-byte message header. */
typedef struct __attribute__((packed)) {
    uint8_t  type;
    uint16_t length;
} csi_msg_header_t;
_Static_assert(sizeof(csi_msg_header_t) == 3, "csi_msg_header_t must be 3 bytes");

/* SESSION_INFO payload (26 bytes). */
typedef struct __attribute__((packed)) {
    uint8_t  chip_id;          /* 1=ESP32 classic, 2=ESP32-S3 */
    uint8_t  csi_format;       /* 0=legacy HT LLTF */
    uint16_t csi_bytes;        /* 128 for LLTF 64-subcarrier */
    uint8_t  mac[6];
    uint8_t  channel;
    uint8_t  reserved;
    uint16_t sample_rate_hz;
    uint32_t boot_id;          /* random per boot; lets host distinguish
                                  reboot from TCP reconnect */
    uint64_t esp_time_us;
} csi_session_info_t;
_Static_assert(sizeof(csi_session_info_t) == 26, "csi_session_info_t size drift");

/* CSI_FRAME meta block (14 bytes). The csi byte payload (length = `len`)
 * follows this struct in the message. */
typedef struct __attribute__((packed)) {
    uint32_t local_timestamp_us;
    uint32_t seq;
    int8_t   rssi;
    int8_t   noise_floor;
    uint8_t  rate;
    uint8_t  first_word_invalid;
    uint16_t len;
} csi_frame_meta_t;
_Static_assert(sizeof(csi_frame_meta_t) == 14, "csi_frame_meta_t size drift");

/* HEARTBEAT payload (19 bytes). */
typedef struct __attribute__((packed)) {
    uint64_t esp_time_us;
    int8_t   rssi;
    uint8_t  channel;
    uint32_t uptime_s;
    uint32_t reconnect_count;
    uint8_t  last_disc_reason;
} csi_heartbeat_t;
_Static_assert(sizeof(csi_heartbeat_t) == 19, "csi_heartbeat_t size drift");
