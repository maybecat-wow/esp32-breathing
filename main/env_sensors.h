/*
 * SPDX-FileCopyrightText: 2026
 * SPDX-License-Identifier: Apache-2.0
 *
 * Wired environment sensors: LDR (CdS, ADC1) + AM2302 (DHT22, RMT).
 * Sampled in a dedicated task and emitted as MSG_ENV (see csi_protocol.h).
 * Entirely independent of the CSI capture path.
 */
#pragma once

#include <stdbool.h>
#include "csi_protocol.h"

/* Configure ADC1 (LDR) and the RMT RX channel (AM2302). Safe to call once at
 * boot. No-op for any sensor not enabled in menuconfig. */
void env_sensors_init(void);

/* SESSION_INFO.sensor_flags bitmap for the compiled-in sensors (SENSOR_FLAG_*). */
uint8_t env_sensor_flags(void);

/* Fill *out with a fresh reading: LDR is read every call; the AM2302 is read
 * from hardware at most once per 2 s (sensor minimum) and served from cache in
 * between. Returns true if at least one sensor produced data. */
bool env_read(csi_env_t *out);
