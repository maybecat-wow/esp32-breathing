/*
 * plug_relay — relay GPIO drive + fail-closed watchdog (the safety core).
 *
 * Invariants enforced here (see SPEC.md / csi-app FIRMWARE.md):
 *   V49 fail-closed boot   — GPIO forced OFF in plug_relay_init(), no NVS.
 *   V50 watchdog           — relay forced OFF if no /relay|/heartbeat within
 *                            CONFIG_PLUG_WATCHDOG_S.
 *   V51 atomic toggle      — GPIO write + read-back + watchdog reset under one
 *                            mutex; plug_relay_set returns the read-back state.
 *   V52 heartbeat-only     — plug_relay_heartbeat resets the watchdog only.
 *
 * All state is guarded by a single mutex shared with the watchdog timer.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/* Configure the relay GPIO, drive it OFF, and start the watchdog timer.
   MUST be called before any network service comes up. */
void plug_relay_init(void);

/* Atomically drive the relay to `on`, reset the watchdog, and return the
   actual post-toggle state read back from the GPIO (V51). `ts_ms_out`, if
   non-NULL, receives the ms-since-boot timestamp of the toggle. */
bool plug_relay_set(bool on, int64_t *ts_ms_out);

/* Reset the watchdog only. MUST NOT change relay state (V52). */
void plug_relay_heartbeat(void);

/* Current relay state, read back from the GPIO. */
bool plug_relay_get(void);

/* Milliseconds since the last /relay or /heartbeat. */
int64_t plug_relay_last_cmd_age_ms(void);

/* Milliseconds until the watchdog fires, clamped to [0, WATCHDOG_S*1000]. */
int64_t plug_relay_watchdog_remaining_ms(void);
