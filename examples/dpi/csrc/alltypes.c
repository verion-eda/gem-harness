// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
/*
 * DPI-C implementations for the all-types example.
 *
 * Identity functions pass through the raw bit pattern.
 * Scale functions interpret bits as float/double, multiply by 2, and return.
 *
 * Signature: void harness_dpi_<name>(const uint32_t* in, size_t n_in,
 *                                   uint32_t* out, size_t n_out);
 */

#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

/* identity_byte: 1 word in (8-bit), 1 word out (8-bit) */
void harness_dpi_identity_byte(const uint32_t* in, size_t n_in,
                              uint32_t* out, size_t n_out) {
    assert(n_in == 1 && n_out == 1);
    out[0] = in[0] & 0xFF;
}

/* identity_shortint: 1 word in (16-bit), 1 word out (16-bit) */
void harness_dpi_identity_shortint(const uint32_t* in, size_t n_in,
                                  uint32_t* out, size_t n_out) {
    assert(n_in == 1 && n_out == 1);
    out[0] = in[0] & 0xFFFF;
}

/* identity_int: 1 word in (32-bit), 1 word out (32-bit) */
void harness_dpi_identity_int(const uint32_t* in, size_t n_in,
                             uint32_t* out, size_t n_out) {
    assert(n_in == 1 && n_out == 1);
    out[0] = in[0];
}

/* identity_longint: 2 words in (64-bit LE), 2 words out (64-bit LE) */
void harness_dpi_identity_longint(const uint32_t* in, size_t n_in,
                                 uint32_t* out, size_t n_out) {
    assert(n_in == 2 && n_out == 2);
    out[0] = in[0];
    out[1] = in[1];
}

/* identity_chandle: 2 words in (64-bit LE void*), 2 words out */
void harness_dpi_identity_chandle(const uint32_t* in, size_t n_in,
                                 uint32_t* out, size_t n_out) {
    assert(n_in == 2 && n_out == 2);
    out[0] = in[0];
    out[1] = in[1];
}

/* identity_string: 2 words in (64-bit LE char*), 2 words out */
void harness_dpi_identity_string(const uint32_t* in, size_t n_in,
                                uint32_t* out, size_t n_out) {
    assert(n_in == 2 && n_out == 2);
    out[0] = in[0];
    out[1] = in[1];
}

/* scale_real: 2 words in (64-bit LE double), 2 words out — doubles the value */
void harness_dpi_scale_real(const uint32_t* in, size_t n_in,
                           uint32_t* out, size_t n_out) {
    assert(n_in == 2 && n_out == 2);
    double val;
    memcpy(&val, in, sizeof(double));
    val *= 2.0;
    memcpy(out, &val, sizeof(double));
}

/* scale_shortreal: 1 word in (32-bit float), 1 word out — doubles the value */
void harness_dpi_scale_shortreal(const uint32_t* in, size_t n_in,
                                uint32_t* out, size_t n_out) {
    assert(n_in == 1 && n_out == 1);
    float val;
    memcpy(&val, in, sizeof(float));
    val *= 2.0f;
    memcpy(out, &val, sizeof(float));
}

/* pack_wide: 3 words in (byte, shortint, int), 2 words out (longint LE) */
void harness_dpi_pack_wide(const uint32_t* in, size_t n_in,
                           uint32_t* out, size_t n_out) {
    assert(n_in == 3 && n_out == 2);
    uint64_t a = in[0] & 0xFF;
    uint64_t b = in[1] & 0xFFFF;
    uint64_t c = in[2];
    uint64_t result = a | (b << 8) | (c << 24);
    out[0] = (uint32_t)result;
    out[1] = (uint32_t)(result >> 32);
}

/* narrow_add: 2 words in (shortint, shortint), 1 word out (byte) */
void harness_dpi_narrow_add(const uint32_t* in, size_t n_in,
                            uint32_t* out, size_t n_out) {
    assert(n_in == 2 && n_out == 1);
    uint16_t x = (uint16_t)(in[0] & 0xFFFF);
    uint16_t y = (uint16_t)(in[1] & 0xFFFF);
    out[0] = (uint32_t)((x + y) & 0xFF);
}
