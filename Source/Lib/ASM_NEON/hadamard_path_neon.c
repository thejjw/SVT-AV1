/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "coding_loop.h"
#include "definitions.h"
#include "mem_neon.h"
#include "transpose_neon.h"

static inline void hadamard8x8_one_pass(int16x8_t *a) {
    const int16x8_t b0 = vaddq_s16(a[0], a[1]);
    const int16x8_t b1 = vsubq_s16(a[0], a[1]);
    const int16x8_t b2 = vaddq_s16(a[2], a[3]);
    const int16x8_t b3 = vsubq_s16(a[2], a[3]);
    const int16x8_t b4 = vaddq_s16(a[4], a[5]);
    const int16x8_t b5 = vsubq_s16(a[4], a[5]);
    const int16x8_t b6 = vaddq_s16(a[6], a[7]);
    const int16x8_t b7 = vsubq_s16(a[6], a[7]);

    const int16x8_t c0 = vaddq_s16(b0, b2);
    const int16x8_t c1 = vaddq_s16(b1, b3);
    const int16x8_t c2 = vsubq_s16(b0, b2);
    const int16x8_t c3 = vsubq_s16(b1, b3);
    const int16x8_t c4 = vaddq_s16(b4, b6);
    const int16x8_t c5 = vaddq_s16(b5, b7);
    const int16x8_t c6 = vsubq_s16(b4, b6);
    const int16x8_t c7 = vsubq_s16(b5, b7);

    a[0] = vaddq_s16(c0, c4);
    a[1] = vsubq_s16(c2, c6);
    a[2] = vsubq_s16(c0, c4);
    a[3] = vaddq_s16(c2, c6);
    a[4] = vaddq_s16(c3, c7);
    a[5] = vsubq_s16(c3, c7);
    a[6] = vsubq_s16(c1, c5);
    a[7] = vaddq_s16(c1, c5);
}

void svt_aom_hadamard_8x8_neon(const int16_t *src_diff, ptrdiff_t src_stride, int32_t *coeff) {
    int16x8_t a[8];

    a[0] = vld1q_s16(src_diff);
    a[1] = vld1q_s16(src_diff + src_stride);
    a[2] = vld1q_s16(src_diff + 2 * src_stride);
    a[3] = vld1q_s16(src_diff + 3 * src_stride);
    a[4] = vld1q_s16(src_diff + 4 * src_stride);
    a[5] = vld1q_s16(src_diff + 5 * src_stride);
    a[6] = vld1q_s16(src_diff + 6 * src_stride);
    a[7] = vld1q_s16(src_diff + 7 * src_stride);

    hadamard8x8_one_pass(a);
    transpose_elems_inplace_s16_8x8(a + 0, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6, a + 7);
    hadamard8x8_one_pass(a);

    store_s16q_to_tran_low(coeff + 0, a[0]);
    store_s16q_to_tran_low(coeff + 8, a[1]);
    store_s16q_to_tran_low(coeff + 16, a[2]);
    store_s16q_to_tran_low(coeff + 24, a[3]);
    store_s16q_to_tran_low(coeff + 32, a[4]);
    store_s16q_to_tran_low(coeff + 40, a[5]);
    store_s16q_to_tran_low(coeff + 48, a[6]);
    store_s16q_to_tran_low(coeff + 56, a[7]);
}

void svt_aom_hadamard_16x16_neon(const int16_t *src_diff, ptrdiff_t src_stride, tran_low_t *coeff) {
    /* Rearrange 16x16 to 8x32 and remove stride.
     * Top left first. */
    svt_aom_hadamard_8x8_neon(src_diff + 0 + 0 * src_stride, src_stride, coeff + 0);
    /* Top right. */
    svt_aom_hadamard_8x8_neon(src_diff + 8 + 0 * src_stride, src_stride, coeff + 64);
    /* Bottom left. */
    svt_aom_hadamard_8x8_neon(src_diff + 0 + 8 * src_stride, src_stride, coeff + 128);
    /* Bottom right. */
    svt_aom_hadamard_8x8_neon(src_diff + 8 + 8 * src_stride, src_stride, coeff + 192);

    // Each iteration of the loop operates on entire rows (16 samples each)
    // because we need to swap the second and third quarters of every row in the
    // output to match AVX2 output (i.e., aom_hadamard_16x16_avx2). See the for
    // loop at the end of aom_hadamard_16x16_c.
    for (int i = 0; i < 64; i += 16) {
        const int32x4_t a00 = vld1q_s32(coeff + 0);
        const int32x4_t a01 = vld1q_s32(coeff + 64);
        const int32x4_t a02 = vld1q_s32(coeff + 128);
        const int32x4_t a03 = vld1q_s32(coeff + 192);

        const int32x4_t b00 = vhaddq_s32(a00, a01);
        const int32x4_t b01 = vhsubq_s32(a00, a01);
        const int32x4_t b02 = vhaddq_s32(a02, a03);
        const int32x4_t b03 = vhsubq_s32(a02, a03);

        const int32x4_t c00 = vaddq_s32(b00, b02);
        const int32x4_t c01 = vaddq_s32(b01, b03);
        const int32x4_t c02 = vsubq_s32(b00, b02);
        const int32x4_t c03 = vsubq_s32(b01, b03);

        const int32x4_t a10 = vld1q_s32(coeff + 4 + 0);
        const int32x4_t a11 = vld1q_s32(coeff + 4 + 64);
        const int32x4_t a12 = vld1q_s32(coeff + 4 + 128);
        const int32x4_t a13 = vld1q_s32(coeff + 4 + 192);

        const int32x4_t b10 = vhaddq_s32(a10, a11);
        const int32x4_t b11 = vhsubq_s32(a10, a11);
        const int32x4_t b12 = vhaddq_s32(a12, a13);
        const int32x4_t b13 = vhsubq_s32(a12, a13);

        const int32x4_t c10 = vaddq_s32(b10, b12);
        const int32x4_t c11 = vaddq_s32(b11, b13);
        const int32x4_t c12 = vsubq_s32(b10, b12);
        const int32x4_t c13 = vsubq_s32(b11, b13);

        const int32x4_t a20 = vld1q_s32(coeff + 8 + 0);
        const int32x4_t a21 = vld1q_s32(coeff + 8 + 64);
        const int32x4_t a22 = vld1q_s32(coeff + 8 + 128);
        const int32x4_t a23 = vld1q_s32(coeff + 8 + 192);

        const int32x4_t b20 = vhaddq_s32(a20, a21);
        const int32x4_t b21 = vhsubq_s32(a20, a21);
        const int32x4_t b22 = vhaddq_s32(a22, a23);
        const int32x4_t b23 = vhsubq_s32(a22, a23);

        const int32x4_t c20 = vaddq_s32(b20, b22);
        const int32x4_t c21 = vaddq_s32(b21, b23);
        const int32x4_t c22 = vsubq_s32(b20, b22);
        const int32x4_t c23 = vsubq_s32(b21, b23);

        const int32x4_t a30 = vld1q_s32(coeff + 12 + 0);
        const int32x4_t a31 = vld1q_s32(coeff + 12 + 64);
        const int32x4_t a32 = vld1q_s32(coeff + 12 + 128);
        const int32x4_t a33 = vld1q_s32(coeff + 12 + 192);

        const int32x4_t b30 = vhaddq_s32(a30, a31);
        const int32x4_t b31 = vhsubq_s32(a30, a31);
        const int32x4_t b32 = vhaddq_s32(a32, a33);
        const int32x4_t b33 = vhsubq_s32(a32, a33);

        const int32x4_t c30 = vaddq_s32(b30, b32);
        const int32x4_t c31 = vaddq_s32(b31, b33);
        const int32x4_t c32 = vsubq_s32(b30, b32);
        const int32x4_t c33 = vsubq_s32(b31, b33);

        vst1q_s32(coeff + 0 + 0, c00);
        vst1q_s32(coeff + 0 + 4, c20);
        vst1q_s32(coeff + 0 + 8, c10);
        vst1q_s32(coeff + 0 + 12, c30);

        vst1q_s32(coeff + 64 + 0, c01);
        vst1q_s32(coeff + 64 + 4, c21);
        vst1q_s32(coeff + 64 + 8, c11);
        vst1q_s32(coeff + 64 + 12, c31);

        vst1q_s32(coeff + 128 + 0, c02);
        vst1q_s32(coeff + 128 + 4, c22);
        vst1q_s32(coeff + 128 + 8, c12);
        vst1q_s32(coeff + 128 + 12, c32);

        vst1q_s32(coeff + 192 + 0, c03);
        vst1q_s32(coeff + 192 + 4, c23);
        vst1q_s32(coeff + 192 + 8, c13);
        vst1q_s32(coeff + 192 + 12, c33);

        coeff += 16;
    }
}

void svt_aom_hadamard_32x32_neon(const int16_t *src_diff, ptrdiff_t src_stride, tran_low_t *coeff) {
    /* Top left first. */
    svt_aom_hadamard_16x16_neon(src_diff + 0 + 0 * src_stride, src_stride, coeff + 0);
    /* Top right. */
    svt_aom_hadamard_16x16_neon(src_diff + 16 + 0 * src_stride, src_stride, coeff + 256);
    /* Bottom left. */
    svt_aom_hadamard_16x16_neon(src_diff + 0 + 16 * src_stride, src_stride, coeff + 512);
    /* Bottom right. */
    svt_aom_hadamard_16x16_neon(src_diff + 16 + 16 * src_stride, src_stride, coeff + 768);

    for (int i = 0; i < 256; i += 4) {
        const int32x4_t a0 = vld1q_s32(coeff);
        const int32x4_t a1 = vld1q_s32(coeff + 256);
        const int32x4_t a2 = vld1q_s32(coeff + 512);
        const int32x4_t a3 = vld1q_s32(coeff + 768);

        const int32x4_t b0 = vshrq_n_s32(vaddq_s32(a0, a1), 2);
        const int32x4_t b1 = vshrq_n_s32(vsubq_s32(a0, a1), 2);
        const int32x4_t b2 = vshrq_n_s32(vaddq_s32(a2, a3), 2);
        const int32x4_t b3 = vshrq_n_s32(vsubq_s32(a2, a3), 2);

        const int32x4_t c0 = vaddq_s32(b0, b2);
        const int32x4_t c1 = vaddq_s32(b1, b3);
        const int32x4_t c2 = vsubq_s32(b0, b2);
        const int32x4_t c3 = vsubq_s32(b1, b3);

        vst1q_s32(coeff + 0, c0);
        vst1q_s32(coeff + 256, c1);
        vst1q_s32(coeff + 512, c2);
        vst1q_s32(coeff + 768, c3);

        coeff += 4;
    }
}

uint32_t hadamard_path_neon(Buf2D residualBuf, Buf2D coeffBuf, Buf2D inputBuf, Buf2D predBuf, BlockSize bsize) {
    assert(residualBuf.buf != NULL && residualBuf.buf0 == NULL && residualBuf.width == 0 && residualBuf.height == 0 &&
           residualBuf.stride != 0);
    assert(coeffBuf.buf != NULL && coeffBuf.buf0 == NULL && coeffBuf.width == 0 && coeffBuf.height == 0 &&
           coeffBuf.stride == block_size_wide[bsize]);
    assert(inputBuf.buf != NULL && inputBuf.buf0 == NULL && inputBuf.width == 0 && inputBuf.height == 0 &&
           inputBuf.stride != 0);
    assert(predBuf.buf != NULL && predBuf.buf0 == NULL && predBuf.width == 0 && predBuf.height == 0 &&
           predBuf.stride != 0);
    uint32_t input_idx, pred_idx, res_idx;

    uint32_t satd_cost = 0;

    const TxSize tx_size = AOMMIN(TX_32X32, eb_max_txsize_lookup[bsize]);

    const int stepr = eb_tx_size_high_unit[tx_size];
    const int stepc = eb_tx_size_wide_unit[tx_size];
    const int txbw  = tx_size_wide[tx_size];
    const int txbh  = tx_size_high[tx_size];

    const int max_blocks_wide = block_size_wide[bsize] >> MI_SIZE_LOG2;
    const int max_blocks_high = block_size_wide[bsize] >> MI_SIZE_LOG2;
    int       row, col;

    for (row = 0; row < max_blocks_high; row += stepr) {
        for (col = 0; col < max_blocks_wide; col += stepc) {
            input_idx = ((row * inputBuf.stride) + col) << 2;
            pred_idx  = ((row * predBuf.stride) + col) << 2;
            res_idx   = 0;

            svt_aom_residual_kernel(inputBuf.buf,
                                    input_idx,
                                    inputBuf.stride,
                                    predBuf.buf,
                                    pred_idx,
                                    predBuf.stride,
                                    (int16_t *)residualBuf.buf,
                                    res_idx,
                                    residualBuf.stride,
                                    0, // inputBuf.buf and predBuf.buf 8-bit
                                    txbw,
                                    txbh);

            switch (tx_size) {
            case TX_4X4:
                svt_aom_hadamard_4x4((int16_t *)residualBuf.buf, residualBuf.stride, &(((int32_t *)coeffBuf.buf)[0]));
                break;

            case TX_8X8:
                svt_aom_hadamard_8x8_neon(
                    (int16_t *)residualBuf.buf, residualBuf.stride, &(((int32_t *)coeffBuf.buf)[0]));
                break;

            case TX_16X16:
                svt_aom_hadamard_16x16_neon(
                    (int16_t *)residualBuf.buf, residualBuf.stride, &(((int32_t *)coeffBuf.buf)[0]));
                break;

            case TX_32X32:
                svt_aom_hadamard_32x32_neon(
                    (int16_t *)residualBuf.buf, residualBuf.stride, &(((int32_t *)coeffBuf.buf)[0]));
                break;

            default: assert(0);
            }
            satd_cost += svt_aom_satd_neon(&(((int32_t *)coeffBuf.buf)[0]), tx_size_2d[tx_size]);
        }
    }
    return (satd_cost);
}
