/*
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef COMPUTE_SAD_NEON_H
#define COMPUTE_SAD_NEON_H

#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "mem_neon.h"
#include "sum_neon.h"
#include "utility.h"

#if __GNUC__
#define svt_ctzll(id, x) id = (unsigned long)__builtin_ctzll(x)
#elif defined(_MSC_VER)
#include <intrin.h>

#define svt_ctzll(id, x) _BitScanForward64(&id, x)
#endif

/* Find the position of the first occurrence of 'value' in the vector 'x'.
 * Returns the position (index) of the first occurrence of 'value' in the vector 'x'. */
static inline uint16_t findposq_u32(uint32x4_t x, uint32_t value) {
    uint32x4_t val_mask = vdupq_n_u32(value);

    /* Pack the information in the lower 64 bits of the register by considering only alternate
     * 16-bit lanes. */
    uint16x4_t is_one = vmovn_u32(vceqq_u32(x, val_mask));

    /* Get the lower 64 bits from the 128-bit register. */
    uint64_t idx = vget_lane_u64(vreinterpret_u64_u16(is_one), 0);

    /* Calculate the position as an index, dividing by 16 to account for 16-bit lanes. */
    uint64_t res;
    svt_ctzll(res, idx);
    return res >> 4;
}

static inline void update_best_sad_u32(uint32x4_t sad4, uint64_t* best_sad, int16_t* x_search_center,
                                       int16_t* y_search_center, int16_t x_search_index, int16_t y_search_index) {
    /* Find the minimum SAD value out of the 4 search spaces. */
    uint64_t temp_sad = vminvq_u32(sad4);

    if (temp_sad < *best_sad) {
        *best_sad        = temp_sad;
        *x_search_center = (int16_t)(x_search_index + findposq_u32(sad4, temp_sad));
        *y_search_center = y_search_index;
    }
}

/* Find the position of the first occurrence of 'value' in the vector 'x'.
 * Returns the position (index) of the first occurrence of 'value' in the vector 'x'. */
static inline uint16_t findposq_u16(uint16x8_t x, uint16_t value) {
    uint16x8_t val_mask = vdupq_n_u16(value);

    /* Pack the information in the lower 64 bits of the register by considering only alternate
     * 8-bit lanes. */
    uint8x8_t is_one = vmovn_u16(vceqq_u16(x, val_mask));

    /* Get the lower 64 bits from the 128-bit register. */
    uint64_t idx = vget_lane_u64(vreinterpret_u64_u8(is_one), 0);

    /* Calculate the position as an index, dividing by 8 to account for 8-bit lanes. */
    uint64_t res;
    svt_ctzll(res, idx);
    return res >> 3;
}

static inline void update_best_sad_u16(uint16x8_t sad8, uint64_t* best_sad, int16_t* x_search_center,
                                       int16_t* y_search_center, int16_t x_search_index, int16_t y_search_index) {
    /* Find the minimum SAD value out of the 8 search spaces. */
    uint64_t temp_sad = vminvq_u16(sad8);

    if (temp_sad < *best_sad) {
        *best_sad        = temp_sad;
        *x_search_center = (int16_t)(x_search_index + findposq_u16(sad8, temp_sad));
        *y_search_center = y_search_index;
    }
}

static inline void update_best_sad(uint64_t temp_sad, uint64_t* best_sad, int16_t* x_search_center,
                                   int16_t* y_search_center, int16_t x_search_index, int16_t y_search_index) {
    if (temp_sad < *best_sad) {
        *best_sad        = temp_sad;
        *x_search_center = x_search_index;
        *y_search_center = y_search_index;
    }
}

/* Return a uint16x8 vector with 'n' lanes filled with 0 and the others filled with 65535
 * The valid range for 'n' is 0 to 7 */
static inline uint16x8_t prepare_maskq_u16(uint16_t n) {
    uint64_t mask    = UINT64_MAX;
    mask             = mask << (8 * n);
    uint8x16_t mask8 = vcombine_u8(vcreate_u8(mask), vdup_n_u8(0));
    return vreinterpretq_u16_u8(vzip1q_u8(mask8, mask8));
}

/* Return a uint32x4 vector with 'n' lanes filled with 0 and the others filled with 4294967295
 * The valid range for 'n' is 0 to 4 */
static inline uint32x4_t prepare_maskq_u32(uint16_t n) {
    uint64_t mask    = UINT64_MAX;
    mask             = n < 4 ? (mask << (16 * n)) : 0;
    uint16x8_t mask8 = vcombine_u16(vcreate_u16(mask), vdup_n_u16(0));
    return vreinterpretq_u32_u16(vzip1q_u16(mask8, mask8));
}

static inline uint32_t sad8xh_neon(const uint8_t* src_ptr, uint32_t src_stride, const uint8_t* ref_ptr,
                                   uint32_t ref_stride, uint32_t h) {
    uint16x8_t sum = vdupq_n_u16(0);
    do {
        uint8x8_t s = vld1_u8(src_ptr);
        uint8x8_t r = vld1_u8(ref_ptr);

        sum = vabal_u8(sum, s, r);

        src_ptr += src_stride;
        ref_ptr += ref_stride;
    } while (--h != 0);

    return vaddlvq_u16(sum);
}

static inline uint32x4_t sad8xhx4d_neon(const uint8_t* src, uint32_t src_stride, const uint8_t* ref,
                                        uint32_t ref_stride, uint32_t h) {
    uint16x8_t sum[4];
    sum[0] = vdupq_n_u16(0);
    sum[1] = vdupq_n_u16(0);
    sum[2] = vdupq_n_u16(0);
    sum[3] = vdupq_n_u16(0);

    do {
        uint8x8_t        s   = vld1_u8(src);
        const uint8x16_t r   = vld1q_u8(ref);
        const uint8x8_t  r_l = vget_low_u8(r);
        const uint8x8_t  r_h = vget_high_u8(r);

        sum[0] = vabal_u8(sum[0], s, r_l);
        sum[1] = vabal_u8(sum[1], s, vext_u8(r_l, r_h, 1));
        sum[2] = vabal_u8(sum[2], s, vext_u8(r_l, r_h, 2));
        sum[3] = vabal_u8(sum[3], s, vext_u8(r_l, r_h, 3));

        src += src_stride;
        ref += ref_stride;
    } while (--h != 0);

    return horizontal_add_4d_u16x8(sum);
}

static inline void svt_sad_loop_kernel8xh_neon(uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride,
                                               uint32_t block_height, uint64_t* best_sad, int16_t* x_search_center,
                                               int16_t* y_search_center, uint32_t src_stride_raw,
                                               int16_t search_area_width, int16_t search_area_height) {
    for (int y_search_index = 0; y_search_index < search_area_height; y_search_index++) {
        int16_t x_search_index;
        for (x_search_index = 0; x_search_index <= search_area_width - 4; x_search_index += 4) {
            /* Get the SAD of 4 search spaces aligned along the width and store it in 'sad4'. */
            uint32x4_t sad4 = sad8xhx4d_neon(src, src_stride, ref + x_search_index, ref_stride, block_height);
            update_best_sad_u32(sad4, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }

        for (; x_search_index < search_area_width; x_search_index++) {
            /* Get the SAD of 1 search spaces aligned along the width and store it in 'temp_sad'. */
            uint64_t temp_sad = sad8xh_neon(src, src_stride, ref + x_search_index, ref_stride, block_height);
            update_best_sad(temp_sad, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }
        ref += src_stride_raw;
    }
}

static inline uint16x8_t sad16_neon_init(uint8x16_t src, uint8x16_t ref) {
    const uint8x16_t abs_diff = vabdq_u8(src, ref);
    return vpaddlq_u8(abs_diff);
}

static inline void sad16_neon(uint8x16_t src, uint8x16_t ref, uint16x8_t* const sad_sum) {
    const uint8x16_t abs_diff = vabdq_u8(src, ref);
    *sad_sum                  = vpadalq_u8(*sad_sum, abs_diff);
}

static inline uint32_t sad128xh_neon(const uint8_t* src_ptr, uint32_t src_stride, const uint8_t* ref_ptr,
                                     uint32_t ref_stride, uint32_t h) {
    // We use 8 accumulators to prevent overflow for large values of 'h', as well
    // as enabling optimal UADALP instruction throughput on CPUs that have either
    // 2 or 4 Neon pipes.
    uint16x8_t sum[8];
    uint8x16_t s0, s1, s2, s3, s4, s5, s6, s7;
    uint8x16_t r0, r1, r2, r3, r4, r5, r6, r7;
    uint8x16_t diff0, diff1, diff2, diff3, diff4, diff5, diff6, diff7;
    uint32x4_t sum_u32;
    uint32_t   i;

    for (i = 0; i < 8; i++) {
        sum[i] = vdupq_n_u16(0);
    }

    i = h;
    do {
        s0     = vld1q_u8(src_ptr);
        r0     = vld1q_u8(ref_ptr);
        diff0  = vabdq_u8(s0, r0);
        sum[0] = vpadalq_u8(sum[0], diff0);

        s1     = vld1q_u8(src_ptr + 16);
        r1     = vld1q_u8(ref_ptr + 16);
        diff1  = vabdq_u8(s1, r1);
        sum[1] = vpadalq_u8(sum[1], diff1);

        s2     = vld1q_u8(src_ptr + 32);
        r2     = vld1q_u8(ref_ptr + 32);
        diff2  = vabdq_u8(s2, r2);
        sum[2] = vpadalq_u8(sum[2], diff2);

        s3     = vld1q_u8(src_ptr + 48);
        r3     = vld1q_u8(ref_ptr + 48);
        diff3  = vabdq_u8(s3, r3);
        sum[3] = vpadalq_u8(sum[3], diff3);

        s4     = vld1q_u8(src_ptr + 64);
        r4     = vld1q_u8(ref_ptr + 64);
        diff4  = vabdq_u8(s4, r4);
        sum[4] = vpadalq_u8(sum[4], diff4);

        s5     = vld1q_u8(src_ptr + 80);
        r5     = vld1q_u8(ref_ptr + 80);
        diff5  = vabdq_u8(s5, r5);
        sum[5] = vpadalq_u8(sum[5], diff5);

        s6     = vld1q_u8(src_ptr + 96);
        r6     = vld1q_u8(ref_ptr + 96);
        diff6  = vabdq_u8(s6, r6);
        sum[6] = vpadalq_u8(sum[6], diff6);

        s7     = vld1q_u8(src_ptr + 112);
        r7     = vld1q_u8(ref_ptr + 112);
        diff7  = vabdq_u8(s7, r7);
        sum[7] = vpadalq_u8(sum[7], diff7);

        src_ptr += src_stride;
        ref_ptr += ref_stride;
    } while (--i != 0);

    sum_u32 = vpaddlq_u16(sum[0]);
    sum_u32 = vpadalq_u16(sum_u32, sum[1]);
    sum_u32 = vpadalq_u16(sum_u32, sum[2]);
    sum_u32 = vpadalq_u16(sum_u32, sum[3]);
    sum_u32 = vpadalq_u16(sum_u32, sum[4]);
    sum_u32 = vpadalq_u16(sum_u32, sum[5]);
    sum_u32 = vpadalq_u16(sum_u32, sum[6]);
    sum_u32 = vpadalq_u16(sum_u32, sum[7]);

    return vaddvq_u32(sum_u32);
}

static inline uint32_t sad64xh_neon(const uint8_t* src_ptr, uint32_t src_stride, const uint8_t* ref_ptr,
                                    uint32_t ref_stride, uint32_t h) {
    uint16x8_t sum[4];
    uint8x16_t s0, s1, s2, s3, r0, r1, r2, r3;
    uint8x16_t diff0, diff1, diff2, diff3;
    uint32x4_t sum_u32;
    uint32_t   i;

    sum[0] = vdupq_n_u16(0);
    sum[1] = vdupq_n_u16(0);
    sum[2] = vdupq_n_u16(0);
    sum[3] = vdupq_n_u16(0);

    i = h;
    do {
        s0     = vld1q_u8(src_ptr);
        r0     = vld1q_u8(ref_ptr);
        diff0  = vabdq_u8(s0, r0);
        sum[0] = vpadalq_u8(sum[0], diff0);

        s1     = vld1q_u8(src_ptr + 16);
        r1     = vld1q_u8(ref_ptr + 16);
        diff1  = vabdq_u8(s1, r1);
        sum[1] = vpadalq_u8(sum[1], diff1);

        s2     = vld1q_u8(src_ptr + 32);
        r2     = vld1q_u8(ref_ptr + 32);
        diff2  = vabdq_u8(s2, r2);
        sum[2] = vpadalq_u8(sum[2], diff2);

        s3     = vld1q_u8(src_ptr + 48);
        r3     = vld1q_u8(ref_ptr + 48);
        diff3  = vabdq_u8(s3, r3);
        sum[3] = vpadalq_u8(sum[3], diff3);

        src_ptr += src_stride;
        ref_ptr += ref_stride;
    } while (--i != 0);

    sum_u32 = vpaddlq_u16(sum[0]);
    sum_u32 = vpadalq_u16(sum_u32, sum[1]);
    sum_u32 = vpadalq_u16(sum_u32, sum[2]);
    sum_u32 = vpadalq_u16(sum_u32, sum[3]);

    return vaddvq_u32(sum_u32);
}

static inline uint32_t sad32xh_neon(const uint8_t* src_ptr, uint32_t src_stride, const uint8_t* ref_ptr,
                                    uint32_t ref_stride, uint32_t h) {
    uint16x8_t sum[2];
    uint8x16_t s0, r0, s1, r1, diff0, diff1;
    uint32_t   i;

    sum[0] = vdupq_n_u16(0);
    sum[1] = vdupq_n_u16(0);

    i = h;
    do {
        s0     = vld1q_u8(src_ptr);
        r0     = vld1q_u8(ref_ptr);
        diff0  = vabdq_u8(s0, r0);
        sum[0] = vpadalq_u8(sum[0], diff0);

        s1     = vld1q_u8(src_ptr + 16);
        r1     = vld1q_u8(ref_ptr + 16);
        diff1  = vabdq_u8(s1, r1);
        sum[1] = vpadalq_u8(sum[1], diff1);

        src_ptr += src_stride;
        ref_ptr += ref_stride;
    } while (--i != 0);

    return vaddlvq_u16(vaddq_u16(sum[0], sum[1]));
}

static inline uint32_t sad16xh_neon(const uint8_t* src_ptr, uint32_t src_stride, const uint8_t* ref_ptr,
                                    uint32_t ref_stride, uint32_t h) {
    uint16x8_t sum;
    uint8x16_t s, r, diff;
    uint32_t   i;

    sum = vdupq_n_u16(0);

    i = h;
    do {
        s = vld1q_u8(src_ptr);
        r = vld1q_u8(ref_ptr);

        diff = vabdq_u8(s, r);
        sum  = vpadalq_u8(sum, diff);

        src_ptr += src_stride;
        ref_ptr += ref_stride;
    } while (--i != 0);

    return vaddlvq_u16(sum);
}

static inline uint32_t sad4xh_neon(const uint8_t* src_ptr, uint32_t src_stride, const uint8_t* ref_ptr,
                                   uint32_t ref_stride, uint32_t h) {
    uint16x8_t sum;
    uint8x8_t  s, r;
    uint32_t   i;

    sum = vdupq_n_u16(0);
    i   = h / 2;
    do {
        s = load_u8_4x2(src_ptr, src_stride);
        r = load_u8_4x2(ref_ptr, ref_stride);

        sum = vabal_u8(sum, s, r); // add and accumulate

        src_ptr += 2 * src_stride;
        ref_ptr += 2 * ref_stride;
    } while (--i != 0);
    return vaddlvq_u16(sum);
}

static inline uint32_t sad24xh_neon(const uint8_t* src_ptr, uint32_t src_stride, const uint8_t* ref_ptr,
                                    uint32_t ref_stride, uint32_t h) {
    uint32_t temp_sad;
    temp_sad = sad16xh_neon(src_ptr + 0, src_stride, ref_ptr + 0, ref_stride, h);
    temp_sad += sad8xh_neon(src_ptr + 16, src_stride, ref_ptr + 16, ref_stride, h);
    return temp_sad;
}

static inline uint32_t sad48xh_neon(const uint8_t* src_ptr, uint32_t src_stride, const uint8_t* ref_ptr,
                                    uint32_t ref_stride, uint32_t h) {
    uint32_t temp_sad;
    temp_sad = sad32xh_neon(src_ptr + 0, src_stride, ref_ptr + 0, ref_stride, h);
    temp_sad += sad16xh_neon(src_ptr + 32, src_stride, ref_ptr + 32, ref_stride, h);
    return temp_sad;
}

DECLARE_ALIGNED(16, static const uint8_t, kPermTable2xh[48]) = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,  6,  7,  7,  8,
                                                                2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,  8,  9,  9,  10,
                                                                4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12};

static inline uint16x8_t sad4xhx8d_neon(const uint8_t* src, uint32_t src_stride, const uint8_t* ref,
                                        uint32_t ref_stride, uint32_t h) {
    /* Initialize 'res' to store the sum of absolute differences (SAD) of 8 search spaces. */
    uint16x8_t   res      = vdupq_n_u16(0);
    uint8x16x2_t perm_tbl = vld1q_u8_x2(kPermTable2xh);

    do {
        uint8x16_t src0 = vreinterpretq_u8_u16(vld1q_dup_u16((const uint16_t*)src));
        uint8x16_t src1 = vreinterpretq_u8_u16(vld1q_dup_u16((const uint16_t*)(src + 2)));

        uint8x16_t ref0 = vqtbl1q_u8(vld1q_u8(ref), perm_tbl.val[0]);
        uint8x16_t ref1 = vqtbl1q_u8(vld1q_u8(ref), perm_tbl.val[1]);

        uint8x16_t abs0 = vabdq_u8(src0, ref0);
        uint8x16_t abs1 = vabdq_u8(src1, ref1);

        res = vpadalq_u8(res, abs0);
        res = vpadalq_u8(res, abs1);

        src += src_stride;
        ref += ref_stride;
    } while (--h != 0);
    return res;
}

static inline uint16x8_t sad6xhx8d_neon(const uint8_t* src, uint32_t src_stride, const uint8_t* ref,
                                        uint32_t ref_stride, uint32_t h) {
    /* Initialize 'res' to store the sum of absolute differences (SAD) of 8 search spaces. */
    uint16x8_t   res      = vdupq_n_u16(0);
    uint8x16x3_t perm_tbl = vld1q_u8_x3(kPermTable2xh);

    do {
        uint8x16_t src0 = vreinterpretq_u8_u16(vld1q_dup_u16((const uint16_t*)src));
        uint8x16_t src1 = vreinterpretq_u8_u16(vld1q_dup_u16((const uint16_t*)(src + 2)));
        uint8x16_t src2 = vreinterpretq_u8_u16(vld1q_dup_u16((const uint16_t*)(src + 4)));

        uint8x16_t ref0 = vqtbl1q_u8(vld1q_u8(ref), perm_tbl.val[0]);
        uint8x16_t ref1 = vqtbl1q_u8(vld1q_u8(ref), perm_tbl.val[1]);
        uint8x16_t ref2 = vqtbl1q_u8(vld1q_u8(ref), perm_tbl.val[2]);

        uint8x16_t abs0 = vabdq_u8(src0, ref0);
        uint8x16_t abs1 = vabdq_u8(src1, ref1);
        uint8x16_t abs2 = vabdq_u8(src2, ref2);

        res = vpadalq_u8(res, abs0);
        res = vpadalq_u8(res, abs1);
        res = vpadalq_u8(res, abs2);

        src += src_stride;
        ref += ref_stride;
    } while (--h != 0);
    return res;
}

static inline void sad12xhx8d_neon(const uint8_t* src, uint32_t src_stride, const uint8_t* ref, uint32_t ref_stride,
                                   uint32_t h, uint32x4_t* res) {
    /* 'sad8' will store 8d SAD for block_width = 4 */
    uint16x8_t sad8;
    sad8   = sad4xhx8d_neon(src + 0, src_stride, ref + 0, ref_stride, h);
    res[0] = sad8xhx4d_neon(src + 4, src_stride, ref + 4, ref_stride, h);
    res[1] = sad8xhx4d_neon(src + 4, src_stride, ref + 8, ref_stride, h);
    res[0] = vaddw_u16(res[0], vget_low_u16(sad8));
    res[1] = vaddw_high_u16(res[1], sad8);
}

static inline void svt_sad_loop_kernel4xh_neon(uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride,
                                               uint32_t block_height, uint64_t* best_sad, int16_t* x_search_center,
                                               int16_t* y_search_center, uint32_t src_stride_raw,
                                               int16_t search_area_width, int16_t search_area_height) {
    int16_t    x_search_index, y_search_index;
    uint16x8_t sad8;
    uint32_t   leftover      = search_area_width & 7;
    uint16x8_t leftover_mask = prepare_maskq_u16(leftover);

    for (y_search_index = 0; y_search_index < search_area_height; y_search_index++) {
        for (x_search_index = 0; x_search_index <= search_area_width - 8; x_search_index += 8) {
            /* Get the SAD of 8 search spaces aligned along the width and store it in 'sad8'. */
            sad8 = sad4xhx8d_neon(src, src_stride, ref + x_search_index, ref_stride, block_height);

            /* Update 'best_sad' */
            update_best_sad_u16(sad8, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }

        if (leftover) {
            /* Get the SAD of 8 search spaces aligned along the width and store it in 'sad8'. */
            sad8 = sad4xhx8d_neon(src, src_stride, ref + x_search_index, ref_stride, block_height);

            /* set undesired lanes to maximum value */
            sad8 = vorrq_u16(sad8, leftover_mask);

            /* Update 'best_sad' */
            update_best_sad_u16(sad8, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }

        ref += src_stride_raw;
    }
}

static inline void svt_sad_loop_kernel6xh_neon(uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride,
                                               uint32_t block_height, uint64_t* best_sad, int16_t* x_search_center,
                                               int16_t* y_search_center, uint32_t src_stride_raw,
                                               int16_t search_area_width, int16_t search_area_height) {
    int16_t    x_search_index, y_search_index;
    uint16x8_t sad8;
    uint32_t   leftover      = search_area_width & 7;
    uint16x8_t leftover_mask = prepare_maskq_u16(leftover);

    for (y_search_index = 0; y_search_index < search_area_height; y_search_index++) {
        for (x_search_index = 0; x_search_index <= search_area_width - 8; x_search_index += 8) {
            /* Get the SAD of 8 search spaces aligned along the width and store it in 'sad8'. */
            sad8 = sad6xhx8d_neon(src, src_stride, ref + x_search_index, ref_stride, block_height);

            /* Update 'best_sad' */
            update_best_sad_u16(sad8, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }

        if (leftover) {
            /* Get the SAD of 8 search spaces aligned along the width and store it in 'sad8'. */
            sad8 = sad6xhx8d_neon(src, src_stride, ref + x_search_index, ref_stride, block_height);

            /* set undesired lanes to maximum value */
            sad8 = vorrq_u16(sad8, leftover_mask);

            /* Update 'best_sad' */
            update_best_sad_u16(sad8, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }

        ref += src_stride_raw;
    }
}

static inline void svt_sad_loop_kernel12xh_neon(uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride,
                                                uint32_t block_height, uint64_t* best_sad, int16_t* x_search_center,
                                                int16_t* y_search_center, uint32_t src_stride_raw,
                                                int16_t search_area_width, int16_t search_area_height) {
    int16_t x_search_index, y_search_index;

    /* To accumulate the SAD of 8 search spaces */
    uint32x4_t sad8[2];

    uint32_t   leftover = search_area_width & 7;
    uint32x4_t leftover_mask[2];
    leftover_mask[0] = prepare_maskq_u32(MIN(leftover, 4));
    leftover_mask[1] = prepare_maskq_u32(leftover - MIN(leftover, 4));
    for (y_search_index = 0; y_search_index < search_area_height; y_search_index++) {
        for (x_search_index = 0; x_search_index <= search_area_width - 8; x_search_index += 8) {
            /* Get the SAD of 8 search spaces aligned along the width and store it in 'sad8'. */
            sad12xhx8d_neon(src, src_stride, ref + x_search_index, ref_stride, block_height, sad8);

            /* Update 'best_sad' */
            update_best_sad_u32(sad8[0], best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
            update_best_sad_u32(
                sad8[1], best_sad, x_search_center, y_search_center, x_search_index + 4, y_search_index);
        }

        if (leftover) {
            /* Get the SAD of 8 search spaces aligned along the width and store it in 'sad8'. */
            sad12xhx8d_neon(src, src_stride, ref + x_search_index, ref_stride, block_height, sad8);

            /* set undesired lanes to maximum value */
            sad8[0] = vorrq_u32(sad8[0], leftover_mask[0]);
            sad8[1] = vorrq_u32(sad8[1], leftover_mask[1]);

            /* Update 'best_sad' */
            update_best_sad_u32(sad8[0], best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
            update_best_sad_u32(
                sad8[1], best_sad, x_search_center, y_search_center, x_search_index + 4, y_search_index);
        }

        ref += src_stride_raw;
    }
}

static inline uint32_t sadwxh_neon(const uint8_t* src, uint32_t src_stride, const uint8_t* ref, uint32_t ref_stride,
                                   uint32_t width, uint32_t height) {
    uint32x4_t sum_u32 = vdupq_n_u32(0);
    uint32_t   sum     = 0;

    // We can accumulate 257 absolute differences in a 16-bit element before
    // it overflows. Given that we're accumulating in 8 16-bit elements we
    // therefore need width * height < (257 * 8). (This isn't quite true as some
    // elements in the tail loops have a different accumulator, but it's a good
    // enough approximation).
    uint32_t h_overflow = (257 * 8) / width;
    uint32_t h_limit    = h_overflow >= height ? height : h_overflow;

    uint32_t i = 0;
    do {
        uint16x8_t sum_u16 = vdupq_n_u16(0);

        do {
            int w = width;

            const uint8_t* src_ptr = src;
            const uint8_t* ref_ptr = ref;

            while (w >= 16) {
                const uint8x16_t s = vld1q_u8(src_ptr);
                sad16_neon(s, vld1q_u8(ref_ptr), &sum_u16);

                src_ptr += 16;
                ref_ptr += 16;
                w -= 16;
            }

            if (w >= 8) {
                const uint8x8_t s = vld1_u8(src_ptr);
                sum_u16           = vabal_u8(sum_u16, s, vld1_u8(ref_ptr));

                src_ptr += 8;
                ref_ptr += 8;
                w -= 8;
            }

            while (w != 0) {
                sum += EB_ABS_DIFF(src_ptr[w - 1], ref_ptr[w - 1]);

                w--;
            }
            src += src_stride;
            ref += ref_stride;
        } while (++i < h_limit);

        sum_u32 = vpadalq_u16(sum_u32, sum_u16);

        uint32_t h_inc = h_limit + h_overflow < height ? h_overflow : height - h_limit;
        h_limit += h_inc;
    } while (i < height);

    return sum + vaddvq_u32(sum_u32);
}

// ============================================================
// sad{W}xh_indep4d_neon — 4-way parallel SAD with independent reference pointers
//
// Unlike sad{W}xhx4d_neon (which assumes 4 adjacent reference offsets for motion
// search), these helpers accept 4 fully-independent reference pointers as required
// by the RTCD svt_aom_sadMxNx4d API.
//
// Why sad4d_reduce_u16x8 instead of horizontal_add_4d_u16x8 for w>=16:
//
// horizontal_add_4d_u16x8() reduces via two stages of vpaddq_u16 (uint16 pairwise),
// so intermediate values reach 4 × per_lane before the final vpaddlq_u16 widening.
// When per_lane > 16383, the product 4 × per_lane exceeds 65535 and wraps in uint16.
//
// For w=16, h=64:  per_lane = 64 × 510 = 32640   →  4 × 32640 = 130560  OVERFLOW
// For w=32, h=64:  per_lane = 128 × 510 = 65280   →  4 × 65280 = 261120  OVERFLOW
//   (note: 65280 < 65535 so the accumulator itself is safe, but horizontal_add
//    is still not usable — sad4d_reduce_u16x8 is required, not merely preferred)
//
// For w=4, h<=16:  per_lane = 8 × 255 = 2040      →  4 × 2040 = 8160  safe
// For w=8, h<=32:  per_lane = 32 × 255 = 8160     →  4 × 8160 = 32640  safe
// ============================================================

static inline uint32x4_t sad4d_reduce_u16x8(const uint16x8_t sum[4]) {
    // Pairwise promote each accumulator to uint32, then reduce across refs.
    const uint32x4_t a0 = vpaddlq_u16(sum[0]);
    const uint32x4_t a1 = vpaddlq_u16(sum[1]);
    const uint32x4_t a2 = vpaddlq_u16(sum[2]);
    const uint32x4_t a3 = vpaddlq_u16(sum[3]);
    // vpaddq_u32: [sum[0]_half0, sum[0]_half1, sum[1]_half0, sum[1]_half1]
    //           + [sum[2]_half0, sum[2]_half1, sum[3]_half0, sum[3]_half1]
    // → [total(sum[0]), total(sum[1]), total(sum[2]), total(sum[3])]
    return vpaddq_u32(vpaddq_u32(a0, a1), vpaddq_u32(a2, a3));
}

// w=4: 2 rows per iteration via load_u8_4x2; per lane = (h/2)*255*2 = h*255.
// For h<=16 (max AV1 w=4 height): per_lane<=4080, 4×4080=16320 — safe for
// horizontal_add_4d_u16x8 (threshold: per_lane <= 16383).
static inline uint32x4_t sad4xh_indep4d_neon(const uint8_t* src, uint32_t src_stride, const uint8_t* ref0,
                                             const uint8_t* ref1, const uint8_t* ref2, const uint8_t* ref3,
                                             uint32_t ref_stride, uint32_t h) {
    assert(h > 0 && (h & 1) == 0); // AV1 w=4 blocks have h in {4,8,16}
    uint16x8_t sum[4];
    sum[0] = vdupq_n_u16(0);
    sum[1] = vdupq_n_u16(0);
    sum[2] = vdupq_n_u16(0);
    sum[3] = vdupq_n_u16(0);

    uint32_t i = h / 2;
    do {
        const uint8x8_t s = load_u8_4x2(src, src_stride);
        sum[0]            = vabal_u8(sum[0], s, load_u8_4x2(ref0, ref_stride));
        sum[1]            = vabal_u8(sum[1], s, load_u8_4x2(ref1, ref_stride));
        sum[2]            = vabal_u8(sum[2], s, load_u8_4x2(ref2, ref_stride));
        sum[3]            = vabal_u8(sum[3], s, load_u8_4x2(ref3, ref_stride));
        src += 2 * src_stride;
        ref0 += 2 * ref_stride;
        ref1 += 2 * ref_stride;
        ref2 += 2 * ref_stride;
        ref3 += 2 * ref_stride;
    } while (--i != 0);

    return horizontal_add_4d_u16x8(sum);
}

// w=8: 1 row/iter; max per lane = h*255. For h<=32: per lane<=8160, 4x<=32640 — safe.
static inline uint32x4_t sad8xh_indep4d_neon(const uint8_t* src, uint32_t src_stride, const uint8_t* ref0,
                                             const uint8_t* ref1, const uint8_t* ref2, const uint8_t* ref3,
                                             uint32_t ref_stride, uint32_t h) {
    uint16x8_t sum[4];
    sum[0] = vdupq_n_u16(0);
    sum[1] = vdupq_n_u16(0);
    sum[2] = vdupq_n_u16(0);
    sum[3] = vdupq_n_u16(0);

    do {
        const uint8x8_t s = vld1_u8(src);
        sum[0]            = vabal_u8(sum[0], s, vld1_u8(ref0));
        sum[1]            = vabal_u8(sum[1], s, vld1_u8(ref1));
        sum[2]            = vabal_u8(sum[2], s, vld1_u8(ref2));
        sum[3]            = vabal_u8(sum[3], s, vld1_u8(ref3));
        src += src_stride;
        ref0 += ref_stride;
        ref1 += ref_stride;
        ref2 += ref_stride;
        ref3 += ref_stride;
    } while (--h != 0);

    return horizontal_add_4d_u16x8(sum);
}

// w=16: 1 row/iter, 1 chunk; max per lane = h*510.
// For h<=64: per lane<=32640. Use sad4d_reduce_u16x8 (safe reduction via uint32).
static inline uint32x4_t sad16xh_indep4d_neon(const uint8_t* src, uint32_t src_stride, const uint8_t* ref0,
                                              const uint8_t* ref1, const uint8_t* ref2, const uint8_t* ref3,
                                              uint32_t ref_stride, uint32_t h) {
    uint16x8_t sum[4];
    sum[0] = vdupq_n_u16(0);
    sum[1] = vdupq_n_u16(0);
    sum[2] = vdupq_n_u16(0);
    sum[3] = vdupq_n_u16(0);

    do {
        const uint8x16_t s = vld1q_u8(src);
        sum[0]             = vpadalq_u8(sum[0], vabdq_u8(s, vld1q_u8(ref0)));
        sum[1]             = vpadalq_u8(sum[1], vabdq_u8(s, vld1q_u8(ref1)));
        sum[2]             = vpadalq_u8(sum[2], vabdq_u8(s, vld1q_u8(ref2)));
        sum[3]             = vpadalq_u8(sum[3], vabdq_u8(s, vld1q_u8(ref3)));
        src += src_stride;
        ref0 += ref_stride;
        ref1 += ref_stride;
        ref2 += ref_stride;
        ref3 += ref_stride;
    } while (--h != 0);

    return sad4d_reduce_u16x8(sum);
}

// w=32: 1 row/iter, 2 chunks accumulated into a single sum[4].
// Per lane = 2 chunks × h × 510 = 128 × 510 = 65280 < 65535 — uint16 accumulator safe.
// However horizontal_add_4d_u16x8 is NOT usable: its intermediate 4×65280=261120
// overflows uint16. sad4d_reduce_u16x8 (vpaddlq_u16 first) is required.
static inline uint32x4_t sad32xh_indep4d_neon(const uint8_t* src, uint32_t src_stride, const uint8_t* ref0,
                                              const uint8_t* ref1, const uint8_t* ref2, const uint8_t* ref3,
                                              uint32_t ref_stride, uint32_t h) {
    uint16x8_t sum[4];
    sum[0] = vdupq_n_u16(0);
    sum[1] = vdupq_n_u16(0);
    sum[2] = vdupq_n_u16(0);
    sum[3] = vdupq_n_u16(0);

    do {
        const uint8x16_t s0 = vld1q_u8(src);
        sum[0]              = vpadalq_u8(sum[0], vabdq_u8(s0, vld1q_u8(ref0)));
        sum[1]              = vpadalq_u8(sum[1], vabdq_u8(s0, vld1q_u8(ref1)));
        sum[2]              = vpadalq_u8(sum[2], vabdq_u8(s0, vld1q_u8(ref2)));
        sum[3]              = vpadalq_u8(sum[3], vabdq_u8(s0, vld1q_u8(ref3)));

        const uint8x16_t s1 = vld1q_u8(src + 16);
        sum[0]              = vpadalq_u8(sum[0], vabdq_u8(s1, vld1q_u8(ref0 + 16)));
        sum[1]              = vpadalq_u8(sum[1], vabdq_u8(s1, vld1q_u8(ref1 + 16)));
        sum[2]              = vpadalq_u8(sum[2], vabdq_u8(s1, vld1q_u8(ref2 + 16)));
        sum[3]              = vpadalq_u8(sum[3], vabdq_u8(s1, vld1q_u8(ref3 + 16)));

        src += src_stride;
        ref0 += ref_stride;
        ref1 += ref_stride;
        ref2 += ref_stride;
        ref3 += ref_stride;
    } while (--h != 0);

    return sad4d_reduce_u16x8(sum);
}

// w=64: 1 row/iter, 4 chunks; 4*h*510 per uint16 lane would overflow for large h.
// Use separate uint16x8_t accumulator per chunk per ref (16 total); each chunk
// accumulates at most h*510 <= 128*510 = 65280 per lane, safely under 65535.
// Reduce each chunk group with sad4d_reduce_u16x8, then sum across chunks.
static inline uint32x4_t sad64xh_indep4d_neon(const uint8_t* src, uint32_t src_stride, const uint8_t* ref0,
                                              const uint8_t* ref1, const uint8_t* ref2, const uint8_t* ref3,
                                              uint32_t ref_stride, uint32_t h) {
    // chunk{0..3}[ref{0..3}]: separate accumulator per 16-byte chunk per ref
    uint16x8_t chunk0[4], chunk1[4], chunk2[4], chunk3[4];
    chunk0[0] = chunk0[1] = chunk0[2] = chunk0[3] = vdupq_n_u16(0);
    chunk1[0] = chunk1[1] = chunk1[2] = chunk1[3] = vdupq_n_u16(0);
    chunk2[0] = chunk2[1] = chunk2[2] = chunk2[3] = vdupq_n_u16(0);
    chunk3[0] = chunk3[1] = chunk3[2] = chunk3[3] = vdupq_n_u16(0);

    do {
        uint8x16_t s;

        s         = vld1q_u8(src);
        chunk0[0] = vpadalq_u8(chunk0[0], vabdq_u8(s, vld1q_u8(ref0)));
        chunk0[1] = vpadalq_u8(chunk0[1], vabdq_u8(s, vld1q_u8(ref1)));
        chunk0[2] = vpadalq_u8(chunk0[2], vabdq_u8(s, vld1q_u8(ref2)));
        chunk0[3] = vpadalq_u8(chunk0[3], vabdq_u8(s, vld1q_u8(ref3)));

        s         = vld1q_u8(src + 16);
        chunk1[0] = vpadalq_u8(chunk1[0], vabdq_u8(s, vld1q_u8(ref0 + 16)));
        chunk1[1] = vpadalq_u8(chunk1[1], vabdq_u8(s, vld1q_u8(ref1 + 16)));
        chunk1[2] = vpadalq_u8(chunk1[2], vabdq_u8(s, vld1q_u8(ref2 + 16)));
        chunk1[3] = vpadalq_u8(chunk1[3], vabdq_u8(s, vld1q_u8(ref3 + 16)));

        s         = vld1q_u8(src + 32);
        chunk2[0] = vpadalq_u8(chunk2[0], vabdq_u8(s, vld1q_u8(ref0 + 32)));
        chunk2[1] = vpadalq_u8(chunk2[1], vabdq_u8(s, vld1q_u8(ref1 + 32)));
        chunk2[2] = vpadalq_u8(chunk2[2], vabdq_u8(s, vld1q_u8(ref2 + 32)));
        chunk2[3] = vpadalq_u8(chunk2[3], vabdq_u8(s, vld1q_u8(ref3 + 32)));

        s         = vld1q_u8(src + 48);
        chunk3[0] = vpadalq_u8(chunk3[0], vabdq_u8(s, vld1q_u8(ref0 + 48)));
        chunk3[1] = vpadalq_u8(chunk3[1], vabdq_u8(s, vld1q_u8(ref1 + 48)));
        chunk3[2] = vpadalq_u8(chunk3[2], vabdq_u8(s, vld1q_u8(ref2 + 48)));
        chunk3[3] = vpadalq_u8(chunk3[3], vabdq_u8(s, vld1q_u8(ref3 + 48)));

        src += src_stride;
        ref0 += ref_stride;
        ref1 += ref_stride;
        ref2 += ref_stride;
        ref3 += ref_stride;
    } while (--h != 0);

    return vaddq_u32(vaddq_u32(sad4d_reduce_u16x8(chunk0), sad4d_reduce_u16x8(chunk1)),
                     vaddq_u32(sad4d_reduce_u16x8(chunk2), sad4d_reduce_u16x8(chunk3)));
}

// w=128: process as two 64-wide halves to reuse sad64xh_indep4d_neon.
static inline uint32x4_t sad128xh_indep4d_neon(const uint8_t* src, uint32_t src_stride, const uint8_t* ref0,
                                               const uint8_t* ref1, const uint8_t* ref2, const uint8_t* ref3,
                                               uint32_t ref_stride, uint32_t h) {
    const uint32x4_t lo = sad64xh_indep4d_neon(src, src_stride, ref0, ref1, ref2, ref3, ref_stride, h);
    const uint32x4_t hi = sad64xh_indep4d_neon(
        src + 64, src_stride, ref0 + 64, ref1 + 64, ref2 + 64, ref3 + 64, ref_stride, h);
    return vaddq_u32(lo, hi);
}

#endif // COMPUTE_SAD_NEON_H
