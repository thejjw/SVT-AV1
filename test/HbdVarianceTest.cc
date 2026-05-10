/*
 * Copyright(c) 2019 Netflix, Inc.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

/******************************************************************************
 * @file HbdVarianceTest.cc
 *
 * @brief Unit test for HBD variance
 * functions:
 * - svt_aom_highbd_BD{8,10,12}_varianceW{8,16,32,64}xH{4,8,16,32,64}
 *
 * @author  Cidana-Wenyao, Cidana-Edmond
 *
 ******************************************************************************/
#include <algorithm>
#include <array>
#include <cstdint>
#include "gtest/gtest.h"
#include "aligned_allocator.hpp"
#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "random.hpp"
#include "util.h"

namespace {
constexpr auto MAX_BLOCK_SIZE = 128 * 128;
using svt_av1_test_tool::SVTRandom;  // to generate the random

#if CONFIG_ENABLE_HIGH_BIT_DEPTH

using HighBdGetVarianceFunc = void (*)(const uint8_t *src8, int32_t src_stride,
                                       const uint8_t *ref8, int32_t ref_stride,
                                       uint32_t *sse, int32_t *sum);

using HighBdVarianceFunc = uint32_t (*)(const uint8_t *src8, int32_t src_stride,
                                        const uint8_t *ref8, int32_t ref_stride,
                                        uint32_t *sse);

// Truncate high bit depth results by downshifting (with rounding) by:
// 2 * (bit_depth - 8) for sse
// (bit_depth - 8) for se
static void round_hbd(const uint8_t bd, int64_t &se, uint64_t &sse) {
    switch (bd) {
    case 12:
        sse = (sse + 128) >> 8;
        se = (se + 8) >> 4;
        break;
    case 10:
        sse = (sse + 8) >> 4;
        se = (se + 2) >> 2;
        break;
    case 8:
    default: break;
    }
}

static void hbd_get_variance_ref(const uint32_t width, const uint32_t height,
                                 const uint8_t bd, const uint8_t *src8,
                                 int32_t src_stride, const uint8_t *ref8,
                                 int32_t ref_stride, uint32_t *sse,
                                 int32_t &sum) {
    int64_t sum_tmp = 0;
    uint64_t sse_tmp = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            int diff = CONVERT_TO_SHORTPTR(src8)[y * src_stride + x] -
                       CONVERT_TO_SHORTPTR(ref8)[y * ref_stride + x];
            sum_tmp += diff;
            sse_tmp += diff * diff;
        }
    }
    round_hbd(bd, sum_tmp, sse_tmp);
    *sse = static_cast<uint32_t>(sse_tmp);
    sum = static_cast<int32_t>(sum_tmp);
}

static uint32_t hbd_variance_ref(const uint32_t width, const uint32_t height,
                                 const uint8_t bd, const uint8_t *src8,
                                 int32_t src_stride, const uint8_t *ref8,
                                 int32_t ref_stride, uint32_t &sse) {
    int32_t sum = 0;
    hbd_get_variance_ref(
        width, height, bd, src8, src_stride, ref8, ref_stride, &sse, sum);
    return static_cast<uint32_t>(
        sse - ((((int64_t)sum * sum)) / ((int64_t)width * height)));
}

// High bit-depth variance test
using HbdVarianceParam = std::tuple<uint32_t,            /**< width */
                                    uint32_t,            /**< height */
                                    uint32_t,            /**< bit-depth */
                                    HighBdVarianceFunc>; /**< test function */

/**
 * @brief Unit test for HBD variance
 * functions:
 * - svt_aom_highbd_BD{8,10,12}_varianceW{8,16,32,64}xH{4,8,16,32,64}
 *
 * Test strategy:
 *  This test case use random source, max source, zero source as test
 * pattern.
 *
 *
 * Expected result:
 *  Results come from reference function and target function are
 * equal.
 *
 * Test cases:
 * - ZeroTest
 * - MaximumTest
 * - MatchTest
 */
class HbdVarianceTest : public ::testing::TestWithParam<HbdVarianceParam> {
  public:
    DEFINE_ALIGNED_NEW_DELETE(HbdVarianceTest)

    template <int times>
    void run_zero_test() {
        for (int i = 0; i < times; ++i) {
            std::generate_n(src_data_.begin(), MAX_BLOCK_SIZE, [this]() {
                return rnd_.random() & ((1 << bd_) - 1);
            });
            ref_data_ = src_data_;
            uint32_t sse_tst = 0;
            uint32_t var_tst = tst_func_(CONVERT_TO_BYTEPTR(src_data_.data()),
                                         width_,
                                         CONVERT_TO_BYTEPTR(ref_data_.data()),
                                         width_,
                                         &sse_tst);
            ASSERT_EQ(var_tst, 0u) << "Expect 0 variance, got: " << var_tst;
        }
    }

    void run_maximum_test() {
        src_data_.fill(0);
        ref_data_.fill((1 << bd_) - 1);
        uint32_t sse_tst = 0, sse_ref = 0;
        uint32_t var_tst = tst_func_(CONVERT_TO_BYTEPTR(src_data_.data()),
                                     width_,
                                     CONVERT_TO_BYTEPTR(ref_data_.data()),
                                     width_,
                                     &sse_tst);
        uint32_t var_ref =
            hbd_variance_ref(width_,
                             height_,
                             bd_,
                             CONVERT_TO_BYTEPTR(src_data_.data()),
                             width_,
                             CONVERT_TO_BYTEPTR(ref_data_.data()),
                             width_,
                             sse_ref);
        ASSERT_EQ(var_tst, var_ref)
            << "Expect var " << var_ref << " got " << var_tst
            << " size: " << width_ << "x" << height_;
        ASSERT_EQ(sse_tst, sse_ref)
            << "Expect sse " << sse_ref << " got " << sse_tst
            << " size: " << width_ << "x" << height_;
    }

    template <int times>
    void run_match_test() {
        for (int i = 0; i < times; ++i) {
            for (int j = 0; j < MAX_BLOCK_SIZE; ++j) {
                src_data_[j] = rnd_.random() & ((1 << bd_) - 1);
                ref_data_[j] = rnd_.random() & ((1 << bd_) - 1);
            }
            uint32_t sse_tst = 0, sse_ref = 0;
            uint32_t var_tst = tst_func_(CONVERT_TO_BYTEPTR(src_data_.data()),
                                         width_,
                                         CONVERT_TO_BYTEPTR(ref_data_.data()),
                                         width_,
                                         &sse_tst);
            uint32_t var_ref =
                hbd_variance_ref(width_,
                                 height_,
                                 bd_,
                                 CONVERT_TO_BYTEPTR(src_data_.data()),
                                 width_,
                                 CONVERT_TO_BYTEPTR(ref_data_.data()),
                                 width_,
                                 sse_ref);
            ASSERT_EQ(var_tst, var_ref)
                << "Error at variance test index: " << i << " size: " << width_
                << "x" << height_;
            ASSERT_EQ(sse_tst, sse_ref)
                << "Error at sse test index: " << i << " size: " << width_
                << "x" << height_;
        }
    }

  private:
    SVTRandom rnd_{16, false};
    alignas(32) std::array<uint16_t, 2 * MAX_BLOCK_SIZE> src_data_{};
    alignas(32) std::array<uint16_t, 2 * MAX_BLOCK_SIZE> ref_data_{};
    const uint32_t width_{TEST_GET_PARAM(0)};
    const uint32_t height_{TEST_GET_PARAM(1)};
    const uint32_t bd_{TEST_GET_PARAM(2)};
    const HighBdVarianceFunc tst_func_{TEST_GET_PARAM(3)};
};

TEST_P(HbdVarianceTest, ZeroTest) {
    run_zero_test<10>();
};

TEST_P(HbdVarianceTest, MaximumTest) {
    run_maximum_test();
};

TEST_P(HbdVarianceTest, MatchTest) {
    run_match_test<10>();
};

#ifdef ARCH_X86_64

const HbdVarianceParam HbdTestVector_sse2[] = {
    HbdVarianceParam(8, 8, 10, svt_aom_highbd_10_variance8x8_sse2),
    HbdVarianceParam(8, 16, 10, svt_aom_highbd_10_variance8x16_sse2),
    HbdVarianceParam(8, 32, 10, svt_aom_highbd_10_variance8x32_sse2),
    HbdVarianceParam(16, 4, 10, svt_aom_highbd_10_variance16x4_sse2),
    HbdVarianceParam(16, 8, 10, svt_aom_highbd_10_variance16x8_sse2),
    HbdVarianceParam(16, 16, 10, svt_aom_highbd_10_variance16x16_sse2),
    HbdVarianceParam(16, 32, 10, svt_aom_highbd_10_variance16x32_sse2),
    HbdVarianceParam(16, 64, 10, svt_aom_highbd_10_variance16x64_sse2),
    HbdVarianceParam(32, 8, 10, svt_aom_highbd_10_variance32x8_sse2),
    HbdVarianceParam(32, 16, 10, svt_aom_highbd_10_variance32x16_sse2),
    HbdVarianceParam(32, 32, 10, svt_aom_highbd_10_variance32x32_sse2),
    HbdVarianceParam(32, 64, 10, svt_aom_highbd_10_variance32x64_sse2),
    HbdVarianceParam(64, 16, 10, svt_aom_highbd_10_variance64x16_sse2),
    HbdVarianceParam(64, 32, 10, svt_aom_highbd_10_variance64x32_sse2),
    HbdVarianceParam(64, 64, 10, svt_aom_highbd_10_variance64x64_sse2),
    HbdVarianceParam(64, 128, 10, svt_aom_highbd_10_variance64x128_sse2),
    HbdVarianceParam(128, 64, 10, svt_aom_highbd_10_variance128x64_sse2),
    HbdVarianceParam(128, 128, 10, svt_aom_highbd_10_variance128x128_sse2),
};

INSTANTIATE_TEST_SUITE_P(SSE2, HbdVarianceTest,
                         ::testing::ValuesIn(HbdTestVector_sse2));

const HbdVarianceParam HbdTestVector_avx2[] = {
    HbdVarianceParam(8, 8, 10, svt_aom_highbd_10_variance8x8_avx2),
    HbdVarianceParam(8, 16, 10, svt_aom_highbd_10_variance8x16_avx2),
    HbdVarianceParam(8, 32, 10, svt_aom_highbd_10_variance8x32_avx2),
    HbdVarianceParam(16, 8, 10, svt_aom_highbd_10_variance16x8_avx2),
    HbdVarianceParam(16, 16, 10, svt_aom_highbd_10_variance16x16_avx2),
    HbdVarianceParam(16, 32, 10, svt_aom_highbd_10_variance16x32_avx2),
    HbdVarianceParam(16, 64, 10, svt_aom_highbd_10_variance16x64_avx2),
    HbdVarianceParam(32, 8, 10, svt_aom_highbd_10_variance32x8_avx2),
    HbdVarianceParam(32, 16, 10, svt_aom_highbd_10_variance32x16_avx2),
    HbdVarianceParam(32, 32, 10, svt_aom_highbd_10_variance32x32_avx2),
    HbdVarianceParam(32, 64, 10, svt_aom_highbd_10_variance32x64_avx2),
    HbdVarianceParam(64, 16, 10, svt_aom_highbd_10_variance64x16_avx2),
    HbdVarianceParam(64, 32, 10, svt_aom_highbd_10_variance64x32_avx2),
    HbdVarianceParam(64, 64, 10, svt_aom_highbd_10_variance64x64_avx2),
    HbdVarianceParam(64, 128, 10, svt_aom_highbd_10_variance64x128_avx2),
    HbdVarianceParam(128, 64, 10, svt_aom_highbd_10_variance128x64_avx2),
    HbdVarianceParam(128, 128, 10, svt_aom_highbd_10_variance128x128_avx2),
};

INSTANTIATE_TEST_SUITE_P(AVX2, HbdVarianceTest,
                         ::testing::ValuesIn(HbdTestVector_avx2));

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64

const HbdVarianceParam HbdTestVector_neon[] = {
    HbdVarianceParam(4, 4, 10, svt_aom_highbd_10_variance4x4_neon),
    HbdVarianceParam(4, 8, 10, svt_aom_highbd_10_variance4x8_neon),
    HbdVarianceParam(4, 16, 10, svt_aom_highbd_10_variance4x16_neon),
    HbdVarianceParam(8, 4, 10, svt_aom_highbd_10_variance8x4_neon),
    HbdVarianceParam(8, 8, 10, svt_aom_highbd_10_variance8x8_neon),
    HbdVarianceParam(8, 16, 10, svt_aom_highbd_10_variance8x16_neon),
    HbdVarianceParam(8, 32, 10, svt_aom_highbd_10_variance8x32_neon),
    HbdVarianceParam(16, 4, 10, svt_aom_highbd_10_variance16x4_neon),
    HbdVarianceParam(16, 8, 10, svt_aom_highbd_10_variance16x8_neon),
    HbdVarianceParam(16, 16, 10, svt_aom_highbd_10_variance16x16_neon),
    HbdVarianceParam(16, 32, 10, svt_aom_highbd_10_variance16x32_neon),
    HbdVarianceParam(16, 64, 10, svt_aom_highbd_10_variance16x64_neon),
    HbdVarianceParam(32, 8, 10, svt_aom_highbd_10_variance32x8_neon),
    HbdVarianceParam(32, 16, 10, svt_aom_highbd_10_variance32x16_neon),
    HbdVarianceParam(32, 32, 10, svt_aom_highbd_10_variance32x32_neon),
    HbdVarianceParam(32, 64, 10, svt_aom_highbd_10_variance32x64_neon),
    HbdVarianceParam(64, 16, 10, svt_aom_highbd_10_variance64x16_neon),
    HbdVarianceParam(64, 32, 10, svt_aom_highbd_10_variance64x32_neon),
    HbdVarianceParam(64, 64, 10, svt_aom_highbd_10_variance64x64_neon),
    HbdVarianceParam(64, 128, 10, svt_aom_highbd_10_variance64x128_neon),
    HbdVarianceParam(128, 64, 10, svt_aom_highbd_10_variance128x64_neon),
    HbdVarianceParam(128, 128, 10, svt_aom_highbd_10_variance128x128_neon),
};

INSTANTIATE_TEST_SUITE_P(NEON, HbdVarianceTest,
                         ::testing::ValuesIn(HbdTestVector_neon));

#if HAVE_SVE

const HbdVarianceParam HbdTestVector_sve[] = {
    HbdVarianceParam(4, 4, 10, svt_aom_highbd_10_variance4x4_sve),
    HbdVarianceParam(4, 8, 10, svt_aom_highbd_10_variance4x8_sve),
    HbdVarianceParam(4, 16, 10, svt_aom_highbd_10_variance4x16_sve),
    HbdVarianceParam(8, 4, 10, svt_aom_highbd_10_variance8x4_sve),
    HbdVarianceParam(8, 8, 10, svt_aom_highbd_10_variance8x8_sve),
    HbdVarianceParam(8, 16, 10, svt_aom_highbd_10_variance8x16_sve),
    HbdVarianceParam(8, 32, 10, svt_aom_highbd_10_variance8x32_sve),
    HbdVarianceParam(16, 4, 10, svt_aom_highbd_10_variance16x4_sve),
    HbdVarianceParam(16, 8, 10, svt_aom_highbd_10_variance16x8_sve),
    HbdVarianceParam(16, 16, 10, svt_aom_highbd_10_variance16x16_sve),
    HbdVarianceParam(16, 32, 10, svt_aom_highbd_10_variance16x32_sve),
    HbdVarianceParam(16, 64, 10, svt_aom_highbd_10_variance16x64_sve),
    HbdVarianceParam(32, 8, 10, svt_aom_highbd_10_variance32x8_sve),
    HbdVarianceParam(32, 16, 10, svt_aom_highbd_10_variance32x16_sve),
    HbdVarianceParam(32, 32, 10, svt_aom_highbd_10_variance32x32_sve),
    HbdVarianceParam(32, 64, 10, svt_aom_highbd_10_variance32x64_sve),
    HbdVarianceParam(64, 16, 10, svt_aom_highbd_10_variance64x16_sve),
    HbdVarianceParam(64, 32, 10, svt_aom_highbd_10_variance64x32_sve),
    HbdVarianceParam(64, 64, 10, svt_aom_highbd_10_variance64x64_sve),
    HbdVarianceParam(64, 128, 10, svt_aom_highbd_10_variance64x128_sve),
    HbdVarianceParam(128, 64, 10, svt_aom_highbd_10_variance128x64_sve),
    HbdVarianceParam(128, 128, 10, svt_aom_highbd_10_variance128x128_sve),
};

INSTANTIATE_TEST_SUITE_P(SVE, HbdVarianceTest,
                         ::testing::ValuesIn(HbdTestVector_sve));
#endif  // HAVE_SVE

#endif  // ARCH_AARCH64

/**
 * @brief Unit test for different implementation of HBD variance with size 16x16
 * and 32x32 functions:
 * - svt_aom_variance_highbd_c
 * - svt_aom_variance_highbd_sse4_1
 * - svt_aom_variance_highbd_avx2
 *
 * Test strategy:
 *  This test case use random source, max source, zero source as test
 * pattern.
 *
 * Expected result:
 *  Results come from reference function and target function are
 * equal.
 *
 * Test cases:
 * - ZeroTest
 * - MaximumTest
 * - MatchTest
 *
 * @author  intel tszumski
 *
 */
using HbdSquareVarianceNoRoundFunc = decltype(&svt_aom_variance_highbd_c);

using HbdSquareVarianceNoRoundParam =
    std::tuple<uint32_t,                      /**< square length */
               HbdSquareVarianceNoRoundFunc>; /**< test function */

class HbdSquareVarianceNoRoundTest
    : public ::testing::TestWithParam<HbdSquareVarianceNoRoundParam> {
  public:
    void *operator new(size_t size) {
        if (void *ptr =
                svt_aom_memalign(alignof(HbdSquareVarianceNoRoundTest), size))
            return ptr;
        throw std::bad_alloc();
    }

    void operator delete(void *ptr) {
        svt_aom_free(ptr);
    }

  protected:
    template <int times>
    void run_zero_test() {
        const HbdSquareVarianceNoRoundFunc tst_func{TEST_GET_PARAM(1)};
        for (int i = 0; i < times; ++i) {
            std::generate_n(src_data_.begin(), MAX_BLOCK_SIZE, [this]() {
                return rnd_.random() & ((1 << bd_) - 1);
            });
            ref_data_ = src_data_;
            uint32_t sse_tst = 0;
            int32_t distortion_tst = tst_func(src_data_.data(),
                                              length_,
                                              ref_data_.data(),
                                              length_,
                                              length_,
                                              length_,
                                              &sse_tst);
            ASSERT_EQ(sse_tst, 0u) << "Expect 0 sse, got: " << sse_tst;
            ASSERT_EQ(distortion_tst, 0)
                << "Expect 0 distortion, got: " << distortion_tst;
        }
    }

    void run_maximum_test() {
        src_data_.fill(0);
        ref_data_.fill((1 << bd_) - 1);

        uint32_t sse_tst = 0, sse_ref = 0;
        const HbdSquareVarianceNoRoundFunc tst_func{TEST_GET_PARAM(1)};

        int32_t distortion_tst = tst_func(src_data_.data(),
                                          length_,
                                          ref_data_.data(),
                                          length_,
                                          length_,
                                          length_,
                                          &sse_tst);

        int32_t distortion_ref = svt_aom_variance_highbd_c(src_data_.data(),
                                                           length_,
                                                           ref_data_.data(),
                                                           length_,
                                                           length_,
                                                           length_,
                                                           &sse_ref);
        ASSERT_EQ(distortion_tst, distortion_ref)
            << "Error at distortion in variance test";
        ASSERT_EQ(sse_tst, sse_ref) << "Error at error sse in variance test";
    }
    template <int times>
    void run_match_test() {
        const HbdSquareVarianceNoRoundFunc tst_func{TEST_GET_PARAM(1)};
        for (int i = 0; i < times; ++i) {
            for (int j = 0; j < MAX_BLOCK_SIZE; ++j) {
                src_data_[j] = rnd_.random() & ((1 << bd_) - 1);
                ref_data_[j] = rnd_.random() & ((1 << bd_) - 1);
            }

            uint32_t sse_tst = 0, sse_ref = 0;

            int32_t distortion_tst = tst_func(src_data_.data(),
                                              length_,
                                              ref_data_.data(),
                                              length_,
                                              length_,
                                              length_,
                                              &sse_tst);

            int32_t distortion_ref = svt_aom_variance_highbd_c(src_data_.data(),
                                                               length_,
                                                               ref_data_.data(),
                                                               length_,
                                                               length_,
                                                               length_,
                                                               &sse_ref);
            ASSERT_EQ(distortion_tst, distortion_ref)
                << "Error at distortion in variance test";
            ASSERT_EQ(sse_tst, sse_ref)
                << "Error at error sse in variance test";
        }
    }

  private:
    SVTRandom rnd_{16, false};
    static constexpr uint32_t bd_{10};
    const uint32_t length_{TEST_GET_PARAM(0)};
    alignas(32) std::array<uint16_t, 2 * MAX_BLOCK_SIZE> src_data_{};
    alignas(32) std::array<uint16_t, 2 * MAX_BLOCK_SIZE> ref_data_{};
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HbdSquareVarianceNoRoundTest);

TEST_P(HbdSquareVarianceNoRoundTest, ZeroTest) {
    run_zero_test<10>();
};

TEST_P(HbdSquareVarianceNoRoundTest, MaximumTest) {
    run_maximum_test();
};

TEST_P(HbdSquareVarianceNoRoundTest, MatchTest) {
    run_match_test<10>();
};

#ifdef ARCH_X86_64

INSTANTIATE_TEST_SUITE_P(
    SSE4_1, HbdSquareVarianceNoRoundTest,
    ::testing::Combine(::testing::Values(16, 32),
                       ::testing::Values(svt_aom_variance_highbd_sse4_1)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, HbdSquareVarianceNoRoundTest,
    ::testing::Combine(::testing::Values(16, 32),
                       ::testing::Values(svt_aom_variance_highbd_avx2)));
#endif

#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

}  // namespace
