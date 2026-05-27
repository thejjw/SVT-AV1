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
 * @file VarianceTest.cc
 *
 * @brief Unit test for variance, mse, sum square functions:
 * - svt_aom_variance{4-128}x{4-128}_{c,sse2,avx2}
 * - svt_aom_get_mb_ss_sse2
 * - aom_mse16x16_{c,avx2}
 * - highbd_variance64_{c,avx2}
 *
 * @author Cidana-Ryan,Cidana-Ivy
 *
 ******************************************************************************/
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "aligned_allocator.hpp"
#include "aom_dsp_rtcd.h"
#include "random.hpp"
#include "util.h"
#include "gtest/gtest.h"

namespace {
using svt_av1_test_tool::aligned_allocator;
using svt_av1_test_tool::SizeOnlyVec;
using svt_av1_test_tool::SVTRandom;  // to generate the random
const auto MAX_BLOCK_SIZE = 128 * 128;

using MSE_NXM_FUNC = decltype(&svt_aom_mse16x16_c);
using TestMseParam = std::tuple<uint32_t, uint32_t, MSE_NXM_FUNC, MSE_NXM_FUNC>;

/**
 * @brief Unit test for mse functions, target functions include:
 *  - aom_mse16x16_{c,avx2}
 *
 * Test strategy:
 *  This test case use random source, max source, zero source as test
 * pattern.
 *
 *
 * Expect result:
 *  Results come from reference functon and target function are
 * equal.
 *
 * Test coverage:
 *
 * Test cases:
 *
 */
class MseTest : public ::testing::TestWithParam<TestMseParam> {
  public:
    DEFINE_ALIGNED_NEW_DELETE(MseTest)
    void run_match_test() {
        constexpr int32_t mask = (1 << 8) - 1;
        SVTRandom rnd{0, mask};
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < MAX_BLOCK_SIZE; ++j) {
                src_data_[j] = rnd.random();
                ref_data_[j] = rnd.random();
            }

            unsigned int res_tst =
                mse_tst_(src_data_.data(), width_, ref_data_.data(), height_);
            unsigned int res_ref =
                mse_ref_(src_data_.data(), width_, ref_data_.data(), height_);
            ASSERT_EQ(res_tst, res_ref) << "Return value error at index: " << i;
        }
    }

    void run_max_test() {
        src_data_.fill(255);
        ref_data_.fill(0);
        unsigned int res_tst =
            mse_tst_(src_data_.data(), width_, ref_data_.data(), width_);
        const uint32_t expected = width_ * height_ * 255 * 255;
        ASSERT_EQ(res_tst, expected)
            << "Return value error at MSE maximum test";
    }

  private:
    alignas(32) std::array<uint8_t, MAX_BLOCK_SIZE> src_data_{};
    alignas(32) std::array<uint8_t, MAX_BLOCK_SIZE> ref_data_{};
    const uint32_t width_{TEST_GET_PARAM(0)};
    const uint32_t height_{TEST_GET_PARAM(1)};
    const MSE_NXM_FUNC mse_tst_{TEST_GET_PARAM(2)};
    const MSE_NXM_FUNC mse_ref_{TEST_GET_PARAM(3)};
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(MseTest);

TEST_P(MseTest, MatchTest) {
    run_match_test();
};

TEST_P(MseTest, MaxTest) {
    run_max_test();
};

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(SSE2, MseTest,
                         ::testing::Values(TestMseParam(16, 16,
                                                        &svt_aom_mse16x16_sse2,
                                                        &svt_aom_mse16x16_c)));

INSTANTIATE_TEST_SUITE_P(AVX2, MseTest,
                         ::testing::Values(TestMseParam(16, 16,
                                                        &svt_aom_mse16x16_avx2,
                                                        &svt_aom_mse16x16_c)));

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(NEON, MseTest,
                         ::testing::Values(TestMseParam(16, 16,
                                                        &svt_aom_mse16x16_neon,
                                                        &svt_aom_mse16x16_c)));

#if HAVE_NEON_DOTPROD
INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, MseTest,
    ::testing::Values(TestMseParam(16, 16, &svt_aom_mse16x16_neon_dotprod,
                                   &svt_aom_mse16x16_c)));
#endif  // HAVE_NEON_DOTPROD

#endif  // ARCH_AARCH64

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
using MSE_HIGHBD_NXM_FUNC = decltype(&svt_aom_highbd_mse16x16_c);
using TestMseParamHighbd =
    std::tuple<uint32_t, uint32_t, MSE_HIGHBD_NXM_FUNC, MSE_HIGHBD_NXM_FUNC>;
class MseTestHighbd : public ::testing::TestWithParam<TestMseParamHighbd> {
  public:
    DEFINE_ALIGNED_NEW_DELETE(MseTestHighbd)
    void run_match_test() {
        constexpr int32_t mask = (1 << 10) - 1;
        SVTRandom rnd{0, mask};
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < MAX_BLOCK_SIZE; ++j) {
                src_data_[j] = rnd.random();
                ref_data_[j] = rnd.random();
            }

            uint32_t sse_ref = mse_ref_(CONVERT_TO_BYTEPTR(src_data_.data()),
                                        width_,
                                        CONVERT_TO_BYTEPTR(ref_data_.data()),
                                        height_);
            uint32_t sse_tst = mse_tst_(CONVERT_TO_BYTEPTR(src_data_.data()),
                                        width_,
                                        CONVERT_TO_BYTEPTR(ref_data_.data()),
                                        height_);
            ASSERT_EQ(sse_tst, sse_ref) << "SSE Error at index: " << i;
        }
    }

    void run_max_test() {
        constexpr int32_t mask = (1 << 10) - 1;
        src_data_.fill(mask);
        ref_data_.fill(0);
        uint32_t sse_ref = mse_ref_(CONVERT_TO_BYTEPTR(src_data_.data()),
                                    width_,
                                    CONVERT_TO_BYTEPTR(ref_data_.data()),
                                    height_);
        uint32_t sse_tst = mse_tst_(CONVERT_TO_BYTEPTR(src_data_.data()),
                                    width_,
                                    CONVERT_TO_BYTEPTR(ref_data_.data()),
                                    height_);

        ASSERT_EQ(sse_tst, sse_ref) << "Error at MSE maximum test ";
    }

  private:
    alignas(32) std::array<uint16_t, MAX_BLOCK_SIZE * 2> src_data_{};
    alignas(32) std::array<uint16_t, MAX_BLOCK_SIZE * 2> ref_data_{};
    const uint32_t width_{TEST_GET_PARAM(0)};
    const uint32_t height_{TEST_GET_PARAM(1)};
    const MSE_HIGHBD_NXM_FUNC mse_tst_{TEST_GET_PARAM(2)};
    const MSE_HIGHBD_NXM_FUNC mse_ref_{TEST_GET_PARAM(3)};
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(MseTestHighbd);

TEST_P(MseTestHighbd, MatchTest) {
    run_match_test();
};

TEST_P(MseTestHighbd, MaxTest) {
    run_max_test();
};

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE2, MseTestHighbd,
    ::testing::Values(TestMseParamHighbd(16, 16, &svt_aom_highbd_mse16x16_sse2,
                                         &svt_aom_highbd_mse16x16_c)));

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, MseTestHighbd,
    ::testing::Values(TestMseParamHighbd(16, 16, &svt_aom_highbd_mse16x16_neon,
                                         &svt_aom_highbd_mse16x16_c)));

#endif  // ARCH_AARCH64

#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

// Variance test
using VARIANCE_NXM_FUNC = uint32_t (*)(const uint8_t *src_ptr,
                                       int32_t src_stride,
                                       const uint8_t *ref_ptr,
                                       int32_t recon_stride, uint32_t *sse);

using VarianceParam =
    std::tuple<uint32_t, uint32_t, VARIANCE_NXM_FUNC, VARIANCE_NXM_FUNC>;

/**
 * @brief Unit test for variance functions, target functions include:
 *  - - svt_aom_variance{4-128}x{4-128}_{c,avx2}
 *
 * Test strategy:
 *  This test case contains zero test, random value test, one quarter test as
 * test pattern.
 *
 *
 * Expect result:
 *  Results come from reference functon and target function are
 * equal.
 *
 * Test coverage:
 *
 * Test cases:
 *
 */
class VarianceTest : public ::testing::TestWithParam<VarianceParam> {
  public:
    DEFINE_ALIGNED_NEW_DELETE(VarianceTest)
    void run_zero_test() {
        for (int i = 0; i < 256; ++i) {
            src_data_.fill(i);
            for (int j = 0; j < 256; ++j) {
                ref_data_.fill(j);

                uint32_t sse;
                uint32_t var = func_asm_(
                    src_data_.data(), width_, ref_data_.data(), width_, &sse);
                ASSERT_EQ(var, 0u)
                    << "Variance is mismatched, src values: " << i
                    << " ref values: " << j;
            }
        }
    }

    void run_match_test() {
        constexpr int32_t mask = (1 << 8) - 1;
        SVTRandom rnd{0, mask};
        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < MAX_BLOCK_SIZE; j++) {
                src_data_[j] = rnd.random();
                ref_data_[j] = rnd.random();
            }
            uint32_t sse_c, sse_asm;

            uint32_t var_c = func_c_(
                src_data_.data(), width_, ref_data_.data(), width_ + 1, &sse_c);
            uint32_t var_asm = func_asm_(src_data_.data(),
                                         width_,
                                         ref_data_.data(),
                                         width_ + 1,
                                         &sse_asm);
            ASSERT_EQ(sse_c, sse_asm) << "Error at test index: " << i;
            ASSERT_EQ(var_c, var_asm) << "Error at test index: " << i;
        }
    }

    void run_one_quarter_test() {
        const int half = width_ * height_ / 2;
        src_data_.fill(255);
        ref_data_.fill(255);
        std::fill(ref_data_.begin() + half, ref_data_.end(), 0);
        uint32_t sse_asm;
        uint32_t var_asm = func_asm_(
            src_data_.data(), width_, ref_data_.data(), width_, &sse_asm);
        const uint32_t expected = width_ * height_ * 255 * 255 / 4;
        ASSERT_EQ(var_asm, expected);
    }

  private:
    alignas(32) std::array<uint8_t, MAX_BLOCK_SIZE> src_data_{};
    alignas(32) std::array<uint8_t, MAX_BLOCK_SIZE> ref_data_{};
    const uint32_t width_{TEST_GET_PARAM(0)};
    const uint32_t height_{TEST_GET_PARAM(1)};
    const VARIANCE_NXM_FUNC func_c_{TEST_GET_PARAM(2)};
    const VARIANCE_NXM_FUNC func_asm_{TEST_GET_PARAM(3)};
};

TEST_P(VarianceTest, ZeroTest) {
    run_zero_test();
};

TEST_P(VarianceTest, MatchTest) {
    run_match_test();
};

TEST_P(VarianceTest, OneQuarterTest) {
    run_one_quarter_test();
};

#ifdef ARCH_X86_64
const VarianceParam variance_func_sse2[] = {
    VarianceParam(4, 4, &svt_aom_variance4x4_c, &svt_aom_variance4x4_sse2),
    VarianceParam(4, 8, &svt_aom_variance4x8_c, &svt_aom_variance4x8_sse2),
    VarianceParam(4, 16, &svt_aom_variance4x16_c, &svt_aom_variance4x16_sse2),
    VarianceParam(8, 4, &svt_aom_variance8x4_c, &svt_aom_variance8x4_sse2),
    VarianceParam(8, 8, &svt_aom_variance8x8_c, &svt_aom_variance8x8_sse2),
    VarianceParam(8, 16, &svt_aom_variance8x16_c, &svt_aom_variance8x16_sse2),
    VarianceParam(8, 32, &svt_aom_variance8x32_c, &svt_aom_variance8x32_sse2),
    VarianceParam(16, 4, &svt_aom_variance16x4_c, &svt_aom_variance16x4_sse2),
    VarianceParam(16, 8, &svt_aom_variance16x8_c, &svt_aom_variance16x8_sse2),
    VarianceParam(16, 16, &svt_aom_variance16x16_c,
                  &svt_aom_variance16x16_sse2),
    VarianceParam(16, 32, &svt_aom_variance16x32_c,
                  &svt_aom_variance16x32_sse2),
    VarianceParam(16, 64, &svt_aom_variance16x64_c,
                  &svt_aom_variance16x64_sse2),
    VarianceParam(32, 8, &svt_aom_variance32x8_c, &svt_aom_variance32x8_sse2),
    VarianceParam(32, 16, &svt_aom_variance32x16_c,
                  &svt_aom_variance32x16_sse2),
    VarianceParam(32, 32, &svt_aom_variance32x32_c,
                  &svt_aom_variance32x32_sse2),
    VarianceParam(32, 64, &svt_aom_variance32x64_c,
                  &svt_aom_variance32x64_sse2),
    VarianceParam(64, 16, &svt_aom_variance64x16_c,
                  &svt_aom_variance64x16_sse2),
    VarianceParam(64, 32, &svt_aom_variance64x32_c,
                  &svt_aom_variance64x32_sse2),
    VarianceParam(64, 64, &svt_aom_variance64x64_c,
                  &svt_aom_variance64x64_sse2),
    VarianceParam(64, 128, &svt_aom_variance64x128_c,
                  &svt_aom_variance64x128_sse2),
    VarianceParam(128, 64, &svt_aom_variance128x64_c,
                  &svt_aom_variance128x64_sse2),
    VarianceParam(128, 128, &svt_aom_variance128x128_c,
                  &svt_aom_variance128x128_sse2),
};

const VarianceParam variance_func_avx2[] = {
    VarianceParam(8, 4, &svt_aom_variance8x4_c, &svt_aom_variance8x4_avx2),
    VarianceParam(8, 8, &svt_aom_variance8x8_c, &svt_aom_variance8x8_avx2),
    VarianceParam(8, 16, &svt_aom_variance8x16_c, &svt_aom_variance8x16_avx2),
    VarianceParam(8, 32, &svt_aom_variance8x32_c, &svt_aom_variance8x32_avx2),
    VarianceParam(16, 4, &svt_aom_variance16x4_c, &svt_aom_variance16x4_avx2),
    VarianceParam(16, 8, &svt_aom_variance16x8_c, &svt_aom_variance16x8_avx2),
    VarianceParam(16, 16, &svt_aom_variance16x16_c,
                  &svt_aom_variance16x16_avx2),
    VarianceParam(16, 32, &svt_aom_variance16x32_c,
                  &svt_aom_variance16x32_avx2),
    VarianceParam(16, 64, &svt_aom_variance16x64_c,
                  &svt_aom_variance16x64_avx2),
    VarianceParam(32, 8, &svt_aom_variance32x8_c, &svt_aom_variance32x8_avx2),
    VarianceParam(32, 16, &svt_aom_variance32x16_c,
                  &svt_aom_variance32x16_avx2),
    VarianceParam(32, 32, &svt_aom_variance32x32_c,
                  &svt_aom_variance32x32_avx2),
    VarianceParam(32, 64, &svt_aom_variance32x64_c,
                  &svt_aom_variance32x64_avx2),
    VarianceParam(64, 16, &svt_aom_variance64x16_c,
                  &svt_aom_variance64x16_avx2),
    VarianceParam(64, 32, &svt_aom_variance64x32_c,
                  &svt_aom_variance64x32_avx2),
    VarianceParam(64, 64, &svt_aom_variance64x64_c,
                  &svt_aom_variance64x64_avx2),
    VarianceParam(64, 128, &svt_aom_variance64x128_c,
                  &svt_aom_variance64x128_avx2),
    VarianceParam(128, 64, &svt_aom_variance128x64_c,
                  &svt_aom_variance128x64_avx2),
    VarianceParam(128, 128, &svt_aom_variance128x128_c,
                  &svt_aom_variance128x128_avx2),
};

INSTANTIATE_TEST_SUITE_P(SSE2, VarianceTest,
                         ::testing::ValuesIn(variance_func_sse2));

INSTANTIATE_TEST_SUITE_P(AVX2, VarianceTest,
                         ::testing::ValuesIn(variance_func_avx2));

#if EN_AVX512_SUPPORT
const VarianceParam variance_func_avx512[] = {
    VarianceParam(32, 8, &svt_aom_variance32x8_c, &svt_aom_variance32x8_avx512),
    VarianceParam(32, 16, &svt_aom_variance32x16_c,
                  &svt_aom_variance32x16_avx512),
    VarianceParam(32, 32, &svt_aom_variance32x32_c,
                  &svt_aom_variance32x32_avx512),
    VarianceParam(32, 64, &svt_aom_variance32x64_c,
                  &svt_aom_variance32x64_avx512),
    VarianceParam(64, 16, &svt_aom_variance64x16_c,
                  &svt_aom_variance64x16_avx512),
    VarianceParam(64, 32, &svt_aom_variance64x32_c,
                  &svt_aom_variance64x32_avx512),
    VarianceParam(64, 64, &svt_aom_variance64x64_c,
                  &svt_aom_variance64x64_avx512),
    VarianceParam(64, 128, &svt_aom_variance64x128_c,
                  &svt_aom_variance64x128_avx512),
    VarianceParam(128, 64, &svt_aom_variance128x64_c,
                  &svt_aom_variance128x64_avx512),
    VarianceParam(128, 128, &svt_aom_variance128x128_c,
                  &svt_aom_variance128x128_avx512),
};

INSTANTIATE_TEST_SUITE_P(AVX512, VarianceTest,
                         ::testing::ValuesIn(variance_func_avx512));
#endif

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64

const VarianceParam variance_func_neon[] = {
    VarianceParam(4, 4, &svt_aom_variance4x4_c, &svt_aom_variance4x4_neon),
    VarianceParam(4, 8, &svt_aom_variance4x8_c, &svt_aom_variance4x8_neon),
    VarianceParam(4, 16, &svt_aom_variance4x16_c, &svt_aom_variance4x16_neon),
    VarianceParam(8, 4, &svt_aom_variance8x4_c, &svt_aom_variance8x4_neon),
    VarianceParam(8, 8, &svt_aom_variance8x8_c, &svt_aom_variance8x8_neon),
    VarianceParam(8, 16, &svt_aom_variance8x16_c, &svt_aom_variance8x16_neon),
    VarianceParam(8, 32, &svt_aom_variance8x32_c, &svt_aom_variance8x32_neon),
    VarianceParam(16, 4, &svt_aom_variance16x4_c, &svt_aom_variance16x4_neon),
    VarianceParam(16, 8, &svt_aom_variance16x8_c, &svt_aom_variance16x8_neon),
    VarianceParam(16, 16, &svt_aom_variance16x16_c,
                  &svt_aom_variance16x16_neon),
    VarianceParam(16, 32, &svt_aom_variance16x32_c,
                  &svt_aom_variance16x32_neon),
    VarianceParam(16, 64, &svt_aom_variance16x64_c,
                  &svt_aom_variance16x64_neon),
    VarianceParam(32, 8, &svt_aom_variance32x8_c, &svt_aom_variance32x8_neon),
    VarianceParam(32, 16, &svt_aom_variance32x16_c,
                  &svt_aom_variance32x16_neon),
    VarianceParam(32, 32, &svt_aom_variance32x32_c,
                  &svt_aom_variance32x32_neon),
    VarianceParam(32, 64, &svt_aom_variance32x64_c,
                  &svt_aom_variance32x64_neon),
    VarianceParam(64, 16, &svt_aom_variance64x16_c,
                  &svt_aom_variance64x16_neon),
    VarianceParam(64, 32, &svt_aom_variance64x32_c,
                  &svt_aom_variance64x32_neon),
    VarianceParam(64, 64, &svt_aom_variance64x64_c,
                  &svt_aom_variance64x64_neon),
    VarianceParam(64, 128, &svt_aom_variance64x128_c,
                  &svt_aom_variance64x128_neon),
    VarianceParam(128, 64, &svt_aom_variance128x64_c,
                  &svt_aom_variance128x64_neon),
    VarianceParam(128, 128, &svt_aom_variance128x128_c,
                  &svt_aom_variance128x128_neon),
};

INSTANTIATE_TEST_SUITE_P(NEON, VarianceTest,
                         ::testing::ValuesIn(variance_func_neon));

#if HAVE_NEON_DOTPROD
const VarianceParam variance_func_neon_dotprod[] = {
    VarianceParam(4, 8, &svt_aom_variance4x8_c,
                  &svt_aom_variance4x8_neon_dotprod),
    VarianceParam(4, 16, &svt_aom_variance4x16_c,
                  &svt_aom_variance4x16_neon_dotprod),
    VarianceParam(8, 4, &svt_aom_variance8x4_c,
                  &svt_aom_variance8x4_neon_dotprod),
    VarianceParam(8, 8, &svt_aom_variance8x8_c,
                  &svt_aom_variance8x8_neon_dotprod),
    VarianceParam(8, 16, &svt_aom_variance8x16_c,
                  &svt_aom_variance8x16_neon_dotprod),
    VarianceParam(8, 32, &svt_aom_variance8x32_c,
                  &svt_aom_variance8x32_neon_dotprod),
    VarianceParam(16, 4, &svt_aom_variance16x4_c,
                  &svt_aom_variance16x4_neon_dotprod),
    VarianceParam(16, 8, &svt_aom_variance16x8_c,
                  &svt_aom_variance16x8_neon_dotprod),
    VarianceParam(16, 16, &svt_aom_variance16x16_c,
                  &svt_aom_variance16x16_neon_dotprod),
    VarianceParam(16, 32, &svt_aom_variance16x32_c,
                  &svt_aom_variance16x32_neon_dotprod),
    VarianceParam(16, 64, &svt_aom_variance16x64_c,
                  &svt_aom_variance16x64_neon_dotprod),
    VarianceParam(32, 8, &svt_aom_variance32x8_c,
                  &svt_aom_variance32x8_neon_dotprod),
    VarianceParam(32, 16, &svt_aom_variance32x16_c,
                  &svt_aom_variance32x16_neon_dotprod),
    VarianceParam(32, 32, &svt_aom_variance32x32_c,
                  &svt_aom_variance32x32_neon_dotprod),
    VarianceParam(32, 64, &svt_aom_variance32x64_c,
                  &svt_aom_variance32x64_neon_dotprod),
    VarianceParam(64, 16, &svt_aom_variance64x16_c,
                  &svt_aom_variance64x16_neon_dotprod),
    VarianceParam(64, 32, &svt_aom_variance64x32_c,
                  &svt_aom_variance64x32_neon_dotprod),
    VarianceParam(64, 64, &svt_aom_variance64x64_c,
                  &svt_aom_variance64x64_neon_dotprod),
    VarianceParam(64, 128, &svt_aom_variance64x128_c,
                  &svt_aom_variance64x128_neon_dotprod),
    VarianceParam(128, 64, &svt_aom_variance128x64_c,
                  &svt_aom_variance128x64_neon_dotprod),
    VarianceParam(128, 128, &svt_aom_variance128x128_c,
                  &svt_aom_variance128x128_neon_dotprod),
};

INSTANTIATE_TEST_SUITE_P(NEON_DOTPROD, VarianceTest,
                         ::testing::ValuesIn(variance_func_neon_dotprod));
#endif  // HAVE_NEON_DOTPROD

#endif  // ARCH_AARCH64

using SubpixVarMxNFunc = unsigned int (*)(const uint8_t *a, int a_stride,
                                          int xoffset, int yoffset,
                                          const uint8_t *b, int b_stride,
                                          unsigned int *sse);

using TestParams =
    std::tuple<uint32_t, uint32_t, SubpixVarMxNFunc, SubpixVarMxNFunc>;

class SubpelVarianceTest : public ::testing::TestWithParam<TestParams> {
  protected:
    void RefTest();
    void ExtremeRefTest();

    const uint32_t log2width{TEST_GET_PARAM(0)}, log2height{TEST_GET_PARAM(1)};
    const SubpixVarMxNFunc func_tst{TEST_GET_PARAM(2)};
    const SubpixVarMxNFunc func_ref{TEST_GET_PARAM(3)};
    const int width{1 << log2width}, height{1 << log2height};
    const int block_size{width * height};
    static constexpr int mask{(1 << EB_EIGHT_BIT) - 1};

    using AlignedVec = SizeOnlyVec<uint8_t, aligned_allocator<uint8_t>>;
    AlignedVec src_{std::size_t(block_size)};
    AlignedVec ref_{std::size_t(block_size + width + height + 1)};
    AlignedVec sec_{std::size_t(block_size)};
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SubpelVarianceTest);

void SubpelVarianceTest::RefTest() {
#if ARCH_AARCH64
    // Neon implementation of sub_pixel variance functions call variance
    // functions through the pointers setup by rtcd, so set them up properly.
    svt_aom_setup_rtcd_internal(EB_CPU_FLAGS_NEON);
#endif
    SVTRandom rnd_{0, mask};

    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            std::generate(
                src_.begin(), src_.end(), [&]() { return rnd_.Rand8(); });
            std::generate(
                ref_.begin(), ref_.end(), [&]() { return rnd_.Rand8(); });
            unsigned int sse1, sse2;
            unsigned int var1 = func_ref(
                ref_.data(), width + 1, x, y, src_.data(), width, &sse1);
            unsigned int var2 = func_tst(
                ref_.data(), width + 1, x, y, src_.data(), width, &sse2);
            EXPECT_EQ(sse1, sse2) << "at position " << x << ", " << y;
            EXPECT_EQ(var1, var2) << "at position " << x << ", " << y;
        }
    }
}

void SubpelVarianceTest::ExtremeRefTest() {
// TODO remove once PRs are merged
#if ARCH_AARCH64
    // Neon implementation of sub_pixel variance functions call variance
    // functions through the pointers setup by rtcd, so set them up properly.
    svt_aom_setup_rtcd_internal(EB_CPU_FLAGS_NEON);
#endif
    // Compare against reference.
    // Src: Set the first half of values to 0, the second half to the maximum.
    // Ref: Set the first half of values to the maximum, the second half to 0.
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            const int half = block_size / 2;
            std::fill(src_.begin(), src_.begin() + half, 0);
            std::fill(src_.begin() + half, src_.end(), 255);
            std::fill(ref_.begin(), ref_.begin() + half, 255);
            std::fill(ref_.begin() + half, ref_.end(), 0);
            unsigned int sse1, sse2;
            unsigned int var1 = func_ref(
                ref_.data(), width + 1, x, y, src_.data(), width, &sse1);
            unsigned int var2 = func_tst(
                ref_.data(), width + 1, x, y, src_.data(), width, &sse2);
            EXPECT_EQ(sse1, sse2)
                << "for xoffset " << x << " and yoffset " << y;
            EXPECT_EQ(var1, var2)
                << "for xoffset " << x << " and yoffset " << y;
        }
    }
}

TEST_P(SubpelVarianceTest, Ref) {
    RefTest();
}
TEST_P(SubpelVarianceTest, ExtremeRef) {
    ExtremeRefTest();
}

#ifdef ARCH_X86_64
const TestParams kArraySubpelVariance_sse2[] = {
    // clang-format off
    { 7, 7, &svt_aom_sub_pixel_variance128x128_sse2,
      &svt_aom_sub_pixel_variance128x128_c },
    { 7, 6, &svt_aom_sub_pixel_variance128x64_sse2,
      &svt_aom_sub_pixel_variance128x64_c },
    { 6, 7, &svt_aom_sub_pixel_variance64x128_sse2,
      &svt_aom_sub_pixel_variance64x128_c },
    { 6, 6, &svt_aom_sub_pixel_variance64x64_sse2,
      &svt_aom_sub_pixel_variance64x64_c },
    { 6, 5, &svt_aom_sub_pixel_variance64x32_sse2,
      &svt_aom_sub_pixel_variance64x32_c },
    { 5, 6, &svt_aom_sub_pixel_variance32x64_sse2,
      &svt_aom_sub_pixel_variance32x64_c },
    { 5, 5, &svt_aom_sub_pixel_variance32x32_sse2,
      &svt_aom_sub_pixel_variance32x32_c },
    { 5, 4, &svt_aom_sub_pixel_variance32x16_sse2,
      &svt_aom_sub_pixel_variance32x16_c },
    { 4, 5, &svt_aom_sub_pixel_variance16x32_sse2,
      &svt_aom_sub_pixel_variance16x32_c },
    { 4, 4, &svt_aom_sub_pixel_variance16x16_sse2,
      &svt_aom_sub_pixel_variance16x16_c },
    { 4, 3, &svt_aom_sub_pixel_variance16x8_sse2,
      &svt_aom_sub_pixel_variance16x8_c },
    { 3, 4, &svt_aom_sub_pixel_variance8x16_sse2,
      &svt_aom_sub_pixel_variance8x16_c },
    { 3, 3, &svt_aom_sub_pixel_variance8x8_sse2,
      &svt_aom_sub_pixel_variance8x8_c },
    { 3, 2, &svt_aom_sub_pixel_variance8x4_sse2,
      &svt_aom_sub_pixel_variance8x4_c },
    { 2, 3, &svt_aom_sub_pixel_variance4x8_sse2,
      &svt_aom_sub_pixel_variance4x8_c },
    { 2, 2, &svt_aom_sub_pixel_variance4x4_sse2,
      &svt_aom_sub_pixel_variance4x4_c },
    { 6, 4, &svt_aom_sub_pixel_variance64x16_sse2,
      &svt_aom_sub_pixel_variance64x16_c },
    { 4, 6, &svt_aom_sub_pixel_variance16x64_sse2,
      &svt_aom_sub_pixel_variance16x64_c },
    { 5, 3, &svt_aom_sub_pixel_variance32x8_sse2,
      &svt_aom_sub_pixel_variance32x8_c },
    { 3, 5, &svt_aom_sub_pixel_variance8x32_sse2,
      &svt_aom_sub_pixel_variance8x32_c },
    { 4, 2, &svt_aom_sub_pixel_variance16x4_sse2,
      &svt_aom_sub_pixel_variance16x4_c },
    { 2, 4, &svt_aom_sub_pixel_variance4x16_sse2,
      &svt_aom_sub_pixel_variance4x16_c }
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(SSE2, SubpelVarianceTest,
                         ::testing::ValuesIn(kArraySubpelVariance_sse2));

const TestParams kArraySubpelVariance_ssse3[] = {
    // clang-format off
    { 7, 7, &svt_aom_sub_pixel_variance128x128_ssse3,
      &svt_aom_sub_pixel_variance128x128_c },
    { 7, 6, &svt_aom_sub_pixel_variance128x64_ssse3,
      &svt_aom_sub_pixel_variance128x64_c },
    { 6, 7, &svt_aom_sub_pixel_variance64x128_ssse3,
      &svt_aom_sub_pixel_variance64x128_c },
    { 6, 6, &svt_aom_sub_pixel_variance64x64_ssse3,
      &svt_aom_sub_pixel_variance64x64_c },
    { 6, 5, &svt_aom_sub_pixel_variance64x32_ssse3,
      &svt_aom_sub_pixel_variance64x32_c },
    { 5, 6, &svt_aom_sub_pixel_variance32x64_ssse3,
      &svt_aom_sub_pixel_variance32x64_c },
    { 5, 5, &svt_aom_sub_pixel_variance32x32_ssse3,
      &svt_aom_sub_pixel_variance32x32_c },
    { 5, 4, &svt_aom_sub_pixel_variance32x16_ssse3,
      &svt_aom_sub_pixel_variance32x16_c },
    { 4, 5, &svt_aom_sub_pixel_variance16x32_ssse3,
      &svt_aom_sub_pixel_variance16x32_c },
    { 4, 4, &svt_aom_sub_pixel_variance16x16_ssse3,
      &svt_aom_sub_pixel_variance16x16_c },
    { 4, 3, &svt_aom_sub_pixel_variance16x8_ssse3,
      &svt_aom_sub_pixel_variance16x8_c },
    { 3, 4, &svt_aom_sub_pixel_variance8x16_ssse3,
      &svt_aom_sub_pixel_variance8x16_c },
    { 3, 3, &svt_aom_sub_pixel_variance8x8_ssse3,
      &svt_aom_sub_pixel_variance8x8_c },
    { 3, 2, &svt_aom_sub_pixel_variance8x4_ssse3,
      &svt_aom_sub_pixel_variance8x4_c },
    { 2, 3, &svt_aom_sub_pixel_variance4x8_ssse3,
      &svt_aom_sub_pixel_variance4x8_c },
    { 2, 2, &svt_aom_sub_pixel_variance4x4_ssse3,
      &svt_aom_sub_pixel_variance4x4_c },
    { 6, 4, &svt_aom_sub_pixel_variance64x16_ssse3,
      &svt_aom_sub_pixel_variance64x16_c },
    { 4, 6, &svt_aom_sub_pixel_variance16x64_ssse3,
      &svt_aom_sub_pixel_variance16x64_c },
    { 5, 3, &svt_aom_sub_pixel_variance32x8_ssse3,
      &svt_aom_sub_pixel_variance32x8_c },
    { 3, 5, &svt_aom_sub_pixel_variance8x32_ssse3,
      &svt_aom_sub_pixel_variance8x32_c },
    { 4, 2, &svt_aom_sub_pixel_variance16x4_ssse3,
      &svt_aom_sub_pixel_variance16x4_c },
    { 2, 4, &svt_aom_sub_pixel_variance4x16_ssse3,
      &svt_aom_sub_pixel_variance4x16_c }
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(SSSE3, SubpelVarianceTest,
                         ::testing::ValuesIn(kArraySubpelVariance_ssse3));

const TestParams kArraySubpelVariance_avx2[] = {
    // clang-format off
    { 7, 7, &svt_aom_sub_pixel_variance128x128_avx2,
      &svt_aom_sub_pixel_variance128x128_c },
    { 7, 6, &svt_aom_sub_pixel_variance128x64_avx2,
      &svt_aom_sub_pixel_variance128x64_c },
    { 6, 7, &svt_aom_sub_pixel_variance64x128_avx2,
      &svt_aom_sub_pixel_variance64x128_c },
    { 6, 6, &svt_aom_sub_pixel_variance64x64_avx2,
      &svt_aom_sub_pixel_variance64x64_c },
    { 6, 5, &svt_aom_sub_pixel_variance64x32_avx2,
      &svt_aom_sub_pixel_variance64x32_c },
    { 5, 6, &svt_aom_sub_pixel_variance32x64_avx2,
      &svt_aom_sub_pixel_variance32x64_c },
    { 5, 5, &svt_aom_sub_pixel_variance32x32_avx2,
      &svt_aom_sub_pixel_variance32x32_c },
    { 5, 4, &svt_aom_sub_pixel_variance32x16_avx2,
      &svt_aom_sub_pixel_variance32x16_c },
    { 4, 5, &svt_aom_sub_pixel_variance16x32_avx2,
      &svt_aom_sub_pixel_variance16x32_c },
    { 4, 4, &svt_aom_sub_pixel_variance16x16_avx2,
      &svt_aom_sub_pixel_variance16x16_c },
    { 4, 3, &svt_aom_sub_pixel_variance16x8_avx2,
      &svt_aom_sub_pixel_variance16x8_c },
    { 4, 6, &svt_aom_sub_pixel_variance16x64_avx2,
      &svt_aom_sub_pixel_variance16x64_c },
    { 4, 2, &svt_aom_sub_pixel_variance16x4_avx2,
      &svt_aom_sub_pixel_variance16x4_c }
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(AVX2, SubpelVarianceTest,
                         ::testing::ValuesIn(kArraySubpelVariance_avx2));

#if EN_AVX512_SUPPORT
const TestParams kArraySubpelVariance_avx512[] = {
    // clang-format off
    { 7, 7, &svt_aom_sub_pixel_variance128x128_avx512,
      &svt_aom_sub_pixel_variance128x128_c },
    { 7, 6, &svt_aom_sub_pixel_variance128x64_avx512,
      &svt_aom_sub_pixel_variance128x64_c },
    { 6, 7, &svt_aom_sub_pixel_variance64x128_avx512,
      &svt_aom_sub_pixel_variance64x128_c },
    { 6, 6, &svt_aom_sub_pixel_variance64x64_avx512,
      &svt_aom_sub_pixel_variance64x64_c },
    { 6, 5, &svt_aom_sub_pixel_variance64x32_avx512,
      &svt_aom_sub_pixel_variance64x32_c },
    { 5, 6, &svt_aom_sub_pixel_variance32x64_avx512,
      &svt_aom_sub_pixel_variance32x64_c },
    { 5, 5, &svt_aom_sub_pixel_variance32x32_avx512,
      &svt_aom_sub_pixel_variance32x32_c },
    { 5, 4, &svt_aom_sub_pixel_variance32x16_avx512,
      &svt_aom_sub_pixel_variance32x16_c }
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(AVX512, SubpelVarianceTest,
                         ::testing::ValuesIn(kArraySubpelVariance_avx512));
#endif
#endif  // ARCH_X86_64

#if ARCH_AARCH64
const TestParams kArraySubpelVariance_neon[] = {
    // clang-format off
    { 7, 7, &svt_aom_sub_pixel_variance128x128_neon,
      &svt_aom_sub_pixel_variance128x128_c },
    { 7, 6, &svt_aom_sub_pixel_variance128x64_neon,
      &svt_aom_sub_pixel_variance128x64_c },
    { 6, 7, &svt_aom_sub_pixel_variance64x128_neon,
      &svt_aom_sub_pixel_variance64x128_c },
    { 6, 6, &svt_aom_sub_pixel_variance64x64_neon,
      &svt_aom_sub_pixel_variance64x64_c },
    { 6, 5, &svt_aom_sub_pixel_variance64x32_neon,
      &svt_aom_sub_pixel_variance64x32_c },
    { 5, 6, &svt_aom_sub_pixel_variance32x64_neon,
      &svt_aom_sub_pixel_variance32x64_c },
    { 5, 5, &svt_aom_sub_pixel_variance32x32_neon,
      &svt_aom_sub_pixel_variance32x32_c },
    { 5, 4, &svt_aom_sub_pixel_variance32x16_neon,
      &svt_aom_sub_pixel_variance32x16_c },
    { 4, 5, &svt_aom_sub_pixel_variance16x32_neon,
      &svt_aom_sub_pixel_variance16x32_c },
    { 4, 4, &svt_aom_sub_pixel_variance16x16_neon,
      &svt_aom_sub_pixel_variance16x16_c },
    { 4, 3, &svt_aom_sub_pixel_variance16x8_neon,
      &svt_aom_sub_pixel_variance16x8_c },
    { 3, 4, &svt_aom_sub_pixel_variance8x16_neon,
      &svt_aom_sub_pixel_variance8x16_c },
    { 3, 3, &svt_aom_sub_pixel_variance8x8_neon,
      &svt_aom_sub_pixel_variance8x8_c },
    { 3, 2, &svt_aom_sub_pixel_variance8x4_neon,
      &svt_aom_sub_pixel_variance8x4_c },
    { 2, 3, &svt_aom_sub_pixel_variance4x8_neon,
      &svt_aom_sub_pixel_variance4x8_c },
    { 2, 2, &svt_aom_sub_pixel_variance4x4_neon,
      &svt_aom_sub_pixel_variance4x4_c },
    { 6, 4, &svt_aom_sub_pixel_variance64x16_neon,
      &svt_aom_sub_pixel_variance64x16_c },
    { 4, 6, &svt_aom_sub_pixel_variance16x64_neon,
      &svt_aom_sub_pixel_variance16x64_c },
    { 5, 3, &svt_aom_sub_pixel_variance32x8_neon,
      &svt_aom_sub_pixel_variance32x8_c },
    { 3, 5, &svt_aom_sub_pixel_variance8x32_neon,
      &svt_aom_sub_pixel_variance8x32_c },
    { 4, 2, &svt_aom_sub_pixel_variance16x4_neon,
      &svt_aom_sub_pixel_variance16x4_c },
    { 2, 4, &svt_aom_sub_pixel_variance4x16_neon,
      &svt_aom_sub_pixel_variance4x16_c }
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(NEON, SubpelVarianceTest,
                         ::testing::ValuesIn(kArraySubpelVariance_neon));

#if HAVE_NEON_DOTPROD
const TestParams kArraySubpelVariance_neon_dotprod[] = {
    // clang-format off
    { 7, 7, &svt_aom_sub_pixel_variance128x128_neon_dotprod,
      &svt_aom_sub_pixel_variance128x128_c },
    { 7, 6, &svt_aom_sub_pixel_variance128x64_neon_dotprod,
      &svt_aom_sub_pixel_variance128x64_c },
    { 6, 7, &svt_aom_sub_pixel_variance64x128_neon_dotprod,
      &svt_aom_sub_pixel_variance64x128_c },
    { 6, 6, &svt_aom_sub_pixel_variance64x64_neon_dotprod,
      &svt_aom_sub_pixel_variance64x64_c },
    { 6, 5, &svt_aom_sub_pixel_variance64x32_neon_dotprod,
      &svt_aom_sub_pixel_variance64x32_c },
    { 5, 6, &svt_aom_sub_pixel_variance32x64_neon_dotprod,
      &svt_aom_sub_pixel_variance32x64_c },
    { 5, 5, &svt_aom_sub_pixel_variance32x32_neon_dotprod,
      &svt_aom_sub_pixel_variance32x32_c },
    { 5, 4, &svt_aom_sub_pixel_variance32x16_neon_dotprod,
      &svt_aom_sub_pixel_variance32x16_c },
    { 4, 5, &svt_aom_sub_pixel_variance16x32_neon_dotprod,
      &svt_aom_sub_pixel_variance16x32_c },
    { 4, 4, &svt_aom_sub_pixel_variance16x16_neon_dotprod,
      &svt_aom_sub_pixel_variance16x16_c },
    { 4, 3, &svt_aom_sub_pixel_variance16x8_neon_dotprod,
      &svt_aom_sub_pixel_variance16x8_c },
    { 3, 4, &svt_aom_sub_pixel_variance8x16_neon_dotprod,
      &svt_aom_sub_pixel_variance8x16_c },
    { 3, 3, &svt_aom_sub_pixel_variance8x8_neon_dotprod,
      &svt_aom_sub_pixel_variance8x8_c },
    { 3, 2, &svt_aom_sub_pixel_variance8x4_neon_dotprod,
      &svt_aom_sub_pixel_variance8x4_c },
    { 2, 3, &svt_aom_sub_pixel_variance4x8_neon_dotprod,
      &svt_aom_sub_pixel_variance4x8_c },
    { 6, 4, &svt_aom_sub_pixel_variance64x16_neon_dotprod,
      &svt_aom_sub_pixel_variance64x16_c },
    { 4, 6, &svt_aom_sub_pixel_variance16x64_neon_dotprod,
      &svt_aom_sub_pixel_variance16x64_c },
    { 5, 3, &svt_aom_sub_pixel_variance32x8_neon_dotprod,
      &svt_aom_sub_pixel_variance32x8_c },
    { 3, 5, &svt_aom_sub_pixel_variance8x32_neon_dotprod,
      &svt_aom_sub_pixel_variance8x32_c },
    { 4, 2, &svt_aom_sub_pixel_variance16x4_neon_dotprod,
      &svt_aom_sub_pixel_variance16x4_c },
    { 2, 4, &svt_aom_sub_pixel_variance4x16_neon_dotprod,
      &svt_aom_sub_pixel_variance4x16_c }
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, SubpelVarianceTest,
    ::testing::ValuesIn(kArraySubpelVariance_neon_dotprod));
#endif  // HAVE_NEON_DOTPROD
#endif  // ARCH_AARCH64

}  // namespace
