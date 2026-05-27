/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

#include "gtest/gtest.h"
#include <array>
#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "acm_random.h"

namespace {
using libaom_test::ACMRandom;
constexpr int kNumIterations = 1000;

using BlockErrorFunc = decltype(&svt_av1_block_error_c);

class BlockErrorTest : public ::testing::TestWithParam<BlockErrorFunc> {
  protected:
    BlockErrorFunc test_func_{GetParam()};
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(BlockErrorTest);

TEST_P(BlockErrorTest, OperationCheck) {
    ACMRandom rnd{ACMRandom::DeterministicSeed()};
    int err_count_total = 0;
    int first_failure = -1;

    for (int i = 0; i < kNumIterations; ++i) {
        constexpr int bit_depth = 8;
        constexpr int msb = bit_depth + 8 - 1;
        alignas(16) std::array<TranLow, 4096> coeff;
        alignas(16) std::array<TranLow, 4096> dqcoeff;
        // All block sizes from 4x4, 8x4 ..64x64
        const intptr_t block_size = 16 << (i % 9);
        for (int j = 0; j < block_size; j++) {
            // coeff and dqcoeff will always have at least the same sign, and
            // this can be used for optimization, so generate test input
            // precisely.
            if (rnd(2)) {
                // Positive number
                coeff[j] = rnd(1 << msb);
                dqcoeff[j] = rnd(1 << msb);
            } else {
                // Negative number
                coeff[j] = -rnd(1 << msb);
                dqcoeff[j] = -rnd(1 << msb);
            }
        }
        int64_t ssz;
        int64_t ref_ssz;
        int64_t ref_ret = svt_av1_block_error_c(
            coeff.data(), dqcoeff.data(), block_size, &ref_ssz);
        int64_t ret =
            test_func_(coeff.data(), dqcoeff.data(), block_size, &ssz);

        int err_count = (ref_ret != ret) || (ref_ssz != ssz);
        if (err_count && !err_count_total) {
            first_failure = i;
        }
        err_count_total += err_count;
    }
    EXPECT_EQ(0, err_count_total)
        << "Error: Error Block Test, C output doesn't match optimized output. "
        << "First failed at test case " << first_failure;
}

TEST_P(BlockErrorTest, ExtremeValues) {
    ACMRandom rnd{ACMRandom::DeterministicSeed()};
    int err_count_total = 0;
    int first_failure = -1;
    constexpr int bit_depth = 8;
    constexpr int msb = bit_depth + 8 - 1;
    int max_val = ((1 << msb) - 1);
    for (int i = 0; i < kNumIterations; ++i) {
        alignas(16) std::array<TranLow, 4096> coeff;
        alignas(16) std::array<TranLow, 4096> dqcoeff;
        int k = (i / 9) % 9;

        // Change the maximum coeff value, to test different bit boundaries
        if (k == 8 && (i % 9) == 0) {
            max_val >>= 1;
        }
        const intptr_t block_size =
            16 << (i % 9);  // All block sizes from 4x4, 8x4 ..64x64
        for (int j = 0; j < block_size; j++) {
            if (k < 4) {
                // Test at positive maximum values
                coeff[j] = k % 2 ? max_val : 0;
                dqcoeff[j] = (k >> 1) % 2 ? max_val : 0;
            } else if (k < 8) {
                // Test at negative maximum values
                coeff[j] = k % 2 ? -max_val : 0;
                dqcoeff[j] = (k >> 1) % 2 ? -max_val : 0;
            } else {
                if (rnd(2)) {
                    // Positive number
                    coeff[j] = rnd(1 << 14);
                    dqcoeff[j] = rnd(1 << 14);
                } else {
                    // Negative number
                    coeff[j] = -rnd(1 << 14);
                    dqcoeff[j] = -rnd(1 << 14);
                }
            }
        }
        int64_t ssz;
        int64_t ref_ssz;
        int64_t ref_ret = svt_av1_block_error_c(
            coeff.data(), dqcoeff.data(), block_size, &ref_ssz);
        int64_t ret =
            test_func_(coeff.data(), dqcoeff.data(), block_size, &ssz);
        int err_count = (ref_ret != ret) || (ref_ssz != ssz);
        if (err_count && !err_count_total) {
            first_failure = i;
        }
        err_count_total += err_count;
    }
    EXPECT_EQ(0, err_count_total)
        << "Error: Error Block Test, C output doesn't match optimized output. "
        << "First failed at test case " << first_failure;
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(AVX2, BlockErrorTest,
                         ::testing::Values(svt_av1_block_error_avx2));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(NEON, BlockErrorTest,
                         ::testing::Values(svt_av1_block_error_neon));
#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(SVE, BlockErrorTest,
                         ::testing::Values(svt_av1_block_error_sve));
#endif  // HAVE_SVE
#endif  // HAVE_AARCH64
}  // namespace
