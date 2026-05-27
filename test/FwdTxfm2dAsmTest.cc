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
 * @file FwdTxfm2dAsmTest.c
 *
 * @brief Unit test for forward 2d transform functions written in assembly code:
 * - svt_av1_fwd_txfm2d_{4, 8, 16, 32, 64}x{4, 8, 16, 32, 64}_avx2
 *
 * @author Cidana-Wenyao
 *
 ******************************************************************************/
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstdint>

#include "aligned_allocator.hpp"
#include "definitions.h"
#include "random.hpp"
#include "util.h"
#include "aom_dsp_rtcd.h"
#include "TxfmCommon.h"

namespace {
using svt_av1_test_tool::SVTRandom;
constexpr auto TEST_OFFSET = 10;
using FwdTxfm2dAsmParam =
    std::tuple<TxSize, EbBitDepth, TxCoeffShape, const FwdTxfm2dFunc *,
               const FwdTxfm2dFunc *>;

#ifdef ARCH_X86_64

constexpr FwdTxfm2dFunc fwd_txfm_2d_avx2_func[TX_SIZES_ALL] = {
    nullptr,
    svt_av1_fwd_txfm2d_8x8_avx2,
    svt_av1_fwd_txfm2d_16x16_avx2,
    svt_av1_fwd_txfm2d_32x32_avx2,
    svt_av1_fwd_txfm2d_64x64_avx2,
    svt_av1_fwd_txfm2d_4x8_avx2,
    svt_av1_fwd_txfm2d_8x4_avx2,
    svt_av1_fwd_txfm2d_8x16_avx2,
    svt_av1_fwd_txfm2d_16x8_avx2,
    svt_av1_fwd_txfm2d_16x32_avx2,
    svt_av1_fwd_txfm2d_32x16_avx2,
    svt_av1_fwd_txfm2d_32x64_avx2,
    svt_av1_fwd_txfm2d_64x32_avx2,
    svt_av1_fwd_txfm2d_4x16_avx2,
    svt_av1_fwd_txfm2d_16x4_avx2,
    svt_av1_fwd_txfm2d_8x32_avx2,
    svt_av1_fwd_txfm2d_32x8_avx2,
    svt_av1_fwd_txfm2d_16x64_avx2,
    svt_av1_fwd_txfm2d_64x16_avx2,
};

constexpr FwdTxfm2dFunc fwd_txfm_2d_sse4_1_func[TX_SIZES_ALL] = {
    svt_av1_fwd_txfm2d_4x4_sse4_1,   svt_av1_fwd_txfm2d_8x8_sse4_1,
    svt_av1_fwd_txfm2d_16x16_sse4_1, svt_av1_fwd_txfm2d_32x32_sse4_1,
    svt_av1_fwd_txfm2d_64x64_sse4_1, svt_av1_fwd_txfm2d_4x8_sse4_1,
    svt_av1_fwd_txfm2d_8x4_sse4_1,   svt_av1_fwd_txfm2d_8x16_sse4_1,
    svt_av1_fwd_txfm2d_16x8_sse4_1,  svt_av1_fwd_txfm2d_16x32_sse4_1,
    svt_av1_fwd_txfm2d_32x16_sse4_1, svt_av1_fwd_txfm2d_32x64_sse4_1,
    svt_av1_fwd_txfm2d_64x32_sse4_1, svt_av1_fwd_txfm2d_4x16_sse4_1,
    svt_av1_fwd_txfm2d_16x4_sse4_1,  svt_av1_fwd_txfm2d_8x32_sse4_1,
    svt_av1_fwd_txfm2d_32x8_sse4_1,  svt_av1_fwd_txfm2d_16x64_sse4_1,
    svt_av1_fwd_txfm2d_64x16_sse4_1,
};

constexpr FwdTxfm2dFunc fwd_txfm_2d_N2_avx2_func[TX_SIZES_ALL] = {
    nullptr,
    svt_av1_fwd_txfm2d_8x8_N2_avx2,
    svt_av1_fwd_txfm2d_16x16_N2_avx2,
    svt_av1_fwd_txfm2d_32x32_N2_avx2,
    svt_av1_fwd_txfm2d_64x64_N2_avx2,
    svt_av1_fwd_txfm2d_4x8_N2_avx2,
    svt_av1_fwd_txfm2d_8x4_N2_avx2,
    svt_av1_fwd_txfm2d_8x16_N2_avx2,
    svt_av1_fwd_txfm2d_16x8_N2_avx2,
    svt_av1_fwd_txfm2d_16x32_N2_avx2,
    svt_av1_fwd_txfm2d_32x16_N2_avx2,
    svt_av1_fwd_txfm2d_32x64_N2_avx2,
    svt_av1_fwd_txfm2d_64x32_N2_avx2,
    svt_av1_fwd_txfm2d_4x16_N2_avx2,
    svt_av1_fwd_txfm2d_16x4_N2_avx2,
    svt_av1_fwd_txfm2d_8x32_N2_avx2,
    svt_av1_fwd_txfm2d_32x8_N2_avx2,
    svt_av1_fwd_txfm2d_16x64_N2_avx2,
    svt_av1_fwd_txfm2d_64x16_N2_avx2,
};

constexpr FwdTxfm2dFunc fwd_txfm_2d_N4_avx2_func[TX_SIZES_ALL] = {
    nullptr,
    svt_av1_fwd_txfm2d_8x8_N4_avx2,
    svt_av1_fwd_txfm2d_16x16_N4_avx2,
    svt_av1_fwd_txfm2d_32x32_N4_avx2,
    svt_av1_fwd_txfm2d_64x64_N4_avx2,
    svt_av1_fwd_txfm2d_4x8_N4_avx2,
    svt_av1_fwd_txfm2d_8x4_N4_avx2,
    svt_av1_fwd_txfm2d_8x16_N4_avx2,
    svt_av1_fwd_txfm2d_16x8_N4_avx2,
    svt_av1_fwd_txfm2d_16x32_N4_avx2,
    svt_av1_fwd_txfm2d_32x16_N4_avx2,
    svt_av1_fwd_txfm2d_32x64_N4_avx2,
    svt_av1_fwd_txfm2d_64x32_N4_avx2,
    svt_av1_fwd_txfm2d_4x16_N4_avx2,
    svt_av1_fwd_txfm2d_16x4_N4_avx2,
    svt_av1_fwd_txfm2d_8x32_N4_avx2,
    svt_av1_fwd_txfm2d_32x8_N4_avx2,
    svt_av1_fwd_txfm2d_16x64_N4_avx2,
    svt_av1_fwd_txfm2d_64x16_N4_avx2,
};

constexpr FwdTxfm2dFunc fwd_txfm_2d_N2_sse4_1_func[TX_SIZES_ALL] = {
    svt_av1_fwd_txfm2d_4x4_N2_sse4_1,   svt_av1_fwd_txfm2d_8x8_N2_sse4_1,
    svt_av1_fwd_txfm2d_16x16_N2_sse4_1, svt_av1_fwd_txfm2d_32x32_N2_sse4_1,
    svt_av1_fwd_txfm2d_64x64_N2_sse4_1, svt_av1_fwd_txfm2d_4x8_N2_sse4_1,
    svt_av1_fwd_txfm2d_8x4_N2_sse4_1,   svt_av1_fwd_txfm2d_8x16_N2_sse4_1,
    svt_av1_fwd_txfm2d_16x8_N2_sse4_1,  svt_av1_fwd_txfm2d_16x32_N2_sse4_1,
    svt_av1_fwd_txfm2d_32x16_N2_sse4_1, svt_av1_fwd_txfm2d_32x64_N2_sse4_1,
    svt_av1_fwd_txfm2d_64x32_N2_sse4_1, svt_av1_fwd_txfm2d_4x16_N2_sse4_1,
    svt_av1_fwd_txfm2d_16x4_N2_sse4_1,  svt_av1_fwd_txfm2d_8x32_N2_sse4_1,
    svt_av1_fwd_txfm2d_32x8_N2_sse4_1,  svt_av1_fwd_txfm2d_16x64_N2_sse4_1,
    svt_av1_fwd_txfm2d_64x16_N2_sse4_1,
};

constexpr FwdTxfm2dFunc fwd_txfm_2d_N4_sse4_1_func[TX_SIZES_ALL] = {
    svt_av1_fwd_txfm2d_4x4_N4_sse4_1,   svt_av1_fwd_txfm2d_8x8_N4_sse4_1,
    svt_av1_fwd_txfm2d_16x16_N4_sse4_1, svt_av1_fwd_txfm2d_32x32_N4_sse4_1,
    svt_av1_fwd_txfm2d_64x64_N4_sse4_1, svt_av1_fwd_txfm2d_4x8_N4_sse4_1,
    svt_av1_fwd_txfm2d_8x4_N4_sse4_1,   svt_av1_fwd_txfm2d_8x16_N4_sse4_1,
    svt_av1_fwd_txfm2d_16x8_N4_sse4_1,  svt_av1_fwd_txfm2d_16x32_N4_sse4_1,
    svt_av1_fwd_txfm2d_32x16_N4_sse4_1, svt_av1_fwd_txfm2d_32x64_N4_sse4_1,
    svt_av1_fwd_txfm2d_64x32_N4_sse4_1, svt_av1_fwd_txfm2d_4x16_N4_sse4_1,
    svt_av1_fwd_txfm2d_16x4_N4_sse4_1,  svt_av1_fwd_txfm2d_8x32_N4_sse4_1,
    svt_av1_fwd_txfm2d_32x8_N4_sse4_1,  svt_av1_fwd_txfm2d_16x64_N4_sse4_1,
    svt_av1_fwd_txfm2d_64x16_N4_sse4_1,
};

#if EN_AVX512_SUPPORT
constexpr FwdTxfm2dFunc fwd_txfm_2d_avx512_func[TX_SIZES_ALL] = {
    nullptr,
    nullptr,
    svt_av1_fwd_txfm2d_16x16_avx512,
    svt_av1_fwd_txfm2d_32x32_avx512,
    svt_av1_fwd_txfm2d_64x64_avx512,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    svt_av1_fwd_txfm2d_16x32_avx512,
    svt_av1_fwd_txfm2d_32x16_avx512,
    svt_av1_fwd_txfm2d_32x64_avx512,
    svt_av1_fwd_txfm2d_64x32_avx512,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    svt_av1_fwd_txfm2d_16x64_avx512,
    svt_av1_fwd_txfm2d_64x16_avx512,
};

constexpr FwdTxfm2dFunc fwd_txfm_2d_N2_avx512_func[TX_SIZES_ALL] = {
    nullptr,
    nullptr,
    nullptr,
    av1_fwd_txfm2d_32x32_N2_avx512,
    av1_fwd_txfm2d_64x64_N2_avx512,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    av1_fwd_txfm2d_32x64_N2_avx512,
    av1_fwd_txfm2d_64x32_N2_avx512,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

constexpr FwdTxfm2dFunc fwd_txfm_2d_N4_avx512_func[TX_SIZES_ALL] = {
    nullptr, nullptr, nullptr, nullptr, av1_fwd_txfm2d_64x64_N4_avx512,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
};
#endif /*EN_AVX512_SUPPORT*/

#endif

#ifdef ARCH_AARCH64

constexpr FwdTxfm2dFunc fwd_txfm_2d_neon_func[TX_SIZES_ALL] = {
    svt_av1_fwd_txfm2d_4x4_neon,   svt_av1_fwd_txfm2d_8x8_neon,
    svt_av1_fwd_txfm2d_16x16_neon, svt_av1_fwd_txfm2d_32x32_neon,
    svt_av1_fwd_txfm2d_64x64_neon, svt_av1_fwd_txfm2d_4x8_neon,
    svt_av1_fwd_txfm2d_8x4_neon,   svt_av1_fwd_txfm2d_8x16_neon,
    svt_av1_fwd_txfm2d_16x8_neon,  svt_av1_fwd_txfm2d_16x32_neon,
    svt_av1_fwd_txfm2d_32x16_neon, svt_av1_fwd_txfm2d_32x64_neon,
    svt_av1_fwd_txfm2d_64x32_neon, svt_av1_fwd_txfm2d_4x16_neon,
    svt_av1_fwd_txfm2d_16x4_neon,  svt_av1_fwd_txfm2d_8x32_neon,
    svt_av1_fwd_txfm2d_32x8_neon,  svt_av1_fwd_txfm2d_16x64_neon,
    svt_av1_fwd_txfm2d_64x16_neon,
};

constexpr FwdTxfm2dFunc fwd_txfm_2d_N4_neon_func[TX_SIZES_ALL] = {
    svt_av1_fwd_txfm2d_4x4_N4_neon,   svt_av1_fwd_txfm2d_8x8_N4_neon,
    svt_av1_fwd_txfm2d_16x16_N4_neon, svt_av1_fwd_txfm2d_32x32_N4_neon,
    svt_av1_fwd_txfm2d_64x64_N4_neon, svt_av1_fwd_txfm2d_4x8_N4_neon,
    svt_av1_fwd_txfm2d_8x4_N4_neon,   svt_av1_fwd_txfm2d_8x16_N4_neon,
    svt_av1_fwd_txfm2d_16x8_N4_neon,  svt_av1_fwd_txfm2d_16x32_N4_neon,
    svt_av1_fwd_txfm2d_32x16_N4_neon, svt_av1_fwd_txfm2d_32x64_N4_neon,
    svt_av1_fwd_txfm2d_64x32_N4_neon, svt_av1_fwd_txfm2d_4x16_N4_neon,
    svt_av1_fwd_txfm2d_16x4_N4_neon,  svt_av1_fwd_txfm2d_8x32_N4_neon,
    svt_av1_fwd_txfm2d_32x8_N4_neon,  svt_av1_fwd_txfm2d_16x64_N4_neon,
    svt_av1_fwd_txfm2d_64x16_N4_neon,
};

constexpr FwdTxfm2dFunc fwd_txfm_2d_N2_neon_func[TX_SIZES_ALL] = {
    svt_av1_fwd_txfm2d_4x4_N2_neon,   svt_av1_fwd_txfm2d_8x8_N2_neon,
    svt_av1_fwd_txfm2d_16x16_N2_neon, svt_av1_fwd_txfm2d_32x32_N2_neon,
    svt_av1_fwd_txfm2d_64x64_N2_neon, svt_av1_fwd_txfm2d_4x8_N2_neon,
    svt_av1_fwd_txfm2d_8x4_N2_neon,   svt_av1_fwd_txfm2d_8x16_N2_neon,
    svt_av1_fwd_txfm2d_16x8_N2_neon,  svt_av1_fwd_txfm2d_16x32_N2_neon,
    svt_av1_fwd_txfm2d_32x16_N2_neon, svt_av1_fwd_txfm2d_32x64_N2_neon,
    svt_av1_fwd_txfm2d_64x32_N2_neon, svt_av1_fwd_txfm2d_4x16_N2_neon,
    svt_av1_fwd_txfm2d_16x4_N2_neon,  svt_av1_fwd_txfm2d_8x32_N2_neon,
    svt_av1_fwd_txfm2d_32x8_N2_neon,  svt_av1_fwd_txfm2d_16x64_N2_neon,
    svt_av1_fwd_txfm2d_64x16_N2_neon,
};

#endif /* ARCH_AARCH64*/

/**
 * @brief Unit test for fwd tx 2d avx2 functions:
 * - svt_av1_fwd_txfm2d_{4, 8, 16, 32, 64}x{4, 8, 16, 32, 64}_avx2
 *
 * Test strategy:
 * Verify this assembly code by comparing with reference c implementation.
 * Feed the same data and check test output and reference output.
 * The test output and reference output are different at the beginning.
 *
 * Expect result:
 * Output from assemble function should be exactly same as output from c.
 *
 * Test coverage:
 * Test cases:
 * Input buffer: Fill with random values
 * TxSize: all the valid TxSize and TxType allowed.
 * BitDepth: 8bit and 10bit.
 *
 */
class FwdTxfm2dAsmTest : public ::testing::TestWithParam<FwdTxfm2dAsmParam> {
  public:
    DEFINE_ALIGNED_NEW_DELETE(FwdTxfm2dAsmTest)

    FwdTxfm2dAsmTest() {
        const auto total_size = width_ * height_;
        // fill out all of the bits
        std::fill_n(output_test_.begin(), total_size, 0xffffffff);
        // pad the rest of the output buffer with a known value to detect
        // out-of-bound write
        std::fill(
            output_test_.begin() + total_size, output_test_.end(), 0xcdcdcdcd);
        std::fill(
            output_ref_.begin() + total_size, output_ref_.end(), 0xcdcdcdcd);
    }

    void run_match_test() {
        const auto test_funcs = TEST_GET_PARAM(3);
        const auto ref_funcs = TEST_GET_PARAM(4);
        execute_test(test_funcs[tx_size_], ref_funcs[tx_size_], shape_);
    }

  private:
    void execute_test(FwdTxfm2dFunc test_func, FwdTxfm2dFunc ref_func,
                      TxCoeffShape shape) {
        if (ref_func == nullptr || test_func == nullptr)
            return;

        for (int tx_type = 0; tx_type < TX_TYPES; ++tx_type) {
            TxType type = static_cast<TxType>(tx_type);
            // tx_type and tx_size are not compatible in the av1-spec.
            // like the max size of adst transform is 16, and max size of
            // identity transform is 32.
            if (!is_txfm_allowed(type, tx_size_))
                continue;

            constexpr int loops = 100;
            for (int k = 0; k < loops; k++) {
                populate_with_random();

                ref_func(input_.data(),
                         output_ref_.data(),
                         stride_,
                         type,
                         (uint8_t)bd_);
                if (shape == N2_SHAPE || shape == N4_SHAPE) {
                    const auto shift = shape == N2_SHAPE ? 1 : 2;
                    const auto tx_width = tx_size_wide[tx_size_];
                    const auto tx_height = tx_size_high[tx_size_];
                    for (int i = 0; i < (tx_width * tx_height); i++) {
                        if (i % tx_width >= (tx_width >> shift) ||
                            i / tx_width >= (tx_height >> shift)) {
                            output_ref_[i] = 0;
                        }
                    }
                }
                test_func(input_.data(),
                          output_test_.data(),
                          stride_,
                          type,
                          (uint8_t)bd_);

                if (output_test_ != output_ref_) {
                    for (int i = 0; i < height_; i++)
                        for (int j = 0; j < width_; j++) {
                            ASSERT_EQ(output_ref_[i * width_ + j],
                                      output_test_[i * width_ + j])
                                << "loop: " << k << " tx_type: " << tx_type
                                << " tx_size: " << tx_size_ << " Mismatch at ("
                                << j << " x " << i << ")";
                        }

                    GTEST_FAIL() << "Output mismatch between reference and "
                                    "test function.";
                }
            }
        }
    }
    void populate_with_random() {
        auto *row = input_.data();
        for (int i = 0; i < height_; ++i, row += stride_) {
            std::generate_n(row, width_, [&]() { return rnd_.Rand16(); });
        }
    }

  private:
    const TxSize tx_size_{TEST_GET_PARAM(0)}; /**< input param tx_size */
    const EbBitDepth bd_{TEST_GET_PARAM(1)};  /**< input param 8bit or 10bit */
    const TxCoeffShape shape_{TEST_GET_PARAM(2)};
    const int width_{tx_size_wide[tx_size_]};
    const int height_{tx_size_high[tx_size_]};
    // input are signed value with bitdepth + 1 bits
    SVTRandom rnd_{-(1 << bd_) + 1, (1 << bd_) - 1};
    static constexpr int stride_ = MAX_TX_SIZE;
    alignas(ALIGNMENT) std::array<int16_t, MAX_TX_SQUARE> input_{};
    alignas(ALIGNMENT)
        std::array<int32_t, MAX_TX_SQUARE + TEST_OFFSET> output_test_{};
    alignas(ALIGNMENT)
        std::array<int32_t, MAX_TX_SQUARE + TEST_OFFSET> output_ref_{};
};

TEST_P(FwdTxfm2dAsmTest, match_test) {
    run_match_test();
}

const auto TxSizeRange = ::testing::Range(TX_4X4, TX_64X16);
const auto BdRange = ::testing::Values(EB_EIGHT_BIT, EB_TEN_BIT);

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, FwdTxfm2dAsmTest,
    ::testing::Combine(TxSizeRange, BdRange, ::testing::Values(DEFAULT_SHAPE),
                       ::testing::Values(fwd_txfm_2d_c_func),
                       ::testing::Values(fwd_txfm_2d_sse4_1_func)));

INSTANTIATE_TEST_SUITE_P(
    N2_SSE4_1, FwdTxfm2dAsmTest,
    ::testing::Combine(TxSizeRange, BdRange, ::testing::Values(N2_SHAPE),
                       ::testing::Values(fwd_txfm_2d_N2_c_func),
                       ::testing::Values(fwd_txfm_2d_N2_sse4_1_func)));

INSTANTIATE_TEST_SUITE_P(
    N4_SSE4_1, FwdTxfm2dAsmTest,
    ::testing::Combine(TxSizeRange, BdRange, ::testing::Values(N4_SHAPE),
                       ::testing::Values(fwd_txfm_2d_N4_c_func),
                       ::testing::Values(fwd_txfm_2d_N4_sse4_1_func)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, FwdTxfm2dAsmTest,
    ::testing::Combine(TxSizeRange, BdRange, ::testing::Values(DEFAULT_SHAPE),
                       ::testing::Values(fwd_txfm_2d_c_func),
                       ::testing::Values(fwd_txfm_2d_avx2_func)));

INSTANTIATE_TEST_SUITE_P(
    N2_AVX2, FwdTxfm2dAsmTest,
    ::testing::Combine(TxSizeRange, BdRange, ::testing::Values(N2_SHAPE),
                       ::testing::Values(fwd_txfm_2d_N2_c_func),
                       ::testing::Values(fwd_txfm_2d_N2_avx2_func)));

INSTANTIATE_TEST_SUITE_P(
    N4_AVX2, FwdTxfm2dAsmTest,
    ::testing::Combine(TxSizeRange, BdRange, ::testing::Values(N4_SHAPE),
                       ::testing::Values(fwd_txfm_2d_N4_c_func),
                       ::testing::Values(fwd_txfm_2d_N4_avx2_func)));

#if EN_AVX512_SUPPORT
INSTANTIATE_TEST_SUITE_P(
    AVX512, FwdTxfm2dAsmTest,
    ::testing::Combine(TxSizeRange, BdRange, ::testing::Values(DEFAULT_SHAPE),
                       ::testing::Values(fwd_txfm_2d_c_func),
                       ::testing::Values(fwd_txfm_2d_avx512_func)));

INSTANTIATE_TEST_SUITE_P(
    N2_AVX512, FwdTxfm2dAsmTest,
    ::testing::Combine(TxSizeRange, BdRange, ::testing::Values(N2_SHAPE),
                       ::testing::Values(fwd_txfm_2d_N2_c_func),
                       ::testing::Values(fwd_txfm_2d_N2_avx512_func)));

INSTANTIATE_TEST_SUITE_P(
    N4_AVX512, FwdTxfm2dAsmTest,
    ::testing::Combine(TxSizeRange, BdRange, ::testing::Values(N4_SHAPE),
                       ::testing::Values(fwd_txfm_2d_N4_c_func),
                       ::testing::Values(fwd_txfm_2d_N4_avx512_func)));

#endif

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, FwdTxfm2dAsmTest,
    ::testing::Combine(TxSizeRange, BdRange, ::testing::Values(DEFAULT_SHAPE),
                       ::testing::Values(fwd_txfm_2d_c_func),
                       ::testing::Values(fwd_txfm_2d_neon_func)));

INSTANTIATE_TEST_SUITE_P(
    N4_NEON, FwdTxfm2dAsmTest,
    ::testing::Combine(TxSizeRange, BdRange, ::testing::Values(N4_SHAPE),
                       ::testing::Values(fwd_txfm_2d_N4_c_func),
                       ::testing::Values(fwd_txfm_2d_N4_neon_func)));

INSTANTIATE_TEST_SUITE_P(
    N2_NEON, FwdTxfm2dAsmTest,
    ::testing::Combine(TxSizeRange, BdRange, ::testing::Values(N2_SHAPE),
                       ::testing::Values(fwd_txfm_2d_N2_c_func),
                       ::testing::Values(fwd_txfm_2d_N2_neon_func)));
#endif
}  // namespace
