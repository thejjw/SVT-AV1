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
 * @file OBMCVarianceTest.cc
 *
 * @brief Unit test for obmc variance functions:
 * - svt_aom_obmc_variance{4-128}x{4-128}_{c, avx2}
 * - svt_aom_obmc_sub_pixel_variance{4-128}x{4-128}_{c, sse4_1}
 *
 * @author Cidana-Edmond
 *
 ******************************************************************************/
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include "aligned_allocator.hpp"
#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "random.hpp"
#include "util.h"
#include "filter.h"

#include "enc_inter_prediction.h"

namespace {
#if CONFIG_ENABLE_OBMC
using svt_av1_test_tool::SVTRandom;  // to generate the random
constexpr int MaskMax = 64;

using ObmcVarFunc = unsigned int (*)(const uint8_t *pre, int pre_stride,
                                     const int32_t *wsrc, const int32_t *mask,
                                     unsigned int *sse);
using ObmcVarParam = std::tuple<ObmcVarFunc, ObmcVarFunc>;

class OBMCVarianceTest : public ::testing::TestWithParam<ObmcVarParam> {
  public:
    DEFINE_ALIGNED_NEW_DELETE(OBMCVarianceTest)

  protected:
    template <size_t test_num>
    void run_test() {
        for (size_t i = 0; i < test_num; i++) {
            for (size_t j = 0; j < MAX_SB_SQUARE; j++) {
                pre_[j] = rnd_.random();
                wsrc_buf_[j] = rnd_.random() * rnd_msk_.random();
                mask_buf_[j] = rnd_msk_.random();
            }

            unsigned int sse_ref = 0, sse_tst = 0;
            uint32_t var_ref = func_ref_(pre_.data(),
                                         MAX_SB_SIZE,
                                         wsrc_buf_.data(),
                                         mask_buf_.data(),
                                         &sse_ref);
            uint32_t var_tst = func_tst_(pre_.data(),
                                         MAX_SB_SIZE,
                                         wsrc_buf_.data(),
                                         mask_buf_.data(),
                                         &sse_tst);

            ASSERT_EQ(var_tst, var_ref) << "compare var error";
            ASSERT_EQ(sse_tst, sse_ref) << "compare sse error";
        }
    }

  protected:
    SVTRandom rnd_{8, false};
    SVTRandom rnd_msk_{0, MaskMax *MaskMax + 1};
    const ObmcVarFunc func_ref_{TEST_GET_PARAM(0)};
    const ObmcVarFunc func_tst_{TEST_GET_PARAM(1)};
    alignas(32) std::array<uint8_t, MAX_SB_SQUARE> pre_{};
    alignas(32) std::array<int32_t, MAX_SB_SQUARE> wsrc_buf_{};
    alignas(32) std::array<int32_t, MAX_SB_SQUARE> mask_buf_{};
};

TEST_P(OBMCVarianceTest, RunCheckOutput) {
    run_test<1000>();
};

#define OBMC_VAR_FUNC(W, H, opt) svt_aom_obmc_variance##W##x##H##_##opt
#define GEN_OBMC_VAR_TEST_PARAM(W, H, opt) \
    ObmcVarParam(OBMC_VAR_FUNC(W, H, c), OBMC_VAR_FUNC(W, H, opt))
#define GEN_TEST_PARAMS(GEN_PARAM, opt)                    \
    {                                                      \
        GEN_PARAM(128, 128, opt), GEN_PARAM(128, 64, opt), \
        GEN_PARAM(64, 128, opt),  GEN_PARAM(64, 64, opt),  \
        GEN_PARAM(64, 32, opt),   GEN_PARAM(32, 64, opt),  \
        GEN_PARAM(32, 32, opt),   GEN_PARAM(32, 16, opt),  \
        GEN_PARAM(16, 32, opt),   GEN_PARAM(16, 16, opt),  \
        GEN_PARAM(16, 8, opt),    GEN_PARAM(8, 16, opt),   \
        GEN_PARAM(8, 8, opt),     GEN_PARAM(8, 4, opt),    \
        GEN_PARAM(4, 8, opt),     GEN_PARAM(4, 4, opt),    \
        GEN_PARAM(4, 16, opt),    GEN_PARAM(16, 4, opt),   \
        GEN_PARAM(8, 32, opt),    GEN_PARAM(32, 8, opt),   \
        GEN_PARAM(16, 64, opt),   GEN_PARAM(64, 16, opt)}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, OBMCVarianceTest,
    ::testing::ValuesIn(GEN_TEST_PARAMS(GEN_OBMC_VAR_TEST_PARAM, sse4_1)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, OBMCVarianceTest,
    ::testing::ValuesIn(GEN_TEST_PARAMS(GEN_OBMC_VAR_TEST_PARAM, avx2)));
#endif  // ARCH_x86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, OBMCVarianceTest,
    ::testing::ValuesIn(GEN_TEST_PARAMS(GEN_OBMC_VAR_TEST_PARAM, neon)));
#endif  // ARCH_AARCH64

using ObmcSubPixVarFunc = unsigned int (*)(const uint8_t *pre, int pre_stride,
                                           int xoffset, int yoffset,
                                           const int32_t *wsrc,
                                           const int32_t *mask,
                                           unsigned int *sse);
using ObmcSubPixVarParam = std::tuple<ObmcSubPixVarFunc, ObmcSubPixVarFunc>;

class OBMCSubPixelVarianceTest
    : public ::testing::TestWithParam<ObmcSubPixVarParam> {
  public:
    DEFINE_ALIGNED_NEW_DELETE(OBMCSubPixelVarianceTest)

  protected:
    template <size_t test_num>
    void run_test() {
        for (size_t i = 0; i < test_num; i++) {
            for (size_t j = 0; j < MAX_SB_SQUARE; j++) {
                pre_[j] = rnd_.random();
                wsrc_buf_[j] = rnd_.random() * rnd_msk_.random();
                mask_buf_[j] = rnd_msk_.random();
            }

            const int offset_x = rnd_offset_.random();
            const int offset_y = rnd_offset_.random();
            unsigned int sse_ref = 0, sse_tst = 0;
            uint32_t var_ref = func_ref_(pre_.data(),
                                         MAX_SB_SIZE,
                                         offset_x,
                                         offset_y,
                                         wsrc_buf_.data(),
                                         mask_buf_.data(),
                                         &sse_ref);
            uint32_t var_tst = func_tst_(pre_.data(),
                                         MAX_SB_SIZE,
                                         offset_x,
                                         offset_y,
                                         wsrc_buf_.data(),
                                         mask_buf_.data(),
                                         &sse_tst);

            ASSERT_EQ(var_tst, var_ref)
                << "compare var error at offset x=" << offset_x
                << " y=" << offset_y;
            ASSERT_EQ(sse_tst, sse_ref)
                << "compare sse error at offset x=" << offset_x
                << " y=" << offset_y;
        }
    }

  protected:
    SVTRandom rnd_{8, false};
    SVTRandom rnd_msk_{0, MaskMax *MaskMax + 1};
    SVTRandom rnd_offset_{0, BIL_SUBPEL_SHIFTS - 1};
    const ObmcSubPixVarFunc func_ref_{TEST_GET_PARAM(0)};
    const ObmcSubPixVarFunc func_tst_{TEST_GET_PARAM(1)};
    alignas(32) std::array<uint8_t, MAX_SB_SQUARE> pre_{};
    alignas(32) std::array<int32_t, MAX_SB_SQUARE> wsrc_buf_{};
    alignas(32) std::array<int32_t, MAX_SB_SQUARE> mask_buf_{};
};

TEST_P(OBMCSubPixelVarianceTest, RunCheckOutput) {
    run_test<1000>();
};

#define OBMC_SUB_PIX_VAR_FUNC(W, H, opt) \
    svt_aom_obmc_sub_pixel_variance##W##x##H##_##opt
#define GEN_OBMC_SUB_PIX_VAR_TEST_PARAM(W, H, opt)     \
    ObmcSubPixVarParam(OBMC_SUB_PIX_VAR_FUNC(W, H, c), \
                       OBMC_SUB_PIX_VAR_FUNC(W, H, opt))

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(SSE4_1, OBMCSubPixelVarianceTest,
                         ::testing::ValuesIn(GEN_TEST_PARAMS(
                             GEN_OBMC_SUB_PIX_VAR_TEST_PARAM, sse4_1)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(NEON, OBMCSubPixelVarianceTest,
                         ::testing::ValuesIn(GEN_TEST_PARAMS(
                             GEN_OBMC_SUB_PIX_VAR_TEST_PARAM, neon)));
#endif  // ARCH_AARCH64
using CalcTargetWeightedPredFn = void (*)(uint8_t, MacroBlockD *, int, uint8_t,
                                          MbModeInfo *, void *);
using CalcTargetWeightedPredParam =
    std::tuple<int, CalcTargetWeightedPredFn, CalcTargetWeightedPredFn>;

class CalcTargetWeightedPredTest
    : public ::testing::TestWithParam<CalcTargetWeightedPredParam> {
  protected:
    template <size_t test_num = 10>
    void run_test() {
        xd.n4_w = 5 + (width_ >> MI_SIZE_LOG2);
        calc_target_weighted_pred_ctxt ctxt_ref = {mask_buf_ref.data(),
                                                   wsrc_buf_ref.data(),
                                                   tmp_ref.data(),
                                                   stride,
                                                   width_};
        calc_target_weighted_pred_ctxt ctxt_tst = {mask_buf_tst.data(),
                                                   wsrc_buf_tst.data(),
                                                   tmp_tst.data(),
                                                   stride,
                                                   width_};
        for (uint32_t i = 0; i < test_num; i++) {
            for (uint32_t j = 0; j < 2 * MAX_SB_SQUARE; j++) {
                mask_buf_ref[j] = mask_buf_tst[j] = rnd_.random();
                wsrc_buf_ref[j] = wsrc_buf_tst[j] = rnd_.random();
                tmp_ref[j] = tmp_tst[j] = rnd_.random() % 255;
            }
            uint8_t size = (width_ >> 1) < 1 ? 1 : (width_ >> 1);
            func_ref_(0, &xd, 0, size, NULL, &ctxt_ref);
            func_tst_(0, &xd, 0, size, NULL, &ctxt_tst);

            auto mismatch_mask = std::mismatch(
                mask_buf_ref.begin(), mask_buf_ref.end(), mask_buf_tst.begin());
            if (mismatch_mask.first != mask_buf_ref.end()) {
                size_t idx =
                    std::distance(mask_buf_ref.begin(), mismatch_mask.first);
                ASSERT_EQ(*mismatch_mask.first, *mismatch_mask.second)
                    << "Mismatch for mask_buf at idx " << idx;
            }
            auto mismatch_wsrc = std::mismatch(
                wsrc_buf_ref.begin(), wsrc_buf_ref.end(), wsrc_buf_tst.begin());
            if (mismatch_wsrc.first != wsrc_buf_ref.end()) {
                size_t idx =
                    std::distance(wsrc_buf_ref.begin(), mismatch_wsrc.first);
                ASSERT_EQ(*mismatch_wsrc.first, *mismatch_wsrc.second)
                    << "Mismatch for wsrc_buf at idx " << idx;
            }
            auto mismatch_tmp =
                std::mismatch(tmp_ref.begin(), tmp_ref.end(), tmp_tst.begin());
            if (mismatch_tmp.first != tmp_ref.end()) {
                size_t idx = std::distance(tmp_ref.begin(), mismatch_tmp.first);
                ASSERT_EQ(*mismatch_tmp.first, *mismatch_tmp.second)
                    << "Mismatch for tmp at idx " << idx;
            }
        }
    }

  protected:
    SVTRandom rnd_{10, false};
    const int width_{TEST_GET_PARAM(0)};
    const CalcTargetWeightedPredFn func_ref_{TEST_GET_PARAM(1)};
    const CalcTargetWeightedPredFn func_tst_{TEST_GET_PARAM(2)};
    MacroBlockD xd{};
    std::array<int32_t, 2 * MAX_SB_SQUARE> mask_buf_ref{};
    std::array<int32_t, 2 * MAX_SB_SQUARE> mask_buf_tst{};
    std::array<int32_t, 2 * MAX_SB_SQUARE> wsrc_buf_ref{};
    std::array<int32_t, 2 * MAX_SB_SQUARE> wsrc_buf_tst{};
    std::array<uint8_t, 2 * MAX_SB_SQUARE> tmp_ref{};
    std::array<uint8_t, 2 * MAX_SB_SQUARE> tmp_tst{};
    const int stride{MAX_SB_SIZE};
};

using CalcTargetWeightedPredTestAbove = CalcTargetWeightedPredTest;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(CalcTargetWeightedPredTestAbove);

TEST_P(CalcTargetWeightedPredTestAbove, RunCheckOutput) {
    run_test();
};

using CalcTargetWeightedPredTestLeft = CalcTargetWeightedPredTest;

TEST_P(CalcTargetWeightedPredTestLeft, RunCheckOutput) {
    run_test();
};

constexpr int overlap_tab[] = {2, 4, 8, 16, 32};

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, CalcTargetWeightedPredTestAbove,
    ::testing::Combine(
        ::testing::ValuesIn(overlap_tab),
        ::testing::Values(svt_av1_calc_target_weighted_pred_above_c),
        ::testing::Values(svt_av1_calc_target_weighted_pred_above_avx2)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, CalcTargetWeightedPredTestLeft,
    ::testing::Combine(
        ::testing::ValuesIn(overlap_tab),
        ::testing::Values(svt_av1_calc_target_weighted_pred_left_c),
        ::testing::Values(svt_av1_calc_target_weighted_pred_left_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, CalcTargetWeightedPredTestLeft,
    ::testing::Combine(
        ::testing::ValuesIn(overlap_tab),
        ::testing::Values(svt_av1_calc_target_weighted_pred_left_c),
        ::testing::Values(svt_av1_calc_target_weighted_pred_left_neon)));
#endif  // ARCH_AARCH64

#endif  // CONFIG_ENABLE_OBMC

}  // namespace
