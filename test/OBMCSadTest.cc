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
 * @file OBMCsad_Test.cc
 *
 * @brief Unit test for obmc sad functions:
 * - obmc_sad_w4_avx2
 * - obmc_sad_w8n_avx2
 *
 * @author Cidana-Ivy
 *
 ******************************************************************************/
#include "gtest/gtest.h"
#include <array>
#include "aligned_allocator.hpp"
#include "aom_dsp_rtcd.h"
#include "random.hpp"
#include "util.h"

namespace {
using svt_av1_test_tool::SVTRandom;  // to generate the random
#if CONFIG_ENABLE_OBMC
constexpr int MaskMax = 64;

using ObmcSadFunc = uint32_t (*)(const uint8_t* pre, int pre_stride,
                                 const int32_t* wsrc, const int32_t* mask);
using ObmcSadParam = std::tuple<ObmcSadFunc, ObmcSadFunc>;

class OBMCSadTest : public ::testing::TestWithParam<ObmcSadParam> {
  public:
    DEFINE_ALIGNED_NEW_DELETE(OBMCSadTest)
  protected:
    template <size_t test_num>
    void run_test() {
        for (size_t i = 0; i < test_num; i++) {
            for (size_t j = 0; j < MAX_SB_SQUARE; j++) {
                pre_[j] = rnd_.random();
                wsrc_buf_[j] = rnd_.random() * rnd_msk_.random();
                mask_buf_[j] = rnd_msk_.random();
            }

            uint32_t sad_ref = func_ref_(
                pre_.data(), MAX_SB_SIZE, wsrc_buf_.data(), mask_buf_.data());
            uint32_t sad_tst = func_tst_(
                pre_.data(), MAX_SB_SIZE, wsrc_buf_.data(), mask_buf_.data());

            ASSERT_EQ(sad_tst, sad_ref) << "compare SAD error";
        }
    }

  protected:
    SVTRandom rnd_{8, false};
    SVTRandom rnd_msk_{0, MaskMax* MaskMax + 1};
    const ObmcSadFunc func_ref_{TEST_GET_PARAM(0)};
    const ObmcSadFunc func_tst_{TEST_GET_PARAM(1)};
    alignas(32) std::array<uint8_t, MAX_SB_SQUARE> pre_{};
    alignas(32) std::array<int32_t, MAX_SB_SQUARE> wsrc_buf_{};
    alignas(32) std::array<int32_t, MAX_SB_SQUARE> mask_buf_{};
};

TEST_P(OBMCSadTest, RunCheckOutput) {
    run_test<1000>();
};

#define OBMC_SAD_FUNC(W, H, opt) svt_aom_obmc_sad##W##x##H##_##opt
#define GEN_OBMC_SAD_TEST_PARAM(W, H, opt) \
    ObmcSadParam(OBMC_SAD_FUNC(W, H, c), OBMC_SAD_FUNC(W, H, opt))
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
    AVX2, OBMCSadTest,
    ::testing::ValuesIn(GEN_TEST_PARAMS(GEN_OBMC_SAD_TEST_PARAM, avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, OBMCSadTest,
    ::testing::ValuesIn(GEN_TEST_PARAMS(GEN_OBMC_SAD_TEST_PARAM, neon)));
#endif  // ARCH_AARCH64

#endif  // CONFIG_ENABLE_OBMC

}  // namespace
