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
 * @file pixel_proj_err_test.cc
 *
 * @brief Unit test of project-related test in selfguided filter:
 *
 * - av1_lowbd_pixel_proj_error_avx2
 * - av1_highbd_pixel_proj_error_avx2
 * - svt_get_proj_subspace_avx2
 *
 * @author Cidana-Edmond, Cidana-Wenyao
 *
 ******************************************************************************/

#include "gtest/gtest.h"
#include <array>
#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "restoration.h"
#include "random.hpp"
#include "aligned_allocator.hpp"
#include "util.h"

namespace {
constexpr auto MAX_DATA_BLOCK = 384;
using svt_av1_test_tool::SVTRandom;

constexpr int min_test_times = 10;

using PixelProjFunc = decltype(&svt_av1_lowbd_pixel_proj_error_c);

using PixelProjErrorTestParam =
    std::tuple<const PixelProjFunc, const PixelProjFunc>;

/**
 * @brief Unit test for pixel projection error:
 * - av1_lowbd_pixel_proj_error_avx2
 * - av1_highbd_pixel_proj_error_avx2
 *
 * Test strategy:
 * Verify this assembly code by comparing with reference c implementation.
 * Feed the same random data and check test output and reference output.
 * Define a template class to handle the common process, and
 * declare sub class to handle different bitdepth.
 *
 * Expected result:
 * Output from assemble functions should be the same with output from c.
 *
 * Test coverage:
 * Test cases:
 * input value: Fill with random values
 * test mode: fixed block size, random block size and extreme data check
 *
 */
template <typename Sample>
class PixelProjErrorTest
    : public ::testing::TestWithParam<PixelProjErrorTestParam> {
  public:
    virtual void prepare_random_data() = 0;
    virtual void prepare_extreme_data() = 0;
    void run_and_check_data(const int index, const int fixed_size) {
        constexpr int dgd_stride = MAX_DATA_BLOCK;
        constexpr int src_stride = MAX_DATA_BLOCK;
        constexpr int flt0_stride = MAX_DATA_BLOCK;
        constexpr int flt1_stride = MAX_DATA_BLOCK;
        int h_end = fixed_size;
        int v_end = fixed_size;
        bool is_fixed_size = true;
        if (fixed_size == 0) {
            h_end = rnd_blk_size_.random();
            v_end = rnd_blk_size_.random();
            is_fixed_size = false;
        }

        int xq[2] = {
            rnd8_.random() % (1 << SGRPROJ_PRJ_BITS),
            rnd8_.random() % (1 << SGRPROJ_PRJ_BITS),
        };
        SgrParamsType params;
        params.r[0] =
            !is_fixed_size ? (rnd8_.random() % MAX_RADIUS) : (index % 2);
        params.r[1] =
            !is_fixed_size ? (rnd8_.random() % MAX_RADIUS) : (index / 2);
        params.s[0] =
            !is_fixed_size ? (rnd8_.random() % MAX_RADIUS) : (index % 2);
        params.s[1] =
            !is_fixed_size ? (rnd8_.random() % MAX_RADIUS) : (index / 2);
        uint8_t *dgd = (sizeof(*dgd_.data()) == 2)
                           ? (CONVERT_TO_BYTEPTR(dgd_.data()))
                           : reinterpret_cast<uint8_t *>(dgd_.data());
        uint8_t *src = (sizeof(*src_.data()) == 2)
                           ? (CONVERT_TO_BYTEPTR(src_.data()))
                           : reinterpret_cast<uint8_t *>(src_.data());

        int64_t err_ref = ref_func_(src,
                                    h_end,
                                    v_end,
                                    src_stride,
                                    dgd,
                                    dgd_stride,
                                    flt0_.data(),
                                    flt0_stride,
                                    flt1_.data(),
                                    flt1_stride,
                                    xq,
                                    &params);
        int64_t err_test = tst_func_(src,
                                     h_end,
                                     v_end,
                                     src_stride,
                                     dgd,
                                     dgd_stride,
                                     flt0_.data(),
                                     flt0_stride,
                                     flt1_.data(),
                                     flt1_stride,
                                     xq,
                                     &params);
        ASSERT_EQ(err_ref, err_test);
    }

    virtual void run_random_test(const int run_times,
                                 const bool is_fixed_size) {
        const int iters = AOMMAX(run_times, min_test_times);
        for (int iter = 0; iter < iters && !HasFatalFailure(); ++iter) {
            prepare_random_data();
            run_and_check_data(iter, is_fixed_size ? 128 : 0);
        }
    }

    virtual void run_extreme_test() {
        const int iters = min_test_times;
        for (int iter = 0; iter < iters && !HasFatalFailure(); ++iter) {
            prepare_extreme_data();
            run_and_check_data(iter, 192);
        }
    }

  protected:
    PixelProjFunc tst_func_{TEST_GET_PARAM(0)};
    PixelProjFunc ref_func_{TEST_GET_PARAM(1)};
    alignas(16) std::array<Sample, MAX_DATA_BLOCK * MAX_DATA_BLOCK> src_{};
    alignas(16) std::array<Sample, MAX_DATA_BLOCK * MAX_DATA_BLOCK> dgd_{};
    alignas(16) std::array<int32_t, MAX_DATA_BLOCK * MAX_DATA_BLOCK> flt0_{};
    alignas(16) std::array<int32_t, MAX_DATA_BLOCK * MAX_DATA_BLOCK> flt1_{};

    SVTRandom rnd8_{8, false};
    SVTRandom rnd16_{16, false};
    SVTRandom rnd15s_{15, true};
    SVTRandom rnd_blk_size_{1, MAX_DATA_BLOCK};
};  // namespace

class PixelProjErrorLbdTest : public PixelProjErrorTest<uint8_t> {
    void prepare_random_data() override {
        for (int i = 0; i < MAX_DATA_BLOCK * MAX_DATA_BLOCK; ++i) {
            dgd_[i] = rnd8_.random();
            src_[i] = rnd8_.random();
            flt0_[i] = rnd15s_.random();
            flt1_[i] = rnd15s_.random();
        }
    }

    void prepare_extreme_data() override {
        dgd_.fill(0);
        src_.fill(255);
        for (int i = 0; i < MAX_DATA_BLOCK * MAX_DATA_BLOCK; ++i) {
            flt0_[i] = rnd15s_.random();
            flt1_[i] = rnd15s_.random();
        }
    }
};

TEST_P(PixelProjErrorLbdTest, MatchTestWithRandomValue) {
    run_random_test(50, true);
}
TEST_P(PixelProjErrorLbdTest, MatchTestWithRandomSizeAndValue) {
    run_random_test(50, false);
}
TEST_P(PixelProjErrorLbdTest, MatchTestWithExtremeValue) {
    run_extreme_test();
}

#if ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, PixelProjErrorLbdTest,
    ::testing::Values(std::make_tuple(svt_av1_lowbd_pixel_proj_error_sse4_1,
                                      svt_av1_lowbd_pixel_proj_error_c)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, PixelProjErrorLbdTest,
    ::testing::Values(std::make_tuple(svt_av1_lowbd_pixel_proj_error_avx2,
                                      svt_av1_lowbd_pixel_proj_error_c)));

#if EN_AVX512_SUPPORT
INSTANTIATE_TEST_SUITE_P(
    AVX512, PixelProjErrorLbdTest,
    ::testing::Values(std::make_tuple(svt_av1_lowbd_pixel_proj_error_avx512,
                                      svt_av1_lowbd_pixel_proj_error_c)));
#endif
#endif  // ARCH_X86_64

#if ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, PixelProjErrorLbdTest,
    ::testing::Values(std::make_tuple(svt_av1_lowbd_pixel_proj_error_neon,
                                      svt_av1_lowbd_pixel_proj_error_c)));

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, PixelProjErrorLbdTest,
    ::testing::Values(std::make_tuple(svt_av1_lowbd_pixel_proj_error_sve,
                                      svt_av1_lowbd_pixel_proj_error_c)));
#endif  // HAVE_SVE
#endif  // ARCH_AARCH64

#if CONFIG_ENABLE_HIGH_BIT_DEPTH

class PixelProjErrorHbdTest : public PixelProjErrorTest<uint16_t> {
  protected:
    void prepare_random_data() override {
        for (int i = 0; i < MAX_DATA_BLOCK * MAX_DATA_BLOCK; ++i) {
            dgd_[i] = rnd12_.random();
            src_[i] = rnd12_.random();
            flt0_[i] = rnd15s_.random();
            flt1_[i] = rnd15s_.random();
        }
    }

    void prepare_extreme_data() override {
        dgd_.fill(0);
        src_.fill((1 << 12) - 1);
        for (int i = 0; i < MAX_DATA_BLOCK * MAX_DATA_BLOCK; ++i) {
            flt0_[i] = rnd15s_.random();
            flt1_[i] = rnd15s_.random();
        }
    }

  private:
    SVTRandom rnd12_{12, false};
};

TEST_P(PixelProjErrorHbdTest, MatchTestWithRandomValue) {
    run_random_test(50, true);
}
TEST_P(PixelProjErrorHbdTest, MatchTestWithRandomSizeAndValue) {
    run_random_test(50, false);
}
TEST_P(PixelProjErrorHbdTest, MatchTestWithExtremeValue) {
    run_extreme_test();
}

#if ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, PixelProjErrorHbdTest,
    ::testing::Values(std::make_tuple(svt_av1_highbd_pixel_proj_error_sse4_1,
                                      svt_av1_highbd_pixel_proj_error_c)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, PixelProjErrorHbdTest,
    ::testing::Values(std::make_tuple(svt_av1_highbd_pixel_proj_error_avx2,
                                      svt_av1_highbd_pixel_proj_error_c)));
#endif  // ARCH_X86_64

#if ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, PixelProjErrorHbdTest,
    ::testing::Values(std::make_tuple(svt_av1_highbd_pixel_proj_error_neon,
                                      svt_av1_highbd_pixel_proj_error_c)));

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, PixelProjErrorHbdTest,
    ::testing::Values(std::make_tuple(svt_av1_highbd_pixel_proj_error_sve,
                                      svt_av1_highbd_pixel_proj_error_c)));
#endif  // HAVE_SVE
#endif  // ARCH_AARCH64

#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

using GetProjSubspaceFunc = decltype(&svt_get_proj_subspace_c);

template <typename Sample>
class GetProjSubspaceTest
    : public ::testing::TestWithParam<GetProjSubspaceFunc> {
  public:
    DEFINE_ALIGNED_NEW_DELETE(GetProjSubspaceTest)

    void run_test() {
        constexpr int32_t pu_width = RESTORATION_PROC_UNIT_SIZE;
        constexpr int32_t pu_height = RESTORATION_PROC_UNIT_SIZE;
        constexpr int NUM_ITERS = 2000;
        Sample *input = input_.data() + stride * 16 + 16;
        Sample *output = output_.data() + out_stride * 16 + 16;
        for (const int32_t width : {128, 192, 256, 270}) {
            int32_t *flt0 = tmpbuf_.data();
            int32_t *flt1 = flt0 + RESTORATION_UNITPELS_MAX;
            const int32_t flt_stride = ((width + 7) & ~7) + 8;

            // check all the sg params
            SVTRandom rnd{8, false};
            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                // prepare src data and recon data
                for (int i = -16; i < height + 16; ++i) {
                    for (int j = -16; j < width + 16; ++j) {
                        if (iter == 0)
                            output[i * stride + j] = input[i * stride + j] =
                                rnd.random();
                        else if (iter == 1)
                            output[i * stride + j] = input[i * stride + j] = 0;
                        else {
                            input[i * stride + j] = rnd.random();
                            output[i * stride + j] = rnd.random();
                        }
                    }
                }

                for (int32_t ep = 0; ep < SGRPROJ_PARAMS; ++ep) {
                    // apply selfguided filter to get A and b
                    for (int k = 0; k < height; k += pu_height) {
                        for (int j = 0; j < width; j += pu_width) {
                            const int32_t w = AOMMIN(pu_width, width - j);
                            const int32_t h = AOMMIN(pu_height, height - k);
                            Sample *output_p = output + k * out_stride + j;
                            int32_t *flt0_p = flt0 + k * flt_stride + j;
                            int32_t *flt1_p = flt1 + k * flt_stride + j;
                            assert(w * h <= RESTORATION_UNITPELS_MAX);

                            svt_av1_selfguided_restoration_c(
                                reinterpret_cast<uint8_t *>(output_p),
                                w,
                                h,
                                out_stride,
                                flt0_p,
                                flt1_p,
                                flt_stride,
                                ep,
                                8,
                                0);
                        }
                    }

                    int32_t xqd_c[2] = {0};
                    int32_t xqd_asm[2] = {0};
                    const SgrParamsType *const params =
                        &svt_aom_eb_sgr_params[ep];
                    const uint8_t *input_p =
                        sizeof(*input) == sizeof(uint16_t)
                            ? CONVERT_TO_BYTEPTR(input)
                            : reinterpret_cast<const uint8_t *>(input);
                    const uint8_t *output_p =
                        sizeof(*output) == sizeof(uint16_t)
                            ? CONVERT_TO_BYTEPTR(output)
                            : reinterpret_cast<const uint8_t *>(output);
                    constexpr int32_t use_highbitdepth =
                        sizeof(*input) == sizeof(uint16_t) ? 1 : 0;

                    svt_get_proj_subspace_c(input_p,
                                            width,
                                            height,
                                            stride,
                                            output_p,
                                            out_stride,
                                            use_highbitdepth,
                                            flt0,
                                            flt_stride,
                                            flt1,
                                            flt_stride,
                                            xqd_c,
                                            params);
                    test_impl_(input_p,
                               width,
                               height,
                               stride,
                               output_p,
                               out_stride,
                               use_highbitdepth,
                               flt0,
                               flt_stride,
                               flt1,
                               flt_stride,
                               xqd_asm,
                               params);
                    ASSERT_EQ(xqd_c[0], xqd_asm[0])
                        << "xqd_asm[0] does not match with xqd_asm[0] with "
                           "iter "
                        << iter << " ep " << ep;
                    ASSERT_EQ(xqd_c[1], xqd_asm[1])
                        << "xqd_asm[1] does not match with xqd_asm[1] with "
                           "iter "
                        << iter << " ep " << ep;
                }
            }
        }
    }

  private:
    static constexpr int32_t height = 256, stride = 300, out_stride = 300;
    const GetProjSubspaceFunc test_impl_{GetParam()};
    alignas(32) std::array<Sample, stride *(height + 32)> input_{};
    alignas(32) std::array<Sample, out_stride *(height + 32)> output_{};
    alignas(32) std::array<int32_t, RESTORATION_TMPBUF_SIZE> tmpbuf_{};
};

using GetProjSubspaceTestLbd = GetProjSubspaceTest<uint8_t>;
using GetProjSubspaceTestHbd = GetProjSubspaceTest<uint16_t>;

TEST_P(GetProjSubspaceTestLbd, MatchTest) {
    run_test();
}
TEST_P(GetProjSubspaceTestHbd, MatchTest) {
    run_test();
}

#if ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(AVX2, GetProjSubspaceTestLbd,
                         ::testing::Values(svt_get_proj_subspace_avx2));

INSTANTIATE_TEST_SUITE_P(AVX2, GetProjSubspaceTestHbd,
                         ::testing::Values(svt_get_proj_subspace_avx2));
#endif  // ARCH_X86_64

#if ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(NEON, GetProjSubspaceTestLbd,
                         ::testing::Values(svt_get_proj_subspace_neon));
INSTANTIATE_TEST_SUITE_P(NEON, GetProjSubspaceTestHbd,
                         ::testing::Values(svt_get_proj_subspace_neon));
#endif  // ARCH_AARCH64

}  // namespace
