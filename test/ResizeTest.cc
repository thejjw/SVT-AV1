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
 * @file ResizeTest.cc
 *
 * @brief Unit test for resize of downsampling functions:
 * - svt_av1_resize_plane
 * - svt_av1_highbd_resize_plane
 *
 * @author Cidana-Edmond
 *
 ******************************************************************************/

#include "gtest/gtest.h"
#include <algorithm>
#include <cstdint>
#include "TestEnv.h"
#include "aligned_allocator.hpp"
#include "aom_dsp_rtcd.h"
#include "random.hpp"
#include "resize.h"
#include "super_res.h"
#include "util.h"

namespace {
using std::make_tuple;
using svt_av1_test_tool::aligned_allocator;
using svt_av1_test_tool::SizeOnlyVec;
using svt_av1_test_tool::SVTRandom;

constexpr auto min_test_times = 10;
constexpr auto REF_STUFF = 0xAA;
constexpr auto TST_STUFF = 0xBB;

using PicSizeParam =
    std::tuple<uint16_t,  /**< width of source/upscaled picture */
               uint16_t,  /**< height of source picture */
               uint16_t>; /**< stride of source picture */

using ResizeTestParam =
    std::tuple<PicSizeParam, uint8_t, /**< denominator of scaling: 8~16 */
               uint8_t>;              /**< bit depth: 8, 10, 12 */

/**
 * @brief Unit test for resize down sampling:
 * - svt_av1_resize_plane
 * - svt_av1_highbd_resize_plane
 *
 * Test strategy:
 * Verify this assembly code by comparing with reference c implementation.
 * Feed the same random data and check test output and reference output.
 * Define a template class to handle the common process, and
 * declare sub class to handle different bit depth.
 *
 * Expected result:
 * Output from assemble functions should be the same with output from c.
 *
 * Test coverage:
 * Test cases:
 * input value: Fill with zero, extreme values and random values
 * test mode: common video frame resolution, 2D data blocks with random
 * rectangle size
 *
 */
template <typename Sample, Sample ref_stuff = REF_STUFF,
          Sample tst_stuff = TST_STUFF>
class ResizePlaneTest : public ::testing::TestWithParam<ResizeTestParam> {
  public:
    ResizePlaneTest() {
        setup_test_env();
    }

    virtual void SetUp() override {
        ASSERT_LE(src_width_, src_stride_)
            << "picture width must be less equal than stride";
        ASSERT_LE(scaled_width_, src_width_)
            << "width of scaled picture must be less equal than source";
    }

    void prepare_zero_data() {
        std::fill_n(src_.begin(), src_stride_ * src_height_, 0);
    }

    void prepare_random_data() {
        std::generate_n(src_.begin(), src_stride_ * src_height_, [this]() {
            return rnd_.random();
        });
    }

    void prepare_extreme_data() {
        std::fill_n(src_.begin(), src_stride_ * src_height_, (1 << bd_) - 1);
    }

    virtual void run_test(bool width_only = false) = 0;

    void check_data(const int index) {
        const uint16_t value_limit = 1 << bd_;

        for (uint16_t y = 0; y < scaled_height_; y++) {
            // check upscaled data
            for (uint16_t x = 0; x < scaled_width_; x++) {
                ASSERT_LT(scaled_ref_[y * src_stride_ + x], value_limit);
                ASSERT_LT(scaled_tst_[y * src_stride_ + x], value_limit);
                ASSERT_EQ(scaled_ref_[y * src_stride_ + x],
                          scaled_tst_[y * src_stride_ + x])
                    << "scaled pixel mismatch at test(" << index
                    << ") row: " << y << ", col: " << x << ", "
                    << scaled_ref_[y * src_stride_ + x] << "<-->"
                    << scaled_tst_[y * src_stride_ + x];
            }
            // check padding data
            for (uint16_t x = scaled_width_; x < src_stride_; x++) {
                EXPECT_EQ(scaled_ref_[y * src_stride_ + x], ref_stuff);
                EXPECT_EQ(scaled_tst_[y * src_stride_ + x], tst_stuff);
            }
        }
    }

    template <bool width_only = false>
    void run_zero_test() {
        constexpr auto iters = min_test_times;
        for (int iter = 0; iter < iters && !HasFatalFailure(); ++iter) {
            prepare_zero_data();
            run_test(width_only);
            check_data(iter);
        }
    }

    template <int run_times, bool width_only = false>
    void run_random_test() {
        constexpr auto iters = AOMMAX(run_times, min_test_times);
        for (int iter = 0; iter < iters && !HasFatalFailure(); ++iter) {
            prepare_random_data();
            run_test(width_only);
            check_data(iter);
        }
    }

    template <bool width_only = false>
    void run_extreme_test() {
        constexpr auto iters = min_test_times;
        for (int iter = 0; iter < iters && !HasFatalFailure(); ++iter) {
            prepare_extreme_data();
            run_test(width_only);
            check_data(iter);
        }
    }

  protected:
    const uint16_t src_width_{std::get<0>(TEST_GET_PARAM(0))};
    const uint16_t src_height_{std::get<1>(TEST_GET_PARAM(0))};
    const uint16_t src_stride_{std::get<2>(TEST_GET_PARAM(0))};
    const uint8_t denom_{TEST_GET_PARAM(1)};
    const uint8_t bd_{TEST_GET_PARAM(2)};
    const uint16_t scaled_width_{
        svt_aom_calc_scaled_size_helper(src_width_, denom_)};
    const uint16_t scaled_height_{
        svt_aom_calc_scaled_size_helper(src_height_, denom_)};

    using SampleVector = SizeOnlyVec<Sample, aligned_allocator<Sample>>;

    SampleVector src_{std::size_t(src_stride_ * src_height_)};
    SampleVector scaled_ref_{std::size_t(src_stride_ * src_height_), ref_stuff};
    SampleVector scaled_tst_{std::size_t(src_stride_ * src_height_), tst_stuff};
    SVTRandom rnd_{bd_, false};
};

class ResizePlaneLbdTest : public ResizePlaneTest<uint8_t> {
  public:
    void run_test(bool width_only) override {
        // setup using c code
        reset_test_env();
        if (width_only) {
            svt_av1_resize_plane_horizontal(src_.data(),
                                            src_height_,
                                            src_width_,
                                            src_stride_,
                                            scaled_ref_.data(),
                                            src_height_,
                                            scaled_width_,
                                            src_stride_);
        } else {
            svt_av1_resize_plane(src_.data(),
                                 src_height_,
                                 src_width_,
                                 src_stride_,
                                 scaled_ref_.data(),
                                 scaled_height_,
                                 scaled_width_,
                                 src_stride_);
        }
        // setup using simd accelerating
        setup_test_env();
        if (width_only) {
            svt_av1_resize_plane_horizontal(src_.data(),
                                            src_height_,
                                            src_width_,
                                            src_stride_,
                                            scaled_tst_.data(),
                                            src_height_,
                                            scaled_width_,
                                            src_stride_);
        } else {
            svt_av1_resize_plane(src_.data(),
                                 src_height_,
                                 src_width_,
                                 src_stride_,
                                 scaled_tst_.data(),
                                 scaled_height_,
                                 scaled_width_,
                                 src_stride_);
        }
    }
};

TEST_P(ResizePlaneLbdTest, MatchTestWithZeroValue) {
    run_zero_test();
    run_zero_test<true>();
}
TEST_P(ResizePlaneLbdTest, MatchTestWithRandomValue) {
    run_random_test<10>();
    run_random_test<10, true>();
}
TEST_P(ResizePlaneLbdTest, MatchTestWithExtremeValue) {
    run_extreme_test();
    run_extreme_test<true>();
}

static PicSizeParam pic_size_vector[] = {
    make_tuple(1280, 720, 1520),
    make_tuple(1920, 1080, 2240),
    make_tuple(3840, 2160, 4480),
};

INSTANTIATE_TEST_SUITE_P(
    Resize, ResizePlaneLbdTest,
    ::testing::Combine(::testing::ValuesIn(pic_size_vector),
                       ::testing::Range<uint8_t>(8, 16),
                       ::testing::Values<uint8_t>(8)));

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
class ResizePlaneHbdTest
    : public ResizePlaneTest<uint16_t, REF_STUFF << 8 | REF_STUFF,
                             TST_STUFF << 8 | TST_STUFF> {
  protected:
    void run_test(bool width_only) override {
        // setup using c code
        reset_test_env();
        if (width_only) {
            svt_av1_highbd_resize_plane_horizontal(src_.data(),
                                                   src_height_,
                                                   src_width_,
                                                   src_stride_,
                                                   scaled_ref_.data(),
                                                   src_height_,
                                                   scaled_width_,
                                                   src_stride_,
                                                   bd_);
        } else {
            svt_av1_highbd_resize_plane(src_.data(),
                                        src_height_,
                                        src_width_,
                                        src_stride_,
                                        scaled_ref_.data(),
                                        scaled_height_,
                                        scaled_width_,
                                        src_stride_,
                                        bd_);
        }
        // setup using simd accelerating
        setup_test_env();
        if (width_only) {
            svt_av1_highbd_resize_plane_horizontal(src_.data(),
                                                   src_height_,
                                                   src_width_,
                                                   src_stride_,
                                                   scaled_tst_.data(),
                                                   src_height_,
                                                   scaled_width_,
                                                   src_stride_,
                                                   bd_);
        } else {
            svt_av1_highbd_resize_plane(src_.data(),
                                        src_height_,
                                        src_width_,
                                        src_stride_,
                                        scaled_tst_.data(),
                                        scaled_height_,
                                        scaled_width_,
                                        src_stride_,
                                        bd_);
        }
    }
};

TEST_P(ResizePlaneHbdTest, MatchTestWithZeroValue) {
    run_zero_test();
    run_zero_test<true>();
}
TEST_P(ResizePlaneHbdTest, MatchTestWithRandomValue) {
    run_random_test<10>();
    run_random_test<10, true>();
}
TEST_P(ResizePlaneHbdTest, MatchTestWithExtremeValue) {
    run_extreme_test();
    run_extreme_test<true>();
}

INSTANTIATE_TEST_SUITE_P(
    Resize, ResizePlaneHbdTest,
    ::testing::Combine(::testing::ValuesIn(pic_size_vector),
                       ::testing::Range<uint8_t>(8, 16),
                       ::testing::Values<uint8_t>(10, 12)));
#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH
}  // namespace
