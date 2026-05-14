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
 * @file PaletteModeUtilTest.cc
 *
 * - svt_aom_count_colors
 * @brief Unit test for util functions in palette mode:
 * - svt_aom_count_colors_highbd
 * - av1_k_means_dim1
 * - av1_k_means_dim2
 *
 * @author Cidana-Edmond
 *
 ******************************************************************************/
#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <vector>

#include "gtest/gtest.h"
#include "aligned_allocator.hpp"
#include "definitions.h"
#include "pic_analysis_process.h"
#include "random.hpp"
#include "util.h"
#include "aom_dsp_rtcd.h"

namespace {
using svt_av1_test_tool::aligned_allocator;
using svt_av1_test_tool::SizeOnlyVec;
using svt_av1_test_tool::SVTRandom;

/**
 * @brief Unit test for counting colors:
 * - svt_aom_count_colors
 * - svt_aom_count_colors_highbd
 *
 * Test strategy:
 * Feeds the random value both into test function and the vector without
 * duplicated, then compares the count of result and the individual item
 * count in vector.
 *
 * Expected result:
 * The count numbers from test function and vector are the same.
 *
 * Test coverage:
 * The input can be 8-bit and 8-bit/10-bit/12-bit for HBD cases
 */
template <typename Sample, uint8_t bd = 8>
class ColorCountTest : public ::testing::Test {
  public:
    DEFINE_ALIGNED_NEW_DELETE(ColorCountTest)

  protected:
    using ValCountVec = std::vector<int, aligned_allocator<int>>;

    void prepare_data() {
        input_.fill(0);
        std::generate(input_.begin(), input_.end(), [&]() {
            constexpr int32_t mask = (1 << bd) - 1;
            return rnd_.random() & mask;
        });
        /** store the same value for reference */
        std::set<Sample> unique_values{input_.begin(), input_.end()};
        ref_.assign(unique_values.begin(), unique_values.end());
    }

    template <size_t times>
    void run_test() {
        constexpr int max_colors = (1 << bd);
        val_count_.resize(max_colors);
        for (size_t i = 0; i < times; i++) {
            prepare_data();
            ASSERT_EQ(count_color(), ref_.size())
                << "color count failed at: " << i;
        }
        val_count_.clear();
    }

    virtual unsigned int count_color() = 0;

  protected:
    SVTRandom rnd_{16, false};
    alignas(32) std::array<Sample, MAX_PALETTE_SQUARE> input_{};
    std::vector<int> ref_{};
    ValCountVec val_count_{};
};

class ColorCountLbdTest : public ColorCountTest<uint8_t> {
  protected:
    unsigned int count_color() override {
        constexpr int max_colors = 8;
        std::fill_n(val_count_.begin(), max_colors, 0);
        unsigned int colors = (unsigned int)svt_aom_count_colors(
            input_.data(), 64, 64, 64, val_count_.data());
        return colors;
    }
};

TEST_F(ColorCountLbdTest, MatchTest) {
    run_test<1000>();
}

template <uint8_t bd>
class ColorCountHbdTest : public ColorCountTest<uint16_t, bd> {
  protected:
    unsigned int count_color() override {
        constexpr int max_colors = (1 << bd);
        std::fill_n(this->val_count_.begin(), max_colors, 0);
        unsigned int colors = (unsigned int)svt_aom_count_colors_highbd(
            this->input_.data(), 64, 64, 64, bd, this->val_count_.data());
        return colors;
    }
};

using ColorCountHbdTest8 = ColorCountHbdTest<8>;
using ColorCountHbdTest10 = ColorCountHbdTest<10>;
using ColorCountHbdTest12 = ColorCountHbdTest<12>;

TEST_F(ColorCountHbdTest8, MatchTest8Bit) {
    run_test<1000>();
}

TEST_F(ColorCountHbdTest10, MatchTest10Bit) {
    run_test<1000>();
}

TEST_F(ColorCountHbdTest12, MatchTest12Bit) {
    run_test<1000>();
}

constexpr int MaxItr = 50;

/**
 * @brief Unit test for kmeans functions:
 * - av1_k_means_dim1
 * - av1_k_means_dim2
 *
 * Test strategy:
 * Feeds the plane buffer with random colors into kmeans function and get the
 * centroids and indices, verifies each color being the closest to the centroid
 * in all candidates.
 *
 * Expected result:
 * Every pixels are closest to their centroid in all candidates
 *
 * Test coverage:
 * Tests for K from PALETTE_MIN_SIZE to PALETTE_MAX_SIZE
 */
class KMeansTest : public ::testing::TestWithParam<int> {
  protected:
    /** functions for 1d test */
    int prepare_data(const int max_colors) {
        assert(max_colors > 0);
        std::vector<uint8_t> palette(max_colors);
        std::generate(
            palette.begin(), palette.end(), [&]() { return rnd_.random(); });
        std::array<uint8_t, MAX_PALETTE_SQUARE> tmp{0};
        std::generate(tmp.begin(), tmp.end(), [&]() {
            return palette[rnd_.random() % max_colors];
        });
        std::copy(tmp.begin(), tmp.end(), data_.begin());
        std::fill(data_.begin() + tmp.size(), data_.end(), 0);
        int val_count[MAX_PALETTE_SQUARE]{0};
        return svt_aom_count_colors(tmp.data(), 64, 64, 64, val_count);
    }

    template <size_t times>
    void run_test() {
        uint8_t indices[MAX_PALETTE_SQUARE] = {0};
        for (size_t i = 0; i < times; i++) {
            const int max_colors = palette_rnd_.random();
            const int colors = prepare_data(max_colors);
            int centroids[PALETTE_MAX_SIZE] = {0};
            const int k = AOMMIN(colors, k_);
            svt_av1_k_means_dim1_c(data_.data(),
                                   centroids,
                                   indices,
                                   MAX_PALETTE_SQUARE,
                                   k,
                                   MaxItr);
            check_output(centroids, k, data_, indices);
        }
    }

    template <size_t N1, size_t N2>
    static void check_output(const int* centroids, const int k,
                             const std::array<int, N1>& data,
                             const uint8_t (&indices)[N2]) {
        for (size_t i = 0; i < N2; i++) {
            const int min_delta = std::abs(data[i] - centroids[indices[i]]);
            for (int j = 0; j < k; j++) {
                const int delta = std::abs(data[i] - centroids[j]);
                ASSERT_GE(delta, min_delta)
                    << "index error at " << i << ", value is " << data[i]
                    << ", distance to centroid( " << centroids[indices[i]]
                    << ") is greater than to " << centroids[j];
            }
        }
    }

    /** functions for 2d test */
    int prepare_data_2d(const int max_colors) {
        std::vector<uint16_t> palette(max_colors);
        std::generate(palette.begin(), palette.end(), [&]() {
            return (rnd_.random() << 8) + rnd_.random();
        });
        std::set<uint16_t> val_vec;
        for (size_t i = 0; i < MAX_PALETTE_SQUARE; i++) {
            uint16_t tmp = palette[rnd_.random() % max_colors];
            data_[2 * i] = tmp >> 8;
            data_[2 * i + 1] = tmp & 0xFF;
            val_vec.insert(tmp);
        }
        std::fill(data_.begin() + 2 * MAX_PALETTE_SQUARE, data_.end(), 0);
        return (int)val_vec.size();
    }

    template <size_t times>
    void run_test_2d() {
        uint8_t indices[2 * MAX_PALETTE_SQUARE] = {0};
        for (size_t i = 0; i < times; i++) {
            const int max_colors = palette_rnd_.random();
            const int colors = prepare_data_2d(max_colors);
            int centroids[2 * PALETTE_MAX_SIZE] = {0};
            const int k = AOMMIN(colors, k_);
            svt_av1_k_means_dim2_c(data_.data(),
                                   centroids,
                                   indices,
                                   MAX_PALETTE_SQUARE,
                                   k,
                                   MaxItr);
            check_output_2d(centroids, k, data_, indices);
        }
    }

    static double distance_2d(int x1, int y1, int x2, int y2) {
        int x_d = x1 - x2;
        int y_d = y1 - y2;
        return std::sqrt(x_d * x_d + y_d * y_d);
    }

    template <size_t N1, size_t N2>
    static void check_output_2d(const int* centroids, const int k,
                                const std::array<int, N1>& data,
                                const uint8_t (&indices)[N2]) {
        for (size_t i = 0; i < N2 / 2; i++) {
            const double min_delta = distance_2d(data[2 * i],
                                                 data[2 * i + 1],
                                                 centroids[2 * indices[i]],
                                                 centroids[2 * indices[i] + 1]);
            for (int j = 0; j < k; j++) {
                const double delta = distance_2d(data[2 * i],
                                                 data[2 * i + 1],
                                                 centroids[2 * j],
                                                 centroids[2 * j + 1]);
                ASSERT_GE(delta, min_delta)
                    << "index error at " << i << ", value is " << data[i]
                    << ", distance to centroid( " << centroids[indices[i]]
                    << ") is greater than to " << centroids[j];
            }
        }
    }

  protected:
    std::array<int, 2 * MAX_PALETTE_SQUARE> data_{};
    const int k_{GetParam()};
    SVTRandom rnd_{8, false};
    SVTRandom palette_rnd_{2, 64};
};

TEST_P(KMeansTest, CheckOutput) {
    run_test<1000>();
};

TEST_P(KMeansTest, CheckOutput2D) {
    run_test_2d<1000>();
};

INSTANTIATE_TEST_SUITE_P(PalleteMode, KMeansTest,
                         ::testing::Range(PALETTE_MIN_SIZE, PALETTE_MAX_SIZE));

using av1_k_means_func = void (*)(const int* data, int* centroids,
                                  uint8_t* indices, int n, int k, int max_itr);
using av1_k_means_indices_func = void (*)(const int* data, const int* centroids,
                                          uint8_t* indices, int n, int k);

using BlockSize = std::tuple<int, int>;

enum TestPattern { MIN, MAX, RANDOM };

const BlockSize TEST_BLOCK_SIZES[] = {
    BlockSize(4, 4),
    BlockSize(4, 8),
    BlockSize(8, 8),
    BlockSize(8, 16),
    BlockSize(8, 32),
    BlockSize(16, 4),
    BlockSize(16, 16),
    BlockSize(16, 32),
    BlockSize(32, 8),
    BlockSize(32, 32),
    BlockSize(32, 64),
    BlockSize(64, 16),
    BlockSize(64, 64),
    BlockSize(64, 128),
    BlockSize(128, 128),
};
constexpr TestPattern TEST_PATTERNS[] = {MIN, MAX, RANDOM};

#if ARCH_X86_64
static void av1_k_means_wrapper(av1_k_means_func func, const int* data,
                                int* centroids, uint8_t* indices, int n, int k,
                                int max_itr) {
    func(data, centroids, indices, n, k, max_itr);
}
#endif

static void av1_k_means_wrapper(av1_k_means_indices_func func, const int* data,
                                int* centroids, uint8_t* indices, int n, int k,
                                int max_itr) {
    (void)max_itr;
    func(data, centroids, indices, n, k);
}

template <typename FuncType>
using Av1KMeansDimParam =
    std::tuple<TestPattern, BlockSize, std::tuple<FuncType, FuncType>>;

// Additional *2 to account possibility of write into extra memory
constexpr auto centroids_size = 2 * PALETTE_MAX_SIZE * 2;
constexpr auto indices_size = MAX_SB_SQUARE * 2;

template <typename FuncType>
class Av1KMeansDim
    : public ::testing::TestWithParam<Av1KMeansDimParam<FuncType>> {
  protected:
    void prepare_data() {
        if (pattern_ == MIN) {
            std::fill(data_.begin(), data_.end(), 0);
            std::fill(centroids_tst_.begin(), centroids_tst_.end(), 0);
            std::fill(centroids_ref_.begin(), centroids_ref_.end(), 0);
            std::fill(indices_ref_.begin(), indices_ref_.end(), 0);
            std::fill(indices_tst_.begin(), indices_tst_.end(), 0);
        } else if (pattern_ == MAX) {
            std::fill(data_.begin(), data_.end(), 0xff);
            std::fill(centroids_ref_.begin(), centroids_ref_.end(), 0xff);
            std::fill(centroids_tst_.begin(), centroids_tst_.end(), 0xff);
            std::fill(indices_ref_.begin(), indices_ref_.end(), 0xff);
            std::fill(indices_tst_.begin(), indices_tst_.end(), 0xff);
        } else {  // pattern_ == RANDOM
            std::generate(
                data_.begin(), data_.end(), [&]() { return rnd32_.random(); });
            std::generate(centroids_ref_.begin(), centroids_ref_.end(), [&]() {
                return rnd32_.random();
            });
            centroids_tst_ = centroids_ref_;
            std::generate(indices_ref_.begin(), indices_ref_.end(), [&]() {
                return rnd8_.random();
            });
            indices_tst_ = indices_ref_;
        }
    }

    void check_output() {
        ASSERT_EQ(centroids_ref_, centroids_tst_)
            << "Compare Centroids array error";
        ASSERT_EQ(indices_ref_, indices_tst_) << "Compare indices array error";
    }

    void run_test() {
        const size_t test_num = pattern_ == MIN || pattern_ == MAX ? 1 : 100;

        for (int k = PALETTE_MIN_SIZE; k <= PALETTE_MAX_SIZE; k++) {
            for (size_t i = 0; i < test_num; i++) {
                prepare_data();
                av1_k_means_wrapper(func_ref_,
                                    data_.data(),
                                    centroids_ref_.data(),
                                    indices_ref_.data(),
                                    n_,
                                    k,
                                    MaxItr);
                av1_k_means_wrapper(func_tst_,
                                    data_.data(),
                                    centroids_tst_.data(),
                                    indices_tst_.data(),
                                    n_,
                                    k,
                                    MaxItr);
                check_output();
            }
        }
    }

  protected:
    SVTRandom rnd32_{-((1 << 14) - 1), ((1 << 14) - 1)};
    SVTRandom rnd8_{0, ((1 << 8) - 1)};
    const FuncType func_ref_{std::get<0>(TEST_GET_PARAM(2))};
    const FuncType func_tst_{std::get<1>(TEST_GET_PARAM(2))};

    SizeOnlyVec<int> centroids_ref_{std::size_t{centroids_size}};
    SizeOnlyVec<int> centroids_tst_{std::size_t{centroids_size}};
    SizeOnlyVec<uint8_t> indices_ref_{std::size_t{indices_size}};
    SizeOnlyVec<uint8_t> indices_tst_{std::size_t{indices_size}};

    const TestPattern pattern_{TEST_GET_PARAM(0)};
    const BlockSize block_{TEST_GET_PARAM(1)};
    const int n_{std::get<0>(block_) * std::get<1>(block_)};
    //*2 to account of AV1_K_MEANS_DIM = 2
    SizeOnlyVec<int> data_{std::size_t(n_ * 2)};
};

using Av1KMeansIndicesDimTest = Av1KMeansDim<av1_k_means_indices_func>;

TEST_P(Av1KMeansIndicesDimTest, RunCheckOutput) {
    run_test();
};

#if ARCH_X86_64

using Av1KMeansDimTest = Av1KMeansDim<av1_k_means_func>;

TEST_P(Av1KMeansDimTest, RunCheckOutput) {
    run_test();
};

const std::tuple<av1_k_means_func, av1_k_means_func> TEST_FUNC_PAIRS[] = {
    std::make_tuple(svt_av1_k_means_dim1_c, svt_av1_k_means_dim1_avx2),
    std::make_tuple(svt_av1_k_means_dim2_c, svt_av1_k_means_dim2_avx2)};

const std::tuple<av1_k_means_indices_func, av1_k_means_indices_func>
    TEST_INDICES_FUNC_PAIRS[] = {
        std::make_tuple(svt_av1_calc_indices_dim1_c,
                        svt_av1_calc_indices_dim1_avx2),
        std::make_tuple(svt_av1_calc_indices_dim2_c,
                        svt_av1_calc_indices_dim2_avx2)};

INSTANTIATE_TEST_SUITE_P(
    AVX2, Av1KMeansDimTest,
    ::testing::Combine(::testing::ValuesIn(TEST_PATTERNS),
                       ::testing::ValuesIn(TEST_BLOCK_SIZES),
                       ::testing::ValuesIn(TEST_FUNC_PAIRS)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, Av1KMeansIndicesDimTest,
    ::testing::Combine(::testing::ValuesIn(TEST_PATTERNS),
                       ::testing::ValuesIn(TEST_BLOCK_SIZES),
                       ::testing::ValuesIn(TEST_INDICES_FUNC_PAIRS)));

#endif  // ARCH_X86_64

#if ARCH_AARCH64
const std::tuple<av1_k_means_indices_func, av1_k_means_indices_func>
    TEST_INDICES_FUNC_PAIRS[] = {
        std::make_tuple(svt_av1_calc_indices_dim1_c,
                        svt_av1_calc_indices_dim1_neon),
        std::make_tuple(svt_av1_calc_indices_dim2_c,
                        svt_av1_calc_indices_dim2_neon)};

INSTANTIATE_TEST_SUITE_P(
    NEON, Av1KMeansIndicesDimTest,
    ::testing::Combine(::testing::ValuesIn(TEST_PATTERNS),
                       ::testing::ValuesIn(TEST_BLOCK_SIZES),
                       ::testing::ValuesIn(TEST_INDICES_FUNC_PAIRS)));
#endif  // ARCH_AARCH64
}  // namespace
