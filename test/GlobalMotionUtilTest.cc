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
 * @file GlobalMotionUtilTest.cc
 *
 * @brief Unit test for utility functions in global motion:
 * - ransac_affine
 * - ransac_rotzoom
 * - ransac_translation
 * - ransac_affine_double_prec
 * - ransac_rotzoom_double_prec
 * - ransac_translation_double_prec
 *
 * @author Cidana-Edmond
 *
 ******************************************************************************/
#undef NOMINMAX
#define NOMINMAX

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "definitions.h"
#include "ransac.h"
#include "random.hpp"
#include "util.h"
namespace {
using std::tuple;
using std::vector;
using svt_av1_test_tool::SVTRandom;
struct AffineMat {
    /* clang-format off */
    /* affine matrix is defined as
     * | m0  m1  m2 |
     * | m3  m4  m5 |
     * |  0   0   1 |
     */
    /* clang-format on */
    double m0;
    double m1;
    double m2;
    double m3;
    double m4;
    double m5;

    /* clang-format off */
    /* the transform matrix in motion parameters is defined:
     * | m2  m3  m0 |
     * | m4  m5  m1 |
     * | m6  m7   1 |
     */
    /* clang-format on */
    void check_transform_matrix(const double params[6]) const {
        ASSERT_NEAR(m0, params[2], 0.0001f);
        ASSERT_NEAR(m1, params[3], 0.0001f);
        ASSERT_NEAR(m2, params[0], 1.0f);
        ASSERT_NEAR(m3, params[4], 0.0001f);
        ASSERT_NEAR(m4, params[5], 0.0001f);
        ASSERT_NEAR(m5, params[1], 1.0f);
    }
};

struct Point {
    double x;
    double y;

    static Point get_random_point(SVTRandom &rnd) {
        // The reason we are using just random and not random_float is that it
        // causes too much drift in m5 with MSVC 2019.
        return {static_cast<double>(rnd.random()),
                static_cast<double>(rnd.random())};
    }

    /* clang-format off */
    /* affine transform
     * | x |   | m0  m1  m2 |   | x*m0 + y*m1 + m2 |
     * | y | X | m3  m4  m5 | = | x*m3 + y*m4 + m5 |
     * | 1 |   |  0   0   1 |   |         1        |
     */
    /* clang-format on */
    Point affine(const AffineMat &mat) const {
        return {(x * mat.m0) + (y * mat.m1) + mat.m2,
                (x * mat.m3) + (y * mat.m4) + mat.m5};
    }
    /* clang-format off */
    /* translation transform
     * | x |   | 1  0  offset_x |   | x+offset_x |
     * | y | X | 0  1  offset_y | = | y+offset_y |
     * | 1 |   | 0  0     1     |   |     1      |
     */
    /* clang-format on */
    Point translate(double offset_x, double offset_y, AffineMat &mat) const {
        mat = {1.0f, 0.0f, offset_x, 0.0f, 1.0f, offset_y};
        return affine(mat);
    }
    /* clang-format off */
    /* zoom transform
     * | x |   | scale  0    0 |   | x*scale |
     * | y | X |   0  scale  0 | = | y*scale |
     * | 1 |   |   0    0    1 |   |    1    |
     */
    /* clang-format on */
    Point zoom(double scale, AffineMat &mat) const {
        mat = {scale, 0.0f, 0.0f, 0.0f, scale, 0.0f};
        return affine(mat);
    }
    /* clang-format off */
    /* rotate transform
     * | x |   | cos(t)  -sin(t)  0 |   | x*cos(t) - y*sin(t) |
     * | y | X | sin(t)  cos(t)   0 | = | x*sin(t) + y*cos(t) |
     * | 1 |   |   0       0      1 |   |          1          |
     */
    /* clang-format on */
    Point rotate(double theta, AffineMat &mat) const {
        const double sin_theta = std::sin(theta);
        const double cos_theta = std::cos(theta);
        mat = {cos_theta, -1 * sin_theta, 0.0f, sin_theta, cos_theta, 0.0f};
        return affine(mat);
    }
};

using TransDataFunc = size_t (*)(SVTRandom &rnd, vector<Point> &data,
                                 vector<Point> &ref, AffineMat &mat);

size_t transform_data_translation(SVTRandom &rnd, vector<Point> &data,
                                  vector<Point> &ref, AffineMat &mat) {
    const int offset_x = (rnd.random() >> 2) * ((rnd.random() % 2) ? -1 : 1);
    const int offset_y = (rnd.random() >> 2) * ((rnd.random() % 2) ? -1 : 1);
    std::transform(data.begin(),
                   data.end(),
                   std::back_inserter(ref),
                   [offset_x, offset_y, &mat](const Point &pt) {
                       return pt.translate(offset_x, offset_y, mat);
                   });
    return ref.size();
}

size_t transform_data_zoom_rotate(SVTRandom &rnd, vector<Point> &data,
                                  vector<Point> &ref, AffineMat &mat) {
    /** limit zoom rate from 50% to 150% */
    const double zoom_scale = (rnd.random() % 1000) / 1000.0f + 0.5f;
    const double theta = PI * (rnd.random() % 360) / 360;
    AffineMat mat_zoom{}, mat_rotate{};
    std::transform(
        data.begin(),
        data.end(),
        std::back_inserter(ref),
        [zoom_scale, theta, &mat_zoom, &mat_rotate](const Point &pt) {
            return pt.zoom(zoom_scale, mat_zoom).rotate(theta, mat_rotate);
        });
    mat.m0 = mat_rotate.m0 * mat_zoom.m0;
    mat.m1 = mat_rotate.m1 * mat_zoom.m0;
    mat.m2 = 0;
    mat.m3 = mat_rotate.m3 * mat_zoom.m4;
    mat.m4 = mat_rotate.m4 * mat_zoom.m4;
    mat.m5 = 0;
    return ref.size();
}

size_t transform_data_affine(SVTRandom &rnd, vector<Point> &data,
                             vector<Point> &ref, AffineMat &mat) {
    mat = {(rnd.random() % 500) / 1000.0f * (((rnd.random() % 2) ? -1 : 1)),
           (rnd.random() % 500) / 1000.0f * (((rnd.random() % 2) ? -1 : 1)),
           (rnd.random() / 4.0) * (((rnd.random() % 2) ? -1 : 1)),
           (rnd.random() % 500) / 1000.0f * (((rnd.random() % 2) ? -1 : 1)),
           (rnd.random() % 500) / 1000.0f * (((rnd.random() % 2) ? -1 : 1)),
           (rnd.random() / 4.0) * (((rnd.random() % 2) ? -1 : 1))};
    std::transform(data.begin(),
                   data.end(),
                   std::back_inserter(ref),
                   [&mat](const Point &pt) { return pt.affine(mat); });
    return ref.size();
}

const std::unordered_map<TransformationType, TransDataFunc> trans_data_func{
    {TRANSLATION, transform_data_translation},
    {ROTZOOM, transform_data_zoom_rotate},
    {AFFINE, transform_data_affine},
};

constexpr int CoordinateMax = (1 << 16) - 1;
constexpr int PointCountMin = 15; /**< 3*MINPTS_MULTIPLIER */

/**
 * @brief Unit test for RANSAC functions:
 * - ransac_affine
 * - ransac_rotzoom
 * - ransac_translation
 * - ransac_affine_double_prec
 * - ransac_rotzoom_double_prec
 * - ransac_translation_double_prec
 *
 * Test strategy:
 * Create a pair of 2D point sets by the matrix of affine transform
 * (translation, zoom, rotate and affine), then add some noise in test data;
 * check the motion parameters in the result of RANSAC function
 *
 * Expected result:
 * The difference between affine transform matrix and the parameters in motion
 * should be less than threshold
 *
 */
template <typename Sample>
class RansacTest : public ::testing::TestWithParam<TransformationType> {
  protected:
    RansacTest() = default;

    template <int count>
    void generate_data(AffineMat &mat) {
        /** ransac function requires more than 15 samples */
        static_assert(count >= PointCountMin,
                      "count must be at least PointCountMin");
        const size_t inliers = std::max(rnd_.random() % count, 10);
        ref_.clear();
        ref_.reserve(inliers);
        data_.resize(inliers, {});
        std::generate(data_.begin(), data_.end(), [&]() {
            return Point::get_random_point(rnd_);
        });
        const auto trans_func = trans_data_func.at(GetParam());
        const auto data_count = trans_func(rnd_, data_, ref_, mat);
        ASSERT_EQ(data_count, inliers);

        /** add noise, less than 25% */
        const int64_t available = count - int64_t(inliers);
        const int64_t max_noise = inliers / 4;
        const int64_t min_noise =
            std::max<int64_t>(0, PointCountMin - int64_t(inliers));

        const size_t noise_count =
            std::max(min_noise, std::min(available, max_noise));
        for (size_t i = 0; i < noise_count; i++) {
            const auto insert_pos = rnd_.random() % data_.size();
            data_.insert(data_.begin() + insert_pos,
                         Point::get_random_point(rnd_));
            ref_.insert(ref_.begin() + insert_pos,
                        Point::get_random_point(rnd_));
        }
    }

    void run_test(size_t times) {
        for (size_t i = 0; i < times; i++) {
            constexpr int max_data_count = MAX_CORNERS;
            AffineMat mat;
            generate_data<max_data_count>(mat);
            do_ransac_check(mat);
        }
    }

    virtual void prepare_input(vector<Sample> &input, size_t npoints) = 0;

    void do_ransac_check(const AffineMat &mat) {
        const int npoints = static_cast<int>(data_.size());
        vector<Sample> points(npoints * 4);
        prepare_input(points, npoints);

        constexpr int num_motions = RANSAC_NUM_MOTIONS;
        std::array<std::vector<int>, num_motions> inliers_storage;
        for (auto &vec : inliers_storage) {
            vec.resize(2 * MAX_CORNERS, 0);
        }
        std::array<MotionModel, num_motions> motions{};
        for (int i = 0; i < num_motions; i++) {
            motions[i].inliers = inliers_storage[i].data();
        }

        bool mem_alloc_failed = false;
        ASSERT_TRUE(
            svt_aom_ransac(reinterpret_cast<Correspondence *>(points.data()),
                           npoints,
                           GetParam(),
                           motions.data(),
                           num_motions,
                           &mem_alloc_failed));

        /** check for the number of inlier */
        ASSERT_NE(motions[0].num_inliers, 0);

        /** check for the transform matrix of motion */
        mat.check_transform_matrix(motions[0].params);
    }

  protected:
    SVTRandom rnd_{0, CoordinateMax};
    vector<Point> data_{};
    vector<Point> ref_{};
};

class RansacIntTest : public RansacTest<int> {
  protected:
    void prepare_input(vector<int> &input, size_t npoints) override {
        for (size_t i = 0; i < npoints; i++) {
            input[4 * i] = std::lround(data_.at(i).x);
            input[4 * i + 1] = std::lround(data_.at(i).y);
            input[4 * i + 2] = std::lround(ref_.at(i).x);
            input[4 * i + 3] = std::lround(ref_.at(i).y);
        }
    }
};

constexpr TransformationType transform_table[]{
    TRANSLATION,
    ROTZOOM,
    AFFINE,
};

TEST_P(RansacIntTest, CheckOutput) {
    run_test(1000);
};

INSTANTIATE_TEST_SUITE_P(GlobalMotion, RansacIntTest,
                         ::testing::ValuesIn(transform_table));

}  // namespace
