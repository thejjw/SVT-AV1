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
 * @file random.hpp
 *
 * @brief Random generator for svt-av1 unit tests
 * - wrap C++11 random generator for different range.
 *
 * @author Cidana-Edmond, Cidana-Wenyao <wenyao.liu@cidana.com>
 *
 ******************************************************************************/

#ifndef _TEST_RANDOM_H_
#define _TEST_RANDOM_H_

#include <cstdint>
#include <cassert>
#include <random>

/** @defgroup svt_av1_test_tool Tool set of test
 *  Defines the tool set of unit test such as random generator, bits shifting
 * and etc...
 *  @{
 */

namespace svt_av1_test_tool {

using std::mt19937;
using std::uniform_int_distribution;
using std::uniform_real_distribution;
using seed_type = mt19937::result_type;

/** SVTRandom defines a tool class for generating random integer as unit test
 * samples and params, the tool can support a random 32-bit integer from
 * [-2^31,2^31).
 */
class SVTRandom {
  public:
    /** contructor with given minimum and maximum bound of random integer*/
    SVTRandom(const int min_bound, const int max_bound) {
        setup(min_bound, max_bound);
        setup(static_cast<float>(min_bound), static_cast<float>(max_bound));
    }

    /** contructor with given limit bits and signed symbol*/
    SVTRandom(const int nbits, const bool is_signed) {
        calculate_bounds(nbits, is_signed);
    }

    /** contructor with given minimum and maximum bound of random real*/
    SVTRandom(const float min_bound, const float max_bound) {
        setup(min_bound, max_bound);
        setup(static_cast<int>(min_bound), static_cast<int>(max_bound));
    }

    /** contructor with given minimum, maximum bound of random integer and seed
     */
    explicit SVTRandom(const int min_bound, const int max_bound,
                       const seed_type seed)
        : gen_(seed) {
        setup(min_bound, max_bound);
        setup(static_cast<float>(min_bound), static_cast<float>(max_bound));
    }

    /** contructor with given limit bits, signed symbol and seed */
    explicit SVTRandom(const int nbits, const bool is_signed,
                       const seed_type seed)
        : gen_(seed) {
        calculate_bounds(nbits, is_signed);
    }

    /** reset generator with new seed
     * @param seed new seed for generator reset
     */
    void reset(seed_type seed) {
        gen_.seed(seed);
    }

    /** reset generator with default seed
     */
    void reset() {
        gen_.seed(deterministic_seed_);
    }

    /** generate a new random integer with minimum and maximum bounds
     * @return:
     * value of random integer
     */
    int random() {
        return dist_nbit_(gen_);
    }

    float random_float() {
        return dist_real_(gen_);
    }

    uint8_t Rand8(void) {
        return static_cast<uint8_t>(random());
    }

    uint16_t Rand16(void) {
        return static_cast<uint16_t>(random());
    }

  private:
    /** setup bounds of generator */
    void setup(const int min_bound, const int max_bound) {
        assert(min_bound <= max_bound);
        decltype(dist_nbit_)::param_type param{min_bound, max_bound};
        dist_nbit_.param(param);
    }

    void setup(const float min_bound, const float max_bound) {
        assert(min_bound <= max_bound);
        decltype(dist_real_)::param_type param{min_bound, max_bound};
        dist_real_.param(param);
    }

    /** calculate and setup bounds of generator */
    void calculate_bounds(const int nbits, const bool is_signed) {
        assert(nbits <= 32);
        const int set_bits = nbits - (is_signed || nbits == 32);
        const int max_bound = (1 << set_bits) - 1;
        const int min_bound = is_signed ? 0 - (1 << (nbits - 1)) : 0;
        setup(min_bound, max_bound);
        setup(static_cast<float>(min_bound), static_cast<float>(max_bound));
    }

  private:
    /**< seed of random generator */
    static constexpr seed_type deterministic_seed_{13596};
    /**< random integer generator */
    std::mt19937 gen_{deterministic_seed_};
    /**< rule of integer generator */
    uniform_int_distribution<int> dist_nbit_;
    /**< rule of real generator */
    uniform_real_distribution<float> dist_real_;
};

}  // namespace svt_av1_test_tool
/** @} */  // end of svt_av1_test_tool

#endif  // _TEST_RANDOM_H_
