/*
* Copyright(c) 2024-2025 Psychovisual Experts Group
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include <math.h>
#include <stdbool.h>
#include "ac_bias.h"
#include "aom_dsp_rtcd.h"

uint64_t svt_psy_distortion(const uint8_t *input, const uint32_t input_stride, const uint8_t *recon,
                            const uint32_t recon_stride, const uint32_t width, const uint32_t height) {
    static uint16_t zero_buffer[8] = {0};
    uint64_t        energy_gap     = 0;

    if (width >= 8 && height >= 8) { /* >8x8 */
        for (uint64_t j = 0; j < height; j += 8) {
            for (uint64_t i = 0; i < width; i += 8) {
                int32_t  coeffs[64];
                uint16_t block_as_16bit[64];
                uint8_t *block_input = (uint8_t *)input + j * input_stride + i;
                uint8_t *recon_input = (uint8_t *)recon + j * recon_stride + i;

                for (int h = 0; h < 8; h++) {
                    for (int w = 0; w < 8; w++) { block_as_16bit[h * 8 + w] = block_input[w]; }

                    block_input += input_stride;
                }

                svt_aom_hadamard_8x8((int16_t *)block_as_16bit, 8, coeffs);

                const int32_t input_energy = ((svt_aom_satd(coeffs, 64) + 2) >> 2) -
                    ((sad_16b_kernel(block_as_16bit, 8, zero_buffer, 0, 8, 8) + 2) >> 2);

                for (int h = 0; h < 8; h++) {
                    for (int w = 0; w < 8; w++) { block_as_16bit[h * 8 + w] = recon_input[w]; }

                    recon_input += recon_stride;
                }

                svt_aom_hadamard_8x8((int16_t *)block_as_16bit, 8, coeffs);

                const int32_t recon_energy = ((svt_aom_satd(coeffs, 64) + 2) >> 2) -
                    ((sad_16b_kernel(block_as_16bit, 8, zero_buffer, 0, 8, 8) + 2) >> 2);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    } else {
        for (uint64_t j = 0; j < height; j += 4) { /* 4x4, 4x8, 4x16, 8x4, and 16x4 */
            for (uint64_t i = 0; i < width; i += 4) {
                int32_t  coeffs[16];
                uint16_t block_as_16bit[16];
                uint8_t *block_input = (uint8_t *)input + j * input_stride + i;
                uint8_t *recon_input = (uint8_t *)recon + j * recon_stride + i;

                for (int h = 0; h < 4; h++) {
                    for (int w = 0; w < 4; w++) { block_as_16bit[h * 4 + w] = block_input[w]; }

                    block_input += input_stride;
                }

                svt_aom_hadamard_4x4((int16_t *)block_as_16bit, 4, coeffs);

                const int32_t input_energy = (svt_aom_satd(coeffs, 16) << 1) -
                    ((sad_16b_kernel(block_as_16bit, 4, zero_buffer, 0, 4, 4) + 2) >> 2);

                for (int h = 0; h < 4; h++) {
                    for (int w = 0; w < 4; w++) { block_as_16bit[h * 4 + w] = recon_input[w]; }

                    block_input += recon_stride;
                }

                svt_aom_hadamard_4x4((int16_t *)block_as_16bit, 4, coeffs);

                const int32_t recon_energy = (svt_aom_satd(coeffs, 16) << 1) -
                    ((sad_16b_kernel(block_as_16bit, 4, zero_buffer, 0, 4, 4) + 2) >> 2);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    }

    return energy_gap;
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
uint64_t svt_psy_distortion_hbd(const uint16_t *input, const uint32_t input_stride, const uint16_t *recon,
                                const uint32_t recon_stride, const uint32_t width, const uint32_t height) {
    static uint16_t zero_buffer[8] = {0};
    uint64_t        energy_gap     = 0;

    if (width >= 8 && height >= 8) { /* >8x8 */
        for (uint64_t j = 0; j < height; j += 8) {
            for (uint64_t i = 0; i < width; i += 8) {
                int32_t coeffs[64];

                svt_aom_highbd_hadamard_8x8((int16_t *)input + j * input_stride + i, input_stride, coeffs);

                const int32_t input_energy = ((svt_aom_satd(coeffs, 64) + 2) >> 2) -
                    ((sad_16b_kernel((uint16_t *)input + j * input_stride + i, input_stride, zero_buffer, 0, 8, 8) +
                      2) >>
                     2);

                svt_aom_highbd_hadamard_8x8((int16_t *)recon + j * recon_stride + i, recon_stride, coeffs);

                const int32_t recon_energy = ((svt_aom_satd(coeffs, 64) + 2) >> 2) -
                    ((sad_16b_kernel((uint16_t *)recon + j * recon_stride + i, recon_stride, zero_buffer, 0, 8, 8) +
                      2) >>
                     2);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    } else {
        for (uint64_t j = 0; j < height; j += 4) { /* 4x4, 4x8, 4x16, 8x4, and 16x4 */
            for (uint64_t i = 0; i < width; i += 4) {
                int32_t coeffs[16];

                // HBD coefficients can fit in 16 bits, so the regular Hadamard 4x4 function can be used here safely
                svt_aom_hadamard_4x4((int16_t *)input + j * input_stride + i, input_stride, coeffs);

                const int32_t input_energy = (svt_aom_satd(coeffs, 16) << 1) -
                    ((sad_16b_kernel((uint16_t *)input + j * input_stride + i, input_stride, zero_buffer, 0, 4, 4) +
                      2) >>
                     2);

                svt_aom_hadamard_4x4((int16_t *)recon + j * recon_stride + i, recon_stride, coeffs);

                const int32_t recon_energy = (svt_aom_satd(coeffs, 16) << 1) -
                    ((sad_16b_kernel((uint16_t *)recon + j * recon_stride + i, recon_stride, zero_buffer, 0, 4, 4) +
                      2) >>
                     2);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    }

    // Energy is scaled to approximately match equivalent 8-bit strengths
    return (energy_gap << 2);
}
#endif

/*
 * Public function that mirrors the arguments of `spatial_full_dist_type_fun()`
 */
uint64_t get_svt_psy_full_dist(const void *s, const uint32_t so, const uint32_t sp, const void *r, const uint32_t ro,
                               const uint32_t rp, const uint32_t w, const uint32_t h, const uint8_t is_hbd,
                               const double ac_bias) {
    if (is_hbd)
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
        return llrint(svt_psy_distortion_hbd((const uint16_t *)s + so, sp, (uint16_t *)r + ro, rp, w, h) * ac_bias);
#else
        return 0;
#endif
    else
        return llrint(svt_psy_distortion((const uint8_t *)s + so, sp, (const uint8_t *)r + ro, rp, w, h) * ac_bias);
}

double get_effective_ac_bias(const double ac_bias, const bool is_islice, const uint8_t temporal_layer_index) {
    if (is_islice)
        return ac_bias * 0.3;
    switch (temporal_layer_index) {
    case 0: return ac_bias * 0.6;
    case 1: return ac_bias * 0.8;
    case 2: return ac_bias * 0.9;
    default: return ac_bias;
    }
}
