/*
* Copyright(c) 2019 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#ifndef EbFullLoop_h
#define EbFullLoop_h

#include "md_process.h"
#include "common_utils.h"
#include "inv_transforms.h"
#include "transforms.h"
#include "definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

void svt_aom_full_loop_chroma_light_pd1(PictureControlSet *pcs, ModeDecisionContext *ctx,
                                        ModeDecisionCandidateBuffer *cand_bf, EbPictureBufferDesc *input_pic,
                                        uint32_t input_cb_origin_in_index, uint32_t blk_chroma_origin_index,
                                        COMPONENT_TYPE component_type, uint32_t chroma_qindex,
                                        uint64_t cb_full_distortion[DIST_CALC_TOTAL],
                                        uint64_t cr_full_distortion[DIST_CALC_TOTAL], uint64_t *cb_coeff_bits,
                                        uint64_t *cr_coeff_bits);
void svt_aom_full_loop_uv(PictureControlSet *pcs, ModeDecisionContext *ctx, ModeDecisionCandidateBuffer *cand_bf,
                          EbPictureBufferDesc *input_pic, COMPONENT_TYPE component_type, uint32_t chroma_qindex,
                          uint64_t cb_full_distortion[DIST_TOTAL][DIST_CALC_TOTAL],
                          uint64_t cr_full_distortion[DIST_TOTAL][DIST_CALC_TOTAL], uint64_t *cb_coeff_bits,
                          uint64_t *cr_coeff_bits, bool is_full_loop);
void svt_aom_inv_transform_recon_wrapper(PictureControlSet *pcs, ModeDecisionContext *ctx, uint8_t *pred_buffer,
                                         uint32_t pred_offset, uint32_t pred_stride, uint8_t *rec_buffer,
                                         uint32_t rec_offset, uint32_t rec_stride, int32_t *rec_coeff_buffer,
                                         uint32_t coeff_offset, bool hbd, TxSize txsize, TxType transform_type,
                                         PlaneType component_type, uint32_t eob);

uint32_t svt_aom_d2_inter_depth_block_decision(PictureControlSet *pcs, ModeDecisionContext *ctx, uint32_t blk_mds);
// compute the cost of curr depth, and the depth above
extern void svt_aom_compute_depth_costs_md_skip(ModeDecisionContext *ctx, PictureParentControlSet *pcs,
                                                uint32_t above_depth_mds, uint32_t step, uint64_t *above_depth_cost,
                                                uint64_t *curr_depth_cost);
void        svt_aom_compute_depth_costs_md_skip_light_pd0(PictureParentControlSet *pcs, ModeDecisionContext *ctx,
                                                          uint32_t above_depth_mds, uint32_t step, uint64_t *above_depth_cost,
                                                          uint64_t *curr_depth_cost);
uint64_t svt_aom_d1_non_square_block_decision(PictureControlSet *pcs, ModeDecisionContext *ctx, uint32_t d1_block_itr);

extern const int av1_get_tx_scale_tab[TX_SIZES_ALL];

extern const TxSize get_txsize_entropy_ctx_tab[TX_SIZES_ALL];

extern const int get_txb_bwl_tab[TX_SIZES_ALL];

extern const int get_txb_wide_tab[TX_SIZES_ALL];

extern const int get_txb_high_tab[TX_SIZES_ALL];
#ifdef __cplusplus
}
#endif
#endif // EbFullLoop_h
