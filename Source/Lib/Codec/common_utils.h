/*
 * Copyright(c) 2019 Netflix, Inc.
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#ifndef EbCommonUtils_h
#define EbCommonUtils_h

#include "definitions.h"
#include "block_structures.h"
#include "cabac_context_model.h"

#define MAX_OFFSET_WIDTH 64
#define MAX_OFFSET_HEIGHT 0

extern const int16_t eb_k_eob_group_start[12];
extern const int16_t eb_k_eob_offset_bits[12];

extern const TxType g_intra_mode_to_tx_type[INTRA_MODES];

extern const PredictionMode g_uv2y[16];
extern const PredictionMode fimode_to_intradir[FILTER_INTRA_MODES];

static INLINE uint8_t *set_levels(uint8_t *const levels_buf, const int32_t width) {
    return levels_buf + TX_PAD_TOP * (width + TX_PAD_HOR);
}
static INLINE int get_txb_bwl(TxSize tx_size) {
    tx_size = av1_get_adjusted_tx_size(tx_size);
    return tx_size_wide_log2[tx_size];
}

static INLINE int get_txb_wide(TxSize tx_size) {
    tx_size = av1_get_adjusted_tx_size(tx_size);
    return tx_size_wide[tx_size];
}

static INLINE int get_txb_high(TxSize tx_size) {
    tx_size = av1_get_adjusted_tx_size(tx_size);
    return tx_size_high[tx_size];
}

static INLINE PredictionMode get_uv_mode(UvPredictionMode mode) {
    assert(mode < UV_INTRA_MODES);
    return g_uv2y[mode];
}

static INLINE TxType intra_mode_to_tx_type(PredictionMode pred_mode, UvPredictionMode pred_mode_uv,
                                           PlaneType plane_type) {
    const PredictionMode mode = (plane_type == PLANE_TYPE_Y) ? pred_mode : get_uv_mode(pred_mode_uv);
    assert(mode < INTRA_MODES);
    return g_intra_mode_to_tx_type[mode];
}

static INLINE int32_t is_chroma_reference(int32_t mi_row, int32_t mi_col, BlockSize bsize, int32_t subsampling_x,
                                          int32_t subsampling_y) {
    const int32_t bw      = mi_size_wide[bsize];
    const int32_t bh      = mi_size_high[bsize];
    int32_t       ref_pos = ((mi_row & 0x01) || !(bh & 0x01) || !subsampling_y) &&
        ((mi_col & 0x01) || !(bw & 0x01) || !subsampling_x);
    return ref_pos;
}

static INLINE int get_segdata(SegmentationParams *seg, int segment_id, SEG_LVL_FEATURES feature_id) {
    return seg->feature_data[segment_id][feature_id];
}

static AOM_FORCE_INLINE int get_br_ctx(const uint8_t *const levels,
                                       const int            c, // raster order
                                       const int bwl, const TxClass tx_class) {
    const int row    = c >> bwl;
    const int col    = c - (row << bwl);
    const int stride = (1 << bwl) + TX_PAD_HOR;
    const int pos    = row * stride + col;
    int       mag    = levels[pos + 1];
    mag += levels[pos + stride];
    switch (tx_class) {
    case TX_CLASS_2D:
        mag += levels[pos + stride + 1];
        mag = AOMMIN((mag + 1) >> 1, 6);
        if (c == 0)
            return mag;
        if ((row < 2) && (col < 2))
            return mag + 7;
        break;
    case TX_CLASS_HORIZ:
        mag += levels[pos + 2];
        mag = AOMMIN((mag + 1) >> 1, 6);
        if (c == 0)
            return mag;
        if (col == 0)
            return mag + 7;
        break;
    case TX_CLASS_VERT:
        mag += levels[pos + (stride << 1)];
        mag = AOMMIN((mag + 1) >> 1, 6);
        if (c == 0)
            return mag;
        if (row == 0)
            return mag + 7;
        break;
    default: break;
    }
    return mag + 14;
}
#endif //EbCommonUtils_h
