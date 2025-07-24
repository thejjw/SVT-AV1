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

#include <stdlib.h>
#include <stdint.h>

#include "pcs.h"
#include "utility.h"
#include "enc_mode_config.h"

void svt_aom_largest_coding_unit_dctor(EbPtr p) {
    SuperBlock *obj = (SuperBlock *)p;
    EB_FREE_ARRAY(obj->av1xd);
    EB_FREE_ARRAY(obj->final_blk_arr);
    EB_FREE_ARRAY(obj->cu_partition_array);
}
/*
Tasks & Questions
    -Need a GetEmptyChain function for testing sub partitions.  Tie it to an Itr?
    -How many empty CUs do we need?  We need to have enough for the max CU count AND
       enough for temp storage when calculating a subdivision.
    -Where do we store temp reconstructed picture data while deciding to subdivide or not?
    -Need a ReconPicture for each candidate.
    -I don't see a way around doing the copies in temp memory and then copying it in...
*/
EbErrorType svt_aom_largest_coding_unit_ctor(SuperBlock *larget_coding_unit_ptr, uint8_t sb_size_pix,
                                             uint16_t sb_origin_x, uint16_t sb_origin_y, uint16_t sb_index,
                                             EncMode enc_mode, bool rtc, uint32_t screen_content_mode,
                                             uint16_t max_block_cnt,

                                             PictureControlSet *picture_control_set)

{
    larget_coding_unit_ptr->dctor = svt_aom_largest_coding_unit_dctor;

    // ************ SB ***************
    // Which borderLargestCuSize is not a power of two

    // Which borderLargestCuSize is not a power of two
    larget_coding_unit_ptr->pcs = picture_control_set;

    larget_coding_unit_ptr->org_x = sb_origin_x;
    larget_coding_unit_ptr->org_y = sb_origin_y;

    larget_coding_unit_ptr->index = sb_index;
    bool disallow_8x4_4x8         = true;
    for (uint8_t is_base = 0; is_base <= 1; is_base++) {
        for (uint8_t is_islice = 0; is_islice <= 1; is_islice++) {
            for (uint8_t coeff_lvl = 0; coeff_lvl <= HIGH_LVL + 1; coeff_lvl++) {
                if (!disallow_8x4_4x8)
                    break;
                const uint8_t nsq_geom_lvl = svt_aom_get_nsq_geom_level(enc_mode, is_base, coeff_lvl, rtc);
                //disallow_4x4 = MIN(disallow_4x4, (nsq_geom_lvl == 0 ? 1 : 0));
                uint8_t allow_HVA_HVB, allow_HV4, min_nsq_bsize;
                svt_aom_set_nsq_geom_ctrls(NULL, nsq_geom_lvl, &allow_HVA_HVB, &allow_HV4, &min_nsq_bsize);
                if (min_nsq_bsize < 8 /*|| (min_nsq_bsize < 16 && allow_HV4)*/)
                    disallow_8x4_4x8 = false;
            }
        }
    }
    bool disallow_16x8_8x16 = true;
    for (uint8_t is_base = 0; is_base <= 1; is_base++) {
        for (uint8_t is_islice = 0; is_islice <= 1; is_islice++) {
            for (uint8_t coeff_lvl = 0; coeff_lvl <= HIGH_LVL + 1; coeff_lvl++) {
                if (!disallow_16x8_8x16)
                    break;
                const uint8_t nsq_geom_lvl = svt_aom_get_nsq_geom_level(enc_mode, is_base, coeff_lvl, rtc);
                //disallow_4x4 = MIN(disallow_4x4, (nsq_geom_lvl == 0 ? 1 : 0));
                uint8_t allow_HVA_HVB, allow_HV4, min_nsq_bsize;
                svt_aom_set_nsq_geom_ctrls(NULL, nsq_geom_lvl, &allow_HVA_HVB, &allow_HV4, &min_nsq_bsize);
                if (min_nsq_bsize < 16 /*|| (min_nsq_bsize < 16 && allow_HV4)*/)
                    disallow_16x8_8x16 = false;
            }
        }
    }
    bool disallow_4x4 = true;
    for (uint8_t is_islice = 0; is_islice <= 1; is_islice++) {
        for (uint8_t is_base = 0; is_base <= 1; is_base++) {
            disallow_4x4 = MIN(disallow_4x4, svt_aom_get_disallow_4x4(enc_mode, is_base));
        }
    }
    bool     disallow_8x8 = svt_aom_get_disallow_8x8(enc_mode, rtc, screen_content_mode);
    uint32_t tot_blk_num;
    if (sb_size_pix == 128)
        if (disallow_8x8 && disallow_16x8_8x16)
            tot_blk_num = 64;
        else if (disallow_8x8 || (disallow_4x4 && disallow_8x4_4x8))
            tot_blk_num = 256;
        else if (disallow_4x4)
            tot_blk_num = 512;
        else
            tot_blk_num = 1024;
    else if (disallow_8x8 && disallow_16x8_8x16)
        tot_blk_num = 16;
    else if (disallow_8x8 || (disallow_4x4 && disallow_8x4_4x8))
        tot_blk_num = 64;
    else if (disallow_4x4)
        tot_blk_num = 128;
    else
        tot_blk_num = 256;
    EB_MALLOC_ARRAY(larget_coding_unit_ptr->final_blk_arr, tot_blk_num);
    EB_MALLOC_ARRAY(larget_coding_unit_ptr->av1xd, 1);
    // Do NOT initialize the final_blk_arr here
    // Malloc maximum but only initialize it only when actually used.
    // This will help to same actually memory usage
    EB_MALLOC_ARRAY(larget_coding_unit_ptr->cu_partition_array, max_block_cnt);
    return EB_ErrorNone;
}
