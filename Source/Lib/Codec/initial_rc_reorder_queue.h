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

#ifndef EbInitialRateControlReorderQueue_h
#define EbInitialRateControlReorderQueue_h

#include "definitions.h"
#if !CLN_REMOVE_IRC_Q // TODO: remove this file
#include "sys_resource_manager.h"
#include "pcs.h"
#include "object.h"
/************************************************
 * Initial Rate Control Reorder Queue Entry
 ************************************************/
typedef struct InitialRateControlReorderEntry {
    EbDctor          dctor;
    uint64_t         picture_number;
    EbObjectWrapper *ppcs_wrapper;
} InitialRateControlReorderEntry;

extern EbErrorType svt_aom_initial_rate_control_reorder_entry_ctor(InitialRateControlReorderEntry *entry_ptr,
                                                                   uint32_t                        picture_number);
#endif

#endif //EbInitialRateControlReorderQueue_h
