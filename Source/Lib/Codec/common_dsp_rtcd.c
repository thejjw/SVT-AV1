// clang-format off
/*
* Copyright(c) 2019 Intel Corporation
* Copyright (c) 2016, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#if HAVE_VALGRIND_H
#include <valgrind/valgrind.h>
#else
// assume the system doesn't have access to valgrind if the header is missing
#define RUNNING_ON_VALGRIND 0
#endif

#define RTCD_C
#include "common_dsp_rtcd.h"
#include "pic_operators.h"
#include "pack_unpack_c.h"
#include "utility.h"

#if defined ARCH_X86_64
#define AOM_ARCH_X86_64 ARCH_X86_64
#include "third_party/aom/aom_ports/x86.h"
#endif

#if defined ARCH_AARCH64

#if defined(__linux__) || HAVE_ELF_AUX_INFO
// For reading the HWCAP flags
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <stdbool.h>
#include <sys/sysctl.h>
#elif defined(_MSC_VER)
#include <windows.h>
#endif
#endif // ARCH_AARCH64

// coeff: 16 bits, dynamic range [-32640, 32640].
// length: value range {16, 64, 256, 1024}.
int svt_aom_satd_c(const TranLow *coeff, int length) {
    int i;
    int satd = 0;
    for (i = 0; i < length; ++i) satd += abs(coeff[i]);

    // satd: 26 bits, dynamic range [-32640 * 1024, 32640 * 1024]
    return satd;
}

int64_t svt_av1_block_error_c(const TranLow *coeff, const TranLow *dqcoeff, intptr_t block_size, int64_t *ssz) {
    int     i;
    int64_t error = 0, sqcoeff = 0;

    for (i = 0; i < block_size; i++) {
        error += SQR(coeff[i] - dqcoeff[i]);
        sqcoeff += SQR(coeff[i]);
    }

    *ssz = sqcoeff;
    return error;
}

/**************************************
 * Instruction Set Support
 **************************************/
#ifdef ARCH_X86_64
EbCpuFlags svt_aom_get_cpu_flags() {
    EbCpuFlags flags     = 0;
    const int  aom_flags = x86_simd_caps();

    flags |= (aom_flags & HAS_MMX) ? EB_CPU_FLAGS_MMX : 0;
    flags |= (aom_flags & HAS_SSE) ? EB_CPU_FLAGS_SSE : 0;
    flags |= (aom_flags & HAS_SSE2) ? EB_CPU_FLAGS_SSE2 : 0;
    flags |= (aom_flags & HAS_SSE3) ? EB_CPU_FLAGS_SSE3 : 0;
    flags |= (aom_flags & HAS_SSSE3) ? EB_CPU_FLAGS_SSSE3 : 0;
    flags |= (aom_flags & HAS_SSE4_1) ? EB_CPU_FLAGS_SSE4_1 : 0;
    flags |= (aom_flags & HAS_SSE4_2) ? EB_CPU_FLAGS_SSE4_2 : 0;
    flags |= (aom_flags & HAS_AVX) ? EB_CPU_FLAGS_AVX : 0;
    flags |= (aom_flags & HAS_AVX2) ? EB_CPU_FLAGS_AVX2 : 0;
    // aom checks for {f,dq,cd,bw,vl} and also {vbmi,vbmi2,gfni,vaes,vpclmulqdq,vnni,bitalg,popcntdq}
    // for avx512 availability. We have the two sections separated since we still need to be able to
    // test avx512 on skylake, which has only the first set of features.
    const EbCpuFlags avx512_flags = EB_CPU_FLAGS_AVX512F | EB_CPU_FLAGS_AVX512CD | EB_CPU_FLAGS_AVX512DQ |
        EB_CPU_FLAGS_AVX512BW | EB_CPU_FLAGS_AVX512VL;
    flags |= (aom_flags & HAS_AVX512) ? avx512_flags : 0;
    flags |= (aom_flags & HAS_AVX512_DL) ? EB_CPU_FLAGS_AVX512ICL : 0;

    return flags;
}

EbCpuFlags svt_aom_get_cpu_flags_to_use() {
    EbCpuFlags flags = svt_aom_get_cpu_flags();
#if !EN_AVX512_SUPPORT
    /* Remove AVX512 flags. */
    flags &= (EB_CPU_FLAGS_AVX512F - 1);
#endif
    return flags;
}
#else
#if defined ARCH_AARCH64

#if defined(__linux__) || HAVE_ELF_AUX_INFO

// Define hwcap values ourselves: building with an old auxv header where these
// hwcap values are not defined should not prevent features from being enabled.
#define AOM_AARCH64_HWCAP_NEON (1 << 1)
#define AOM_AARCH64_HWCAP_CRC32 (1 << 7)
#define AOM_AARCH64_HWCAP_ASIMDDP (1 << 20)
#define AOM_AARCH64_HWCAP_SVE (1 << 22)
#define AOM_AARCH64_HWCAP2_SVE2 (1 << 1)
#define AOM_AARCH64_HWCAP2_I8MM (1 << 13)

EbCpuFlags svt_aom_get_cpu_flags(void) {
#if HAVE_ARM_CRC32 || HAVE_NEON_DOTPROD || HAVE_SVE
#if HAVE_ELF_AUX_INFO
    unsigned long hwcap = 0;
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
#else
    unsigned long hwcap = getauxval(AT_HWCAP);
#endif
#endif
#if HAVE_NEON_I8MM || HAVE_SVE2
#if HAVE_ELF_AUX_INFO
    unsigned long hwcap2 = 0;
    elf_aux_info(AT_HWCAP2, &hwcap2, sizeof(hwcap2));
#else
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
#endif
#endif

#if CONFIG_ARM_NEON_IS_GUARANTEED
    EbCpuFlags flags = EB_CPU_FLAGS_NEON; // Neon is mandatory in Armv8.0-A.
#else
    EbCpuFlags flags = 0;
    if (hwcap & AOM_AARCH64_HWCAP_NEON)
        flags |= EB_CPU_FLAGS_NEON;
#endif

#if HAVE_ARM_CRC32
    if (hwcap & AOM_AARCH64_HWCAP_CRC32)
        flags |= EB_CPU_FLAGS_ARM_CRC32;
#endif // HAVE_ARM_CRC32
#if HAVE_NEON_DOTPROD
    if (hwcap & AOM_AARCH64_HWCAP_ASIMDDP)
        flags |= EB_CPU_FLAGS_NEON_DOTPROD;
#endif // HAVE_NEON_DOTPROD
#if HAVE_NEON_I8MM
    if (hwcap2 & AOM_AARCH64_HWCAP2_I8MM)
        flags |= EB_CPU_FLAGS_NEON_I8MM;
#endif // HAVE_NEON_I8MM
#if HAVE_SVE
    if (hwcap & AOM_AARCH64_HWCAP_SVE)
        flags |= EB_CPU_FLAGS_SVE;
#endif // HAVE_SVE
#if HAVE_SVE2
    if (hwcap2 & AOM_AARCH64_HWCAP2_SVE2)
        flags |= EB_CPU_FLAGS_SVE2;
#endif // HAVE_SVE2
    return flags;
}

#elif defined(__APPLE__) // end __linux__

// sysctlbyname() parameter documentation for instruction set characteristics:
// https://developer.apple.com/documentation/kernel/1387446-sysctlbyname/determining_instruction_set_characteristics
#if HAVE_ARM_CRC32 || HAVE_NEON_DOTPROD || HAVE_NEON_I8MM
static INLINE bool have_feature(const char *feature) {
    int64_t feature_present = 0;
    size_t  size            = sizeof(feature_present);
    if (sysctlbyname(feature, &feature_present, &size, NULL, 0) != 0) {
        return false;
    }
    return feature_present;
}
#endif

EbCpuFlags svt_aom_get_cpu_flags(void) {
#if CONFIG_ARM_NEON_IS_GUARANTEED
    EbCpuFlags flags = EB_CPU_FLAGS_NEON;
#else
    EbCpuFlags flags = 0;
    if (have_feature("hw.optional.neon"))
        flags |= EB_CPU_FLAGS_NEON;
#endif
#if HAVE_ARM_CRC32
    if (have_feature("hw.optional.armv8_crc32"))
        flags |= EB_CPU_FLAGS_ARM_CRC32;
#endif // HAVE_ARM_CRC32
#if HAVE_NEON_DOTPROD
    if (have_feature("hw.optional.arm.FEAT_DotProd"))
        flags |= EB_CPU_FLAGS_NEON_DOTPROD;
#endif // HAVE_NEON_DOTPROD
#if HAVE_NEON_I8MM
    if (have_feature("hw.optional.arm.FEAT_I8MM"))
        flags |= EB_CPU_FLAGS_NEON_I8MM;
#endif // HAVE_NEON_I8MM
    return flags;
}

#elif defined(_MSC_VER) // end __APPLE__

// IsProcessorFeaturePresent() parameter documentation:
// https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent#parameters
EbCpuFlags svt_aom_get_cpu_flags(void) {
#if CONFIG_ARM_NEON_IS_GUARANTEED
    EbCpuFlags flags = EB_CPU_FLAGS_NEON; // Neon is mandatory in Armv8.0-A.
#else
    EbCpuFlags flags = 0;
    if (IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE)) {
        flags |= EB_CPU_FLAGS_NEON;
#endif
#if HAVE_ARM_CRC32
    if (IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE)) {
        flags |= EB_CPU_FLAGS_ARM_CRC32;
    }
#endif // HAVE_ARM_CRC32
#if HAVE_NEON_DOTPROD
// Support for PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE was added in Windows SDK
// 20348, supported by Windows 11 and Windows Server 2022.
#if defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
    if (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)) {
        flags |= EB_CPU_FLAGS_NEON_DOTPROD;
    }
#endif // defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
#endif // HAVE_NEON_DOTPROD
    // No I8MM or SVE feature detection available on Windows at time of writing.
    return flags;
}

#else // end _MSC_VER

EbCpuFlags svt_aom_get_cpu_flags() {
    EbCpuFlags flags = 0;

    // safe to call multiple times, and threadsafe

#if CONFIG_ARM_NEON_IS_GUARANTEED
    flags |= EB_CPU_FLAGS_NEON;
#endif

    return flags;
}

#endif

EbCpuFlags svt_aom_get_cpu_flags_to_use() {
    EbCpuFlags flags = svt_aom_get_cpu_flags();

    // Restrict flags: FEAT_I8MM assumes that FEAT_DotProd is available.
    if (!(flags & EB_CPU_FLAGS_NEON_DOTPROD))
        flags &= ~EB_CPU_FLAGS_NEON_I8MM;

    // Restrict flags: SVE assumes that FEAT_{DotProd,I8MM} are available.
    if (!(flags & EB_CPU_FLAGS_NEON_DOTPROD))
        flags &= ~EB_CPU_FLAGS_SVE;
    if (!(flags & EB_CPU_FLAGS_NEON_I8MM))
        flags &= ~EB_CPU_FLAGS_SVE;

    // Restrict flags: SVE2 assumes that FEAT_SVE is available.
    if (!(flags & EB_CPU_FLAGS_SVE))
        flags &= ~EB_CPU_FLAGS_SVE2;

    return flags;
}

#else
EbCpuFlags svt_aom_get_cpu_flags_to_use() { return 0; }
#endif
#endif

#include "common_dsp_rtcd_template.h"

// clang-format on
