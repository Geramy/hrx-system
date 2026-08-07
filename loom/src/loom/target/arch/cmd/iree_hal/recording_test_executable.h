// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_CMD_IREE_HAL_RECORDING_TEST_EXECUTABLE_H_
#define LOOM_TARGET_ARCH_CMD_IREE_HAL_RECORDING_TEST_EXECUTABLE_H_

#include "iree/hal/local/executable_library.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the static executable used by command-program recording tests.
const iree_hal_executable_library_header_t* const*
loom_cmd_recording_test_executable_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_IREE_HAL_RECORDING_TEST_EXECUTABLE_H_
