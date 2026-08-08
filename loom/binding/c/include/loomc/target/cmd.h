// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_H_
#define LOOMC_TARGET_CMD_H_

#include "loomc/compile.h"

/// @file
/// Command-program compiler capability package.

#ifdef __cplusplus
extern "C" {
#endif

/// Creates a program environment containing command-program compilation.
///
/// Install the returned environment through
/// `loomc_compiler_program_options_t` when creating a compiler that will
/// prepare command-program roots. The environment is independent of hardware
/// target environments and may prepare roots whose dependency units target any
/// architecture linked into the compiler.
LOOMC_API_EXPORT loomc_status_t loomc_program_environment_create_command(
    loomc_allocator_t allocator,
    loomc_program_environment_t** out_program_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CMD_H_
