// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_BINDING_C_TARGET_CMD_PROGRAM_H_
#define LOOM_BINDING_C_TARGET_CMD_PROGRAM_H_

#include "loom/target/arch/cmd/program.h"
#include "loomc/target/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the generic compiled package retained by |command_program|.
const loomc_program_t* loomc_cmd_program_package(
    const loomc_cmd_program_t* command_program);

// Returns the trusted parsed portable program owned by |command_program|.
const loom_cmd_program_t* loomc_cmd_program_parsed(
    const loomc_cmd_program_t* command_program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_BINDING_C_TARGET_CMD_PROGRAM_H_
