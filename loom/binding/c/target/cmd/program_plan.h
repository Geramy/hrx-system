// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-target adapter for root-selected Loomc program plans.

#ifndef LOOM_BINDING_C_TARGET_CMD_PROGRAM_PLAN_H_
#define LOOM_BINDING_C_TARGET_CMD_PROGRAM_PLAN_H_

#include "loomc/compile.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/program_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepares every public command root in an already-linked module.
//
// This is the bounded command-target adapter used while the generic target
// provider preparation route is established. |module| must already carry exact
// target selection and the complete public root set. The returned plan retains
// all source-independent unit modules, |compiler|, and |unit_pass_program|.
//
// An OK return always owns |out_result|. A succeeded result also owns
// |out_program_plan|. Compiler-domain failures are reported through the result.
loomc_status_t loomc_cmd_program_plan_prepare_module(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* unit_pass_program, loomc_module_t* module,
    loomc_allocator_t allocator, loomc_program_plan_t** out_program_plan,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_BINDING_C_TARGET_CMD_PROGRAM_PLAN_H_
