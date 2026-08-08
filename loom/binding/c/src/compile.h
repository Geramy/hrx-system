// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_COMPILE_STORAGE_H_
#define LOOMC_COMPILE_STORAGE_H_

#include "loom/pass/interpreter.h"
#include "loomc/compile.h"
#include "loomc/program_plan.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the program environment retained by |compiler|, or NULL.
LOOMC_API_PRIVATE loomc_program_environment_t*
loomc_compiler_program_environment(const loomc_compiler_t* compiler);

// Compiles |module| while restricting pass.for<func> traversal with
// |function_selector|. Module-root passes retain whole-module semantics.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_module_select_functions(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* pass_program, loomc_module_t* module,
    loom_pass_function_selector_t function_selector,
    const loomc_compile_options_t* options, loomc_allocator_t allocator,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_COMPILE_STORAGE_H_
