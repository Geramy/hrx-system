// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PROGRAM_ENVIRONMENT_H_
#define LOOMC_PROGRAM_ENVIRONMENT_H_

#include "loomc/program_plan.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// One immutable provider of a linked program-root representation.
typedef struct loomc_program_provider_t {
  // Returns true when this provider owns at least one selected module root.
  bool (*matches)(const loomc_module_t* module);

  // Prepares one exact program plan from the selected module roots.
  loomc_status_t (*prepare)(loomc_compiler_t* compiler,
                            loomc_workspace_t* workspace,
                            const loomc_pass_program_t* unit_pass_program,
                            loomc_module_t* module,
                            const loomc_program_plan_options_t* options,
                            loomc_result_t* result, loomc_allocator_t allocator,
                            loomc_program_plan_t** out_program_plan);
} loomc_program_provider_t;

// Creates an immutable program environment from static provider descriptors.
//
// The provider pointer table is cloned. Provider descriptors have process
// lifetime and remain borrowed by the returned environment.
LOOMC_API_PRIVATE loomc_status_t loomc_program_environment_create(
    const loomc_program_provider_t* const* providers,
    loomc_host_size_t provider_count, loomc_allocator_t allocator,
    loomc_program_environment_t** out_program_environment);

// Resolves the one provider that owns roots in |module|.
LOOMC_API_PRIVATE loomc_status_t loomc_program_environment_select_provider(
    const loomc_program_environment_t* program_environment,
    const loomc_module_t* module,
    const loomc_program_provider_t** out_provider);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PROGRAM_ENVIRONMENT_H_
