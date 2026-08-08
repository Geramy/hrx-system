// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PROGRAM_PLAN_STORAGE_H_
#define LOOMC_PROGRAM_PLAN_STORAGE_H_

#include "loomc/program_plan.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// One root-local dependency supplied during plan construction.
typedef struct loomc_program_plan_dependency_t {
  // Dense target-defined dependency slot.
  uint32_t slot;

  // Plan-local unit that populates |slot|.
  uint32_t unit_index;
} loomc_program_plan_dependency_t;

// One selected root supplied during plan construction.
typedef struct loomc_program_plan_root_create_params_t {
  // Public root name cloned into the plan.
  loomc_string_view_t name;

  // Plan-local unit containing this root.
  uint32_t program_unit_index;

  // Root-local dependency slots cloned into the plan.
  const loomc_program_plan_dependency_t* dependencies;

  // Number of entries in |dependencies|.
  loomc_host_size_t dependency_count;
} loomc_program_plan_root_create_params_t;

// One independently producible unit supplied during plan construction.
typedef struct loomc_program_plan_unit_create_params_t {
  // Stable diagnostic identifier cloned into the plan.
  loomc_string_view_t identifier;

  // Exact compiled-program cache key cloned into the plan.
  loomc_byte_span_t cache_key;
} loomc_program_plan_unit_create_params_t;

// Target-owned immutable plan operation table.
typedef struct loomc_program_plan_operations_t {
  // Compiles one plan unit.
  loomc_status_t (*compile_unit)(
      const void* storage, loomc_workspace_t* workspace, uint32_t unit_index,
      const loomc_program_plan_unit_compile_options_t* options,
      loomc_allocator_t allocator, loomc_program_t** out_program,
      loomc_result_t** out_result);

  // Loads one plan unit from cached artifacts.
  loomc_status_t (*load_unit_program)(const void* storage, uint32_t unit_index,
                                      const loomc_artifact_t* artifacts,
                                      loomc_host_size_t artifact_count,
                                      loomc_allocator_t allocator,
                                      loomc_program_t** out_program,
                                      loomc_result_t** out_result);

  // Assembles selected plan roots from a dense unit table.
  loomc_status_t (*assemble)(
      const void* storage, loomc_workspace_t* workspace,
      const loomc_program_plan_root_t* roots, loomc_host_size_t root_count,
      const loomc_program_plan_unit_table_t* unit_table,
      const loomc_program_plan_assembly_options_t* options,
      loomc_allocator_t allocator, loomc_program_t** out_program,
      loomc_result_t** out_result);

  // Releases target-owned immutable plan storage.
  void (*destroy)(void* storage, loomc_allocator_t allocator);
} loomc_program_plan_operations_t;

// Trusted compiler-owned inputs for immutable plan construction.
typedef struct loomc_program_plan_create_params_t {
  // Selected roots cloned into the plan in caller order.
  const loomc_program_plan_root_create_params_t* roots;

  // Number of entries in |roots|.
  loomc_host_size_t root_count;

  // Independently producible units cloned into the plan in token order.
  const loomc_program_plan_unit_create_params_t* units;

  // Number of entries in |units|.
  loomc_host_size_t unit_count;

  // Static target-owned operation table.
  const loomc_program_plan_operations_t* operations;

  // Target-owned immutable plan representation.
  void* target_storage;
} loomc_program_plan_create_params_t;

// Creates an immutable exact program plan from trusted compiler-owned inputs.
//
// Metadata is cloned. Target storage ownership transfers only on success and
// is released through |operations->destroy| with the plan allocator.
LOOMC_API_PRIVATE loomc_status_t loomc_program_plan_create(
    const loomc_program_plan_create_params_t* params,
    loomc_allocator_t allocator, loomc_program_plan_t** out_program_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PROGRAM_PLAN_STORAGE_H_
