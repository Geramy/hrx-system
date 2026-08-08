// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PROGRAM_STORAGE_H_
#define LOOMC_PROGRAM_STORAGE_H_

#include "artifact_storage.h"
#include "loomc/program.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Releases target-owned program storage.
typedef void (*loomc_program_target_storage_release_fn_t)(
    void* storage, loomc_allocator_t allocator);

// One retained ancillary program supplied during program construction.
typedef struct loomc_program_dependency_t {
  // Dense target-defined slot populated by this dependency.
  uint32_t slot;

  // Program retained on successful construction.
  loomc_program_t* program;
} loomc_program_dependency_t;

// Trusted compiler-owned inputs for immutable program construction.
typedef struct loomc_program_create_params_t {
  // Public export names cloned into the program in caller order.
  const loomc_string_view_t* export_names;

  // Number of entries in |export_names|.
  loomc_host_size_t export_count;

  // Shared direct artifact storage retained by the program.
  loomc_artifact_storage_t* artifact_storage;

  // Ancillary program slots retained by the program.
  const loomc_program_dependency_t* dependencies;

  // Number of entries in |dependencies|.
  loomc_host_size_t dependency_count;

  // Optional target-owned immutable program representation.
  void* target_storage;

  // Callback that releases |target_storage| when the program is destroyed.
  loomc_program_target_storage_release_fn_t target_storage_release;
} loomc_program_create_params_t;

// Creates an immutable program from trusted compiler-owned inputs.
//
// Artifact storage and dependencies are retained, export names are cloned, and
// target storage ownership transfers only on success.
LOOMC_API_PRIVATE loomc_status_t loomc_program_create(
    const loomc_program_create_params_t* params, loomc_allocator_t allocator,
    loomc_program_t** out_program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PROGRAM_STORAGE_H_
