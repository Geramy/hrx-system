// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_COMMAND_PACKAGE_H_
#define EXPERIMENTAL_QWEN_RUNTIME_COMMAND_PACKAGE_H_

#include "experimental/qwen/runtime/request.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loomc/sanitizer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reusable materialization of the Qwen3-30B command-program roots.
//
// The package owns one compiled multi-root product, its shared loaded device
// executables, and one reusable command buffer per program. It retains the
// model whose immutable parameter ranges are baked into those recordings.
// Distinct programs may issue concurrently; callers externally synchronize
// issue operations selecting the same program.
typedef struct qwen_command_package_t qwen_command_package_t;

// Program identities available in a prepared package.
typedef enum qwen_command_program_e {
  // Exact 512-row initial prefill with a 512-row attention extent.
  QWEN_COMMAND_PROGRAM_PREFILL_512 = 0,
  // One-token decode over the 576-row context class.
  QWEN_COMMAND_PROGRAM_DECODE_576 = 1,
  // Number of known command programs.
  QWEN_COMMAND_PROGRAM_COUNT = 2,
} qwen_command_program_t;

// Options controlling package compilation and command recording.
typedef struct qwen_command_package_options_t {
  // Size of this structure in bytes.
  iree_host_size_t structure_size;
  // Optional extension chain; must be NULL.
  const void* next;
  // Request token-storage capacity whose field layout is specialized.
  iree_host_size_t request_token_capacity;
  // Request K/V storage capacity shared by every materialized program.
  iree_host_size_t context_capacity;
  // Optional request storage behavior; currently must be none.
  qwen_request_flags_t request_flags;
  // Report-only Loom access sanitizer assertions compiled into every kernel.
  loomc_sanitizer_checks_t sanitizer_checks;
  // Number of task workers compiling independent program units.
  iree_host_size_t compiler_worker_count;
  // HAL command-buffer mode used for every reusable recording.
  iree_hal_command_buffer_mode_t command_buffer_mode;
} qwen_command_package_options_t;

// Immutable metadata for one materialized command program.
typedef struct qwen_command_program_info_t {
  // Size of this structure in bytes.
  iree_host_size_t structure_size;
  // Reserved extension chain; must be NULL.
  void* next;
  // Exact number of active token rows consumed by the program.
  iree_host_size_t token_count;
  // Attention context extent compiled into the program.
  iree_host_size_t context_count;
  // Packed issue-time transient requirement in bytes.
  iree_device_size_t transient_byte_length;
  // Required base alignment of the transient allocation.
  iree_device_size_t transient_minimum_alignment;
} qwen_command_program_info_t;

// Initializes |out_options| for Prefill-512 followed by Decode-576.
IREE_API_EXPORT void qwen_command_package_options_initialize(
    qwen_command_package_options_t* out_options);

// Compiles and materializes all command programs in one shared package.
//
// Preparation is synchronous host work and may overlap the model's
// asynchronous parameter gather. Every compiled parameter range is matched
// against the model slab before immutable ranges are recorded.
IREE_API_EXPORT iree_status_t qwen_command_package_prepare(
    qwen_model_t* model, const qwen_command_package_options_t* options,
    iree_allocator_t host_allocator, qwen_command_package_t** out_package);

// Retains |package| for the caller.
IREE_API_EXPORT void qwen_command_package_retain(
    qwen_command_package_t* package);

// Releases |package| and all materialized command programs.
IREE_API_EXPORT void qwen_command_package_release(
    qwen_command_package_t* package);

// Queries immutable metadata for |program|.
IREE_API_EXPORT iree_status_t qwen_command_package_query_program(
    const qwen_command_package_t* package, qwen_command_program_t program,
    qwen_command_program_info_t* out_info);

// Issues one prepared program against compatible request state.
//
// The issue waits for model residency, request readiness, and caller waits;
// allocates and initializes the program's packed transient slab; executes its
// reusable command buffer; and deallocates the slab before publishing request
// and caller completion. It performs no compilation, linking, parameter
// lookup, launch evaluation, allocation planning, or command recording.
IREE_API_EXPORT iree_status_t qwen_command_package_issue(
    qwen_command_package_t* package, qwen_command_program_t program,
    qwen_request_t* request, iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_COMMAND_PACKAGE_H_
