// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_RUNTIME_DECODE_COMMAND_PROGRAM_H_
#define EXPERIMENTAL_QWEN_RUNTIME_DECODE_COMMAND_PROGRAM_H_

#include "experimental/qwen/runtime/request.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reusable materialization of the Qwen3-30B decode-576 command-program root.
//
// The object owns the compiled command package, loaded device executables, and
// one recorded command buffer. It retains the model whose immutable parameter
// ranges are baked into that recording. Callers externally synchronize issue
// operations on the same object.
typedef struct qwen_decode_command_program_t qwen_decode_command_program_t;

// Options controlling exact program preparation and command recording.
typedef struct qwen_decode_command_program_options_t {
  // Size of this structure in bytes.
  iree_host_size_t structure_size;
  // Optional extension chain; must be NULL.
  const void* next;
  // Request token-storage capacity whose field layout is specialized.
  iree_host_size_t request_token_capacity;
  // Request K/V storage capacity; currently exactly 576 rows.
  iree_host_size_t context_capacity;
  // Optional request storage behavior; currently must be none.
  qwen_request_flags_t request_flags;
  // Number of task workers compiling independent program units.
  iree_host_size_t compiler_worker_count;
  // HAL command-buffer mode used for the reusable recording.
  iree_hal_command_buffer_mode_t command_buffer_mode;
} qwen_decode_command_program_options_t;

// Initializes |out_options| with the supported decode-576 shape.
IREE_API_EXPORT void qwen_decode_command_program_options_initialize(
    qwen_decode_command_program_options_t* out_options);

// Compiles and materializes the complete decode command program.
//
// Preparation is synchronous host work and may overlap the model's
// asynchronous parameter gather. Every compiled parameter range is matched
// against the model slab before immutable ranges are recorded.
IREE_API_EXPORT iree_status_t qwen_decode_command_program_prepare(
    qwen_model_t* model, const qwen_decode_command_program_options_t* options,
    iree_allocator_t host_allocator,
    qwen_decode_command_program_t** out_program);

// Retains |program| for the caller.
IREE_API_EXPORT void qwen_decode_command_program_retain(
    qwen_decode_command_program_t* program);

// Releases |program| and its materialized command package.
IREE_API_EXPORT void qwen_decode_command_program_release(
    qwen_decode_command_program_t* program);

// Issues one decode token against compatible request state.
//
// The issue waits for model residency, request readiness, and caller waits;
// allocates and initializes the packed transient slab; executes the reusable
// command buffer; and deallocates the slab before publishing request and caller
// completion. It performs no compilation, linking, parameter lookup, launch
// evaluation, allocation planning, or command recording.
IREE_API_EXPORT iree_status_t qwen_decode_command_program_issue(
    qwen_decode_command_program_t* program, qwen_request_t* request,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list);

// Returns the packed issue-time transient requirement in bytes.
IREE_API_EXPORT iree_device_size_t
qwen_decode_command_program_transient_byte_length(
    const qwen_decode_command_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_DECODE_COMMAND_PROGRAM_H_
