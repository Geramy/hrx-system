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
#include "loomc/target/cmd/program.h"

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
  // Exact 32-row initial prefill with a 64-row attention extent.
  QWEN_COMMAND_PROGRAM_PREFILL_32 = 0,
  // Exact 64-row initial prefill with a 64-row attention extent.
  QWEN_COMMAND_PROGRAM_PREFILL_64 = 1,
  // Exact 128-row initial prefill with a 128-row attention extent.
  QWEN_COMMAND_PROGRAM_PREFILL_128 = 2,
  // Exact 256-row initial prefill with a 256-row attention extent.
  QWEN_COMMAND_PROGRAM_PREFILL_256 = 3,
  // Exact 512-row initial prefill with a 512-row attention extent.
  QWEN_COMMAND_PROGRAM_PREFILL_512 = 4,
  // One-token decode over the 576-row context class.
  QWEN_COMMAND_PROGRAM_DECODE_576 = 5,
  // Number of known command programs.
  QWEN_COMMAND_PROGRAM_COUNT = 6,
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

// Command-program issue behavior flag bits.
typedef enum qwen_command_issue_flag_bits_e {
  // Materializes the selected program as barrier-wave command buffers and
  // waits for each wave before issuing the next. Commands inside one wave
  // retain their original concurrent recording.
  // Stops immediately on a submission or completion failure without issuing
  // any later wave.
  QWEN_COMMAND_ISSUE_FLAG_DIAGNOSTIC_BARRIER_WAVES = 1u << 0,
} qwen_command_issue_flag_bits_t;

// Bitmask of qwen_command_issue_flag_bits_t values.
typedef uint32_t qwen_command_issue_flags_t;

// Progress event emitted by barrier-wave issue.
typedef enum qwen_command_barrier_wave_event_e {
  // The wave is about to be submitted to the device queue.
  QWEN_COMMAND_BARRIER_WAVE_EVENT_BEFORE_EXECUTE = 0,
  // Queue completion for the wave was observed successfully.
  QWEN_COMMAND_BARRIER_WAVE_EVENT_COMPLETED = 1,
} qwen_command_barrier_wave_event_t;

// One synchronous barrier-wave progress event.
typedef struct qwen_command_barrier_wave_event_info_t {
  // Size of this structure in bytes.
  iree_host_size_t structure_size;
  // Reserved extension chain; must be NULL.
  const void* next;
  // Semantic command program being issued.
  qwen_command_program_t program;
  // Progress event for this wave.
  qwen_command_barrier_wave_event_t event;
  // Zero-based position among the materialized non-empty waves.
  iree_host_size_t wave_index;
  // Number of materialized non-empty waves in the issue.
  iree_host_size_t wave_count;
  // Canonical barrier-wave ordinal carried by operation metadata.
  uint32_t barrier_wave_ordinal;
  // Canonical command range recorded into this wave command buffer.
  loomc_cmd_program_command_range_t command_range;
} qwen_command_barrier_wave_event_info_t;

// Synchronous barrier-wave progress callback.
//
// The callback runs on the issuing thread immediately before submission and
// after successful queue completion. It must not reenter the package.
typedef void(IREE_API_PTR* qwen_command_barrier_wave_callback_t)(
    void* user_data, const qwen_command_barrier_wave_event_info_t* event_info);

// Optional observer invoked during a barrier-wave issue.
typedef struct qwen_command_barrier_wave_observer_t {
  // Callback function, or NULL to disable progress notifications.
  qwen_command_barrier_wave_callback_t callback;
  // Opaque value passed to |callback|.
  void* user_data;
} qwen_command_barrier_wave_observer_t;

// Options controlling one command-program issue.
typedef struct qwen_command_issue_options_t {
  // Size of this structure in bytes.
  iree_host_size_t structure_size;
  // Optional extension chain; must be NULL.
  const void* next;
  // Flags selecting issue behavior.
  qwen_command_issue_flags_t flags;
  // Synchronous progress observer used by barrier-wave issue.
  qwen_command_barrier_wave_observer_t barrier_wave_observer;
} qwen_command_issue_options_t;

// Initializes |out_options| for the exact prefill family and Decode-576.
IREE_API_EXPORT void qwen_command_package_options_initialize(
    qwen_command_package_options_t* out_options);

// Initializes |out_options| for a normal whole-program issue.
IREE_API_EXPORT void qwen_command_issue_options_initialize(
    qwen_command_issue_options_t* out_options);

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

// Executes a canonical program prefix and captures its transient backing root.
//
// The prefix is the half-open command range [0, |first_excluded_command|).
// The function uses the package's original fixed buffers, issue-time binding
// table, transient layout, and command-buffer mode. It waits synchronously for
// the prefix and root readback, but does not commit a model result or advance
// request state. The caller must not issue the same program concurrently and
// should discard the diagnostically mutated request after capture.
//
// |transient_capture| must have exactly the selected program's transient byte
// length. Capturing the complete backing root preserves the alias and offset
// relationships of every transient view for later replay.
IREE_API_EXPORT iree_status_t qwen_command_package_capture_transient_prefix(
    qwen_command_package_t* package, qwen_command_program_t program,
    qwen_request_t* request, uint32_t first_excluded_command,
    iree_byte_span_t transient_capture);

// Issues one prepared program against compatible request state.
//
// A normal issue waits for model residency, request readiness, and caller
// waits; allocates and initializes the program's packed transient slab;
// executes its reusable command buffer; and deallocates the slab before
// publishing request and caller completion. It performs no compilation,
// linking, parameter lookup, launch evaluation, allocation planning, or
// command recording. Diagnostic barrier-wave issue temporarily records the
// selected root in barrier-aligned segments and host-waits between them.
IREE_API_EXPORT iree_status_t qwen_command_package_issue(
    qwen_command_package_t* package, qwen_command_program_t program,
    qwen_request_t* request, const qwen_command_issue_options_t* options,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_RUNTIME_COMMAND_PACKAGE_H_
