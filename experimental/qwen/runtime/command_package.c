// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/command_package.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "experimental/qwen/programs/command_package_source.h"
#include "experimental/qwen/runtime/loom_compile_pool.h"
#include "experimental/qwen/runtime/loom_jit.h"
#include "experimental/qwen/runtime/parameters.h"
#include "experimental/qwen/runtime/request_state.h"
#include "iree/base/internal/atomics.h"
#include "loomc/iree.h"
#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/target/cmd/hal.h"
#include "loomc/target/cmd/program_plan.h"

#define QWEN_COMMAND_FIXED_BUFFER_COUNT 2
#define QWEN_COMMAND_INITIAL_SEMAPHORE_CAPACITY 8

typedef enum qwen_command_binding_e {
  QWEN_COMMAND_BINDING_KEY_CACHE = 0,
  QWEN_COMMAND_BINDING_VALUE_CACHE = 1,
  QWEN_COMMAND_BINDING_REQUEST_STATE = 2,
  QWEN_COMMAND_BINDING_OUTPUT_STAGING = 3,
  QWEN_COMMAND_BINDING_TRANSIENT = 4,
  QWEN_COMMAND_BINDING_LAUNCH_COUNTS = 5,
  QWEN_COMMAND_BINDING_CAPACITY = 6,
} qwen_command_binding_t;

typedef enum qwen_command_context_kind_e {
  // The root requires context base zero and has one exact context extent.
  QWEN_COMMAND_CONTEXT_KIND_FIXED_INITIAL = 0,
  // The root accepts any one-row prefix within the prepared package capacity.
  QWEN_COMMAND_CONTEXT_KIND_CAPACITY_PREFIX = 1,
} qwen_command_context_kind_t;

typedef struct qwen_command_parameter_t {
  // Stable fixed-schema key.
  char key[QWEN_PARAMETER_KEY_CAPACITY];
  // Exact model-slab placement.
  iree_io_parameter_span_t span;
  // True after one compiled requirement claims this parameter.
  bool matched;
} qwen_command_parameter_t;

typedef struct qwen_command_program_descriptor_t {
  // Compiled public command-program export name.
  const char* export_name;
  // Exact active token-row count.
  iree_host_size_t token_count;
  // Exact context extent for QWEN_COMMAND_CONTEXT_KIND_FIXED_INITIAL.
  iree_host_size_t fixed_context_count;
  // Request context contract enforced before issue.
  qwen_command_context_kind_t context_kind;
} qwen_command_program_descriptor_t;

typedef struct qwen_command_launch_state_t {
  // Loaded evaluator for this program's aggregate launch function.
  loomc_launch_config_program_t* program;
  // Module-local aggregate launch-function identity.
  loomc_launch_config_function_t function;
  // Persistently mapped host-local/device-visible launch-count storage.
  iree_hal_buffer_t* buffer;
  // Persistent host mapping of |buffer|.
  iree_hal_buffer_mapping_t mapping;
  // Exact mapped prefix written and flushed before each issue.
  iree_device_size_t byte_length;
  // Rebindable command-program slot occupied by |buffer|.
  uint32_t binding_index;
} qwen_command_launch_state_t;

static const qwen_command_program_descriptor_t
    qwen_command_program_descriptors[QWEN_COMMAND_PROGRAM_COUNT] = {
        [QWEN_COMMAND_PROGRAM_PREFILL_32] =
            {
                .export_name = "qwen3_30b_prefill_32",
                .token_count = 32,
                .fixed_context_count = 64,
                .context_kind = QWEN_COMMAND_CONTEXT_KIND_FIXED_INITIAL,
            },
        [QWEN_COMMAND_PROGRAM_PREFILL_64] =
            {
                .export_name = "qwen3_30b_prefill_64",
                .token_count = 64,
                .fixed_context_count = 64,
                .context_kind = QWEN_COMMAND_CONTEXT_KIND_FIXED_INITIAL,
            },
        [QWEN_COMMAND_PROGRAM_PREFILL_128] =
            {
                .export_name = "qwen3_30b_prefill_128",
                .token_count = 128,
                .fixed_context_count = 128,
                .context_kind = QWEN_COMMAND_CONTEXT_KIND_FIXED_INITIAL,
            },
        [QWEN_COMMAND_PROGRAM_PREFILL_256] =
            {
                .export_name = "qwen3_30b_prefill_256",
                .token_count = 256,
                .fixed_context_count = 256,
                .context_kind = QWEN_COMMAND_CONTEXT_KIND_FIXED_INITIAL,
            },
        [QWEN_COMMAND_PROGRAM_PREFILL_512] =
            {
                .export_name = "qwen3_30b_prefill_512",
                .token_count = QWEN_COMMAND_PREFILL_TOKEN_CAPACITY,
                .fixed_context_count = QWEN_COMMAND_PREFILL_TOKEN_CAPACITY,
                .context_kind = QWEN_COMMAND_CONTEXT_KIND_FIXED_INITIAL,
            },
        [QWEN_COMMAND_PROGRAM_DECODE] =
            {
                .export_name = "qwen3_30b_decode",
                .token_count = 1,
                .fixed_context_count = 0,
                .context_kind = QWEN_COMMAND_CONTEXT_KIND_CAPACITY_PREFIX,
            },
};

typedef struct qwen_command_program_state_t {
  // Static descriptor for this materialized root.
  const qwen_command_program_descriptor_t* descriptor;
  // Selected export in the package retained by the owning Qwen package.
  loomc_cmd_program_export_t program_export;
  // Materialized reusable command program.
  loomc_cmd_hal_program_t* hal_program;
  // Immutable parameter buffer ranges used by every materialization.
  iree_hal_buffer_ref_t fixed_buffers[QWEN_COMMAND_FIXED_BUFFER_COUNT];
  // Internal allocation, execution, and completion timeline.
  iree_hal_semaphore_t* timeline_semaphore;
  // Latest timeline value reserved by a submitted issue.
  uint64_t timeline_value;
  // Packed issue-time transient byte length.
  iree_device_size_t transient_byte_length;
  // Required alignment of the transient slab base.
  iree_device_size_t transient_minimum_alignment;
  // Number of rebindable buffer slots required when issuing the program.
  iree_host_size_t binding_count;
  // Dynamic aggregate launch state, empty for fully static programs.
  qwen_command_launch_state_t launch;
  // Reusable wait-semaphore pointer storage.
  iree_hal_semaphore_t** wait_semaphores;
  // Reusable wait payload storage paired with |wait_semaphores|.
  uint64_t* wait_values;
  // Number of entries available in the wait arrays.
  iree_host_size_t wait_capacity;
  // Reusable signal-semaphore pointer storage.
  iree_hal_semaphore_t** signal_semaphores;
  // Reusable signal payload storage paired with |signal_semaphores|.
  uint64_t* signal_values;
  // Number of entries available in the signal arrays.
  iree_host_size_t signal_capacity;
} qwen_command_program_state_t;

struct qwen_command_package_t {
  // Reference count for shared prepared-program ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for all program-owned host storage.
  iree_allocator_t host_allocator;
  // Model retained for immutable fixed-buffer identity and request matching.
  qwen_model_t* model;
  // Request token capacity compiled into persistent field placement.
  iree_host_size_t request_token_capacity;
  // Request K/V capacity compiled into cache and mask placement.
  iree_host_size_t context_capacity;
  // Request flags accepted by the compiled binding layout.
  qwen_request_flags_t request_flags;
  // Loaded portable command-program package shared by every root.
  loomc_cmd_program_package_t* command_program_package;
  // HAL recording mode used by every materialized command buffer.
  iree_hal_command_buffer_mode_t command_buffer_mode;
  // Materialized program state indexed by qwen_command_program_t.
  qwen_command_program_state_t programs[QWEN_COMMAND_PROGRAM_COUNT];
};

typedef enum qwen_command_unit_kind_e {
  QWEN_COMMAND_UNIT_KIND_UNKNOWN = 0,
  QWEN_COMMAND_UNIT_KIND_PACKAGE = 1,
  QWEN_COMMAND_UNIT_KIND_LAUNCH_CONFIG = 2,
  QWEN_COMMAND_UNIT_KIND_EXECUTABLE = 3,
} qwen_command_unit_kind_t;

typedef struct qwen_command_compile_batch_t {
  // Immutable production plan shared by every independent job.
  loomc_program_plan_t* plan;
  // Immutable prepared compiler shared by every independent job.
  loomc_compiler_t* compiler;
  // Unit compilation extensions shared by every independent job.
  const loomc_program_plan_unit_compile_options_t* compile_options;
  // Empty pass program used by portable package and launch-config units.
  const loomc_pass_program_t* empty_pass_program;
  // Target lowering used by executable units.
  const loomc_pass_program_t* executable_pass_program;
  // Unit roles selecting the pass program for each job.
  const qwen_command_unit_kind_t* unit_kinds;
  // Worker-local mutable workspaces indexed by worker ordinal.
  loomc_workspace_t** worker_workspaces;
  // Dense unit results indexed by job and plan-unit ordinal.
  loomc_result_t** unit_results;
  // Allocator passed to every Loom result and program operation.
  loomc_allocator_t allocator;
} qwen_command_compile_batch_t;

static iree_status_t qwen_command_require_result(iree_string_view_t phase,
                                                 const loomc_result_t* result) {
  if (result && loomc_result_succeeded(result)) return iree_ok_status();

  iree_string_view_t message = IREE_SV("Loom operation failed");
  if (result && loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    if (diagnostic) {
      message = iree_string_view_from_loomc(diagnostic->message);
    }
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "%.*s failed: %.*s",
                          (int)phase.size, phase.data, (int)message.size,
                          message.data);
}

static iree_status_t qwen_command_reserve_semaphore_storage(
    iree_host_size_t capacity, iree_allocator_t host_allocator,
    iree_hal_semaphore_t*** out_semaphores, uint64_t** out_values) {
  *out_semaphores = NULL;
  *out_values = NULL;
  iree_status_t status = iree_allocator_malloc_array(host_allocator, capacity,
                                                     sizeof(**out_semaphores),
                                                     (void**)out_semaphores);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, capacity, sizeof(**out_values), (void**)out_values);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, *out_values);
    iree_allocator_free(host_allocator, *out_semaphores);
    *out_semaphores = NULL;
    *out_values = NULL;
  }
  return status;
}

static iree_status_t qwen_command_ensure_semaphore_storage(
    iree_host_size_t required_capacity, iree_allocator_t host_allocator,
    iree_hal_semaphore_t*** inout_semaphores, uint64_t** inout_values,
    iree_host_size_t* inout_capacity) {
  if (*inout_capacity >= required_capacity) return iree_ok_status();
  iree_host_size_t new_capacity = *inout_capacity;
  while (new_capacity < required_capacity) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      new_capacity = required_capacity;
      break;
    }
    new_capacity *= 2;
  }

  iree_hal_semaphore_t** new_semaphores = NULL;
  uint64_t* new_values = NULL;
  IREE_RETURN_IF_ERROR(qwen_command_reserve_semaphore_storage(
      new_capacity, host_allocator, &new_semaphores, &new_values));
  iree_allocator_free(host_allocator, *inout_values);
  iree_allocator_free(host_allocator, *inout_semaphores);
  *inout_semaphores = new_semaphores;
  *inout_values = new_values;
  *inout_capacity = new_capacity;
  return iree_ok_status();
}

static void qwen_command_package_destroy(qwen_command_package_t* package) {
  iree_allocator_t host_allocator = package->host_allocator;
  for (iree_host_size_t i = 0; i < QWEN_COMMAND_PROGRAM_COUNT; ++i) {
    qwen_command_program_state_t* program = &package->programs[i];
    if (program->launch.mapping.buffer) {
      iree_hal_buffer_unmap_range(&program->launch.mapping);
    }
    iree_hal_buffer_release(program->launch.buffer);
    loomc_launch_config_program_release(program->launch.program);
    iree_allocator_free(host_allocator, program->signal_values);
    iree_allocator_free(host_allocator, program->signal_semaphores);
    iree_allocator_free(host_allocator, program->wait_values);
    iree_allocator_free(host_allocator, program->wait_semaphores);
    iree_hal_semaphore_release(program->timeline_semaphore);
    loomc_cmd_hal_program_release(program->hal_program);
  }
  loomc_cmd_program_package_release(package->command_program_package);
  qwen_model_release(package->model);
  iree_allocator_free(host_allocator, package);
}

void qwen_command_package_options_initialize(
    qwen_command_package_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (qwen_command_package_options_t){
      .structure_size = sizeof(*out_options),
      .next = NULL,
      .request_token_capacity = QWEN_COMMAND_PREFILL_TOKEN_CAPACITY,
      .context_capacity = QWEN_COMMAND_DEFAULT_CONTEXT_CAPACITY,
      .request_flags = QWEN_REQUEST_FLAG_NONE,
      .sanitizer_checks = 0,
      .compiler_worker_count = QWEN_LOOM_JIT_DEFAULT_WORKER_COUNT,
      .command_buffer_mode = IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
  };
}

iree_status_t qwen_command_select_prefill_program(
    iree_host_size_t token_count, qwen_command_program_t* out_program) {
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = QWEN_COMMAND_PROGRAM_COUNT;
  for (qwen_command_program_t program = QWEN_COMMAND_PROGRAM_PREFILL_32;
       program <= QWEN_COMMAND_PROGRAM_PREFILL_512; ++program) {
    if (qwen_command_program_descriptors[program].token_count == token_count) {
      *out_program = program;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "Qwen command prefill supports exactly 32, 64, 128, 256, or 512 "
      "tokens; received %" PRIhsz,
      token_count);
}

void qwen_command_issue_options_initialize(
    qwen_command_issue_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (qwen_command_issue_options_t){
      .structure_size = sizeof(*out_options),
      .next = NULL,
      .flags = 0,
      .barrier_wave_observer = {0},
  };
}

static iree_status_t qwen_command_validate_issue_options(
    const qwen_command_issue_options_t* options) {
  if (!options) return iree_ok_status();
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen command issue options structure is too small");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen command issue option extensions are unsupported");
  }
  const qwen_command_issue_flags_t supported_flags =
      QWEN_COMMAND_ISSUE_FLAG_DIAGNOSTIC_BARRIER_WAVES;
  if (iree_any_bit_set(options->flags, ~supported_flags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen command issue options contain unsupported flags");
  }
  if (!iree_any_bit_set(options->flags,
                        QWEN_COMMAND_ISSUE_FLAG_DIAGNOSTIC_BARRIER_WAVES) &&
      (options->barrier_wave_observer.callback ||
       options->barrier_wave_observer.user_data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen barrier-wave observer requires barrier-wave issue");
  }
  if (!options->barrier_wave_observer.callback &&
      options->barrier_wave_observer.user_data) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen barrier-wave observer user data requires a callback");
  }
  return iree_ok_status();
}

static iree_status_t qwen_command_validate_options(
    qwen_model_t* model, const qwen_command_package_options_t* options) {
  if (!model || !options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen model and command-package options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen command-package options structure is too small");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen command-package option extensions are unsupported");
  }
  if (options->request_token_capacity < QWEN_COMMAND_PREFILL_TOKEN_CAPACITY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen command package requires at least %d request token rows",
        QWEN_COMMAND_PREFILL_TOKEN_CAPACITY);
  }
  if (options->context_capacity < options->request_token_capacity ||
      options->context_capacity > QWEN_COMMAND_MAX_CONTEXT_CAPACITY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen command-package context capacity %" PRIhsz " must cover %" PRIhsz
        " request rows and not exceed %d",
        options->context_capacity, options->request_token_capacity,
        QWEN_COMMAND_MAX_CONTEXT_CAPACITY);
  }
  if (options->request_flags != QWEN_REQUEST_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen command roots do not address optional request storage");
  }
  const loomc_sanitizer_checks_t supported_sanitizer_checks =
      LOOMC_SANITIZER_CHECK_ACCESS;
  if (options->sanitizer_checks & ~supported_sanitizer_checks) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen command package sanitizer checks contain unsupported bits");
  }
  if (options->compiler_worker_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen command-package preparation requires compiler workers");
  }
  const iree_hal_command_buffer_mode_t allowed_mode =
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA |
      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA;
  if (iree_any_bit_set(options->command_buffer_mode, ~allowed_mode)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen reusable command programs accept only profiling metadata "
        "command-buffer mode bits");
  }
  return iree_ok_status();
}

static iree_status_t qwen_command_create_source(iree_allocator_t host_allocator,
                                                loomc_source_t** out_source) {
  *out_source = NULL;
  if (qwen_loom_command_package_source_size() != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen command package source must contain exactly one file");
  }
  const iree_file_toc_t* files = qwen_loom_command_package_source_create();
  const loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .next = NULL,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view(files[0].name),
      .contents = loomc_make_byte_span(files[0].data, files[0].size),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  return iree_status_from_loomc(loomc_source_create(
      &options, loomc_allocator_from_iree(host_allocator), out_source));
}

static iree_status_t qwen_command_query_kernels(
    loomc_module_t* module, iree_allocator_t host_allocator,
    loomc_module_function_t** out_functions,
    iree_host_size_t* out_function_count) {
  *out_functions = NULL;
  *out_function_count = 0;
  const loomc_module_function_query_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      .structure_size = sizeof(options),
      .next = NULL,
      .function_symbol = {0},
      .kind = LOOMC_MODULE_FUNCTION_KIND_KERNEL,
  };
  loomc_result_t* result = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_module_query_functions(
      module, &options, loomc_allocator_from_iree(host_allocator), 0, NULL,
      out_function_count, &result));
  if (iree_status_is_ok(status)) {
    status = qwen_command_require_result(IREE_SV("kernel query"), result);
  }
  loomc_result_release(result);
  result = NULL;
  if (iree_status_is_ok(status) && *out_function_count == 0) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Qwen command source contains no kernel roots");
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, *out_function_count,
                                         sizeof(**out_functions),
                                         (void**)out_functions);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_query_functions(
        module, &options, loomc_allocator_from_iree(host_allocator),
        *out_function_count, *out_functions, out_function_count, &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_command_require_result(IREE_SV("kernel query"), result);
  }
  loomc_result_release(result);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, *out_functions);
    *out_functions = NULL;
    *out_function_count = 0;
  }
  return status;
}

static iree_status_t qwen_command_compile_unit_job(
    void* user_data, iree_host_size_t worker_ordinal,
    iree_host_size_t job_ordinal) {
  qwen_command_compile_batch_t* batch =
      (qwen_command_compile_batch_t*)user_data;
  const loomc_pass_program_t* pass_program =
      batch->unit_kinds[job_ordinal] == QWEN_COMMAND_UNIT_KIND_EXECUTABLE
          ? batch->executable_pass_program
          : batch->empty_pass_program;
  iree_status_t status = iree_status_from_loomc(loomc_program_plan_compile_unit(
      batch->plan, batch->compiler, batch->worker_workspaces[worker_ordinal],
      loomc_program_plan_unit_at(batch->plan, job_ordinal), pass_program,
      batch->compile_options, batch->allocator,
      &batch->unit_results[job_ordinal]));
  if (iree_status_is_ok(status)) {
    status = qwen_command_require_result(IREE_SV("unit compile"),
                                         batch->unit_results[job_ordinal]);
  }
  return status;
}

static iree_status_t qwen_command_compile_units(
    loomc_program_plan_t* plan, loomc_compiler_t* compiler,
    const loomc_pass_program_t* empty_pass_program,
    const loomc_pass_program_t* executable_pass_program,
    const qwen_command_unit_kind_t* unit_kinds,
    iree_host_size_t requested_worker_count,
    const loomc_program_plan_unit_compile_options_t* compile_options,
    iree_allocator_t host_allocator, loomc_result_t*** out_unit_results,
    iree_host_size_t* out_unit_count) {
  *out_unit_results = NULL;
  *out_unit_count = loomc_program_plan_unit_count(plan);
  if (*out_unit_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen command plan contains no program units");
  }
  const iree_host_size_t worker_count =
      iree_min(requested_worker_count, *out_unit_count);
  loomc_workspace_t** worker_workspaces = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, worker_count, sizeof(*worker_workspaces),
      (void**)&worker_workspaces);
  if (iree_status_is_ok(status)) {
    memset(worker_workspaces, 0, worker_count * sizeof(*worker_workspaces));
    status = iree_allocator_malloc_array(host_allocator, *out_unit_count,
                                         sizeof(**out_unit_results),
                                         (void**)out_unit_results);
  }
  if (iree_status_is_ok(status)) {
    memset(*out_unit_results, 0, *out_unit_count * sizeof(**out_unit_results));
  }
  for (iree_host_size_t i = 0; i < worker_count && iree_status_is_ok(status);
       ++i) {
    status = iree_status_from_loomc(loomc_workspace_create(
        /*options=*/NULL, loomc_allocator_from_iree(host_allocator),
        &worker_workspaces[i]));
  }

  qwen_loom_compile_pool_t compile_pool;
  memset(&compile_pool, 0, sizeof(compile_pool));
  if (iree_status_is_ok(status)) {
    status = qwen_loom_compile_pool_initialize(worker_count, host_allocator,
                                               &compile_pool);
  }
  if (iree_status_is_ok(status)) {
    qwen_command_compile_batch_t batch = {
        .plan = plan,
        .compiler = compiler,
        .compile_options = compile_options,
        .empty_pass_program = empty_pass_program,
        .executable_pass_program = executable_pass_program,
        .unit_kinds = unit_kinds,
        .worker_workspaces = worker_workspaces,
        .unit_results = *out_unit_results,
        .allocator = loomc_allocator_from_iree(host_allocator),
    };
    status = qwen_loom_compile_pool_run_batch(
        &compile_pool, *out_unit_count, qwen_command_compile_unit_job, &batch);
  }
  qwen_loom_compile_pool_deinitialize(&compile_pool);
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    loomc_workspace_release(worker_workspaces ? worker_workspaces[i] : NULL);
  }
  iree_allocator_free(host_allocator, worker_workspaces);

  if (!iree_status_is_ok(status)) {
    if (*out_unit_results) {
      for (iree_host_size_t i = 0; i < *out_unit_count; ++i) {
        loomc_result_release((*out_unit_results)[i]);
      }
    }
    iree_allocator_free(host_allocator, *out_unit_results);
    *out_unit_results = NULL;
    *out_unit_count = 0;
  }
  return status;
}

static void qwen_command_release_unit_results(iree_host_size_t unit_count,
                                              loomc_result_t** unit_results,
                                              iree_allocator_t host_allocator) {
  if (unit_results) {
    for (iree_host_size_t i = 0; i < unit_count; ++i) {
      loomc_result_release(unit_results[i]);
    }
  }
  iree_allocator_free(host_allocator, unit_results);
}

static const loomc_artifact_t* qwen_command_find_artifact(
    const loomc_result_t* result, loomc_artifact_kind_t kind,
    const char* format) {
  if (!result) return NULL;
  const loomc_string_view_t expected_format = loomc_make_cstring_view(format);
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact && artifact->kind == kind &&
        loomc_string_view_equal(artifact->format, expected_format)) {
      return artifact;
    }
  }
  return NULL;
}

static iree_status_t qwen_command_mark_unit_kind(
    loomc_program_plan_unit_t unit, qwen_command_unit_kind_t kind,
    iree_host_size_t unit_count, qwen_command_unit_kind_t* unit_kinds) {
  if (!loomc_program_plan_unit_is_valid(unit) || unit.value >= unit_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen command plan contains an invalid unit");
  }
  qwen_command_unit_kind_t* current = &unit_kinds[unit.value];
  if (*current != QWEN_COMMAND_UNIT_KIND_UNKNOWN && *current != kind) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen command plan assigns unit %" PRIu64
                            " conflicting roles",
                            unit.value);
  }
  *current = kind;
  return iree_ok_status();
}

static iree_status_t qwen_command_select_executable_target(
    iree_hal_device_t* device,
    const iree_hal_executable_target_t** out_executable_target) {
  *out_executable_target = NULL;
  const iree_hal_executable_target_selection_t selection = {
      .family = IREE_SV("amdgpu"),
      .target_key = iree_string_view_empty(),
      .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      .physical_device_affinity = 0,
  };
  const iree_hal_executable_target_selection_result_t selection_result =
      iree_hal_device_spec_select_executable_target(
          iree_hal_device_spec(device), &selection);
  if (selection_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "HAL device has no exact AMDGPU executable target");
  }
  if (selection_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL device has multiple exact AMDGPU executable targets");
  }
  *out_executable_target = selection_result.target;
  return iree_ok_status();
}

static iree_status_t qwen_command_load_executable(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_executable_target_t* executable_target,
    const loomc_artifact_t* artifact, iree_hal_executable_t** out_executable) {
  *out_executable = NULL;
  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.executable_data =
      iree_const_byte_span_from_loomc(artifact->contents);
  return iree_hal_device_load_executable(
      device, queue_affinity, executable_target, &load_params, out_executable);
}

static iree_status_t qwen_command_validate_parameter_layout(
    qwen_model_t* model, const loomc_cmd_program_package_t* command_package,
    loomc_cmd_program_export_t program_export,
    const loomc_cmd_program_info_t* info, iree_allocator_t host_allocator,
    iree_hal_buffer_ref_t out_fixed_buffers[QWEN_COMMAND_FIXED_BUFFER_COUNT]) {
  const qwen_parameter_layout_t* model_layout =
      qwen_model_parameter_layout(model);
  qwen_command_parameter_t* model_parameters = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, QWEN_PARAMETER_COUNT, sizeof(*model_parameters),
      (void**)&model_parameters));
  memset(model_parameters, 0, QWEN_PARAMETER_COUNT * sizeof(*model_parameters));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < QWEN_PARAMETER_COUNT && iree_status_is_ok(status); ++i) {
    iree_string_view_t key = iree_string_view_empty();
    status = qwen_parameter_layout_enumerate(model_layout, i,
                                             model_parameters[i].key, &key,
                                             &model_parameters[i].span);
    if (iree_status_is_ok(status)) {
      memmove(model_parameters[i].key, key.data, key.size);
      model_parameters[i].key[key.size] = 0;
    }
  }

  iree_host_size_t main_parameter_count = 0;
  bool matched_auxiliary = false;
  for (iree_host_size_t i = 0;
       i < info->parameter_count && iree_status_is_ok(status); ++i) {
    loomc_cmd_program_parameter_info_t parameter = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_INFO,
        .structure_size = sizeof(parameter),
    };
    status = iree_status_from_loomc(loomc_cmd_program_package_parameter_info(
        command_package, program_export, i, &parameter));
    if (!iree_status_is_ok(status)) break;

    const iree_string_view_t key = iree_string_view_from_loomc(parameter.key);
    if (parameter.fixed_buffer_index == 1) {
      if (matched_auxiliary ||
          !iree_string_view_equal(key, IREE_SV("rope.inverse_frequencies")) ||
          parameter.byte_offset != 0 ||
          parameter.byte_length !=
              model_layout->rope_inverse_frequencies.length) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "compiled auxiliary parameter '%.*s' does not match the model "
            "slab",
            (int)key.size, key.data);
      } else {
        matched_auxiliary = true;
      }
      continue;
    }
    if (parameter.fixed_buffer_index != 0) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "compiled parameter '%.*s' uses unexpected fixed buffer %u",
          (int)key.size, key.data, parameter.fixed_buffer_index);
      break;
    }

    qwen_command_parameter_t* model_parameter = NULL;
    for (iree_host_size_t j = 0; j < QWEN_PARAMETER_COUNT; ++j) {
      const iree_string_view_t model_key =
          iree_make_cstring_view(model_parameters[j].key);
      if (iree_string_view_equal(key, model_key)) {
        model_parameter = &model_parameters[j];
        break;
      }
    }
    if (!model_parameter) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "compiled parameter '%.*s' is absent from the model slab",
          (int)key.size, key.data);
      break;
    }
    if (model_parameter->matched) {
      status =
          iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                           "compiled parameter '%.*s' appears more than once",
                           (int)key.size, key.data);
      break;
    }
    if (parameter.byte_offset != model_parameter->span.buffer_offset ||
        parameter.byte_length != model_parameter->span.length ||
        parameter.minimum_alignment == 0 ||
        parameter.byte_offset % parameter.minimum_alignment != 0) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "compiled parameter '%.*s' range [%" PRIu64 ", %" PRIu64
          ") alignment %" PRIu64 " does not match model range [%" PRIu64
          ", %" PRIu64 ")",
          (int)key.size, key.data, parameter.byte_offset,
          parameter.byte_offset + parameter.byte_length,
          parameter.minimum_alignment, model_parameter->span.buffer_offset,
          model_parameter->span.buffer_offset + model_parameter->span.length);
      break;
    }
    model_parameter->matched = true;
    ++main_parameter_count;
  }

  if (iree_status_is_ok(status) &&
      (main_parameter_count != QWEN_PARAMETER_COUNT || !matched_auxiliary)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "compiled command program matched %" PRIhsz
        " model parameters and auxiliary=%d; expected %d and auxiliary=1",
        main_parameter_count, matched_auxiliary, QWEN_PARAMETER_COUNT);
  }

  loomc_cmd_program_parameter_root_info_t main_root = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      .structure_size = sizeof(main_root),
  };
  loomc_cmd_program_parameter_root_info_t auxiliary_root = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      .structure_size = sizeof(auxiliary_root),
  };
  if (iree_status_is_ok(status)) {
    status =
        iree_status_from_loomc(loomc_cmd_program_package_parameter_root_info(
            command_package, program_export, 0, &main_root));
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_status_from_loomc(loomc_cmd_program_package_parameter_root_info(
            command_package, program_export, 1, &auxiliary_root));
  }
  const qwen_model_statistics_t model_statistics = qwen_model_statistics(model);
  if (iree_status_is_ok(status) &&
      (main_root.fixed_buffer_index != 0 ||
       main_root.required_byte_length > model_statistics.allocation_bytes ||
       auxiliary_root.fixed_buffer_index != 1 ||
       auxiliary_root.required_byte_length !=
           model_layout->rope_inverse_frequencies.length ||
       model_layout->rope_inverse_frequencies.offset %
               auxiliary_root.minimum_alignment !=
           0)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "compiled fixed-buffer roots do not match the model allocation");
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_t* parameter_buffer = qwen_model_parameter_buffer(model);
    out_fixed_buffers[0] = iree_hal_make_buffer_ref(
        parameter_buffer, 0, main_root.required_byte_length);
    out_fixed_buffers[1] = iree_hal_make_buffer_ref(
        parameter_buffer, model_layout->rope_inverse_frequencies.offset,
        auxiliary_root.required_byte_length);
  }
  iree_allocator_free(host_allocator, model_parameters);
  return status;
}

static iree_status_t qwen_command_prepare_launch_state(
    qwen_model_t* model, const qwen_command_program_descriptor_t* descriptor,
    const loomc_cmd_program_info_t* command_info,
    const loomc_artifact_t* launch_artifact, iree_allocator_t host_allocator,
    qwen_command_launch_state_t* out_launch) {
  const bool has_dynamic_launch_counts =
      command_info->config.binding_index != LOOMC_CMD_PROGRAM_BINDING_INVALID;
  if (!has_dynamic_launch_counts) {
    if (launch_artifact) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "static Qwen command program '%s' has a launch-config unit",
          descriptor->export_name);
    }
    return iree_ok_status();
  }
  if (!launch_artifact) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "dynamic Qwen command program '%s' has no launch-config artifact",
        descriptor->export_name);
  }

  out_launch->byte_length = command_info->config.required_byte_length;
  out_launch->binding_index = command_info->config.binding_index;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_launch_config_program_load(
      launch_artifact, /*release=*/NULL, /*release_user_data=*/NULL,
      loomc_allocator_from_iree(host_allocator), &out_launch->program)));
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_launch_config_program_lookup_function(
          out_launch->program, loomc_make_cstring_view(descriptor->export_name),
          &out_launch->function)));

  const iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
               IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = qwen_model_queue_affinity(model),
      .min_alignment = LOOMC_CMD_HAL_CONFIG_ALIGNMENT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(qwen_model_device(model)), buffer_params,
      out_launch->byte_length, &out_launch->buffer));
  return iree_hal_buffer_map_range(
      out_launch->buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_WRITE, /*byte_offset=*/0, out_launch->byte_length,
      &out_launch->mapping);
}

static iree_status_t qwen_command_evaluate_launch(
    qwen_command_program_state_t* program, iree_host_size_t context_base) {
  if (!program->launch.program) return iree_ok_status();

  const uint64_t visible_context_count = (uint64_t)context_base + 1;
  loomc_cmd_launch_config_t launch_config = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_LAUNCH_CONFIG,
      .structure_size = sizeof(launch_config),
      .next = NULL,
      .data = loomc_make_mutable_byte_span(
          program->launch.mapping.contents.data, program->launch.byte_length),
  };
  IREE_RETURN_IF_ERROR(
      iree_status_from_loomc(loomc_launch_config_program_invoke_cmd(
          program->launch.program, program->launch.function,
          &visible_context_count, /*argument_count=*/1, &launch_config)));
  return iree_hal_buffer_mapping_flush_range(
      &program->launch.mapping, /*byte_offset=*/0, program->launch.byte_length);
}

static iree_status_t qwen_command_classify_plan(
    loomc_program_plan_t* plan,
    loomc_program_plan_root_t out_roots[QWEN_COMMAND_PROGRAM_COUNT],
    loomc_cmd_program_plan_root_info_t
        out_root_infos[QWEN_COMMAND_PROGRAM_COUNT],
    qwen_command_unit_kind_t* unit_kinds, iree_host_size_t unit_count,
    loomc_program_plan_unit_t* out_package_unit,
    loomc_program_plan_unit_t* out_launch_config_unit) {
  *out_package_unit = loomc_program_plan_unit_invalid();
  *out_launch_config_unit = loomc_program_plan_unit_invalid();
  for (iree_host_size_t i = 0; i < QWEN_COMMAND_PROGRAM_COUNT; ++i) {
    IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_program_plan_lookup_root(
        plan,
        loomc_make_cstring_view(
            qwen_command_program_descriptors[i].export_name),
        &out_roots[i])));
    IREE_RETURN_IF_ERROR(
        iree_status_from_loomc(loomc_cmd_program_plan_root_info(
            plan, out_roots[i], &out_root_infos[i])));
    const loomc_cmd_program_plan_root_info_t* root_info = &out_root_infos[i];
    if (!loomc_program_plan_unit_is_valid(*out_package_unit)) {
      *out_package_unit = root_info->package_unit;
    } else if (root_info->package_unit.value != out_package_unit->value) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Qwen command roots do not share one portable package unit");
    }
    IREE_RETURN_IF_ERROR(qwen_command_mark_unit_kind(
        root_info->package_unit, QWEN_COMMAND_UNIT_KIND_PACKAGE, unit_count,
        unit_kinds));

    if (loomc_program_plan_unit_is_valid(root_info->launch_config_unit)) {
      if (!loomc_program_plan_unit_is_valid(*out_launch_config_unit)) {
        *out_launch_config_unit = root_info->launch_config_unit;
      } else if (root_info->launch_config_unit.value !=
                 out_launch_config_unit->value) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Qwen command roots do not share one launch-config unit");
      }
      IREE_RETURN_IF_ERROR(qwen_command_mark_unit_kind(
          root_info->launch_config_unit, QWEN_COMMAND_UNIT_KIND_LAUNCH_CONFIG,
          unit_count, unit_kinds));
    }

    for (iree_host_size_t j = 0; j < root_info->executable_requirement_count;
         ++j) {
      const loomc_cmd_program_plan_executable_requirement_t* requirement =
          &root_info->executable_requirements[j];
      if (!loomc_program_plan_unit_is_valid(requirement->unit)) {
        const iree_string_view_t import_name =
            iree_string_view_from_loomc(requirement->import_name);
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "Qwen command root '%s' requires external executable '%.*s'",
            qwen_command_program_descriptors[i].export_name,
            (int)import_name.size, import_name.data);
      }
      IREE_RETURN_IF_ERROR(qwen_command_mark_unit_kind(
          requirement->unit, QWEN_COMMAND_UNIT_KIND_EXECUTABLE, unit_count,
          unit_kinds));
    }
  }

  for (iree_host_size_t i = 0; i < unit_count; ++i) {
    if (unit_kinds[i] == QWEN_COMMAND_UNIT_KIND_UNKNOWN) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Qwen command plan contains unclassified unit %" PRIhsz, i);
    }
  }
  return iree_ok_status();
}

iree_status_t qwen_command_package_prepare(
    qwen_model_t* model, const qwen_command_package_options_t* options,
    iree_allocator_t host_allocator, qwen_command_package_t** out_package) {
  IREE_ASSERT_ARGUMENT(out_package);
  *out_package = NULL;
  IREE_RETURN_IF_ERROR(qwen_command_validate_options(model, options));

  const loomc_allocator_t loom_allocator =
      loomc_allocator_from_iree(host_allocator);
  loomc_target_environment_t* target_environment = NULL;
  loomc_context_t* context = NULL;
  loomc_workspace_t* coordinator_workspace = NULL;
  loomc_source_t* source = NULL;
  loomc_module_t* module = NULL;
  loomc_target_profile_t* target_profile = NULL;
  loomc_compiler_t* compiler = NULL;
  loomc_pass_program_t* preparation_pass_program = NULL;
  loomc_pass_program_t* empty_pass_program = NULL;
  loomc_pass_program_t* executable_pass_program = NULL;
  loomc_module_function_t* kernel_functions = NULL;
  loomc_target_specialization_t* specializations = NULL;
  iree_host_size_t kernel_function_count = 0;
  loomc_program_plan_t* plan = NULL;
  loomc_program_plan_root_t roots[QWEN_COMMAND_PROGRAM_COUNT];
  loomc_cmd_program_plan_root_info_t
      root_plan_infos[QWEN_COMMAND_PROGRAM_COUNT];
  loomc_program_plan_unit_t package_unit = loomc_program_plan_unit_invalid();
  loomc_program_plan_unit_t launch_config_unit =
      loomc_program_plan_unit_invalid();
  qwen_command_unit_kind_t* unit_kinds = NULL;
  loomc_result_t** unit_results = NULL;
  iree_hal_executable_t** unit_executables = NULL;
  iree_host_size_t unit_count = 0;
  loomc_cmd_program_package_t* command_program_package = NULL;
  loomc_cmd_program_export_t program_exports[QWEN_COMMAND_PROGRAM_COUNT];
  loomc_cmd_program_info_t command_program_infos[QWEN_COMMAND_PROGRAM_COUNT];
  iree_hal_buffer_ref_t fixed_buffers[QWEN_COMMAND_PROGRAM_COUNT]
                                     [QWEN_COMMAND_FIXED_BUFFER_COUNT];
  loomc_cmd_hal_program_t* hal_programs[QWEN_COMMAND_PROGRAM_COUNT];
  const loomc_artifact_t* launch_config_artifact = NULL;
  loomc_result_t* result = NULL;
  memset(roots, 0, sizeof(roots));
  memset(root_plan_infos, 0, sizeof(root_plan_infos));
  memset(program_exports, 0, sizeof(program_exports));
  memset(command_program_infos, 0, sizeof(command_program_infos));
  memset(fixed_buffers, 0, sizeof(fixed_buffers));
  memset(hal_programs, 0, sizeof(hal_programs));

  const loomc_sanitizer_options_t sanitizer_options = {
      .type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
      .structure_size = sizeof(sanitizer_options),
      .next = NULL,
      .checks = options->sanitizer_checks,
      .flags = LOOMC_SANITIZER_FLAG_NONE,
      .reporting_mode = LOOMC_SANITIZER_REPORTING_MODE_REPORT_ONLY,
  };

  iree_status_t status =
      iree_status_from_loomc(loomc_target_environment_create_amdgpu(
          loom_allocator, &target_environment));
  if (iree_status_is_ok(status)) {
    const loomc_context_target_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
        .structure_size = sizeof(target_options),
        .next = NULL,
        .target_environment = target_environment,
    };
    const loomc_context_options_t context_options = {
        .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
        .structure_size = sizeof(context_options),
        .next = &target_options,
    };
    status = iree_status_from_loomc(
        loomc_context_create(&context_options, loom_allocator, &context));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_workspace_create(
        /*options=*/NULL, loom_allocator, &coordinator_workspace));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_command_create_source(host_allocator, &source);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_deserialize_from_source(
        context, coordinator_workspace, source, /*options=*/NULL,
        loom_allocator, &module, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_command_require_result(IREE_SV("command package parse"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    const loomc_amdgpu_iree_hal_profile_options_t profile_options = {
        .type = LOOMC_STRUCTURE_TYPE_AMDGPU_IREE_HAL_PROFILE_OPTIONS,
        .structure_size = sizeof(profile_options),
        .next = NULL,
        .identifier = loomc_make_cstring_view("qwen-live-amdgpu-command"),
        .device = qwen_model_device(model),
        .physical_device_affinity = 0,
    };
    status = iree_status_from_loomc(loomc_target_profile_create_amdgpu_iree_hal(
        target_environment, &profile_options, loom_allocator, &target_profile,
        &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_command_require_result(IREE_SV("target profile"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    status = qwen_command_query_kernels(
        module, host_allocator, &kernel_functions, &kernel_function_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, kernel_function_count,
                                         sizeof(*specializations),
                                         (void**)&specializations);
  }
  for (iree_host_size_t i = 0;
       i < kernel_function_count && iree_status_is_ok(status); ++i) {
    specializations[i] = (loomc_target_specialization_t){
        .function_symbol = kernel_functions[i].symbol_name,
        .target_profile = target_profile,
    };
  }

  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_compiler_create(
        context, /*options=*/NULL, loom_allocator, &compiler));
  }
  if (iree_status_is_ok(status)) {
    const loomc_target_pipeline_options_t pipeline_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        .structure_size = sizeof(pipeline_options),
        .next = NULL,
        .identifier = loomc_make_cstring_view("qwen-command-expanded-source"),
        .kind = LOOMC_TARGET_PIPELINE_KIND_EXPANDED_SOURCE,
        .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
        .source_to_low_max_errors = 20,
    };
    status =
        iree_status_from_loomc(loomc_pass_program_create_from_target_pipeline(
            context, &pipeline_options, loom_allocator,
            &preparation_pass_program, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_command_require_result(IREE_SV("preparation pipeline"), result);
  }
  loomc_result_release(result);
  result = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_pass_program_create_empty(
        context, /*options=*/NULL, loom_allocator, &empty_pass_program));
  }
  if (iree_status_is_ok(status)) {
    const loomc_target_pipeline_options_t pipeline_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        .structure_size = sizeof(pipeline_options),
        .next = options->sanitizer_checks ? &sanitizer_options : NULL,
        .identifier = loomc_make_cstring_view("qwen-command-prepared-low"),
        .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
        .source_to_low_max_errors = 20,
    };
    status =
        iree_status_from_loomc(loomc_pass_program_create_from_target_pipeline(
            context, &pipeline_options, loom_allocator,
            &executable_pass_program, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_command_require_result(IREE_SV("executable pipeline"), result);
  }
  loomc_result_release(result);
  result = NULL;

  char request_token_capacity_storage[32];
  const int request_token_capacity_length = snprintf(
      request_token_capacity_storage, sizeof(request_token_capacity_storage),
      "%" PRIhsz, options->request_token_capacity);
  char context_capacity_storage[32];
  const int context_capacity_length =
      snprintf(context_capacity_storage, sizeof(context_capacity_storage),
               "%" PRIhsz, options->context_capacity);
  if (iree_status_is_ok(status) &&
      (request_token_capacity_length < 0 ||
       (iree_host_size_t)request_token_capacity_length >=
           sizeof(request_token_capacity_storage) ||
       context_capacity_length < 0 ||
       (iree_host_size_t)context_capacity_length >=
           sizeof(context_capacity_storage))) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Qwen capacity spelling overflowed");
  }
  if (iree_status_is_ok(status)) {
    const loomc_config_binding_t config_bindings[] = {
        {
            .key = loomc_make_cstring_view("qwen3_30b.request.token_capacity"),
            .value = loomc_make_string_view(
                request_token_capacity_storage,
                (loomc_host_size_t)request_token_capacity_length),
        },
        {
            .key = loomc_make_cstring_view(
                "qwen3_moe.attention.key_value_token_capacity"),
            .value = loomc_make_string_view(
                context_capacity_storage,
                (loomc_host_size_t)context_capacity_length),
        },
    };
    const loomc_target_specialization_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        .structure_size = sizeof(target_options),
        .next = NULL,
        .specializations = specializations,
        .specialization_count = kernel_function_count,
    };
    const loomc_compile_options_t compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
        .structure_size = sizeof(compile_options),
        .next = &target_options,
        .module_name = loomc_make_cstring_view("qwen3_30b_command_programs"),
        .artifact_flags = 0,
        .config =
            {
                .bindings = config_bindings,
                .binding_count = IREE_ARRAYSIZE(config_bindings),
                .json_object = {0},
                .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
            },
    };
    status = iree_status_from_loomc(loomc_compile_module(
        compiler, coordinator_workspace, preparation_pass_program, module,
        &compile_options, loom_allocator, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_command_require_result(IREE_SV("program preparation"), result);
  }
  loomc_result_release(result);
  result = NULL;

  loomc_string_view_t root_names[QWEN_COMMAND_PROGRAM_COUNT];
  for (iree_host_size_t i = 0; i < QWEN_COMMAND_PROGRAM_COUNT; ++i) {
    root_names[i] = loomc_make_cstring_view(
        qwen_command_program_descriptors[i].export_name);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_program_plan_prepare(
        coordinator_workspace, module, root_names, QWEN_COMMAND_PROGRAM_COUNT,
        /*options=*/NULL, loom_allocator, &plan, &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_command_require_result(IREE_SV("program plan"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    unit_count = loomc_program_plan_unit_count(plan);
    if (unit_count == 0) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "Qwen command plan contains no units");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, unit_count, sizeof(*unit_kinds), (void**)&unit_kinds);
  }
  if (iree_status_is_ok(status)) {
    memset(unit_kinds, 0, unit_count * sizeof(*unit_kinds));
    status = qwen_command_classify_plan(plan, roots, root_plan_infos,
                                        unit_kinds, unit_count, &package_unit,
                                        &launch_config_unit);
  }

  if (iree_status_is_ok(status)) {
    const loomc_program_plan_unit_compile_options_t unit_compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_COMPILE_OPTIONS,
        .structure_size = sizeof(unit_compile_options),
        .next = NULL,
    };
    status = qwen_command_compile_units(
        plan, compiler, empty_pass_program, executable_pass_program, unit_kinds,
        options->compiler_worker_count, &unit_compile_options, host_allocator,
        &unit_results, &unit_count);
  }

  if (iree_status_is_ok(status)) {
    const loomc_artifact_t* package_artifact = qwen_command_find_artifact(
        unit_results[package_unit.value], LOOMC_ARTIFACT_KIND_EXECUTABLE,
        LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE);
    if (!package_artifact) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "Qwen command plan did not emit a portable package artifact");
    } else {
      status = iree_status_from_loomc(loomc_cmd_program_package_load(
          package_artifact, /*release=*/NULL, /*release_user_data=*/NULL,
          loom_allocator, &command_program_package));
    }
  }
  if (iree_status_is_ok(status) &&
      loomc_program_plan_unit_is_valid(launch_config_unit)) {
    launch_config_artifact =
        qwen_command_find_artifact(unit_results[launch_config_unit.value],
                                   LOOMC_ARTIFACT_KIND_COMMAND_LAUNCH_CONFIG,
                                   LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
    if (!launch_config_artifact) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "Qwen command plan did not emit its launch-config artifact");
    }
  }

  const iree_hal_executable_target_t* executable_target = NULL;
  if (iree_status_is_ok(status)) {
    status = qwen_command_select_executable_target(qwen_model_device(model),
                                                   &executable_target);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, unit_count,
                                         sizeof(*unit_executables),
                                         (void**)&unit_executables);
  }
  if (iree_status_is_ok(status)) {
    memset(unit_executables, 0, unit_count * sizeof(*unit_executables));
  }
  for (iree_host_size_t i = 0; i < unit_count && iree_status_is_ok(status);
       ++i) {
    if (unit_kinds[i] != QWEN_COMMAND_UNIT_KIND_EXECUTABLE) continue;
    const loomc_artifact_t* executable_artifact = qwen_command_find_artifact(
        unit_results[i], LOOMC_ARTIFACT_KIND_EXECUTABLE,
        LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
    if (!executable_artifact) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "Qwen command executable unit %" PRIhsz " emitted no HSACO", i);
      break;
    }
    status = qwen_command_load_executable(
        qwen_model_device(model), qwen_model_queue_affinity(model),
        executable_target, executable_artifact, &unit_executables[i]);
  }

  for (iree_host_size_t i = 0;
       i < QWEN_COMMAND_PROGRAM_COUNT && iree_status_is_ok(status); ++i) {
    const qwen_command_program_descriptor_t* descriptor =
        &qwen_command_program_descriptors[i];
    program_exports[i] = loomc_cmd_program_export_invalid();
    status = iree_status_from_loomc(loomc_cmd_program_package_lookup_export(
        command_program_package,
        loomc_make_cstring_view(descriptor->export_name), &program_exports[i]));
    command_program_infos[i] = (loomc_cmd_program_info_t){
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO,
        .structure_size = sizeof(command_program_infos[i]),
    };
    if (iree_status_is_ok(status)) {
      status = iree_status_from_loomc(loomc_cmd_program_package_export_info(
          command_program_package, program_exports[i],
          &command_program_infos[i]));
    }

    const loomc_cmd_program_info_t* info = &command_program_infos[i];
    const bool expects_dynamic_config =
        descriptor->context_kind == QWEN_COMMAND_CONTEXT_KIND_CAPACITY_PREFIX;
    const iree_host_size_t expected_binding_count =
        expects_dynamic_config ? QWEN_COMMAND_BINDING_CAPACITY
                               : QWEN_COMMAND_BINDING_LAUNCH_COUNTS;
    const uint32_t expected_config_binding =
        expects_dynamic_config ? QWEN_COMMAND_BINDING_LAUNCH_COUNTS
                               : LOOMC_CMD_PROGRAM_BINDING_INVALID;
    const uint64_t expected_config_byte_length =
        expects_dynamic_config ? sizeof(loomc_dimension3_t) : 0;
    const bool config_alignment_is_valid =
        expects_dynamic_config ? info->config.minimum_alignment != 0 &&
                                     info->config.minimum_alignment <=
                                         LOOMC_CMD_HAL_CONFIG_ALIGNMENT &&
                                     LOOMC_CMD_HAL_CONFIG_ALIGNMENT %
                                             info->config.minimum_alignment ==
                                         0
                               : info->config.minimum_alignment == 0;
    if (iree_status_is_ok(status) &&
        (info->fixed_buffer_count != QWEN_COMMAND_FIXED_BUFFER_COUNT ||
         info->rebindable_binding_count != expected_binding_count ||
         info->executable_count !=
             root_plan_infos[i].executable_requirement_count ||
         info->parameter_root_count != 2 ||
         info->parameter_count != QWEN_PARAMETER_COUNT + 1 ||
         info->transient.binding_index != QWEN_COMMAND_BINDING_TRANSIENT ||
         info->transient.required_byte_length == 0 ||
         info->transient.minimum_alignment == 0 ||
         info->config.binding_index != expected_config_binding ||
         info->config.required_byte_length != expected_config_byte_length ||
         !config_alignment_is_valid)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "compiled Qwen command program '%s' has an incompatible ABI",
          descriptor->export_name);
    }
    const bool has_launch_config_unit =
        loomc_program_plan_unit_is_valid(root_plan_infos[i].launch_config_unit);
    if (iree_status_is_ok(status) &&
        has_launch_config_unit != expects_dynamic_config) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "compiled Qwen command program '%s' has inconsistent dynamic config",
          descriptor->export_name);
    }
    if (iree_status_is_ok(status)) {
      status = qwen_command_validate_parameter_layout(
          model, command_program_package, program_exports[i], info,
          host_allocator, fixed_buffers[i]);
    }
  }

  for (iree_host_size_t i = 0;
       i < QWEN_COMMAND_PROGRAM_COUNT && iree_status_is_ok(status); ++i) {
    const loomc_cmd_program_plan_root_info_t* root_info = &root_plan_infos[i];
    iree_hal_executable_t** root_executables = NULL;
    if (root_info->executable_requirement_count != 0) {
      status = iree_allocator_malloc_array(
          host_allocator, root_info->executable_requirement_count,
          sizeof(*root_executables), (void**)&root_executables);
    }
    for (iree_host_size_t j = 0; j < root_info->executable_requirement_count &&
                                 iree_status_is_ok(status);
         ++j) {
      const loomc_program_plan_unit_t unit =
          root_info->executable_requirements[j].unit;
      if (!loomc_program_plan_unit_is_valid(unit) || unit.value >= unit_count ||
          !unit_executables[unit.value]) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Qwen command root '%s' has an unavailable executable slot",
            qwen_command_program_descriptors[i].export_name);
      } else {
        root_executables[j] = unit_executables[unit.value];
      }
    }
    if (iree_status_is_ok(status)) {
      const loomc_cmd_hal_program_options_t program_options = {
          .type = LOOMC_STRUCTURE_TYPE_CMD_HAL_PROGRAM_OPTIONS,
          .structure_size = sizeof(program_options),
          .next = NULL,
          .command_buffer_mode = options->command_buffer_mode,
          .queue_affinity = qwen_model_queue_affinity(model),
          .fixed_buffers = fixed_buffers[i],
          .fixed_buffer_count = IREE_ARRAYSIZE(fixed_buffers[i]),
          .executables = root_executables,
          .executable_count = root_info->executable_requirement_count,
      };
      status = iree_status_from_loomc(loomc_cmd_hal_program_create(
          command_program_package, program_exports[i], qwen_model_device(model),
          &program_options, loom_allocator, &hal_programs[i]));
    }
    iree_allocator_free(host_allocator, root_executables);
  }

  qwen_command_package_t* package = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*package),
                                   (void**)&package);
  }
  if (iree_status_is_ok(status)) {
    memset(package, 0, sizeof(*package));
    iree_atomic_ref_count_init(&package->ref_count);
    package->host_allocator = host_allocator;
    package->model = model;
    qwen_model_retain(model);
    package->request_token_capacity = options->request_token_capacity;
    package->context_capacity = options->context_capacity;
    package->request_flags = options->request_flags;
    package->command_program_package = command_program_package;
    command_program_package = NULL;
    package->command_buffer_mode = options->command_buffer_mode;
  }
  for (iree_host_size_t i = 0;
       i < QWEN_COMMAND_PROGRAM_COUNT && iree_status_is_ok(status); ++i) {
    qwen_command_program_state_t* program = &package->programs[i];
    program->descriptor = &qwen_command_program_descriptors[i];
    program->program_export = program_exports[i];
    program->hal_program = hal_programs[i];
    hal_programs[i] = NULL;
    memcpy(program->fixed_buffers, fixed_buffers[i],
           sizeof(program->fixed_buffers));
    program->transient_byte_length =
        command_program_infos[i].transient.required_byte_length;
    program->transient_minimum_alignment =
        command_program_infos[i].transient.minimum_alignment;
    program->binding_count = command_program_infos[i].rebindable_binding_count;
    const loomc_artifact_t* root_launch_artifact =
        loomc_program_plan_unit_is_valid(root_plan_infos[i].launch_config_unit)
            ? launch_config_artifact
            : NULL;
    status = qwen_command_prepare_launch_state(
        model, program->descriptor, &command_program_infos[i],
        root_launch_artifact, host_allocator, &program->launch);
    if (iree_status_is_ok(status)) {
      status = qwen_command_reserve_semaphore_storage(
          QWEN_COMMAND_INITIAL_SEMAPHORE_CAPACITY, host_allocator,
          &program->wait_semaphores, &program->wait_values);
    }
    if (iree_status_is_ok(status)) {
      program->wait_capacity = QWEN_COMMAND_INITIAL_SEMAPHORE_CAPACITY;
      status = qwen_command_reserve_semaphore_storage(
          QWEN_COMMAND_INITIAL_SEMAPHORE_CAPACITY, host_allocator,
          &program->signal_semaphores, &program->signal_values);
    }
    if (iree_status_is_ok(status)) {
      program->signal_capacity = QWEN_COMMAND_INITIAL_SEMAPHORE_CAPACITY;
      status = iree_hal_semaphore_create(
          qwen_model_device(model), qwen_model_queue_affinity(model),
          /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
          &program->timeline_semaphore);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_package = package;
    package = NULL;
  }

  if (package) qwen_command_package_destroy(package);
  for (iree_host_size_t i = 0; i < QWEN_COMMAND_PROGRAM_COUNT; ++i) {
    loomc_cmd_hal_program_release(hal_programs[i]);
  }
  loomc_cmd_program_package_release(command_program_package);
  if (unit_executables) {
    for (iree_host_size_t i = 0; i < unit_count; ++i) {
      iree_hal_executable_release(unit_executables[i]);
    }
  }
  iree_allocator_free(host_allocator, unit_executables);
  qwen_command_release_unit_results(unit_count, unit_results, host_allocator);
  iree_allocator_free(host_allocator, unit_kinds);
  loomc_program_plan_release(plan);
  iree_allocator_free(host_allocator, specializations);
  iree_allocator_free(host_allocator, kernel_functions);
  loomc_pass_program_release(executable_pass_program);
  loomc_pass_program_release(empty_pass_program);
  loomc_pass_program_release(preparation_pass_program);
  loomc_compiler_release(compiler);
  loomc_target_profile_release(target_profile);
  loomc_module_release(module);
  loomc_source_release(source);
  loomc_workspace_release(coordinator_workspace);
  loomc_context_release(context);
  loomc_target_environment_release(target_environment);
  return status;
}
void qwen_command_package_retain(qwen_command_package_t* package) {
  if (package) iree_atomic_ref_count_inc(&package->ref_count);
}

void qwen_command_package_release(qwen_command_package_t* package) {
  if (package && iree_atomic_ref_count_dec(&package->ref_count) == 1) {
    qwen_command_package_destroy(package);
  }
}

static iree_status_t qwen_command_build_issue_wait_list(
    qwen_command_package_t* package, qwen_command_program_state_t* program,
    qwen_request_t* request, iree_hal_semaphore_list_t caller_waits,
    iree_hal_semaphore_list_t* out_waits) {
  const iree_hal_semaphore_list_t model_ready =
      qwen_model_ready_semaphore_list(package->model);
  if (caller_waits.count > IREE_HOST_SIZE_MAX - model_ready.count - 1) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen issue wait count overflows");
  }
  const iree_host_size_t wait_count =
      caller_waits.count + model_ready.count + 1;
  IREE_RETURN_IF_ERROR(qwen_command_ensure_semaphore_storage(
      wait_count, package->host_allocator, &program->wait_semaphores,
      &program->wait_values, &program->wait_capacity));

  iree_host_size_t cursor = 0;
  for (iree_host_size_t i = 0; i < model_ready.count; ++i) {
    program->wait_semaphores[cursor] = model_ready.semaphores[i];
    program->wait_values[cursor] = model_ready.payload_values[i];
    ++cursor;
  }
  program->wait_semaphores[cursor] = qwen_request_timeline_semaphore(request);
  program->wait_values[cursor] = qwen_request_timeline_value(request);
  ++cursor;
  for (iree_host_size_t i = 0; i < caller_waits.count; ++i) {
    program->wait_semaphores[cursor] = caller_waits.semaphores[i];
    program->wait_values[cursor] = caller_waits.payload_values[i];
    ++cursor;
  }
  *out_waits = (iree_hal_semaphore_list_t){
      .count = cursor,
      .semaphores = program->wait_semaphores,
      .payload_values = program->wait_values,
  };
  return iree_ok_status();
}

static iree_status_t qwen_command_build_issue_signal_list(
    qwen_command_package_t* package, qwen_command_program_state_t* program,
    qwen_request_t* request, uint64_t program_completion_value,
    uint64_t request_completion_value, iree_hal_semaphore_list_t caller_signals,
    iree_hal_semaphore_list_t* out_signals) {
  if (caller_signals.count > IREE_HOST_SIZE_MAX - 2) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen issue signal count overflows");
  }
  const iree_host_size_t signal_count = caller_signals.count + 2;
  IREE_RETURN_IF_ERROR(qwen_command_ensure_semaphore_storage(
      signal_count, package->host_allocator, &program->signal_semaphores,
      &program->signal_values, &program->signal_capacity));

  program->signal_semaphores[0] = program->timeline_semaphore;
  program->signal_values[0] = program_completion_value;
  program->signal_semaphores[1] = qwen_request_timeline_semaphore(request);
  program->signal_values[1] = request_completion_value;
  for (iree_host_size_t i = 0; i < caller_signals.count; ++i) {
    program->signal_semaphores[i + 2] = caller_signals.semaphores[i];
    program->signal_values[i + 2] = caller_signals.payload_values[i];
  }
  *out_signals = (iree_hal_semaphore_list_t){
      .count = signal_count,
      .semaphores = program->signal_semaphores,
      .payload_values = program->signal_values,
  };
  return iree_ok_status();
}

static void qwen_command_fail_after_partial_submission(
    qwen_command_program_state_t* program, qwen_request_t* request,
    iree_status_t status) {
  qwen_request_fail(request, iree_status_clone(status));
  iree_hal_semaphore_fail(program->timeline_semaphore,
                          iree_status_clone(status));
}

static iree_status_t qwen_command_validate_program_request(
    qwen_command_package_t* package, qwen_command_program_t program,
    qwen_request_t* request, qwen_command_program_state_t** out_program_state) {
  *out_program_state = NULL;
  if (!package || !request) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen command package and request are required");
  }
  if ((uint32_t)program >= QWEN_COMMAND_PROGRAM_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen command program %d is unknown", program);
  }
  if (qwen_request_model(request) != package->model ||
      qwen_request_token_capacity(request) != package->request_token_capacity ||
      qwen_request_context_capacity(request) != package->context_capacity ||
      qwen_request_flags(request) != package->request_flags) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen request model, storage capacities, and flags must match the "
        "prepared command package");
  }
  const iree_host_size_t context_base = qwen_request_context_base(request);
  const iree_host_size_t active_token_count =
      qwen_request_active_token_count(request);
  const qwen_command_program_descriptor_t* descriptor =
      &qwen_command_program_descriptors[program];
  const bool context_base_is_valid =
      (descriptor->context_kind == QWEN_COMMAND_CONTEXT_KIND_FIXED_INITIAL &&
       context_base == 0) ||
      (descriptor->context_kind == QWEN_COMMAND_CONTEXT_KIND_CAPACITY_PREFIX &&
       context_base < package->context_capacity);
  if (active_token_count != descriptor->token_count || !context_base_is_valid) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen command program '%s' requires %" PRIhsz
        " tokens and a context base valid for its prepared context contract",
        descriptor->export_name, descriptor->token_count);
  }
  if (qwen_request_input_kind(request) != QWEN_REQUEST_INPUT_KIND_TOKEN_IDS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen command programs require token-ID request state");
  }

  qwen_command_program_state_t* program_state = &package->programs[program];
  uint64_t completed_program_value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_query(
      program_state->timeline_semaphore, &completed_program_value));
  if (completed_program_value < program_state->timeline_value) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen command program already has an in-flight issue at value %" PRIu64,
        program_state->timeline_value);
  }

  *out_program_state = program_state;
  return iree_ok_status();
}

static iree_hal_buffer_binding_table_t qwen_command_build_binding_table(
    const qwen_command_program_state_t* program, qwen_request_t* request,
    iree_hal_buffer_t* transient_buffer,
    iree_device_size_t transient_byte_length,
    iree_hal_buffer_binding_t out_bindings[QWEN_COMMAND_BINDING_CAPACITY]) {
  const qwen_request_storage_layout_t* request_layout =
      qwen_request_storage_layout(request);
  out_bindings[QWEN_COMMAND_BINDING_KEY_CACHE] = (iree_hal_buffer_binding_t){
      .buffer = qwen_request_storage_buffer(request),
      .offset = request_layout->key_cache.offset,
      .length = request_layout->key_cache.length,
  };
  out_bindings[QWEN_COMMAND_BINDING_VALUE_CACHE] = (iree_hal_buffer_binding_t){
      .buffer = qwen_request_storage_buffer(request),
      .offset = request_layout->value_cache.offset,
      .length = request_layout->value_cache.length,
  };
  out_bindings[QWEN_COMMAND_BINDING_REQUEST_STATE] =
      (iree_hal_buffer_binding_t){
          .buffer = qwen_request_storage_buffer(request),
          .offset = 0,
          .length = request_layout->dispatch_state_byte_length,
      };
  out_bindings[QWEN_COMMAND_BINDING_OUTPUT_STAGING] =
      (iree_hal_buffer_binding_t){
          .buffer = qwen_request_output_staging_buffer(request),
          .offset = 0,
          .length = sizeof(int32_t),
      };
  out_bindings[QWEN_COMMAND_BINDING_TRANSIENT] = (iree_hal_buffer_binding_t){
      .buffer = transient_buffer,
      .offset = 0,
      .length = transient_byte_length,
  };
  if (program->launch.buffer) {
    out_bindings[program->launch.binding_index] = (iree_hal_buffer_binding_t){
        .buffer = program->launch.buffer,
        .offset = 0,
        .length = program->launch.byte_length,
    };
  }
  return (iree_hal_buffer_binding_table_t){
      .count = program->binding_count,
      .bindings = out_bindings,
  };
}

iree_status_t qwen_command_package_query_program(
    const qwen_command_package_t* package, qwen_command_program_t program,
    qwen_command_program_info_t* out_info) {
  if (!package || !out_info) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen command package and output program info are required");
  }
  if ((uint32_t)program >= QWEN_COMMAND_PROGRAM_COUNT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen command program %d is unknown", program);
  }
  if (out_info->structure_size < sizeof(*out_info) || out_info->next) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen command program info structure is invalid");
  }
  const qwen_command_program_state_t* program_state =
      &package->programs[program];
  *out_info = (qwen_command_program_info_t){
      .structure_size = sizeof(*out_info),
      .next = NULL,
      .token_count = program_state->descriptor->token_count,
      .context_count = program_state->descriptor->context_kind ==
                               QWEN_COMMAND_CONTEXT_KIND_FIXED_INITIAL
                           ? program_state->descriptor->fixed_context_count
                           : package->context_capacity,
      .transient_byte_length = program_state->transient_byte_length,
      .transient_minimum_alignment = program_state->transient_minimum_alignment,
  };
  return iree_ok_status();
}

iree_status_t qwen_command_package_capture_transient_prefix(
    qwen_command_package_t* package, qwen_command_program_t program,
    qwen_request_t* request, uint32_t first_excluded_command,
    iree_byte_span_t transient_capture) {
  (void)package;
  (void)program;
  (void)request;
  (void)first_excluded_command;
  (void)transient_capture;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "Qwen command prefix capture requires portable package range "
      "materialization");
}
iree_status_t qwen_command_package_issue(
    qwen_command_package_t* package, qwen_command_program_t program,
    qwen_request_t* request, const qwen_command_issue_options_t* options,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  IREE_RETURN_IF_ERROR(qwen_command_validate_issue_options(options));
  if (options &&
      iree_any_bit_set(options->flags,
                       QWEN_COMMAND_ISSUE_FLAG_DIAGNOSTIC_BARRIER_WAVES)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Qwen barrier-wave issue requires portable package range "
        "materialization");
  }
  qwen_command_program_state_t* program_state = NULL;
  IREE_RETURN_IF_ERROR(qwen_command_validate_program_request(
      package, program, request, &program_state));
  const iree_host_size_t context_base = qwen_request_context_base(request);
  IREE_RETURN_IF_ERROR(
      qwen_command_evaluate_launch(program_state, context_base));

  const uint64_t program_timeline_advance = 4;
  if (program_state->timeline_value >
          IREE_HAL_SEMAPHORE_MAX_VALUE - program_timeline_advance ||
      qwen_request_timeline_value(request) == IREE_HAL_SEMAPHORE_MAX_VALUE) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Qwen issue timeline is exhausted");
  }

  iree_hal_semaphore_list_t alloca_waits;
  iree_status_t status = qwen_command_build_issue_wait_list(
      package, program_state, request, wait_semaphore_list, &alloca_waits);
  if (!iree_status_is_ok(status)) return status;

  uint64_t scratch_ready_value = program_state->timeline_value + 1;
  uint64_t execute_ready_value = program_state->timeline_value + 2;
  uint64_t execute_complete_value =
      program_state->timeline_value + program_timeline_advance - 1;
  const uint64_t program_complete_value =
      program_state->timeline_value + program_timeline_advance;
  const uint64_t request_complete_value =
      qwen_request_timeline_value(request) + 1;
  iree_hal_semaphore_t* program_timeline = program_state->timeline_semaphore;
  const iree_hal_semaphore_list_t scratch_ready = {
      .count = 1,
      .semaphores = &program_timeline,
      .payload_values = &scratch_ready_value,
  };
  const iree_hal_semaphore_list_t execute_ready = {
      .count = 1,
      .semaphores = &program_timeline,
      .payload_values = &execute_ready_value,
  };
  const iree_hal_semaphore_list_t execute_complete = {
      .count = 1,
      .semaphores = &program_timeline,
      .payload_values = &execute_complete_value,
  };
  iree_hal_semaphore_list_t completion_signals;
  status = qwen_command_build_issue_signal_list(
      package, program_state, request, program_complete_value,
      request_complete_value, signal_semaphore_list, &completion_signals);
  if (!iree_status_is_ok(status)) return status;

  iree_hal_device_t* device = qwen_model_device(package->model);
  const iree_hal_queue_affinity_t queue_affinity =
      qwen_model_queue_affinity(package->model);
  const iree_hal_buffer_params_t transient_params = {
      .usage =
          IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_affinity = queue_affinity,
      .min_alignment = program_state->transient_minimum_alignment,
  };
  iree_hal_buffer_t* transient_buffer = NULL;
  status = iree_hal_device_queue_alloca(
      device, queue_affinity, alloca_waits, scratch_ready, /*pool=*/NULL,
      transient_params, program_state->transient_byte_length,
      IREE_HAL_ALLOCA_FLAG_NONE, &transient_buffer);
  if (!iree_status_is_ok(status)) return status;

  const uint32_t zero_pattern = 0;
  status = iree_hal_device_queue_fill(
      device, queue_affinity, scratch_ready, execute_ready, transient_buffer,
      /*target_offset=*/0, program_state->transient_byte_length, &zero_pattern,
      sizeof(zero_pattern), IREE_HAL_FILL_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_status_t cleanup_status = iree_hal_device_queue_dealloca(
        device, queue_affinity, scratch_ready, iree_hal_semaphore_list_empty(),
        transient_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
    iree_hal_buffer_release(transient_buffer);
    qwen_command_fail_after_partial_submission(program_state, request, status);
    return iree_status_join(status, cleanup_status);
  }

  iree_hal_buffer_binding_t bindings[QWEN_COMMAND_BINDING_CAPACITY];
  const iree_hal_buffer_binding_table_t binding_table =
      qwen_command_build_binding_table(program_state, request, transient_buffer,
                                       program_state->transient_byte_length,
                                       bindings);
  status = iree_hal_device_queue_execute(
      device, queue_affinity, execute_ready, execute_complete,
      loomc_cmd_hal_program_command_buffer(program_state->hal_program),
      binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_status_t cleanup_status = iree_hal_device_queue_dealloca(
        device, queue_affinity, execute_ready, iree_hal_semaphore_list_empty(),
        transient_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
    iree_hal_buffer_release(transient_buffer);
    qwen_command_fail_after_partial_submission(program_state, request, status);
    return iree_status_join(status, cleanup_status);
  }

  status = iree_hal_device_queue_dealloca(
      device, queue_affinity, execute_complete, completion_signals,
      transient_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
  iree_hal_buffer_release(transient_buffer);
  if (!iree_status_is_ok(status)) {
    qwen_command_fail_after_partial_submission(program_state, request, status);
    return status;
  }

  program_state->timeline_value = program_complete_value;
  qwen_request_commit_selected_token_signal(
      request, request_complete_value,
      context_base + program_state->descriptor->token_count);
  return iree_ok_status();
}
