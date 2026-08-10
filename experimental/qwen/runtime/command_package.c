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
#include "loomc/target/cmd.h"
#include "loomc/target/cmd/iree_hal.h"

#define QWEN_COMMAND_PREFILL_TOKEN_COUNT 512
#define QWEN_COMMAND_CONTEXT_CAPACITY 576
#define QWEN_COMMAND_FIXED_BUFFER_COUNT 2
#define QWEN_COMMAND_INITIAL_SEMAPHORE_CAPACITY 8

typedef enum qwen_command_binding_e {
  QWEN_COMMAND_BINDING_KEY_CACHE = 0,
  QWEN_COMMAND_BINDING_VALUE_CACHE = 1,
  QWEN_COMMAND_BINDING_REQUEST_STATE = 2,
  QWEN_COMMAND_BINDING_OUTPUT_STAGING = 3,
  QWEN_COMMAND_BINDING_TRANSIENT = 4,
  QWEN_COMMAND_BINDING_COUNT = 5,
} qwen_command_binding_t;

typedef struct qwen_command_parameter_t {
  // Stable fixed-schema key.
  char key[QWEN_PARAMETER_KEY_CAPACITY];
  // Exact model-slab placement.
  iree_io_parameter_span_t span;
  // True after one compiled requirement claims this parameter.
  bool matched;
} qwen_command_parameter_t;

typedef struct qwen_command_program_descriptor_t {
  // Public semantic program identity.
  qwen_command_program_t program;
  // Compiled public command-program export name.
  const char* export_name;
  // Exact active token-row count.
  iree_host_size_t token_count;
  // Exact attention context extent.
  iree_host_size_t context_count;
} qwen_command_program_descriptor_t;

static const qwen_command_program_descriptor_t
    qwen_command_program_descriptors[QWEN_COMMAND_PROGRAM_COUNT] = {
        [QWEN_COMMAND_PROGRAM_PREFILL_512] =
            {
                .program = QWEN_COMMAND_PROGRAM_PREFILL_512,
                .export_name = "qwen3_30b_prefill_512",
                .token_count = QWEN_COMMAND_PREFILL_TOKEN_COUNT,
                .context_count = QWEN_COMMAND_PREFILL_TOKEN_COUNT,
            },
        [QWEN_COMMAND_PROGRAM_DECODE_576] =
            {
                .program = QWEN_COMMAND_PROGRAM_DECODE_576,
                .export_name = "qwen3_30b_decode_576",
                .token_count = 1,
                .context_count = QWEN_COMMAND_CONTEXT_CAPACITY,
            },
};

typedef struct qwen_command_program_state_t {
  // Static descriptor for this materialized root.
  const qwen_command_program_descriptor_t* descriptor;
  // Selected portable command-program root retained for diagnostic ranges.
  loomc_cmd_program_t* command_program;
  // Materialized reusable command program.
  loomc_cmd_iree_hal_program_t* hal_program;
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
  // Loaded executable package shared by full and diagnostic materializations.
  loomc_cmd_iree_hal_package_t* hal_package;
  // HAL recording mode used by every materialized command buffer.
  iree_hal_command_buffer_mode_t command_buffer_mode;
  // Materialized program state indexed by qwen_command_program_t.
  qwen_command_program_state_t programs[QWEN_COMMAND_PROGRAM_COUNT];
};

typedef struct qwen_command_program_segment_t {
  // Canonical barrier wave recorded into this segment.
  loomc_cmd_program_barrier_wave_t barrier_wave;
  // Materialized command buffer containing exactly |barrier_wave|.
  loomc_cmd_iree_hal_program_t* hal_program;
} qwen_command_program_segment_t;

typedef struct qwen_command_compile_batch_t {
  // Immutable production plan shared by every independent job.
  loomc_program_plan_t* plan;
  // Unit compilation extensions shared by every independent job.
  const loomc_program_plan_unit_compile_options_t* compile_options;
  // Worker-local mutable workspaces indexed by worker ordinal.
  loomc_workspace_t** worker_workspaces;
  // Dense unit outputs indexed by job and plan-unit ordinal.
  loomc_program_t** unit_programs;
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
    iree_allocator_free(host_allocator, program->signal_values);
    iree_allocator_free(host_allocator, program->signal_semaphores);
    iree_allocator_free(host_allocator, program->wait_values);
    iree_allocator_free(host_allocator, program->wait_semaphores);
    iree_hal_semaphore_release(program->timeline_semaphore);
    loomc_cmd_iree_hal_program_release(program->hal_program);
    loomc_cmd_program_release(program->command_program);
  }
  loomc_cmd_iree_hal_package_release(package->hal_package);
  qwen_model_release(package->model);
  iree_allocator_free(host_allocator, package);
}

void qwen_command_package_options_initialize(
    qwen_command_package_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (qwen_command_package_options_t){
      .structure_size = sizeof(*out_options),
      .next = NULL,
      .request_token_capacity = QWEN_COMMAND_PREFILL_TOKEN_COUNT,
      .context_capacity = QWEN_COMMAND_CONTEXT_CAPACITY,
      .request_flags = QWEN_REQUEST_FLAG_NONE,
      .sanitizer_checks = 0,
      .compiler_worker_count = QWEN_LOOM_JIT_DEFAULT_WORKER_COUNT,
      .command_buffer_mode = IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
  };
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
  if (options->request_token_capacity < QWEN_COMMAND_PREFILL_TOKEN_COUNT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen command package requires at least %d request token rows",
        QWEN_COMMAND_PREFILL_TOKEN_COUNT);
  }
  if (options->context_capacity != QWEN_COMMAND_CONTEXT_CAPACITY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen command package requires context capacity %d",
                            QWEN_COMMAND_CONTEXT_CAPACITY);
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
  loomc_result_t* result = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_program_plan_compile_unit(
      batch->plan, batch->worker_workspaces[worker_ordinal],
      loomc_program_plan_unit_from_index((uint32_t)job_ordinal),
      batch->compile_options, batch->allocator,
      &batch->unit_programs[job_ordinal], &result));
  if (iree_status_is_ok(status)) {
    status = qwen_command_require_result(IREE_SV("unit compile"), result);
  }
  loomc_result_release(result);
  return status;
}

static iree_status_t qwen_command_compile_units(
    loomc_program_plan_t* plan, iree_host_size_t requested_worker_count,
    const loomc_program_plan_unit_compile_options_t* compile_options,
    iree_allocator_t host_allocator, loomc_program_t*** out_unit_programs,
    iree_host_size_t* out_unit_count) {
  *out_unit_programs = NULL;
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
                                         sizeof(**out_unit_programs),
                                         (void**)out_unit_programs);
  }
  if (iree_status_is_ok(status)) {
    memset(*out_unit_programs, 0,
           *out_unit_count * sizeof(**out_unit_programs));
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
        .compile_options = compile_options,
        .worker_workspaces = worker_workspaces,
        .unit_programs = *out_unit_programs,
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
    if (*out_unit_programs) {
      for (iree_host_size_t i = 0; i < *out_unit_count; ++i) {
        loomc_program_release((*out_unit_programs)[i]);
      }
    }
    iree_allocator_free(host_allocator, *out_unit_programs);
    *out_unit_programs = NULL;
    *out_unit_count = 0;
  }
  return status;
}

static void qwen_command_release_units(iree_host_size_t unit_count,
                                       loomc_program_t** unit_programs,
                                       iree_allocator_t host_allocator) {
  if (unit_programs) {
    for (iree_host_size_t i = 0; i < unit_count; ++i) {
      loomc_program_release(unit_programs[i]);
    }
  }
  iree_allocator_free(host_allocator, unit_programs);
}

static iree_status_t qwen_command_validate_parameter_layout(
    qwen_model_t* model, const loomc_cmd_program_t* command_program,
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
    status = iree_status_from_loomc(
        loomc_cmd_program_parameter_info(command_program, i, &parameter));
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
    status = iree_status_from_loomc(
        loomc_cmd_program_parameter_root_info(command_program, 0, &main_root));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_cmd_program_parameter_root_info(
        command_program, 1, &auxiliary_root));
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
  loomc_program_environment_t* program_environment = NULL;
  loomc_compiler_t* compiler = NULL;
  loomc_pass_program_t* preparation_pass_program = NULL;
  loomc_pass_program_t* unit_pass_program = NULL;
  loomc_module_function_t* kernel_functions = NULL;
  loomc_target_specialization_t* specializations = NULL;
  iree_host_size_t kernel_function_count = 0;
  loomc_program_plan_t* plan = NULL;
  loomc_program_t** unit_programs = NULL;
  iree_host_size_t unit_count = 0;
  loomc_program_t* assembled_program = NULL;
  loomc_program_plan_root_t roots[QWEN_COMMAND_PROGRAM_COUNT];
  loomc_cmd_program_t* command_programs[QWEN_COMMAND_PROGRAM_COUNT];
  loomc_cmd_program_info_t command_program_infos[QWEN_COMMAND_PROGRAM_COUNT];
  iree_hal_buffer_ref_t fixed_buffers[QWEN_COMMAND_PROGRAM_COUNT]
                                     [QWEN_COMMAND_FIXED_BUFFER_COUNT];
  loomc_cmd_iree_hal_package_t* hal_package = NULL;
  loomc_cmd_iree_hal_program_t* hal_programs[QWEN_COMMAND_PROGRAM_COUNT];
  loomc_result_t* result = NULL;
  memset(roots, 0, sizeof(roots));
  memset(command_programs, 0, sizeof(command_programs));
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
    status = iree_status_from_loomc(loomc_program_environment_create_command(
        loom_allocator, &program_environment));
  }
  if (iree_status_is_ok(status)) {
    const loomc_compiler_program_options_t program_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILER_PROGRAM_OPTIONS,
        .structure_size = sizeof(program_options),
        .next = NULL,
        .program_environment = program_environment,
    };
    const loomc_compiler_options_t compiler_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILER_OPTIONS,
        .structure_size = sizeof(compiler_options),
        .next = &program_options,
    };
    status = iree_status_from_loomc(loomc_compiler_create(
        context, &compiler_options, loom_allocator, &compiler));
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
            context, &pipeline_options, loom_allocator, &unit_pass_program,
            &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_command_require_result(IREE_SV("unit pipeline"), result);
  }
  loomc_result_release(result);
  result = NULL;

  char request_token_capacity_storage[32];
  const int request_token_capacity_length = snprintf(
      request_token_capacity_storage, sizeof(request_token_capacity_storage),
      "%" PRIhsz, options->request_token_capacity);
  if (iree_status_is_ok(status) &&
      (request_token_capacity_length < 0 ||
       (iree_host_size_t)request_token_capacity_length >=
           sizeof(request_token_capacity_storage))) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Qwen token capacity spelling overflowed");
  }
  if (iree_status_is_ok(status)) {
    const loomc_config_binding_t config_bindings[] = {
        {
            .key = loomc_make_cstring_view("qwen3_30b.request.token_capacity"),
            .value = loomc_make_string_view(
                request_token_capacity_storage,
                (loomc_host_size_t)request_token_capacity_length),
        },
    };
    const loomc_target_specialization_options_t target_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
        .structure_size = sizeof(target_options),
        .next = NULL,
        .specializations = specializations,
        .specialization_count = kernel_function_count,
    };
    const loomc_cmd_program_plan_options_t command_options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PLAN_OPTIONS,
        .structure_size = sizeof(command_options),
        .next = NULL,
        .dependency_artifact_format =
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
    };
    const loomc_program_plan_options_t plan_options = {
        .type = LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS,
        .structure_size = sizeof(plan_options),
        .next = &command_options,
        .config =
            {
                .bindings = config_bindings,
                .binding_count = IREE_ARRAYSIZE(config_bindings),
                .json_object = {0},
                .flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
            },
        .target_specialization = &target_options,
    };
    status = iree_status_from_loomc(loomc_prepare_programs(
        compiler, coordinator_workspace, preparation_pass_program,
        unit_pass_program, module, &plan_options, loom_allocator, &plan,
        &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_command_require_result(IREE_SV("program plan"), result);
  }
  loomc_result_release(result);
  result = NULL;

  for (iree_host_size_t i = 0;
       i < QWEN_COMMAND_PROGRAM_COUNT && iree_status_is_ok(status); ++i) {
    roots[i] = loomc_program_plan_root_invalid();
    status = iree_status_from_loomc(loomc_program_plan_lookup_root(
        plan,
        loomc_make_cstring_view(
            qwen_command_program_descriptors[i].export_name),
        &roots[i]));
  }
  if (iree_status_is_ok(status)) {
    const loomc_amdgpu_runtime_global_flags_t runtime_globals =
        options->sanitizer_checks
            ? LOOMC_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG |
                  LOOMC_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG
            : LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE;
    const loomc_amdgpu_emit_options_t amdgpu_emit_options = {
        .type = LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
        .structure_size = sizeof(amdgpu_emit_options),
        .next = NULL,
        .runtime_globals = runtime_globals,
    };
    const loomc_program_plan_unit_compile_options_t unit_compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_COMPILE_OPTIONS,
        .structure_size = sizeof(unit_compile_options),
        .next = options->sanitizer_checks ? &amdgpu_emit_options : NULL,
    };
    status = qwen_command_compile_units(plan, options->compiler_worker_count,
                                        &unit_compile_options, host_allocator,
                                        &unit_programs, &unit_count);
  }
  if (iree_status_is_ok(status)) {
    const loomc_program_plan_unit_table_t unit_table = {
        .type = LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_TABLE,
        .structure_size = sizeof(unit_table),
        .next = NULL,
        .programs = unit_programs,
        .program_count = unit_count,
    };
    status = iree_status_from_loomc(loomc_program_plan_assemble(
        plan, coordinator_workspace, roots, QWEN_COMMAND_PROGRAM_COUNT,
        &unit_table, /*options=*/NULL, loom_allocator, &assembled_program,
        &result));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_command_require_result(IREE_SV("program assembly"), result);
  }
  loomc_result_release(result);
  result = NULL;

  for (iree_host_size_t i = 0;
       i < QWEN_COMMAND_PROGRAM_COUNT && iree_status_is_ok(status); ++i) {
    loomc_program_export_t root_export = loomc_program_export_invalid();
    const qwen_command_program_descriptor_t* descriptor =
        &qwen_command_program_descriptors[i];
    status = iree_status_from_loomc(loomc_program_lookup_export(
        assembled_program, loomc_make_cstring_view(descriptor->export_name),
        &root_export));
    if (iree_status_is_ok(status)) {
      status = iree_status_from_loomc(loomc_cmd_program_create_from_export(
          assembled_program, root_export, loom_allocator,
          &command_programs[i]));
    }
    command_program_infos[i] = (loomc_cmd_program_info_t){
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO,
        .structure_size = sizeof(command_program_infos[i]),
    };
    if (iree_status_is_ok(status)) {
      status = iree_status_from_loomc(loomc_cmd_program_info(
          command_programs[i], &command_program_infos[i]));
    }
    const loomc_cmd_program_info_t* info = &command_program_infos[i];
    if (iree_status_is_ok(status) &&
        (info->fixed_buffer_count != QWEN_COMMAND_FIXED_BUFFER_COUNT ||
         info->rebindable_binding_count != QWEN_COMMAND_BINDING_COUNT ||
         info->parameter_root_count != 2 ||
         info->parameter_count != QWEN_PARAMETER_COUNT + 1 ||
         info->transient.binding_index != QWEN_COMMAND_BINDING_TRANSIENT ||
         info->transient.required_byte_length == 0 ||
         info->transient.minimum_alignment == 0 ||
         info->launch_counts.binding_index !=
             LOOMC_CMD_PROGRAM_BINDING_INVALID)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "compiled Qwen command program '%s' has an incompatible ABI",
          descriptor->export_name);
    }
    if (iree_status_is_ok(status)) {
      status = qwen_command_validate_parameter_layout(
          model, command_programs[i], info, host_allocator, fixed_buffers[i]);
    }
  }

  if (iree_status_is_ok(status)) {
    const iree_hal_executable_target_selection_t target_selection = {
        .family = IREE_SV("amdgpu"),
        .target_key = {0},
        .kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
        .physical_device_affinity = 0,
    };
    const loomc_cmd_iree_hal_package_options_t package_options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_IREE_HAL_PACKAGE_OPTIONS,
        .structure_size = sizeof(package_options),
        .next = NULL,
        .device = qwen_model_device(model),
        .executable_queue_affinity = qwen_model_queue_affinity(model),
        .target_selection = target_selection,
        .executable_artifact_format =
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
    };
    status = iree_status_from_loomc(loomc_cmd_iree_hal_package_create(
        assembled_program, &package_options, loom_allocator, &hal_package));
  }
  for (iree_host_size_t i = 0;
       i < QWEN_COMMAND_PROGRAM_COUNT && iree_status_is_ok(status); ++i) {
    const loomc_cmd_iree_hal_program_options_t program_options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_IREE_HAL_PROGRAM_OPTIONS,
        .structure_size = sizeof(program_options),
        .next = NULL,
        .command_buffer_mode = options->command_buffer_mode,
        .queue_affinity = qwen_model_queue_affinity(model),
        .fixed_buffers = fixed_buffers[i],
        .fixed_buffer_count = IREE_ARRAYSIZE(fixed_buffers[i]),
    };
    status = iree_status_from_loomc(loomc_cmd_iree_hal_program_create(
        hal_package, command_programs[i], &program_options, loom_allocator,
        &hal_programs[i]));
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
    package->hal_package = hal_package;
    hal_package = NULL;
    package->command_buffer_mode = options->command_buffer_mode;
  }
  for (iree_host_size_t i = 0;
       i < QWEN_COMMAND_PROGRAM_COUNT && iree_status_is_ok(status); ++i) {
    qwen_command_program_state_t* program = &package->programs[i];
    program->descriptor = &qwen_command_program_descriptors[i];
    program->command_program = command_programs[i];
    command_programs[i] = NULL;
    program->hal_program = hal_programs[i];
    hal_programs[i] = NULL;
    memcpy(program->fixed_buffers, fixed_buffers[i],
           sizeof(program->fixed_buffers));
    program->transient_byte_length =
        command_program_infos[i].transient.required_byte_length;
    program->transient_minimum_alignment =
        command_program_infos[i].transient.minimum_alignment;
    status = qwen_command_reserve_semaphore_storage(
        QWEN_COMMAND_INITIAL_SEMAPHORE_CAPACITY, host_allocator,
        &program->wait_semaphores, &program->wait_values);
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
    loomc_cmd_iree_hal_program_release(hal_programs[i]);
  }
  loomc_cmd_iree_hal_package_release(hal_package);
  for (iree_host_size_t i = 0; i < QWEN_COMMAND_PROGRAM_COUNT; ++i) {
    loomc_cmd_program_release(command_programs[i]);
  }
  loomc_program_release(assembled_program);
  qwen_command_release_units(unit_count, unit_programs, host_allocator);
  loomc_program_plan_release(plan);
  iree_allocator_free(host_allocator, specializations);
  iree_allocator_free(host_allocator, kernel_functions);
  loomc_pass_program_release(unit_pass_program);
  loomc_pass_program_release(preparation_pass_program);
  loomc_compiler_release(compiler);
  loomc_program_environment_release(program_environment);
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

static void qwen_command_release_program_segments(
    iree_host_size_t segment_count, qwen_command_program_segment_t* segments,
    iree_allocator_t host_allocator) {
  for (iree_host_size_t i = 0; i < segment_count; ++i) {
    loomc_cmd_iree_hal_program_release(segments[i].hal_program);
  }
  iree_allocator_free(host_allocator, segments);
}

static iree_status_t qwen_command_materialize_barrier_waves(
    qwen_command_package_t* package, qwen_command_program_state_t* program,
    qwen_command_program_segment_t** out_segments,
    iree_host_size_t* out_segment_count) {
  *out_segments = NULL;
  *out_segment_count = 0;

  loomc_cmd_program_barrier_wave_iterator_t iterator = {0};
  iree_status_t status =
      iree_status_from_loomc(loomc_cmd_program_barrier_wave_iterator_initialize(
          program->command_program, &iterator));
  iree_host_size_t segment_capacity = 0;
  while (iree_status_is_ok(status)) {
    loomc_cmd_program_barrier_wave_t barrier_wave = {0};
    bool has_wave = false;
    status =
        iree_status_from_loomc(loomc_cmd_program_barrier_wave_iterator_next(
            &iterator, &barrier_wave, &has_wave));
    if (!iree_status_is_ok(status) || !has_wave) break;

    if (*out_segment_count == segment_capacity) {
      status = iree_allocator_grow_array(
          package->host_allocator, *out_segment_count + 1,
          sizeof(**out_segments), &segment_capacity, (void**)out_segments);
      if (!iree_status_is_ok(status)) break;
    }

    qwen_command_program_segment_t* segment =
        &(*out_segments)[*out_segment_count];
    *segment = (qwen_command_program_segment_t){
        .barrier_wave = barrier_wave,
        .hal_program = NULL,
    };
    const loomc_cmd_program_range_options_t range_options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_RANGE_OPTIONS,
        .structure_size = sizeof(range_options),
        .next = NULL,
        .command_range = barrier_wave.commands,
    };
    const loomc_cmd_iree_hal_program_options_t program_options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_IREE_HAL_PROGRAM_OPTIONS,
        .structure_size = sizeof(program_options),
        .next = &range_options,
        .command_buffer_mode = package->command_buffer_mode,
        .queue_affinity = qwen_model_queue_affinity(package->model),
        .fixed_buffers = program->fixed_buffers,
        .fixed_buffer_count = IREE_ARRAYSIZE(program->fixed_buffers),
        .flags = LOOMC_CMD_IREE_HAL_PROGRAM_FLAG_RETAIN_RECORDED_OPERATIONS,
    };
    status = iree_status_from_loomc(loomc_cmd_iree_hal_program_create(
        package->hal_package, program->command_program, &program_options,
        loomc_allocator_from_iree(package->host_allocator),
        &segment->hal_program));
    if (iree_status_is_ok(status)) ++*out_segment_count;
  }

  if (iree_status_is_ok(status) && *out_segment_count == 0) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen command program contains no non-empty barrier waves");
  }
  if (!iree_status_is_ok(status)) {
    qwen_command_release_program_segments(*out_segment_count, *out_segments,
                                          package->host_allocator);
    *out_segments = NULL;
    *out_segment_count = 0;
  }
  return status;
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

static void qwen_command_notify_barrier_wave(
    const qwen_command_issue_options_t* options, qwen_command_program_t program,
    const qwen_command_program_segment_t* segment,
    iree_host_size_t segment_index, iree_host_size_t segment_count,
    qwen_command_barrier_wave_event_t event) {
  if (!options->barrier_wave_observer.callback) return;
  const qwen_command_barrier_wave_event_info_t event_info = {
      .structure_size = sizeof(event_info),
      .next = NULL,
      .program = program,
      .event = event,
      .wave_index = segment_index,
      .wave_count = segment_count,
      .barrier_wave_ordinal = segment->barrier_wave.ordinal,
      .command_range = segment->barrier_wave.commands,
  };
  options->barrier_wave_observer.callback(
      options->barrier_wave_observer.user_data, &event_info);
}

static iree_status_t qwen_command_execute_barrier_waves(
    qwen_command_package_t* package, qwen_command_program_t program,
    qwen_command_program_state_t* program_state,
    const qwen_command_issue_options_t* options,
    const qwen_command_program_segment_t* segments,
    iree_host_size_t segment_count, uint64_t execute_ready_value,
    iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_device_t* device = qwen_model_device(package->model);
  const iree_hal_queue_affinity_t queue_affinity =
      qwen_model_queue_affinity(package->model);
  iree_hal_semaphore_t* program_timeline = program_state->timeline_semaphore;
  uint64_t wait_value = execute_ready_value;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < segment_count && iree_status_is_ok(status);
       ++i) {
    uint64_t signal_value = execute_ready_value + i + 1;
    const iree_hal_semaphore_list_t waits = {
        .count = 1,
        .semaphores = &program_timeline,
        .payload_values = &wait_value,
    };
    const iree_hal_semaphore_list_t signals = {
        .count = 1,
        .semaphores = &program_timeline,
        .payload_values = &signal_value,
    };
    qwen_command_notify_barrier_wave(
        options, program, &segments[i], i, segment_count,
        QWEN_COMMAND_BARRIER_WAVE_EVENT_BEFORE_EXECUTE);
    status = iree_hal_device_queue_execute(
        device, queue_affinity, waits, signals,
        loomc_cmd_iree_hal_program_command_buffer(segments[i].hal_program),
        binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_wait(program_timeline, signal_value,
                                       iree_infinite_timeout(),
                                       IREE_ASYNC_WAIT_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      qwen_command_notify_barrier_wave(
          options, program, &segments[i], i, segment_count,
          QWEN_COMMAND_BARRIER_WAVE_EVENT_COMPLETED);
      wait_value = signal_value;
    }
  }
  return status;
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
  if (program == QWEN_COMMAND_PROGRAM_PREFILL_512 &&
      (active_token_count != QWEN_COMMAND_PREFILL_TOKEN_COUNT ||
       context_base != 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen prefill command requires %d tokens at context base zero",
        QWEN_COMMAND_PREFILL_TOKEN_COUNT);
  }
  if (program == QWEN_COMMAND_PROGRAM_DECODE_576 &&
      (active_token_count != 1 ||
       context_base < QWEN_COMMAND_PREFILL_TOKEN_COUNT ||
       context_base >= QWEN_COMMAND_CONTEXT_CAPACITY)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen decode command requires one token in context class [%d, %d)",
        QWEN_COMMAND_PREFILL_TOKEN_COUNT, QWEN_COMMAND_CONTEXT_CAPACITY);
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
    qwen_request_t* request, iree_hal_buffer_t* transient_buffer,
    iree_device_size_t transient_byte_length,
    iree_hal_buffer_binding_t out_bindings[QWEN_COMMAND_BINDING_COUNT]) {
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
  return (iree_hal_buffer_binding_table_t){
      .count = QWEN_COMMAND_BINDING_COUNT,
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
      .context_count = program_state->descriptor->context_count,
      .transient_byte_length = program_state->transient_byte_length,
      .transient_minimum_alignment = program_state->transient_minimum_alignment,
  };
  return iree_ok_status();
}

iree_status_t qwen_command_package_capture_transient_prefix(
    qwen_command_package_t* package, qwen_command_program_t program,
    qwen_request_t* request, uint32_t first_excluded_command,
    iree_byte_span_t transient_capture) {
  qwen_command_program_state_t* program_state = NULL;
  IREE_RETURN_IF_ERROR(qwen_command_validate_program_request(
      package, program, request, &program_state));
  if (program_state->transient_byte_length > IREE_HOST_SIZE_MAX ||
      transient_capture.data_length != program_state->transient_byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen transient capture has %" PRIhsz
                            " bytes; expected %" PRIu64,
                            transient_capture.data_length,
                            (uint64_t)program_state->transient_byte_length);
  }
  if (transient_capture.data_length != 0 && !transient_capture.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen transient capture storage is required");
  }

  const loomc_cmd_program_range_options_t range_options = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_RANGE_OPTIONS,
      .structure_size = sizeof(range_options),
      .next = NULL,
      .command_range =
          {
              .first_command = 0,
              .command_count = first_excluded_command,
          },
  };
  const loomc_cmd_iree_hal_program_options_t program_options = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_IREE_HAL_PROGRAM_OPTIONS,
      .structure_size = sizeof(program_options),
      .next = &range_options,
      .command_buffer_mode = package->command_buffer_mode,
      .queue_affinity = qwen_model_queue_affinity(package->model),
      .fixed_buffers = program_state->fixed_buffers,
      .fixed_buffer_count = IREE_ARRAYSIZE(program_state->fixed_buffers),
      .flags = LOOMC_CMD_IREE_HAL_PROGRAM_FLAG_RETAIN_RECORDED_OPERATIONS,
  };
  loomc_cmd_iree_hal_program_t* prefix_program = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_cmd_iree_hal_program_create(
      package->hal_package, program_state->command_program, &program_options,
      loomc_allocator_from_iree(package->host_allocator), &prefix_program)));

  iree_hal_device_t* device = qwen_model_device(package->model);
  iree_hal_allocator_t* device_allocator = iree_hal_device_allocator(device);
  const iree_hal_queue_affinity_t queue_affinity =
      qwen_model_queue_affinity(package->model);
  iree_hal_semaphore_t* timeline_semaphore = NULL;
  iree_hal_buffer_t* staging_buffer = NULL;
  iree_hal_buffer_t* transient_buffer = NULL;
  bool transient_dealloca_submitted = false;
  uint64_t cleanup_wait_value = 0;

  iree_status_t status = iree_hal_semaphore_create(
      device, queue_affinity, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &timeline_semaphore);
  const iree_hal_buffer_params_t staging_params = {
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
               IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED,
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = queue_affinity,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(
        device_allocator, staging_params, program_state->transient_byte_length,
        &staging_buffer);
  }

  iree_hal_semaphore_list_t alloca_waits = iree_hal_semaphore_list_empty();
  if (iree_status_is_ok(status)) {
    status = qwen_command_build_issue_wait_list(package, program_state, request,
                                                iree_hal_semaphore_list_empty(),
                                                &alloca_waits);
  }

  uint64_t scratch_ready_value = 1;
  uint64_t execute_ready_value = 2;
  uint64_t execute_complete_value = 3;
  uint64_t capture_complete_value = 4;
  uint64_t dealloca_complete_value = 5;
  const iree_hal_semaphore_list_t scratch_ready = {
      .count = 1,
      .semaphores = &timeline_semaphore,
      .payload_values = &scratch_ready_value,
  };
  const iree_hal_semaphore_list_t execute_ready = {
      .count = 1,
      .semaphores = &timeline_semaphore,
      .payload_values = &execute_ready_value,
  };
  const iree_hal_semaphore_list_t execute_complete = {
      .count = 1,
      .semaphores = &timeline_semaphore,
      .payload_values = &execute_complete_value,
  };
  const iree_hal_semaphore_list_t capture_complete = {
      .count = 1,
      .semaphores = &timeline_semaphore,
      .payload_values = &capture_complete_value,
  };
  const iree_hal_semaphore_list_t dealloca_complete = {
      .count = 1,
      .semaphores = &timeline_semaphore,
      .payload_values = &dealloca_complete_value,
  };

  const iree_hal_buffer_params_t transient_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_affinity = queue_affinity,
      .min_alignment = program_state->transient_minimum_alignment,
  };
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_alloca(
        device, queue_affinity, alloca_waits, scratch_ready, /*pool=*/NULL,
        transient_params, program_state->transient_byte_length,
        IREE_HAL_ALLOCA_FLAG_NONE, &transient_buffer);
    if (iree_status_is_ok(status)) cleanup_wait_value = scratch_ready_value;
  }
  const uint32_t zero_pattern = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_fill(
        device, queue_affinity, scratch_ready, execute_ready, transient_buffer,
        /*target_offset=*/0, program_state->transient_byte_length,
        &zero_pattern, sizeof(zero_pattern), IREE_HAL_FILL_FLAG_NONE);
    if (iree_status_is_ok(status)) cleanup_wait_value = execute_ready_value;
  }

  iree_hal_buffer_binding_t bindings[QWEN_COMMAND_BINDING_COUNT];
  const iree_hal_buffer_binding_table_t binding_table =
      qwen_command_build_binding_table(request, transient_buffer,
                                       program_state->transient_byte_length,
                                       bindings);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_execute(
        device, queue_affinity, execute_ready, execute_complete,
        loomc_cmd_iree_hal_program_command_buffer(prefix_program),
        binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
    if (iree_status_is_ok(status)) cleanup_wait_value = execute_complete_value;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_copy(
        device, queue_affinity, execute_complete, capture_complete,
        transient_buffer, /*source_offset=*/0, staging_buffer,
        /*target_offset=*/0, program_state->transient_byte_length,
        IREE_HAL_COPY_FLAG_NONE);
    if (iree_status_is_ok(status)) cleanup_wait_value = capture_complete_value;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        device, queue_affinity, capture_complete, dealloca_complete,
        transient_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
    transient_dealloca_submitted = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(
        timeline_semaphore, dealloca_complete_value, iree_infinite_timeout(),
        IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_map_read(staging_buffer, /*source_offset=*/0,
                                      transient_capture.data,
                                      transient_capture.data_length);
  }

  if (transient_buffer && !transient_dealloca_submitted) {
    const iree_hal_semaphore_list_t cleanup_wait = {
        .count = 1,
        .semaphores = &timeline_semaphore,
        .payload_values = &cleanup_wait_value,
    };
    iree_status_t cleanup_status = iree_hal_device_queue_dealloca(
        device, queue_affinity, cleanup_wait, dealloca_complete,
        transient_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
    if (iree_status_is_ok(cleanup_status)) {
      cleanup_status = iree_hal_semaphore_wait(
          timeline_semaphore, dealloca_complete_value, iree_infinite_timeout(),
          IREE_ASYNC_WAIT_FLAG_NONE);
    }
    status = iree_status_join(status, cleanup_status);
  }
  iree_hal_buffer_release(transient_buffer);
  iree_hal_buffer_release(staging_buffer);
  iree_hal_semaphore_release(timeline_semaphore);
  loomc_cmd_iree_hal_program_release(prefix_program);
  return status;
}

iree_status_t qwen_command_package_issue(
    qwen_command_package_t* package, qwen_command_program_t program,
    qwen_request_t* request, const qwen_command_issue_options_t* options,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  IREE_RETURN_IF_ERROR(qwen_command_validate_issue_options(options));
  qwen_command_program_state_t* program_state = NULL;
  IREE_RETURN_IF_ERROR(qwen_command_validate_program_request(
      package, program, request, &program_state));
  const iree_host_size_t context_base = qwen_request_context_base(request);

  const bool issue_barrier_waves =
      options &&
      iree_any_bit_set(options->flags,
                       QWEN_COMMAND_ISSUE_FLAG_DIAGNOSTIC_BARRIER_WAVES);
  qwen_command_program_segment_t* segments = NULL;
  iree_host_size_t segment_count = 0;
  if (issue_barrier_waves) {
    IREE_RETURN_IF_ERROR(qwen_command_materialize_barrier_waves(
        package, program_state, &segments, &segment_count));
  }
  const uint64_t program_timeline_advance =
      issue_barrier_waves ? (uint64_t)segment_count + 3 : 4;
  if (program_state->timeline_value >
          IREE_HAL_SEMAPHORE_MAX_VALUE - program_timeline_advance ||
      qwen_request_timeline_value(request) == IREE_HAL_SEMAPHORE_MAX_VALUE) {
    qwen_command_release_program_segments(segment_count, segments,
                                          package->host_allocator);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Qwen issue timeline is exhausted");
  }

  iree_hal_semaphore_list_t alloca_waits;
  iree_status_t status = qwen_command_build_issue_wait_list(
      package, program_state, request, wait_semaphore_list, &alloca_waits);
  if (!iree_status_is_ok(status)) {
    qwen_command_release_program_segments(segment_count, segments,
                                          package->host_allocator);
    return status;
  }

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
  if (!iree_status_is_ok(status)) {
    qwen_command_release_program_segments(segment_count, segments,
                                          package->host_allocator);
    return status;
  }

  iree_hal_device_t* device = qwen_model_device(package->model);
  const iree_hal_queue_affinity_t queue_affinity =
      qwen_model_queue_affinity(package->model);
  const iree_hal_buffer_params_t transient_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
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
  if (!iree_status_is_ok(status)) {
    qwen_command_release_program_segments(segment_count, segments,
                                          package->host_allocator);
    return status;
  }

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
    qwen_command_release_program_segments(segment_count, segments,
                                          package->host_allocator);
    return iree_status_join(status, cleanup_status);
  }

  iree_hal_buffer_binding_t bindings[QWEN_COMMAND_BINDING_COUNT];
  const iree_hal_buffer_binding_table_t binding_table =
      qwen_command_build_binding_table(request, transient_buffer,
                                       program_state->transient_byte_length,
                                       bindings);
  if (issue_barrier_waves) {
    status = qwen_command_execute_barrier_waves(
        package, program, program_state, options, segments, segment_count,
        execute_ready_value, binding_table);
  } else {
    status = iree_hal_device_queue_execute(
        device, queue_affinity, execute_ready, execute_complete,
        loomc_cmd_iree_hal_program_command_buffer(program_state->hal_program),
        binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  }
  if (!iree_status_is_ok(status)) {
    iree_status_t cleanup_status = iree_ok_status();
    if (!issue_barrier_waves) {
      cleanup_status = iree_hal_device_queue_dealloca(
          device, queue_affinity, execute_ready,
          iree_hal_semaphore_list_empty(), transient_buffer,
          IREE_HAL_DEALLOCA_FLAG_NONE);
    }
    iree_hal_buffer_release(transient_buffer);
    qwen_command_fail_after_partial_submission(program_state, request, status);
    qwen_command_release_program_segments(segment_count, segments,
                                          package->host_allocator);
    return iree_status_join(status, cleanup_status);
  }

  status = iree_hal_device_queue_dealloca(
      device, queue_affinity, execute_complete, completion_signals,
      transient_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
  iree_hal_buffer_release(transient_buffer);
  qwen_command_release_program_segments(segment_count, segments,
                                        package->host_allocator);
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
