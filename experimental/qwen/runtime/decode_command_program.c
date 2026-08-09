// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/decode_command_program.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "experimental/qwen/programs/decode_source.h"
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

#define QWEN_DECODE_COMMAND_CONTEXT_CAPACITY 576
#define QWEN_DECODE_COMMAND_CONTEXT_CLASS_BASE 512
#define QWEN_DECODE_COMMAND_INITIAL_SEMAPHORE_CAPACITY 8

typedef enum qwen_decode_command_binding_e {
  QWEN_DECODE_COMMAND_BINDING_KEY_CACHE = 0,
  QWEN_DECODE_COMMAND_BINDING_VALUE_CACHE = 1,
  QWEN_DECODE_COMMAND_BINDING_REQUEST_STATE = 2,
  QWEN_DECODE_COMMAND_BINDING_OUTPUT_STAGING = 3,
  QWEN_DECODE_COMMAND_BINDING_TRANSIENT = 4,
  QWEN_DECODE_COMMAND_BINDING_COUNT = 5,
} qwen_decode_command_binding_t;

typedef struct qwen_decode_command_parameter_t {
  // Stable fixed-schema key.
  char key[QWEN_PARAMETER_KEY_CAPACITY];
  // Exact model-slab placement.
  iree_io_parameter_span_t span;
  // True after one compiled requirement claims this parameter.
  bool matched;
} qwen_decode_command_parameter_t;

struct qwen_decode_command_program_t {
  // Reference count for shared prepared-program ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for all program-owned host storage.
  iree_allocator_t host_allocator;
  // Model retained for immutable fixed-buffer identity and request matching.
  qwen_model_t* model;
  // Materialized reusable command program.
  loomc_cmd_iree_hal_program_t* hal_program;
  // Internal allocation, execution, and completion timeline.
  iree_hal_semaphore_t* timeline_semaphore;
  // Latest timeline value reserved by a submitted issue.
  uint64_t timeline_value;
  // Request token capacity compiled into persistent field placement.
  iree_host_size_t request_token_capacity;
  // Request K/V capacity compiled into cache and mask placement.
  iree_host_size_t context_capacity;
  // Request flags accepted by the compiled binding layout.
  qwen_request_flags_t request_flags;
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
};

typedef struct qwen_decode_command_compile_batch_t {
  // Immutable production plan shared by every independent job.
  loomc_program_plan_t* plan;
  // Worker-local mutable workspaces indexed by worker ordinal.
  loomc_workspace_t** worker_workspaces;
  // Dense unit outputs indexed by job and plan-unit ordinal.
  loomc_program_t** unit_programs;
  // Allocator passed to every Loom result and program operation.
  loomc_allocator_t allocator;
} qwen_decode_command_compile_batch_t;

static iree_status_t qwen_decode_command_require_result(
    iree_string_view_t phase, const loomc_result_t* result) {
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

static iree_status_t qwen_decode_command_reserve_semaphore_storage(
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

static iree_status_t qwen_decode_command_ensure_semaphore_storage(
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
  IREE_RETURN_IF_ERROR(qwen_decode_command_reserve_semaphore_storage(
      new_capacity, host_allocator, &new_semaphores, &new_values));
  iree_allocator_free(host_allocator, *inout_values);
  iree_allocator_free(host_allocator, *inout_semaphores);
  *inout_semaphores = new_semaphores;
  *inout_values = new_values;
  *inout_capacity = new_capacity;
  return iree_ok_status();
}

static void qwen_decode_command_program_destroy(
    qwen_decode_command_program_t* program) {
  iree_allocator_t host_allocator = program->host_allocator;
  iree_allocator_free(host_allocator, program->signal_values);
  iree_allocator_free(host_allocator, program->signal_semaphores);
  iree_allocator_free(host_allocator, program->wait_values);
  iree_allocator_free(host_allocator, program->wait_semaphores);
  iree_hal_semaphore_release(program->timeline_semaphore);
  loomc_cmd_iree_hal_program_release(program->hal_program);
  qwen_model_release(program->model);
  iree_allocator_free(host_allocator, program);
}

void qwen_decode_command_program_options_initialize(
    qwen_decode_command_program_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (qwen_decode_command_program_options_t){
      .structure_size = sizeof(*out_options),
      .next = NULL,
      .request_token_capacity = 1,
      .context_capacity = QWEN_DECODE_COMMAND_CONTEXT_CAPACITY,
      .request_flags = QWEN_REQUEST_FLAG_NONE,
      .compiler_worker_count = QWEN_LOOM_JIT_DEFAULT_WORKER_COUNT,
      .command_buffer_mode = IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
  };
}

static iree_status_t qwen_decode_command_validate_options(
    qwen_model_t* model, const qwen_decode_command_program_options_t* options) {
  if (!model || !options) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen model and decode command-program options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen decode command-program options structure is too small");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen decode command-program option extensions are unsupported");
  }
  if (options->request_token_capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request token capacity must not be zero");
  }
  if (options->context_capacity != QWEN_DECODE_COMMAND_CONTEXT_CAPACITY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen decode command root requires context capacity %d",
        QWEN_DECODE_COMMAND_CONTEXT_CAPACITY);
  }
  if (options->request_flags != QWEN_REQUEST_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen decode command root does not address optional request storage");
  }
  if (options->compiler_worker_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen decode command preparation requires compiler workers");
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

static iree_status_t qwen_decode_command_create_source(
    iree_allocator_t host_allocator, loomc_source_t** out_source) {
  *out_source = NULL;
  if (qwen_loom_decode_program_source_size() != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen decode command source must contain exactly one file");
  }
  const iree_file_toc_t* files = qwen_loom_decode_program_source_create();
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

static iree_status_t qwen_decode_command_query_kernels(
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
    status =
        qwen_decode_command_require_result(IREE_SV("kernel query"), result);
  }
  loomc_result_release(result);
  result = NULL;
  if (iree_status_is_ok(status) && *out_function_count == 0) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Qwen decode source contains no kernel roots");
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
    status =
        qwen_decode_command_require_result(IREE_SV("kernel query"), result);
  }
  loomc_result_release(result);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, *out_functions);
    *out_functions = NULL;
    *out_function_count = 0;
  }
  return status;
}

static iree_status_t qwen_decode_command_compile_unit_job(
    void* user_data, iree_host_size_t worker_ordinal,
    iree_host_size_t job_ordinal) {
  qwen_decode_command_compile_batch_t* batch =
      (qwen_decode_command_compile_batch_t*)user_data;
  loomc_result_t* result = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_program_plan_compile_unit(
      batch->plan, batch->worker_workspaces[worker_ordinal],
      loomc_program_plan_unit_from_index((uint32_t)job_ordinal),
      /*options=*/NULL, batch->allocator, &batch->unit_programs[job_ordinal],
      &result));
  if (iree_status_is_ok(status)) {
    status =
        qwen_decode_command_require_result(IREE_SV("unit compile"), result);
  }
  loomc_result_release(result);
  return status;
}

static iree_status_t qwen_decode_command_compile_units(
    loomc_program_plan_t* plan, iree_host_size_t requested_worker_count,
    iree_allocator_t host_allocator, loomc_program_t*** out_unit_programs,
    iree_host_size_t* out_unit_count) {
  *out_unit_programs = NULL;
  *out_unit_count = loomc_program_plan_unit_count(plan);
  if (*out_unit_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Qwen decode plan contains no program units");
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
    qwen_decode_command_compile_batch_t batch = {
        .plan = plan,
        .worker_workspaces = worker_workspaces,
        .unit_programs = *out_unit_programs,
        .allocator = loomc_allocator_from_iree(host_allocator),
    };
    status = qwen_loom_compile_pool_run_batch(
        &compile_pool, *out_unit_count, qwen_decode_command_compile_unit_job,
        &batch);
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

static void qwen_decode_command_release_units(iree_host_size_t unit_count,
                                              loomc_program_t** unit_programs,
                                              iree_allocator_t host_allocator) {
  if (unit_programs) {
    for (iree_host_size_t i = 0; i < unit_count; ++i) {
      loomc_program_release(unit_programs[i]);
    }
  }
  iree_allocator_free(host_allocator, unit_programs);
}

static iree_status_t qwen_decode_command_validate_parameter_layout(
    qwen_model_t* model, const loomc_cmd_program_t* command_program,
    const loomc_cmd_program_info_t* info, iree_allocator_t host_allocator,
    iree_hal_buffer_ref_t out_fixed_buffers[2]) {
  const qwen_parameter_layout_t* model_layout =
      qwen_model_parameter_layout(model);
  qwen_decode_command_parameter_t* model_parameters = NULL;
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

    qwen_decode_command_parameter_t* model_parameter = NULL;
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

iree_status_t qwen_decode_command_program_prepare(
    qwen_model_t* model, const qwen_decode_command_program_options_t* options,
    iree_allocator_t host_allocator,
    qwen_decode_command_program_t** out_program) {
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = NULL;
  IREE_RETURN_IF_ERROR(qwen_decode_command_validate_options(model, options));

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
  loomc_cmd_program_t* command_program = NULL;
  loomc_cmd_iree_hal_package_t* hal_package = NULL;
  loomc_cmd_iree_hal_program_t* hal_program = NULL;
  loomc_result_t* result = NULL;

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
    status = qwen_decode_command_create_source(host_allocator, &source);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_module_deserialize_from_source(
        context, coordinator_workspace, source, /*options=*/NULL,
        loom_allocator, &module, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_decode_command_require_result(IREE_SV("decode parse"), result);
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
    status =
        qwen_decode_command_require_result(IREE_SV("target profile"), result);
  }
  loomc_result_release(result);
  result = NULL;

  if (iree_status_is_ok(status)) {
    status = qwen_decode_command_query_kernels(
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
        .identifier = loomc_make_cstring_view("qwen-decode-expanded-source"),
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
    status = qwen_decode_command_require_result(IREE_SV("preparation pipeline"),
                                                result);
  }
  loomc_result_release(result);
  result = NULL;
  if (iree_status_is_ok(status)) {
    const loomc_target_pipeline_options_t pipeline_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        .structure_size = sizeof(pipeline_options),
        .next = NULL,
        .identifier = loomc_make_cstring_view("qwen-decode-prepared-low"),
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
    status =
        qwen_decode_command_require_result(IREE_SV("unit pipeline"), result);
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
    status =
        qwen_decode_command_require_result(IREE_SV("program plan"), result);
  }
  loomc_result_release(result);
  result = NULL;

  loomc_program_plan_root_t root = loomc_program_plan_root_invalid();
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_program_plan_lookup_root(
        plan, loomc_make_cstring_view("qwen3_30b_decode_576"), &root));
  }
  if (iree_status_is_ok(status)) {
    status = qwen_decode_command_compile_units(
        plan, options->compiler_worker_count, host_allocator, &unit_programs,
        &unit_count);
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
        plan, coordinator_workspace, &root, 1, &unit_table, /*options=*/NULL,
        loom_allocator, &assembled_program, &result));
  }
  if (iree_status_is_ok(status)) {
    status =
        qwen_decode_command_require_result(IREE_SV("program assembly"), result);
  }
  loomc_result_release(result);
  result = NULL;

  loomc_program_export_t root_export = loomc_program_export_invalid();
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_program_lookup_export(
        assembled_program, loomc_make_cstring_view("qwen3_30b_decode_576"),
        &root_export));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_cmd_program_create_from_export(
        assembled_program, root_export, loom_allocator, &command_program));
  }
  loomc_cmd_program_info_t info = {
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO,
      .structure_size = sizeof(info),
  };
  if (iree_status_is_ok(status)) {
    status =
        iree_status_from_loomc(loomc_cmd_program_info(command_program, &info));
  }
  if (iree_status_is_ok(status) &&
      (info.fixed_buffer_count != 2 ||
       info.rebindable_binding_count != QWEN_DECODE_COMMAND_BINDING_COUNT ||
       info.parameter_root_count != 2 ||
       info.parameter_count != QWEN_PARAMETER_COUNT + 1 ||
       info.transient.binding_index != QWEN_DECODE_COMMAND_BINDING_TRANSIENT ||
       info.transient.required_byte_length == 0 ||
       info.transient.minimum_alignment == 0 ||
       info.launch_counts.binding_index != LOOMC_CMD_PROGRAM_BINDING_INVALID)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "compiled Qwen decode command ABI does not match the runtime adapter");
  }

  iree_hal_buffer_ref_t fixed_buffers[2];
  memset(fixed_buffers, 0, sizeof(fixed_buffers));
  if (iree_status_is_ok(status)) {
    status = qwen_decode_command_validate_parameter_layout(
        model, command_program, &info, host_allocator, fixed_buffers);
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
  if (iree_status_is_ok(status)) {
    const loomc_cmd_iree_hal_program_options_t program_options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_IREE_HAL_PROGRAM_OPTIONS,
        .structure_size = sizeof(program_options),
        .next = NULL,
        .command_buffer_mode = options->command_buffer_mode,
        .queue_affinity = qwen_model_queue_affinity(model),
        .fixed_buffers = fixed_buffers,
        .fixed_buffer_count = IREE_ARRAYSIZE(fixed_buffers),
    };
    status = iree_status_from_loomc(loomc_cmd_iree_hal_program_create(
        hal_package, command_program, &program_options, loom_allocator,
        &hal_program));
  }

  qwen_decode_command_program_t* program = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(*program),
                                   (void**)&program);
  }
  if (iree_status_is_ok(status)) {
    memset(program, 0, sizeof(*program));
    iree_atomic_ref_count_init(&program->ref_count);
    program->host_allocator = host_allocator;
    program->model = model;
    qwen_model_retain(model);
    program->hal_program = hal_program;
    hal_program = NULL;
    program->request_token_capacity = options->request_token_capacity;
    program->context_capacity = options->context_capacity;
    program->request_flags = options->request_flags;
    program->transient_byte_length = info.transient.required_byte_length;
    program->transient_minimum_alignment = info.transient.minimum_alignment;
    status = qwen_decode_command_reserve_semaphore_storage(
        QWEN_DECODE_COMMAND_INITIAL_SEMAPHORE_CAPACITY, host_allocator,
        &program->wait_semaphores, &program->wait_values);
  }
  if (iree_status_is_ok(status)) {
    program->wait_capacity = QWEN_DECODE_COMMAND_INITIAL_SEMAPHORE_CAPACITY;
    status = qwen_decode_command_reserve_semaphore_storage(
        QWEN_DECODE_COMMAND_INITIAL_SEMAPHORE_CAPACITY, host_allocator,
        &program->signal_semaphores, &program->signal_values);
  }
  if (iree_status_is_ok(status)) {
    program->signal_capacity = QWEN_DECODE_COMMAND_INITIAL_SEMAPHORE_CAPACITY;
    status = iree_hal_semaphore_create(
        qwen_model_device(model), qwen_model_queue_affinity(model),
        /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &program->timeline_semaphore);
  }
  if (iree_status_is_ok(status)) {
    *out_program = program;
    program = NULL;
  }

  if (program) qwen_decode_command_program_destroy(program);
  loomc_cmd_iree_hal_program_release(hal_program);
  loomc_cmd_iree_hal_package_release(hal_package);
  loomc_cmd_program_release(command_program);
  loomc_program_release(assembled_program);
  qwen_decode_command_release_units(unit_count, unit_programs, host_allocator);
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

void qwen_decode_command_program_retain(
    qwen_decode_command_program_t* program) {
  if (program) iree_atomic_ref_count_inc(&program->ref_count);
}

void qwen_decode_command_program_release(
    qwen_decode_command_program_t* program) {
  if (program && iree_atomic_ref_count_dec(&program->ref_count) == 1) {
    qwen_decode_command_program_destroy(program);
  }
}

static iree_status_t qwen_decode_command_build_issue_wait_list(
    qwen_decode_command_program_t* program, qwen_request_t* request,
    iree_hal_semaphore_list_t caller_waits,
    iree_hal_semaphore_list_t* out_waits) {
  const iree_hal_semaphore_list_t model_ready =
      qwen_model_ready_semaphore_list(program->model);
  if (caller_waits.count > IREE_HOST_SIZE_MAX - model_ready.count - 1) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen issue wait count overflows");
  }
  const iree_host_size_t wait_count =
      caller_waits.count + model_ready.count + 1;
  IREE_RETURN_IF_ERROR(qwen_decode_command_ensure_semaphore_storage(
      wait_count, program->host_allocator, &program->wait_semaphores,
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

static iree_status_t qwen_decode_command_build_issue_signal_list(
    qwen_decode_command_program_t* program, qwen_request_t* request,
    uint64_t program_completion_value, uint64_t request_completion_value,
    iree_hal_semaphore_list_t caller_signals,
    iree_hal_semaphore_list_t* out_signals) {
  if (caller_signals.count > IREE_HOST_SIZE_MAX - 2) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Qwen issue signal count overflows");
  }
  const iree_host_size_t signal_count = caller_signals.count + 2;
  IREE_RETURN_IF_ERROR(qwen_decode_command_ensure_semaphore_storage(
      signal_count, program->host_allocator, &program->signal_semaphores,
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

static void qwen_decode_command_fail_after_partial_submission(
    qwen_decode_command_program_t* program, qwen_request_t* request,
    iree_status_t status) {
  qwen_request_fail(request, iree_status_clone(status));
  iree_hal_semaphore_fail(program->timeline_semaphore,
                          iree_status_clone(status));
}

iree_status_t qwen_decode_command_program_issue(
    qwen_decode_command_program_t* program, qwen_request_t* request,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list) {
  if (!program || !request) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen decode command program and request are required");
  }
  if (qwen_request_model(request) != program->model ||
      qwen_request_token_capacity(request) != program->request_token_capacity ||
      qwen_request_context_capacity(request) != program->context_capacity ||
      qwen_request_flags(request) != program->request_flags) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Qwen request model, storage capacities, and flags must match the "
        "prepared decode command program");
  }
  const iree_host_size_t context_base = qwen_request_context_base(request);
  if (qwen_request_active_token_count(request) != 1 ||
      context_base < QWEN_DECODE_COMMAND_CONTEXT_CLASS_BASE ||
      context_base >= QWEN_DECODE_COMMAND_CONTEXT_CAPACITY) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen request must contain one decode token in context class "
        "[%d, %d)",
        QWEN_DECODE_COMMAND_CONTEXT_CLASS_BASE,
        QWEN_DECODE_COMMAND_CONTEXT_CAPACITY);
  }
  if (qwen_request_input_kind(request) != QWEN_REQUEST_INPUT_KIND_TOKEN_IDS) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen decode command program requires token-ID request state");
  }
  if (program->timeline_value > IREE_HAL_SEMAPHORE_MAX_VALUE - 4 ||
      qwen_request_timeline_value(request) == IREE_HAL_SEMAPHORE_MAX_VALUE) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Qwen issue timeline is exhausted");
  }

  uint64_t completed_program_value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_query(program->timeline_semaphore,
                                                &completed_program_value));
  if (completed_program_value < program->timeline_value) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Qwen command program already has an in-flight issue at value %" PRIu64,
        program->timeline_value);
  }

  iree_hal_semaphore_list_t alloca_waits;
  IREE_RETURN_IF_ERROR(qwen_decode_command_build_issue_wait_list(
      program, request, wait_semaphore_list, &alloca_waits));

  uint64_t scratch_ready_value = program->timeline_value + 1;
  uint64_t execute_ready_value = program->timeline_value + 2;
  uint64_t execute_complete_value = program->timeline_value + 3;
  const uint64_t program_complete_value = program->timeline_value + 4;
  const uint64_t request_complete_value =
      qwen_request_timeline_value(request) + 1;
  iree_hal_semaphore_t* program_timeline = program->timeline_semaphore;
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
  IREE_RETURN_IF_ERROR(qwen_decode_command_build_issue_signal_list(
      program, request, program_complete_value, request_complete_value,
      signal_semaphore_list, &completion_signals));

  iree_hal_device_t* device = qwen_model_device(program->model);
  const iree_hal_queue_affinity_t queue_affinity =
      qwen_model_queue_affinity(program->model);
  const iree_hal_buffer_params_t transient_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .queue_affinity = queue_affinity,
      .min_alignment = program->transient_minimum_alignment,
  };
  iree_hal_buffer_t* transient_buffer = NULL;
  iree_status_t status = iree_hal_device_queue_alloca(
      device, queue_affinity, alloca_waits, scratch_ready, /*pool=*/NULL,
      transient_params, program->transient_byte_length,
      IREE_HAL_ALLOCA_FLAG_NONE, &transient_buffer);
  if (!iree_status_is_ok(status)) return status;

  const uint32_t zero_pattern = 0;
  status = iree_hal_device_queue_fill(
      device, queue_affinity, scratch_ready, execute_ready, transient_buffer,
      /*target_offset=*/0, program->transient_byte_length, &zero_pattern,
      sizeof(zero_pattern), IREE_HAL_FILL_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_status_t cleanup_status = iree_hal_device_queue_dealloca(
        device, queue_affinity, scratch_ready, iree_hal_semaphore_list_empty(),
        transient_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
    iree_hal_buffer_release(transient_buffer);
    qwen_decode_command_fail_after_partial_submission(program, request, status);
    return iree_status_join(status, cleanup_status);
  }

  const qwen_request_storage_layout_t* request_layout =
      qwen_request_storage_layout(request);
  const iree_hal_buffer_binding_t bindings[QWEN_DECODE_COMMAND_BINDING_COUNT] =
      {
          [QWEN_DECODE_COMMAND_BINDING_KEY_CACHE] =
              {
                  .buffer = qwen_request_storage_buffer(request),
                  .offset = request_layout->key_cache.offset,
                  .length = request_layout->key_cache.length,
              },
          [QWEN_DECODE_COMMAND_BINDING_VALUE_CACHE] =
              {
                  .buffer = qwen_request_storage_buffer(request),
                  .offset = request_layout->value_cache.offset,
                  .length = request_layout->value_cache.length,
              },
          [QWEN_DECODE_COMMAND_BINDING_REQUEST_STATE] =
              {
                  .buffer = qwen_request_storage_buffer(request),
                  .offset = 0,
                  .length = request_layout->dispatch_state_byte_length,
              },
          [QWEN_DECODE_COMMAND_BINDING_OUTPUT_STAGING] =
              {
                  .buffer = qwen_request_output_staging_buffer(request),
                  .offset = 0,
                  .length = sizeof(int32_t),
              },
          [QWEN_DECODE_COMMAND_BINDING_TRANSIENT] =
              {
                  .buffer = transient_buffer,
                  .offset = 0,
                  .length = program->transient_byte_length,
              },
      };
  const iree_hal_buffer_binding_table_t binding_table = {
      .count = IREE_ARRAYSIZE(bindings),
      .bindings = bindings,
  };
  status = iree_hal_device_queue_execute(
      device, queue_affinity, execute_ready, execute_complete,
      loomc_cmd_iree_hal_program_command_buffer(program->hal_program),
      binding_table, IREE_HAL_EXECUTE_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_status_t cleanup_status = iree_hal_device_queue_dealloca(
        device, queue_affinity, execute_ready, iree_hal_semaphore_list_empty(),
        transient_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
    iree_hal_buffer_release(transient_buffer);
    qwen_decode_command_fail_after_partial_submission(program, request, status);
    return iree_status_join(status, cleanup_status);
  }

  status = iree_hal_device_queue_dealloca(
      device, queue_affinity, execute_complete, completion_signals,
      transient_buffer, IREE_HAL_DEALLOCA_FLAG_NONE);
  iree_hal_buffer_release(transient_buffer);
  if (!iree_status_is_ok(status)) {
    qwen_decode_command_fail_after_partial_submission(program, request, status);
    return status;
  }

  program->timeline_value = program_complete_value;
  qwen_request_commit_selected_token_signal(request, request_complete_value,
                                            context_base + 1);
  return iree_ok_status();
}

iree_device_size_t qwen_decode_command_program_transient_byte_length(
    const qwen_decode_command_program_t* program) {
  return program ? program->transient_byte_length : 0;
}
