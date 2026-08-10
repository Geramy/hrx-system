// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "benchmark/benchmark.h"
#include "experimental/qwen/runtime/command_package.h"
#include "experimental/qwen/runtime/model.h"
#include "experimental/qwen/runtime/model_shape.h"
#include "experimental/qwen/runtime/program.h"
#include "experimental/qwen/runtime/request.h"
#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/types.h"
#include "iree/tooling/device_util.h"

#define QWEN_PREFILL_FIXTURE_TOKEN_COUNT 512

IREE_FLAG(string, tokens, "",
          "Raw prefill token IDs: exactly 512 little-endian I32 values; the "
          "selected prefill shape consumes its leading values.");
IREE_FLAG(int32_t, prefill_token_count, QWEN_PREFILL_FIXTURE_TOKEN_COUNT,
          "Exact leading fixture token count: 32, 64, 128, 256, or 512.");
IREE_FLAG(int32_t, expected_prefill_token, IREE_TOKENIZER_TOKEN_ID_INVALID,
          "Required token expected from the configured prefill shape.");
IREE_FLAG(int32_t, expected_decode_token, IREE_TOKENIZER_TOKEN_ID_INVALID,
          "Expected token selected after appending the prefill-selected token. "
          "Providing this with a 512-token prefill enables matched Decode-513 "
          "rows.");

namespace {

typedef struct QwenBenchmarkTimepoint {
  // Timeline semaphore carrying this timepoint.
  iree_hal_semaphore_t* semaphore;
  // Monotonically increasing timeline value.
  uint64_t value;
} QwenBenchmarkTimepoint;

typedef enum QwenBenchmarkProgramKind {
  QWEN_BENCHMARK_PROGRAM_KIND_OWNED = 0,
  QWEN_BENCHMARK_PROGRAM_KIND_COMMAND = 1,
} QwenBenchmarkProgramKind;

typedef struct QwenBenchmarkProgram {
  // Runtime implementation owning this benchmark row.
  QwenBenchmarkProgramKind kind;
  // Legacy owned recorder handle when |kind| is OWNED.
  qwen_program_t* owned_program;
  // Prepared command-program identity when |kind| is COMMAND.
  qwen_command_program_t command_program;
  // Packed transient requirement for the selected implementation.
  iree_device_size_t transient_byte_length;
} QwenBenchmarkProgram;

typedef struct QwenPrefillBenchmarkEnvironment {
  // Allocator used for all process-global host objects.
  iree_allocator_t host_allocator;
  // Standard IREE device, parameter, and profiling configuration.
  qwen_tooling_runtime_context_t runtime_context;
  // Caller-owned timeline serializing model, request, reset, and issue work.
  iree_hal_semaphore_t* timeline;
  // Last allocated value on |timeline|.
  uint64_t timeline_value;
  // Externally published completion of the asynchronous model gather.
  QwenBenchmarkTimepoint model_ready;
  // Leading fixture token count consumed by the configured prefill shape.
  iree_host_size_t prefill_token_count;
  // Externally supplied token oracle for the configured prefill shape.
  iree_tokenizer_token_id_t expected_prefill_token;
  // Validated token fixture retained for the process lifetime.
  iree_tokenizer_token_id_t token_ids[QWEN_PREFILL_FIXTURE_TOKEN_COUNT];
  // Resident fixed Qwen model.
  qwen_model_t* model;
  // Shared package containing every reusable command-program root.
  qwen_command_package_t* command_package;
  // Legacy owned-recorder prefill oracle.
  QwenBenchmarkProgram owned_prefill_program;
  // Command-program prefill candidate matched to |prefill_token_count|.
  QwenBenchmarkProgram command_prefill_program;
  // Legacy owned-recorder decode oracle, when requested.
  QwenBenchmarkProgram owned_decode_program;
  // Reusable Decode-576 command-program candidate, when requested.
  QwenBenchmarkProgram command_decode_program;
  // Persistent full-model request state.
  qwen_request_t* request;
  // First or joined terminal error from setup, a row, or profiling.
  iree_status_t terminal_status;
} QwenPrefillBenchmarkEnvironment;

typedef enum QwenBenchmarkInputKind {
  QWEN_BENCHMARK_INPUT_KIND_PREFILL = 0,
  QWEN_BENCHMARK_INPUT_KIND_DECODE = 1,
} QwenBenchmarkInputKind;

static iree_hal_semaphore_list_t QwenBenchmarkTimepointList(
    QwenBenchmarkTimepoint* timepoint) {
  iree_hal_semaphore_list_t list = {
      /*.count=*/1,
      /*.semaphores=*/&timepoint->semaphore,
      /*.payload_values=*/&timepoint->value,
  };
  return list;
}

static QwenBenchmarkTimepoint QwenBenchmarkNextTimepoint(
    QwenPrefillBenchmarkEnvironment* environment) {
  QwenBenchmarkTimepoint timepoint = {
      /*.semaphore=*/environment->timeline,
      /*.value=*/++environment->timeline_value,
  };
  return timepoint;
}

static QwenBenchmarkTimepoint QwenBenchmarkCurrentTimepoint(
    QwenPrefillBenchmarkEnvironment* environment) {
  QwenBenchmarkTimepoint timepoint = {
      /*.semaphore=*/environment->timeline,
      /*.value=*/environment->timeline_value,
  };
  return timepoint;
}

// Transient, non-sanctioned containment for an AMDGPU async file-action
// teardown defect. A host preparation failure must not release the tooling
// runtime while the model gather remains active. Keep this wait tool-only and
// delete it when asynchronous device teardown is safe.
static void qwen_wait_for_model_ready_bringup_workaround(
    QwenPrefillBenchmarkEnvironment* environment) {
  if (!environment->model) return;
  environment->terminal_status = iree_status_join(
      environment->terminal_status,
      iree_hal_semaphore_wait(
          environment->model_ready.semaphore, environment->model_ready.value,
          iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

static void QwenBenchmarkRecordFailure(
    QwenPrefillBenchmarkEnvironment* environment, iree_status_t status,
    benchmark::State* benchmark_state) {
  if (iree_status_is_ok(status)) return;
  environment->terminal_status =
      iree_status_join(environment->terminal_status, status);
  if (benchmark_state) {
    benchmark_state->SkipWithError(
        "Qwen full-model benchmark failed; the process exits nonzero with "
        "details");
  }
}

static iree_status_t QwenBenchmarkLoadTokens(
    QwenPrefillBenchmarkEnvironment* environment) {
  iree_string_view_t path = iree_make_cstring_view(FLAG_tokens);
  if (iree_string_view_is_empty(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--tokens is required");
  }

  iree_io_file_contents_t* contents = nullptr;
  iree_status_t status =
      iree_io_file_contents_read(path, environment->host_allocator, &contents);
  const iree_host_size_t expected_byte_length =
      IREE_ARRAYSIZE(environment->token_ids) *
      sizeof(environment->token_ids[0]);
  if (iree_status_is_ok(status) &&
      contents->const_buffer.data_length != expected_byte_length) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--tokens must contain exactly %" PRIhsz
        " bytes (512 little-endian I32 values); received %" PRIhsz,
        expected_byte_length, contents->const_buffer.data_length);
  }

  if (iree_status_is_ok(status)) {
    const uint8_t* bytes = contents->const_buffer.data;
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(environment->token_ids);
         ++i) {
      const iree_host_size_t byte_offset =
          i * sizeof(environment->token_ids[0]);
      const uint32_t encoded = ((uint32_t)bytes[byte_offset + 0]) |
                               ((uint32_t)bytes[byte_offset + 1] << 8) |
                               ((uint32_t)bytes[byte_offset + 2] << 16) |
                               ((uint32_t)bytes[byte_offset + 3] << 24);
      if (encoded >= QWEN_MODEL_VOCABULARY_SIZE) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "--tokens value at index %" PRIhsz " is %" PRIu32
                             "; expected [0, %" PRIu32 ")",
                             i, encoded, (uint32_t)QWEN_MODEL_VOCABULARY_SIZE);
        break;
      }
      environment->token_ids[i] = (iree_tokenizer_token_id_t)encoded;
    }
  }

  iree_io_file_contents_free(contents);
  return status;
}

static iree_status_t QwenBenchmarkPublishTokens(
    QwenPrefillBenchmarkEnvironment* environment, iree_host_size_t context_base,
    iree_tokenizer_token_id_list_t token_ids) {
  QwenBenchmarkTimepoint wait_timepoint =
      QwenBenchmarkCurrentTimepoint(environment);
  QwenBenchmarkTimepoint signal_timepoint =
      QwenBenchmarkNextTimepoint(environment);
  IREE_RETURN_IF_ERROR(
      qwen_request_reset_tokens(environment->request, context_base, token_ids,
                                QwenBenchmarkTimepointList(&wait_timepoint),
                                QwenBenchmarkTimepointList(&signal_timepoint)));
  return iree_hal_semaphore_wait(environment->timeline, signal_timepoint.value,
                                 iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t QwenBenchmarkPublishPrefillTokens(
    QwenPrefillBenchmarkEnvironment* environment) {
  return QwenBenchmarkPublishTokens(
      environment, /*context_base=*/0,
      iree_tokenizer_make_token_id_list(environment->token_ids,
                                        environment->prefill_token_count));
}

static iree_status_t QwenBenchmarkPublishDecodeToken(
    QwenPrefillBenchmarkEnvironment* environment) {
  return QwenBenchmarkPublishTokens(
      environment, /*context_base=*/environment->prefill_token_count,
      iree_tokenizer_make_token_id_list(&environment->expected_prefill_token,
                                        1));
}

static iree_status_t QwenBenchmarkPublishInput(
    QwenPrefillBenchmarkEnvironment* environment,
    QwenBenchmarkInputKind input_kind) {
  return input_kind == QWEN_BENCHMARK_INPUT_KIND_PREFILL
             ? QwenBenchmarkPublishPrefillTokens(environment)
             : QwenBenchmarkPublishDecodeToken(environment);
}

static QwenBenchmarkProgram QwenBenchmarkOwnedProgram(
    qwen_program_t* owned_program) {
  return QwenBenchmarkProgram{
      /*.kind=*/QWEN_BENCHMARK_PROGRAM_KIND_OWNED,
      /*.owned_program=*/owned_program,
      /*.command_program=*/QWEN_COMMAND_PROGRAM_COUNT,
      /*.transient_byte_length=*/
      qwen_program_transient_byte_length(owned_program),
  };
}

static iree_status_t QwenBenchmarkCommandProgram(
    QwenPrefillBenchmarkEnvironment* environment,
    qwen_command_program_t command_program, QwenBenchmarkProgram* out_program) {
  qwen_command_program_info_t program_info = {
      /*.structure_size=*/sizeof(program_info),
      /*.next=*/nullptr,
  };
  IREE_RETURN_IF_ERROR(qwen_command_package_query_program(
      environment->command_package, command_program, &program_info));
  *out_program = QwenBenchmarkProgram{
      /*.kind=*/QWEN_BENCHMARK_PROGRAM_KIND_COMMAND,
      /*.owned_program=*/nullptr,
      /*.command_program=*/command_program,
      /*.transient_byte_length=*/program_info.transient_byte_length,
  };
  return iree_ok_status();
}

static iree_status_t QwenBenchmarkIssueAndWait(
    QwenPrefillBenchmarkEnvironment* environment,
    const QwenBenchmarkProgram* program, iree_time_t* out_elapsed_time) {
  QwenBenchmarkTimepoint wait_timepoint =
      QwenBenchmarkCurrentTimepoint(environment);
  QwenBenchmarkTimepoint signal_timepoint =
      QwenBenchmarkNextTimepoint(environment);

  const iree_time_t start_time = iree_time_now();
  if (program->kind == QWEN_BENCHMARK_PROGRAM_KIND_OWNED) {
    IREE_RETURN_IF_ERROR(
        qwen_program_issue(program->owned_program, environment->request,
                           QwenBenchmarkTimepointList(&wait_timepoint),
                           QwenBenchmarkTimepointList(&signal_timepoint)));
  } else {
    IREE_RETURN_IF_ERROR(qwen_command_package_issue(
        environment->command_package, program->command_program,
        environment->request, /*options=*/nullptr,
        QwenBenchmarkTimepointList(&wait_timepoint),
        QwenBenchmarkTimepointList(&signal_timepoint)));
  }
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait(
      environment->timeline, signal_timepoint.value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  *out_elapsed_time = iree_time_now() - start_time;
  return iree_ok_status();
}

static iree_status_t QwenBenchmarkReadAndValidate(
    QwenPrefillBenchmarkEnvironment* environment,
    iree_tokenizer_token_id_t expected_token, const char* operation_name) {
  iree_tokenizer_token_id_t selected_token = IREE_TOKENIZER_TOKEN_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      qwen_request_read_selected_token(environment->request, &selected_token));
  if (selected_token != expected_token) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "%s selected token %" PRId32 "; expected %" PRId32,
                            operation_name, selected_token, expected_token);
  }
  return iree_ok_status();
}

static iree_status_t QwenBenchmarkWarmAndValidate(
    QwenPrefillBenchmarkEnvironment* environment,
    const QwenBenchmarkProgram* program, QwenBenchmarkInputKind input_kind,
    iree_tokenizer_token_id_t expected_token, const char* operation_name) {
  IREE_RETURN_IF_ERROR(QwenBenchmarkPublishInput(environment, input_kind));
  iree_time_t warmup_time = 0;
  IREE_RETURN_IF_ERROR(
      QwenBenchmarkIssueAndWait(environment, program, &warmup_time));
  return QwenBenchmarkReadAndValidate(environment, expected_token,
                                      operation_name);
}

static iree_status_t QwenBenchmarkEnvironmentInitialize(
    QwenPrefillBenchmarkEnvironment* environment) {
  environment->host_allocator = iree_allocator_system();
  environment->terminal_status = iree_ok_status();

  iree_status_t status = iree_ok_status();
  qwen_command_program_t command_prefill_program = QWEN_COMMAND_PROGRAM_COUNT;
  const bool decode_is_enabled =
      FLAG_expected_decode_token != IREE_TOKENIZER_TOKEN_ID_INVALID;
  if (FLAG_prefill_token_count <= 0 ||
      FLAG_prefill_token_count > QWEN_PREFILL_FIXTURE_TOKEN_COUNT) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--prefill_token_count must be in [1, %d]; received %" PRId32,
        QWEN_PREFILL_FIXTURE_TOKEN_COUNT, FLAG_prefill_token_count);
  }
  if (iree_status_is_ok(status) &&
      (FLAG_expected_prefill_token < 0 ||
       FLAG_expected_prefill_token >= QWEN_MODEL_VOCABULARY_SIZE)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--expected_prefill_token must be in [0, %u); received %" PRId32,
        (uint32_t)QWEN_MODEL_VOCABULARY_SIZE, FLAG_expected_prefill_token);
  }
  if (iree_status_is_ok(status) &&
      (FLAG_expected_decode_token < IREE_TOKENIZER_TOKEN_ID_INVALID ||
       FLAG_expected_decode_token >= QWEN_MODEL_VOCABULARY_SIZE)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--expected_decode_token must be -1 or in [0, %u); received %" PRId32,
        (uint32_t)QWEN_MODEL_VOCABULARY_SIZE, FLAG_expected_decode_token);
  }
  if (iree_status_is_ok(status)) {
    environment->prefill_token_count =
        (iree_host_size_t)FLAG_prefill_token_count;
    environment->expected_prefill_token = FLAG_expected_prefill_token;
  }
  if (iree_status_is_ok(status)) {
    status = qwen_command_select_prefill_program(
        environment->prefill_token_count, &command_prefill_program);
  }
  if (iree_status_is_ok(status) && decode_is_enabled &&
      environment->prefill_token_count != QWEN_COMMAND_PREFILL_TOKEN_CAPACITY) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "matched owned and command decode rows require a 512-token prefill");
  }
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkLoadTokens(environment);
  }
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_runtime_context_initialize_from_flags(
        environment->host_allocator, &environment->runtime_context);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        qwen_tooling_runtime_context_device(&environment->runtime_context),
        IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &environment->timeline);
  }

  environment->model_ready = QwenBenchmarkNextTimepoint(environment);
  if (iree_status_is_ok(status)) {
    qwen_model_options_t model_options;
    qwen_model_options_initialize(&model_options);
    model_options.device_group = environment->runtime_context.device_group;
    if (environment->runtime_context.jit_worker_count != 0) {
      model_options.jit_worker_count =
          environment->runtime_context.jit_worker_count;
    }
    qwen_parameter_source_t parameter_source = {
        /*.index=*/environment->runtime_context.parameter_index,
        /*.provider=*/environment->runtime_context.parameter_provider,
        /*.scope=*/iree_string_view_empty(),
    };
    status = qwen_model_load(
        &model_options, &parameter_source, iree_hal_semaphore_list_empty(),
        QwenBenchmarkTimepointList(&environment->model_ready),
        environment->host_allocator, &environment->model);
  }

  // Both implementations use the command package's shared request layout so
  // row comparisons differ only in recording/materialization ownership.
  const iree_host_size_t request_token_capacity =
      QWEN_COMMAND_PREFILL_TOKEN_CAPACITY;
  const iree_host_size_t request_context_capacity =
      QWEN_COMMAND_CONTEXT_CAPACITY;

  // Host-side owned-program and package preparation overlap the asynchronous
  // model gather. None of this work enters a benchmark row.
  if (iree_status_is_ok(status)) {
    qwen_program_options_t program_options;
    qwen_program_options_initialize(&program_options);
    program_options.kind = QWEN_PROGRAM_KIND_PREFILL;
    program_options.layer_index = 0;
    program_options.token_count = environment->prefill_token_count;
    program_options.context_count = environment->prefill_token_count;
    program_options.token_capacity = request_token_capacity;
    program_options.context_capacity = request_context_capacity;
    program_options.command_buffer_mode =
        environment->runtime_context.command_buffer_mode;
    status = qwen_program_prepare(
        environment->model, &program_options, environment->host_allocator,
        &environment->owned_prefill_program.owned_program);
  }
  if (iree_status_is_ok(status) && decode_is_enabled) {
    qwen_program_options_t program_options;
    qwen_program_options_initialize(&program_options);
    program_options.kind = QWEN_PROGRAM_KIND_DECODE;
    program_options.layer_index = 0;
    program_options.token_count = 1;
    program_options.context_count = request_context_capacity;
    program_options.token_capacity = request_token_capacity;
    program_options.context_capacity = request_context_capacity;
    program_options.command_buffer_mode =
        environment->runtime_context.command_buffer_mode;
    status = qwen_program_prepare(
        environment->model, &program_options, environment->host_allocator,
        &environment->owned_decode_program.owned_program);
  }
  if (iree_status_is_ok(status)) {
    qwen_command_package_options_t package_options;
    qwen_command_package_options_initialize(&package_options);
    if (environment->runtime_context.jit_worker_count != 0) {
      package_options.compiler_worker_count =
          environment->runtime_context.jit_worker_count;
    }
    package_options.command_buffer_mode =
        environment->runtime_context.command_buffer_mode;
    status = qwen_command_package_prepare(environment->model, &package_options,
                                          environment->host_allocator,
                                          &environment->command_package);
  }
  if (iree_status_is_ok(status)) {
    environment->owned_prefill_program = QwenBenchmarkOwnedProgram(
        environment->owned_prefill_program.owned_program);
    status = QwenBenchmarkCommandProgram(environment, command_prefill_program,
                                         &environment->command_prefill_program);
  }
  if (iree_status_is_ok(status) && decode_is_enabled) {
    environment->owned_decode_program = QwenBenchmarkOwnedProgram(
        environment->owned_decode_program.owned_program);
    status = QwenBenchmarkCommandProgram(environment,
                                         QWEN_COMMAND_PROGRAM_DECODE_576,
                                         &environment->command_decode_program);
  }

  QwenBenchmarkTimepoint request_ready =
      QwenBenchmarkNextTimepoint(environment);
  if (iree_status_is_ok(status)) {
    qwen_request_options_t request_options;
    qwen_request_options_initialize(&request_options);
    request_options.token_capacity = request_token_capacity;
    request_options.context_capacity = request_context_capacity;
    status = qwen_request_create(
        environment->model, &request_options,
        QwenBenchmarkTimepointList(&environment->model_ready),
        QwenBenchmarkTimepointList(&request_ready), environment->host_allocator,
        &environment->request);
  }

  // Warm and validate both ownership paths before Google Benchmark controls
  // the repeated program issues. Each path receives the same fixture reset.
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkWarmAndValidate(
        environment, &environment->owned_prefill_program,
        QWEN_BENCHMARK_INPUT_KIND_PREFILL, environment->expected_prefill_token,
        "owned full-model prefill");
  }
  if (iree_status_is_ok(status)) {
    status = QwenBenchmarkWarmAndValidate(
        environment, &environment->command_prefill_program,
        QWEN_BENCHMARK_INPUT_KIND_PREFILL, environment->expected_prefill_token,
        "command full-model prefill");
  }
  return status;
}

static void QwenBenchmarkEnvironmentDeinitialize(
    QwenPrefillBenchmarkEnvironment* environment) {
  qwen_wait_for_model_ready_bringup_workaround(environment);
  qwen_request_release(environment->request);
  qwen_command_package_release(environment->command_package);
  qwen_program_release(environment->owned_decode_program.owned_program);
  qwen_program_release(environment->owned_prefill_program.owned_program);
  qwen_model_release(environment->model);
  iree_hal_semaphore_release(environment->timeline);
  qwen_tooling_runtime_context_deinitialize(&environment->runtime_context);
}

static void QwenBenchmarkMeasureProgram(
    QwenPrefillBenchmarkEnvironment* environment,
    const QwenBenchmarkProgram* program, QwenBenchmarkInputKind input_kind,
    iree_tokenizer_token_id_t expected_token, const char* operation_name,
    iree_host_size_t logical_token_count, benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    (void)_;

    // Each measured row starts from the same production input state. The
    // reset is outside both profiling and the manual issue interval; generated
    // sequences instead consume the device-published continuation directly.
    iree_status_t status = QwenBenchmarkPublishInput(environment, input_kind);

    // Profiling excludes process setup and warmup and surrounds the measured
    // issue. The manual interval includes submission and user-visible
    // completion; dispatch-only timings come from the device profile.
    iree_hal_profiling_from_flags_t* profiling = nullptr;
    if (iree_status_is_ok(status)) {
      status = iree_hal_begin_device_group_profiling_from_flags(
          environment->runtime_context.device_group,
          environment->host_allocator, &profiling);
    }

    iree_time_t elapsed_time = 0;
    if (iree_status_is_ok(status)) {
      status = QwenBenchmarkIssueAndWait(environment, program, &elapsed_time);
    }
    if (iree_status_is_ok(status)) {
      benchmark_state.SetIterationTime((double)elapsed_time / 1000000000.0);
    }
    if (profiling) {
      status = iree_status_join(status,
                                iree_hal_end_profiling_from_flags(profiling));
    }
    if (iree_status_is_ok(status)) {
      status = QwenBenchmarkReadAndValidate(environment, expected_token,
                                            operation_name);
    }
    if (!iree_status_is_ok(status)) {
      QwenBenchmarkRecordFailure(environment, status, &benchmark_state);
      break;
    }
  }

  if (iree_status_is_ok(environment->terminal_status)) {
    const qwen_model_statistics_t model_statistics =
        qwen_model_statistics(environment->model);
    benchmark_state.SetItemsProcessed(benchmark_state.iterations() *
                                      logical_token_count);
    if (program->kind == QWEN_BENCHMARK_PROGRAM_KIND_OWNED) {
      benchmark_state.counters["dispatches"] =
          (double)qwen_program_dispatch_count(program->owned_program);
    }
    benchmark_state.counters["encoded_parameter_bytes"] =
        (double)model_statistics.encoded_parameter_bytes;
    benchmark_state.counters["persistent_bytes"] =
        (double)qwen_request_persistent_byte_length(environment->request);
    benchmark_state.counters["resident_bytes"] =
        (double)model_statistics.allocation_bytes;
    benchmark_state.counters["submissions"] = 1;
    benchmark_state.counters["tokens"] = (double)logical_token_count;
    benchmark_state.counters["transient_bytes"] =
        (double)program->transient_byte_length;
  }
}

static void QwenFullModelOwnedPrefill(
    QwenPrefillBenchmarkEnvironment* environment,
    benchmark::State& benchmark_state) {
  QwenBenchmarkMeasureProgram(
      environment, &environment->owned_prefill_program,
      QWEN_BENCHMARK_INPUT_KIND_PREFILL, environment->expected_prefill_token,
      "owned full-model prefill", environment->prefill_token_count,
      benchmark_state);
}

static void QwenFullModelCommandPrefill(
    QwenPrefillBenchmarkEnvironment* environment,
    benchmark::State& benchmark_state) {
  QwenBenchmarkMeasureProgram(
      environment, &environment->command_prefill_program,
      QWEN_BENCHMARK_INPUT_KIND_PREFILL, environment->expected_prefill_token,
      "command full-model prefill", environment->prefill_token_count,
      benchmark_state);
}

static void QwenFullModelDecode(QwenPrefillBenchmarkEnvironment* environment,
                                const QwenBenchmarkProgram* program,
                                const char* operation_name,
                                benchmark::State& benchmark_state) {
  iree_status_t status = QwenBenchmarkWarmAndValidate(
      environment, program, QWEN_BENCHMARK_INPUT_KIND_DECODE,
      FLAG_expected_decode_token, operation_name);
  if (!iree_status_is_ok(status)) {
    QwenBenchmarkRecordFailure(environment, status, &benchmark_state);
    return;
  }
  QwenBenchmarkMeasureProgram(environment, program,
                              QWEN_BENCHMARK_INPUT_KIND_DECODE,
                              FLAG_expected_decode_token, operation_name,
                              /*logical_token_count=*/1, benchmark_state);
}

}  // namespace

int main(int argc, char** argv) {
  iree_flags_set_usage(
      "qwen-prefill-benchmark",
      "Compares matched owned-recorder and command-program issues for one "
      "resident Qwen full-model prefill shape and optional Decode-513.");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    benchmark::Shutdown();
    return EXIT_FAILURE;
  }

  QwenPrefillBenchmarkEnvironment environment = {};
  environment.host_allocator = iree_allocator_system();
  environment.terminal_status = iree_ok_status();
  iree_status_t status = QwenBenchmarkEnvironmentInitialize(&environment);
  QwenBenchmarkRecordFailure(&environment, status,
                             /*benchmark_state=*/nullptr);

  if (iree_status_is_ok(environment.terminal_status)) {
    const std::string prefill_suffix =
        "/Prefill/" + std::to_string(environment.prefill_token_count);
    const std::string owned_prefill_name =
        "Qwen/FullModel/Owned" + prefill_suffix;
    const std::string command_prefill_name =
        "Qwen/FullModel/Command" + prefill_suffix;
    benchmark::RegisterBenchmark(
        owned_prefill_name.c_str(),
        [&environment](benchmark::State& benchmark_state) {
          QwenFullModelOwnedPrefill(&environment, benchmark_state);
        })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond);
    benchmark::RegisterBenchmark(
        command_prefill_name.c_str(),
        [&environment](benchmark::State& benchmark_state) {
          QwenFullModelCommandPrefill(&environment, benchmark_state);
        })
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond);
    if (FLAG_expected_decode_token != IREE_TOKENIZER_TOKEN_ID_INVALID) {
      const std::string decode_suffix =
          "/Decode/" + std::to_string(environment.prefill_token_count + 1);
      const std::string owned_decode_name =
          "Qwen/FullModel/Owned" + decode_suffix;
      const std::string command_decode_name =
          "Qwen/FullModel/Command" + decode_suffix;
      benchmark::RegisterBenchmark(
          owned_decode_name.c_str(),
          [&environment](benchmark::State& benchmark_state) {
            QwenFullModelDecode(&environment, &environment.owned_decode_program,
                                "owned full-model decode", benchmark_state);
          })
          ->UseManualTime()
          ->Unit(benchmark::kMillisecond);
      benchmark::RegisterBenchmark(
          command_decode_name.c_str(),
          [&environment](benchmark::State& benchmark_state) {
            QwenFullModelDecode(&environment,
                                &environment.command_decode_program,
                                "command full-model decode", benchmark_state);
          })
          ->UseManualTime()
          ->Unit(benchmark::kMillisecond);
    }
    benchmark::RunSpecifiedBenchmarks();
  }
  benchmark::Shutdown();
  QwenBenchmarkEnvironmentDeinitialize(&environment);

  if (!iree_status_is_ok(environment.terminal_status)) {
    iree_status_fprint(stderr, environment.terminal_status);
    iree_status_free(environment.terminal_status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
