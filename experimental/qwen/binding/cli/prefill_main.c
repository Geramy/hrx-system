// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/qwen/runtime/command_package.h"
#include "experimental/qwen/runtime/model.h"
#include "experimental/qwen/runtime/request.h"
#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/types.h"
#include "iree/tooling/device_util.h"

#define QWEN_PREFILL_MIN_TOKEN_COUNT 32

IREE_FLAG(string, tokens, "",
          "Raw token IDs: exactly 32, 64, 128, 256, or 512 little-endian I32 "
          "values.");
IREE_FLAG(int32_t, expected_token, IREE_TOKENIZER_TOKEN_ID_INVALID,
          "Expected selected prefill token; omit to skip validation.");
IREE_FLAG(bool, decode_one, false,
          "Consume the device-published prefill token at the next position and "
          "execute one exact-count decode issue.");
IREE_FLAG(int32_t, expected_decode_token, IREE_TOKENIZER_TOKEN_ID_INVALID,
          "Expected token selected by --decode_one; omit to report without "
          "external validation.");
IREE_FLAG(bool, command_access_sanitizer, false,
          "Compile command-program kernels with report-only access checks; "
          "requires --amdgpu_asan=true.");
IREE_FLAG(bool, command_barrier_waves, false,
          "Diagnostically issue the selected prefill one barrier wave at a "
          "time and report each completed canonical command range.");
IREE_FLAG(int32_t, command_capture_prefix, -1,
          "Execute canonical commands [0, N) and capture the complete "
          "transient backing root without issuing command N.");
IREE_FLAG(string, command_capture_transient, "",
          "Path receiving the transient backing-root bytes produced by "
          "--command_capture_prefix.");

static const char* const qwen_prefill_cli_usage =
    "Runs one exact Qwen prefill shape and optional one-token Decode-576.\n"
    "\n"
    "Required flags:\n"
    "  --device=<device URI>\n"
    "  --parameters=<GGUF or parameter archive path>\n"
    "  --tokens=<raw 32/64/128/256/512-element little-endian I32 path>\n"
    "\n"
    "Optional validation:\n"
    "  --expected_token=<prefill-selected token ID>\n"
    "  --decode_one\n"
    "  --expected_decode_token=<decode-selected token ID>\n"
    "  --command_barrier_waves\n"
    "  --command_capture_prefix=<first excluded canonical command>\n"
    "  --command_capture_transient=<transient root output path>\n"
    "\n"
    "Decode-576 accepts every exact prefill root. Both stages execute reusable "
    "command programs from one compiled multi-root package. "
    "Profiling flags surround the prefill issue. Barrier-wave and "
    "prefix-capture modes are fault-localization only and invalidate timing."
    "\n";

typedef struct qwen_prefill_cli_timepoint_t {
  // Timeline semaphore carrying this timepoint.
  iree_hal_semaphore_t* semaphore;
  // Monotonically increasing timeline value.
  uint64_t value;
} qwen_prefill_cli_timepoint_t;

static iree_hal_semaphore_list_t qwen_prefill_cli_timepoint_list(
    qwen_prefill_cli_timepoint_t* timepoint) {
  iree_hal_semaphore_list_t list = {
      .count = 1,
      .semaphores = &timepoint->semaphore,
      .payload_values = &timepoint->value,
  };
  return list;
}

static const char* qwen_prefill_cli_program_name(
    qwen_command_program_t program) {
  switch (program) {
    case QWEN_COMMAND_PROGRAM_PREFILL_32:
      return "prefill-32";
    case QWEN_COMMAND_PROGRAM_PREFILL_64:
      return "prefill-64";
    case QWEN_COMMAND_PROGRAM_PREFILL_128:
      return "prefill-128";
    case QWEN_COMMAND_PROGRAM_PREFILL_256:
      return "prefill-256";
    case QWEN_COMMAND_PROGRAM_PREFILL_512:
      return "prefill-512";
    case QWEN_COMMAND_PROGRAM_DECODE_576:
      return "decode-576";
    default:
      return "unknown";
  }
}

static void qwen_prefill_cli_barrier_wave_callback(
    void* user_data, const qwen_command_barrier_wave_event_info_t* event_info) {
  (void)user_data;
  const char* program_name = qwen_prefill_cli_program_name(event_info->program);
  const char* event_name =
      event_info->event == QWEN_COMMAND_BARRIER_WAVE_EVENT_BEFORE_EXECUTE
          ? "issuing"
          : "completed";
  const uint32_t end_command = event_info->command_range.first_command +
                               event_info->command_range.command_count;
  fprintf(stderr,
          "Qwen command %s barrier wave %" PRIhsz "/%" PRIhsz
          " ordinal %" PRIu32 " commands [%" PRIu32 ", %" PRIu32 ") %s\n",
          program_name, event_info->wave_index + 1, event_info->wave_count,
          event_info->barrier_wave_ordinal,
          event_info->command_range.first_command, end_command, event_name);
  fflush(stderr);
}

// Transient, non-sanctioned containment for an AMDGPU async file-action
// teardown defect. A host preparation failure must not release the tooling
// runtime while the model gather remains active. Keep this wait tool-only and
// delete it when asynchronous device teardown is safe.
static iree_status_t qwen_wait_for_model_ready_bringup_workaround(
    iree_status_t status, qwen_model_t* model,
    const qwen_prefill_cli_timepoint_t* model_ready) {
  if (!model) return status;
  return iree_status_join(
      status, iree_hal_semaphore_wait(
                  model_ready->semaphore, model_ready->value,
                  iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

static iree_status_t qwen_prefill_cli_load_tokens(
    iree_string_view_t path, iree_allocator_t host_allocator,
    iree_io_file_contents_t** out_contents,
    iree_tokenizer_token_id_t
        out_token_ids[QWEN_COMMAND_PREFILL_TOKEN_CAPACITY],
    iree_host_size_t* out_token_count) {
  *out_contents = NULL;
  *out_token_count = 0;
  if (iree_string_view_is_empty(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--tokens must specify a token-ID file");
  }

  IREE_RETURN_IF_ERROR(iree_io_file_contents_map(path, IREE_IO_FILE_ACCESS_READ,
                                                 host_allocator, out_contents));
  const iree_host_size_t byte_length =
      (*out_contents)->const_buffer.data_length;
  if (byte_length <
          QWEN_PREFILL_MIN_TOKEN_COUNT * sizeof(iree_tokenizer_token_id_t) ||
      byte_length % sizeof(iree_tokenizer_token_id_t) != 0 ||
      byte_length > QWEN_COMMAND_PREFILL_TOKEN_CAPACITY *
                        sizeof(iree_tokenizer_token_id_t)) {
    const iree_host_size_t actual_byte_length =
        (*out_contents)->const_buffer.data_length;
    iree_io_file_contents_free(*out_contents);
    *out_contents = NULL;
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "token-ID file '%.*s' has %" PRIhsz
        " bytes; expected %d to %d complete little-endian I32 values",
        (int)path.size, path.data, actual_byte_length,
        QWEN_PREFILL_MIN_TOKEN_COUNT, QWEN_COMMAND_PREFILL_TOKEN_CAPACITY);
  }

  *out_token_count = byte_length / sizeof(iree_tokenizer_token_id_t);
  const uint8_t* source_data = (*out_contents)->const_buffer.data;
  for (iree_host_size_t i = 0; i < *out_token_count; ++i) {
    const uint32_t token_bits =
        iree_unaligned_load_le_u32(source_data + i * sizeof(uint32_t));
    memcpy(&out_token_ids[i], &token_bits, sizeof(token_bits));
  }
  return iree_ok_status();
}

static iree_status_t qwen_prefill_cli_run(void) {
  iree_allocator_t host_allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();
  const bool capture_prefix_requested =
      FLAG_command_capture_prefix >= 0 || FLAG_command_capture_transient[0];

  if (FLAG_expected_token < IREE_TOKENIZER_TOKEN_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--expected_token must be nonnegative when validation is requested");
  }
  if (FLAG_expected_decode_token < IREE_TOKENIZER_TOKEN_ID_INVALID) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--expected_decode_token must be nonnegative when validation is "
        "requested");
  }
  if (!FLAG_decode_one &&
      FLAG_expected_decode_token != IREE_TOKENIZER_TOKEN_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--expected_decode_token requires --decode_one");
  }
  if (capture_prefix_requested && (FLAG_command_capture_prefix <= 0 ||
                                   !FLAG_command_capture_transient[0])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prefix capture requires a positive --command_capture_prefix and "
        "--command_capture_transient path");
  }
  if (capture_prefix_requested &&
      (FLAG_command_barrier_waves || FLAG_decode_one ||
       FLAG_expected_token != IREE_TOKENIZER_TOKEN_ID_INVALID ||
       FLAG_expected_decode_token != IREE_TOKENIZER_TOKEN_ID_INVALID)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prefix capture cannot be combined with barrier-wave issue, decode, "
        "or selected-token validation");
  }
  iree_tokenizer_token_id_t token_ids[QWEN_COMMAND_PREFILL_TOKEN_CAPACITY];
  iree_host_size_t prefill_token_count = 0;
  qwen_command_program_t prefill_program = QWEN_COMMAND_PROGRAM_COUNT;
  iree_io_file_contents_t* token_contents = NULL;
  status = qwen_prefill_cli_load_tokens(iree_make_cstring_view(FLAG_tokens),
                                        host_allocator, &token_contents,
                                        token_ids, &prefill_token_count);
  if (iree_status_is_ok(status)) {
    status = qwen_command_select_prefill_program(prefill_token_count,
                                                 &prefill_program);
  }
  qwen_tooling_runtime_context_t runtime_context;
  if (iree_status_is_ok(status)) {
    status = qwen_tooling_runtime_context_initialize_from_flags(
        host_allocator, &runtime_context);
  } else {
    memset(&runtime_context, 0, sizeof(runtime_context));
  }

  iree_hal_semaphore_t* timeline = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        qwen_tooling_runtime_context_device(&runtime_context),
        IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &timeline);
  }

  qwen_prefill_cli_timepoint_t model_ready = {
      .semaphore = timeline,
      .value = 1,
  };
  qwen_model_t* model = NULL;
  if (iree_status_is_ok(status)) {
    qwen_model_options_t model_options;
    qwen_model_options_initialize(&model_options);
    model_options.device_group = runtime_context.device_group;
    if (runtime_context.jit_worker_count != 0) {
      model_options.jit_worker_count = runtime_context.jit_worker_count;
    }
    qwen_parameter_source_t parameter_source = {
        .index = runtime_context.parameter_index,
        .provider = runtime_context.parameter_provider,
        .scope = iree_string_view_empty(),
    };
    status = qwen_model_load(
        &model_options, &parameter_source, iree_hal_semaphore_list_empty(),
        qwen_prefill_cli_timepoint_list(&model_ready), host_allocator, &model);
  }

  // Package preparation is synchronous host work and intentionally overlaps
  // the asynchronous model gather above. Both roots share the same compiled
  // executable set and immutable parameter bindings.
  qwen_command_package_t* command_package = NULL;
  if (iree_status_is_ok(status)) {
    qwen_command_package_options_t package_options;
    qwen_command_package_options_initialize(&package_options);
    if (runtime_context.jit_worker_count != 0) {
      package_options.compiler_worker_count = runtime_context.jit_worker_count;
    }
    if (FLAG_command_access_sanitizer) {
      package_options.sanitizer_checks |= LOOMC_SANITIZER_CHECK_ACCESS;
    }
    package_options.command_buffer_mode = runtime_context.command_buffer_mode;
    status = qwen_command_package_prepare(model, &package_options,
                                          host_allocator, &command_package);
  }

  qwen_prefill_cli_timepoint_t request_ready = {
      .semaphore = timeline,
      .value = 2,
  };
  qwen_request_t* request = NULL;
  if (iree_status_is_ok(status)) {
    qwen_request_options_t request_options;
    qwen_request_options_initialize(&request_options);
    request_options.token_capacity = QWEN_COMMAND_PREFILL_TOKEN_CAPACITY;
    request_options.context_capacity = QWEN_COMMAND_CONTEXT_CAPACITY;
    status = qwen_request_create(
        model, &request_options, qwen_prefill_cli_timepoint_list(&model_ready),
        qwen_prefill_cli_timepoint_list(&request_ready), host_allocator,
        &request);
  }

  qwen_prefill_cli_timepoint_t tokens_ready = {
      .semaphore = timeline,
      .value = 3,
  };
  if (iree_status_is_ok(status)) {
    status = qwen_request_reset_tokens(
        request, /*context_base=*/0,
        iree_tokenizer_make_token_id_list(token_ids, prefill_token_count),
        qwen_prefill_cli_timepoint_list(&request_ready),
        qwen_prefill_cli_timepoint_list(&tokens_ready));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(timeline, tokens_ready.value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }

  qwen_command_program_info_t prefill_info = {
      .structure_size = sizeof(prefill_info),
      .next = NULL,
  };
  if (iree_status_is_ok(status)) {
    status = qwen_command_package_query_program(command_package,
                                                prefill_program, &prefill_info);
  }

  uint8_t* transient_capture = NULL;
  if (iree_status_is_ok(status) && capture_prefix_requested) {
    if (prefill_info.transient_byte_length > IREE_HOST_SIZE_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "Qwen transient capture length %" PRIu64
                                " exceeds host addressable memory",
                                (uint64_t)prefill_info.transient_byte_length);
    } else {
      status = iree_allocator_malloc(
          host_allocator, (iree_host_size_t)prefill_info.transient_byte_length,
          (void**)&transient_capture);
    }
  }
  if (iree_status_is_ok(status) && capture_prefix_requested) {
    status = qwen_command_package_capture_transient_prefix(
        command_package, prefill_program, request,
        (uint32_t)FLAG_command_capture_prefix,
        iree_make_byte_span(
            transient_capture,
            (iree_host_size_t)prefill_info.transient_byte_length));
  }
  if (iree_status_is_ok(status) && capture_prefix_requested) {
    status = iree_io_file_contents_write(
        iree_make_cstring_view(FLAG_command_capture_transient),
        iree_make_const_byte_span(
            transient_capture,
            (iree_host_size_t)prefill_info.transient_byte_length),
        host_allocator);
  }
  if (iree_status_is_ok(status) && capture_prefix_requested) {
    fprintf(stderr,
            "Qwen command prefix [0, %" PRId32 ") captured %" PRIu64
            " transient bytes to %s\n",
            FLAG_command_capture_prefix,
            (uint64_t)prefill_info.transient_byte_length,
            FLAG_command_capture_transient);
  }
  iree_allocator_free(host_allocator, transient_capture);

  // Profiling surrounds only the issue and host-visible completion wait.
  iree_hal_profiling_from_flags_t* profiling = NULL;
  if (iree_status_is_ok(status) && !capture_prefix_requested) {
    status = iree_hal_begin_device_group_profiling_from_flags(
        runtime_context.device_group, host_allocator, &profiling);
  }

  qwen_prefill_cli_timepoint_t issue_complete = {
      .semaphore = timeline,
      .value = 4,
  };
  qwen_command_issue_options_t prefill_issue_options;
  const qwen_command_issue_options_t* prefill_issue_options_ptr = NULL;
  if (FLAG_command_barrier_waves) {
    qwen_command_issue_options_initialize(&prefill_issue_options);
    prefill_issue_options.flags |=
        QWEN_COMMAND_ISSUE_FLAG_DIAGNOSTIC_BARRIER_WAVES;
    prefill_issue_options.barrier_wave_observer.callback =
        qwen_prefill_cli_barrier_wave_callback;
    prefill_issue_options_ptr = &prefill_issue_options;
  }
  if (iree_status_is_ok(status) && !capture_prefix_requested) {
    status = qwen_command_package_issue(
        command_package, prefill_program, request, prefill_issue_options_ptr,
        qwen_prefill_cli_timepoint_list(&tokens_ready),
        qwen_prefill_cli_timepoint_list(&issue_complete));
  }
  if (iree_status_is_ok(status) && !capture_prefix_requested) {
    status = iree_hal_semaphore_wait(timeline, issue_complete.value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  if (profiling) {
    status =
        iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  }

  iree_tokenizer_token_id_t selected_token = IREE_TOKENIZER_TOKEN_ID_INVALID;
  if (iree_status_is_ok(status) && !capture_prefix_requested) {
    status = qwen_request_read_selected_token(request, &selected_token);
  }
  if (iree_status_is_ok(status) && !capture_prefix_requested &&
      FLAG_expected_token != IREE_TOKENIZER_TOKEN_ID_INVALID &&
      selected_token != FLAG_expected_token) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "selected token %" PRId32
                              " differs from expected token %" PRId32,
                              selected_token, FLAG_expected_token);
  }
  if (iree_status_is_ok(status) && !capture_prefix_requested) {
    const qwen_model_statistics_t model_statistics =
        qwen_model_statistics(model);
    fprintf(stdout,
            "Qwen command prefill %" PRIhsz " selected token %" PRId32
            ": %" PRIu64 " resident bytes, %" PRIu64 " transient bytes\n",
            prefill_token_count, selected_token,
            (uint64_t)model_statistics.allocation_bytes,
            (uint64_t)prefill_info.transient_byte_length);
  }

  qwen_prefill_cli_timepoint_t decode_complete = {
      .semaphore = timeline,
      .value = 5,
  };
  if (iree_status_is_ok(status) && FLAG_decode_one) {
    status = qwen_command_package_issue(
        command_package, QWEN_COMMAND_PROGRAM_DECODE_576, request,
        /*options=*/NULL, qwen_prefill_cli_timepoint_list(&issue_complete),
        qwen_prefill_cli_timepoint_list(&decode_complete));
  }
  if (iree_status_is_ok(status) && FLAG_decode_one) {
    status = iree_hal_semaphore_wait(timeline, decode_complete.value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_tokenizer_token_id_t decode_token = IREE_TOKENIZER_TOKEN_ID_INVALID;
  if (iree_status_is_ok(status) && FLAG_decode_one) {
    status = qwen_request_read_selected_token(request, &decode_token);
  }
  if (iree_status_is_ok(status) && FLAG_decode_one &&
      FLAG_expected_decode_token != IREE_TOKENIZER_TOKEN_ID_INVALID &&
      decode_token != FLAG_expected_decode_token) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "decode selected token %" PRId32
                              " differs from expected decode token %" PRId32,
                              decode_token, FLAG_expected_decode_token);
  }
  if (iree_status_is_ok(status) && FLAG_decode_one) {
    qwen_command_program_info_t decode_info = {
        .structure_size = sizeof(decode_info),
        .next = NULL,
    };
    status = qwen_command_package_query_program(
        command_package, QWEN_COMMAND_PROGRAM_DECODE_576, &decode_info);
    if (iree_status_is_ok(status)) {
      fprintf(stdout,
              "Qwen command decode at context %" PRIhsz
              " selected token %" PRId32 ": %" PRIu64 " transient bytes\n",
              prefill_token_count + 1, decode_token,
              (uint64_t)decode_info.transient_byte_length);
    }
  }

  status =
      qwen_wait_for_model_ready_bringup_workaround(status, model, &model_ready);
  qwen_request_release(request);
  qwen_command_package_release(command_package);
  qwen_model_release(model);
  iree_hal_semaphore_release(timeline);
  qwen_tooling_runtime_context_deinitialize(&runtime_context);
  iree_io_file_contents_free(token_contents);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage("qwen-prefill-cli", qwen_prefill_cli_usage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (argc != 1) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "this command accepts flags but no arguments");
  }
  if (iree_status_is_ok(status)) {
    status = qwen_prefill_cli_run();
  }

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
