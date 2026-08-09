// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/cmd/iree_hal.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "loom/binding/c/target/cmd/program.h"
#include "loom/target/arch/cmd/iree_hal/recording.h"
#include "loomc/iree.h"

struct loomc_cmd_iree_hal_package_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for package-owned host storage.
  loomc_allocator_t allocator;

  // Generic compiled program retained for package identity.
  loomc_program_t* program;

  // HAL device retaining every loaded executable.
  iree_hal_device_t* device;

  // Shared immutable host launch module for every package root.
  loomc_launch_config_module_t* launch_module;

  // Loaded executable table indexed by command executable slots.
  iree_hal_executable_t** executables;

  // Reflected entry table indexed by command entry slots.
  loom_cmd_iree_hal_entry_t* entries;

  // Contiguous reflected parameter storage borrowed by entries.
  iree_hal_executable_function_parameter_t* parameters;

  // Number of loaded executables and reflected entries.
  iree_host_size_t executable_count;
};

struct loomc_cmd_iree_hal_program_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release this materialized program.
  loomc_allocator_t allocator;

  // Device-loaded package retained by the command buffer.
  loomc_cmd_iree_hal_package_t* package;

  // Selected portable root retained for its runtime ABI.
  loomc_cmd_program_t* command_program;

  // Reusable recorded command buffer.
  iree_hal_command_buffer_t* command_buffer;

  // Package launch function matching the selected root.
  loomc_launch_config_function_t launch_function;

  // Optional recorder-produced HAL-operation identity table.
  loom_cmd_iree_hal_operation_map_entry_t* recorded_operations;

  // Number of entries in |recorded_operations|.
  iree_host_size_t recorded_operation_count;
};

static_assert((uint32_t)LOOM_CMD_IREE_HAL_OPERATION_PHASE_BARRIER ==
                  (uint32_t)LOOMC_CMD_IREE_HAL_RECORDED_OPERATION_PHASE_BARRIER,
              "internal and public barrier operation phases must match");
static_assert((uint32_t)LOOM_CMD_IREE_HAL_OPERATION_PHASE_PAYLOAD ==
                  (uint32_t)LOOMC_CMD_IREE_HAL_RECORDED_OPERATION_PHASE_PAYLOAD,
              "internal and public payload operation phases must match");

static bool loomc_cmd_iree_hal_string_view_is_well_formed(
    loomc_string_view_t value) {
  return value.data != NULL || value.size == 0;
}

static bool loomc_cmd_iree_hal_iree_string_view_is_well_formed(
    iree_string_view_t value) {
  return value.data != NULL || value.size == 0;
}

typedef struct loomc_cmd_iree_hal_option_prefix_t {
  // Structure type identifying the option descriptor.
  loomc_structure_type_t type;

  // Size of the complete option descriptor in bytes.
  loomc_host_size_t structure_size;

  // Next descriptor in the unordered option chain.
  const void* next;
} loomc_cmd_iree_hal_option_prefix_t;

static loomc_status_t loomc_cmd_iree_hal_validate_package_options(
    const loomc_cmd_iree_hal_package_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "IREE HAL package options must not be NULL");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_CMD_IREE_HAL_PACKAGE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL package options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL package options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "IREE HAL package option extensions are not supported");
  }
  if (options->device == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "IREE HAL package options require a device");
  }
  if (!loomc_cmd_iree_hal_iree_string_view_is_well_formed(
          options->target_selection.family) ||
      !loomc_cmd_iree_hal_iree_string_view_is_well_formed(
          options->target_selection.target_key)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL package target selection contains a malformed string view");
  }
  if (!loomc_cmd_iree_hal_string_view_is_well_formed(
          options->executable_artifact_format) ||
      options->executable_artifact_format.size == 0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL package executable artifact format must be non-empty");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_cmd_iree_hal_resolve_program_options(
    const loomc_cmd_iree_hal_program_options_t* options,
    const loomc_cmd_program_range_options_t** out_range_options) {
  *out_range_options = NULL;
  if (options == NULL) return loomc_ok_status();
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_CMD_IREE_HAL_PROGRAM_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL program options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL program options structure_size is too small");
  }
  const loomc_cmd_iree_hal_program_flags_t known_flags =
      LOOMC_CMD_IREE_HAL_PROGRAM_FLAG_RETAIN_RECORDED_OPERATIONS;
  if ((options->flags & ~known_flags) != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "IREE HAL program options contain unknown flags");
  }
  if (options->fixed_buffer_count != 0 && options->fixed_buffers == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL program fixed_buffer_count is non-zero but fixed_buffers is "
        "NULL");
  }

  const void* next = options->next;
  while (next != NULL) {
    const loomc_cmd_iree_hal_option_prefix_t* prefix =
        (const loomc_cmd_iree_hal_option_prefix_t*)next;
    switch (prefix->type) {
      case LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_RANGE_OPTIONS: {
        if (*out_range_options != NULL) {
          return loomc_make_status(
              LOOMC_STATUS_INVALID_ARGUMENT,
              "IREE HAL program option chain contains duplicate command "
              "ranges");
        }
        const loomc_cmd_program_range_options_t* range_options =
            (const loomc_cmd_program_range_options_t*)next;
        if (range_options->structure_size != 0 &&
            range_options->structure_size < sizeof(*range_options)) {
          return loomc_make_status(
              LOOMC_STATUS_INVALID_ARGUMENT,
              "IREE HAL program range options structure_size is too small");
        }
        *out_range_options = range_options;
        next = range_options->next;
        break;
      }
      case LOOMC_STRUCTURE_TYPE_NONE:
        return loomc_make_status(
            LOOMC_STATUS_INVALID_ARGUMENT,
            "IREE HAL program option extension is missing a structure type");
      default:
        return loomc_make_status(
            LOOMC_STATUS_UNIMPLEMENTED,
            "IREE HAL program option extension type is not supported");
    }
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_cmd_iree_hal_resolve_program_range(
    const loom_cmd_program_t* program,
    const loomc_cmd_program_range_options_t* range_options,
    loom_cmd_program_command_range_t* out_range,
    uint32_t* out_barrier_wave_ordinal) {
  const loom_cmd_program_command_range_t requested = {
      .first_command = range_options->command_range.first_command,
      .command_count = range_options->command_range.command_count,
  };
  if (requested.first_command > program->commands.count ||
      requested.command_count >
          program->commands.count - requested.first_command) {
    return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                             "IREE HAL program command range is out of range");
  }
  *out_range = requested;
  *out_barrier_wave_ordinal = 0;
  if (requested.command_count == 0) return loomc_ok_status();

  loom_cmd_program_barrier_wave_iterator_t iterator;
  loom_cmd_program_barrier_wave_iterator_initialize(program, &iterator);
  loom_cmd_program_barrier_wave_t wave;
  while (loom_cmd_program_barrier_wave_iterator_next(&iterator, &wave)) {
    const uint32_t end_command =
        wave.commands.first_command + wave.commands.command_count;
    if (requested.first_command < end_command) {
      *out_barrier_wave_ordinal = wave.ordinal;
      return loomc_ok_status();
    }
  }
  IREE_ASSERT_UNREACHABLE(
      "validated command range must be contained in a barrier wave");
  IREE_BUILTIN_UNREACHABLE();
}

static loomc_status_t loomc_cmd_iree_hal_find_artifact(
    const loomc_program_t* program, loomc_artifact_kind_t kind,
    loomc_string_view_t format, loomc_string_view_t identifier,
    const loomc_artifact_t** out_artifact) {
  *out_artifact = NULL;
  for (loomc_host_size_t i = 0; i < loomc_program_artifact_count(program);
       ++i) {
    const loomc_artifact_t* artifact = loomc_program_artifact_at(program, i);
    if (artifact->kind != kind ||
        !loomc_string_view_equal(artifact->format, format) ||
        !loomc_string_view_equal(artifact->identifier, identifier)) {
      continue;
    }
    if (*out_artifact != NULL) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "compiled program contains duplicate required artifacts");
    }
    *out_artifact = artifact;
  }
  if (*out_artifact == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compiled program does not contain its required artifact");
  }
  return loomc_ok_status();
}

static void loomc_cmd_iree_hal_package_destroy(
    loomc_cmd_iree_hal_package_t* package) {
  loomc_allocator_t allocator = package->allocator;
  if (package->executables != NULL) {
    for (iree_host_size_t i = 0; i < package->executable_count; ++i) {
      iree_hal_executable_release(package->executables[i]);
    }
  }
  loomc_allocator_free(allocator, package->parameters);
  loomc_allocator_free(allocator, package->entries);
  loomc_allocator_free(allocator, package->executables);
  loomc_launch_config_module_release(package->launch_module);
  iree_hal_device_release(package->device);
  loomc_program_release(package->program);
  loomc_allocator_free(allocator, package);
}

static loomc_status_t loomc_cmd_iree_hal_package_load_executables(
    loomc_cmd_iree_hal_package_t* package,
    const loomc_cmd_iree_hal_package_options_t* options,
    const iree_hal_executable_target_t* target) {
  const loomc_host_size_t dependency_count =
      loomc_program_dependency_count(package->program);
  if (dependency_count > UINT32_MAX ||
      dependency_count > LOOMC_HOST_SIZE_MAX / sizeof(*package->executables) ||
      dependency_count > LOOMC_HOST_SIZE_MAX / sizeof(*package->entries)) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "command package dependency table is too large");
  }
  package->executable_count = dependency_count;
  loomc_status_t status = loomc_ok_status();
  if (dependency_count != 0) {
    status = loomc_allocator_malloc(
        package->allocator, dependency_count * sizeof(*package->executables),
        (void**)&package->executables);
  }
  if (loomc_status_is_ok(status) && dependency_count != 0) {
    memset(package->executables, 0,
           dependency_count * sizeof(*package->executables));
    status = loomc_allocator_malloc(
        package->allocator, dependency_count * sizeof(*package->entries),
        (void**)&package->entries);
  }
  if (loomc_status_is_ok(status) && dependency_count != 0) {
    memset(package->entries, 0, dependency_count * sizeof(*package->entries));
  }

  iree_host_size_t parameter_count = 0;
  for (iree_host_size_t i = 0;
       i < dependency_count && loomc_status_is_ok(status); ++i) {
    loomc_program_dependency_info_t dependency_info = {
        .type = LOOMC_STRUCTURE_TYPE_PROGRAM_DEPENDENCY_INFO,
        .structure_size = sizeof(dependency_info),
    };
    status =
        loomc_program_dependency_info(package->program, i, &dependency_info);
    if (!loomc_status_is_ok(status)) break;
    if (dependency_info.slot >= dependency_count ||
        package->executables[dependency_info.slot] != NULL) {
      status = loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "command package dependency slots are not dense and unique");
      break;
    }
    if (loomc_program_export_count(dependency_info.program) != 1) {
      status = loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "command package dependency must export exactly one entry");
      break;
    }
    loomc_program_export_info_t export_info = {
        .type = LOOMC_STRUCTURE_TYPE_PROGRAM_EXPORT_INFO,
        .structure_size = sizeof(export_info),
    };
    status = loomc_program_export_info(dependency_info.program,
                                       loomc_program_export_from_index(0),
                                       &export_info);
    if (!loomc_status_is_ok(status)) break;

    const loomc_artifact_t* executable_artifact = NULL;
    status = loomc_cmd_iree_hal_find_artifact(
        dependency_info.program, LOOMC_ARTIFACT_KIND_EXECUTABLE,
        options->executable_artifact_format, export_info.name,
        &executable_artifact);
    if (!loomc_status_is_ok(status)) break;

    iree_hal_executable_load_params_t load_params;
    iree_hal_executable_load_params_initialize(&load_params);
    load_params.executable_data =
        iree_const_byte_span_from_loomc(executable_artifact->contents);
    status = loomc_status_from_iree(iree_hal_device_load_executable(
        package->device, options->executable_queue_affinity, target,
        &load_params, &package->executables[dependency_info.slot]));
    if (!loomc_status_is_ok(status)) break;

    loom_cmd_iree_hal_entry_t* entry = &package->entries[dependency_info.slot];
    entry->executable_index = dependency_info.slot;
    entry->function = iree_hal_executable_function_from_index(0);
    status = loomc_status_from_iree(iree_hal_executable_function_info(
        package->executables[dependency_info.slot], entry->function,
        &entry->info));
    if (!loomc_status_is_ok(status)) break;
    if (entry->info.parameter_count > LOOMC_HOST_SIZE_MAX - parameter_count) {
      status = loomc_make_status(
          LOOMC_STATUS_RESOURCE_EXHAUSTED,
          "command package reflected parameter table is too large");
      break;
    }
    parameter_count += entry->info.parameter_count;
  }

  for (iree_host_size_t i = 0;
       i < dependency_count && loomc_status_is_ok(status); ++i) {
    if (package->executables[i] == NULL) {
      status =
          loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                            "command package dependency slots contain a gap");
    }
  }
  if (loomc_status_is_ok(status) &&
      parameter_count > LOOMC_HOST_SIZE_MAX / sizeof(*package->parameters)) {
    status = loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "command package reflected parameter storage is too large");
  }
  if (loomc_status_is_ok(status) && parameter_count != 0) {
    status = loomc_allocator_malloc(
        package->allocator, parameter_count * sizeof(*package->parameters),
        (void**)&package->parameters);
  }

  iree_host_size_t parameter_offset = 0;
  for (iree_host_size_t i = 0;
       i < dependency_count && loomc_status_is_ok(status); ++i) {
    loom_cmd_iree_hal_entry_t* entry = &package->entries[i];
    if (entry->info.parameter_count == 0) continue;
    iree_hal_executable_function_parameter_t* parameters =
        package->parameters + parameter_offset;
    entry->parameters = parameters;
    status = loomc_status_from_iree(iree_hal_executable_function_parameters(
        package->executables[i], entry->function, entry->info.parameter_count,
        parameters));
    parameter_offset += entry->info.parameter_count;
  }
  return status;
}

loomc_status_t loomc_cmd_iree_hal_package_create(
    loomc_program_t* program,
    const loomc_cmd_iree_hal_package_options_t* options,
    loomc_allocator_t allocator, loomc_cmd_iree_hal_package_t** out_package) {
  if (out_package == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_package must not be NULL");
  }
  *out_package = NULL;
  if (program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_cmd_iree_hal_validate_package_options(options));

  const iree_hal_device_spec_t* device_spec =
      iree_hal_device_spec(options->device);
  if (device_spec == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNAVAILABLE,
        "IREE HAL device does not expose immutable target facts");
  }
  const iree_hal_executable_target_selection_result_t target_result =
      iree_hal_device_spec_select_executable_target(device_spec,
                                                    &options->target_selection);
  if (target_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return loomc_make_status(
        LOOMC_STATUS_UNAVAILABLE,
        "IREE HAL device does not advertise the requested executable target");
  }
  if (target_result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return loomc_make_status(
        LOOMC_STATUS_FAILED_PRECONDITION,
        "IREE HAL executable target selection is ambiguous");
  }

  const loomc_artifact_t* launch_artifact = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_cmd_iree_hal_find_artifact(
      program, LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
      loomc_make_cstring_view("command-program-launch.loombc"),
      &launch_artifact));

  loomc_cmd_iree_hal_package_t* package = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*package), (void**)&package));
  memset(package, 0, sizeof(*package));
  iree_atomic_ref_count_init(&package->ref_count);
  package->allocator = allocator;
  package->program = program;
  loomc_program_retain(program);
  package->device = options->device;
  iree_hal_device_retain(package->device);

  loomc_status_t status = loomc_launch_config_module_load(
      launch_artifact, /*options=*/NULL, allocator, &package->launch_module);
  if (loomc_status_is_ok(status)) {
    status = loomc_cmd_iree_hal_package_load_executables(package, options,
                                                         target_result.target);
  }
  if (loomc_status_is_ok(status)) {
    *out_package = package;
  } else {
    loomc_cmd_iree_hal_package_destroy(package);
  }
  return status;
}

void loomc_cmd_iree_hal_package_retain(loomc_cmd_iree_hal_package_t* package) {
  if (package == NULL) return;
  iree_atomic_ref_count_inc(&package->ref_count);
}

void loomc_cmd_iree_hal_package_release(loomc_cmd_iree_hal_package_t* package) {
  if (package == NULL) return;
  if (iree_atomic_ref_count_dec(&package->ref_count) == 1) {
    loomc_cmd_iree_hal_package_destroy(package);
  }
}

static void loomc_cmd_iree_hal_program_destroy(
    loomc_cmd_iree_hal_program_t* hal_program) {
  loomc_allocator_t allocator = hal_program->allocator;
  iree_hal_command_buffer_release(hal_program->command_buffer);
  loomc_cmd_program_release(hal_program->command_program);
  loomc_cmd_iree_hal_package_release(hal_program->package);
  loomc_allocator_free(allocator, hal_program->recorded_operations);
  loomc_allocator_free(allocator, hal_program);
}

loomc_status_t loomc_cmd_iree_hal_program_create(
    loomc_cmd_iree_hal_package_t* package, loomc_cmd_program_t* command_program,
    const loomc_cmd_iree_hal_program_options_t* options,
    loomc_allocator_t allocator,
    loomc_cmd_iree_hal_program_t** out_hal_program) {
  if (out_hal_program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_hal_program must not be NULL");
  }
  *out_hal_program = NULL;
  if (package == NULL || command_program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "package and command_program must not be NULL");
  }
  const loomc_cmd_program_range_options_t* range_options = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_cmd_iree_hal_resolve_program_options(options, &range_options));
  if (loomc_cmd_program_package(command_program) != package->program) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "selected command program does not belong to the loaded package");
  }

  loomc_cmd_program_info_t command_info = {0};
  LOOMC_RETURN_IF_ERROR(loomc_cmd_program_info(command_program, &command_info));
  loomc_launch_config_function_t launch_function =
      loomc_launch_config_function_invalid();
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_lookup_function_by_name(
      package->launch_module, command_info.name, &launch_function));
  loomc_launch_config_function_info_t launch_info = {0};
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_function_info(
      package->launch_module, launch_function, &launch_info));
  if (command_info.launch_counts.required_byte_length > LOOMC_HOST_SIZE_MAX ||
      command_info.launch_counts.minimum_alignment > LOOMC_HOST_SIZE_MAX) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "command launch-count storage exceeds the host address domain");
  }
  const loomc_host_size_t expected_launch_byte_length =
      command_info.launch_counts.binding_index ==
              LOOMC_CMD_PROGRAM_BINDING_INVALID
          ? 0
          : (loomc_host_size_t)command_info.launch_counts.required_byte_length;
  const loomc_host_size_t expected_launch_alignment =
      command_info.launch_counts.binding_index ==
              LOOMC_CMD_PROGRAM_BINDING_INVALID
          ? 0
          : (loomc_host_size_t)command_info.launch_counts.minimum_alignment;
  if (launch_info.output_byte_length != expected_launch_byte_length ||
      (expected_launch_byte_length != 0 &&
       launch_info.output_alignment != expected_launch_alignment)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command root and launch module declare incompatible output storage");
  }

  const loomc_cmd_iree_hal_program_options_t default_options = {0};
  if (options == NULL) options = &default_options;
  const loom_cmd_program_t* parsed = loomc_cmd_program_parsed(command_program);
  loom_cmd_program_command_range_t materialization_range =
      loom_cmd_program_command_range_all(parsed);
  uint32_t first_barrier_wave_ordinal = 0;
  if (range_options != NULL) {
    LOOMC_RETURN_IF_ERROR(loomc_cmd_iree_hal_resolve_program_range(
        parsed, range_options, &materialization_range,
        &first_barrier_wave_ordinal));
  }
  const loom_cmd_iree_hal_inputs_t inputs = {
      .binding_count = parsed->requirements.rebindable_binding_count,
      .fixed_buffer_count = options->fixed_buffer_count,
      .fixed_buffers = options->fixed_buffers,
      .executable_count = package->executable_count,
      .executables = package->executables,
      .entry_count = package->executable_count,
      .entries = package->entries,
  };

  loomc_cmd_iree_hal_program_t* hal_program = NULL;
  loomc_status_t status = loomc_allocator_malloc(
      allocator, sizeof(*hal_program), (void**)&hal_program);
  if (loomc_status_is_ok(status)) {
    memset(hal_program, 0, sizeof(*hal_program));
    hal_program->allocator = allocator;
  }

  loom_cmd_iree_hal_operation_map_t operation_map = {0};
  loom_cmd_iree_hal_operation_map_t* operation_map_ptr = NULL;
  if (loomc_status_is_ok(status) &&
      (options->flags &
       LOOMC_CMD_IREE_HAL_PROGRAM_FLAG_RETAIN_RECORDED_OPERATIONS) != 0) {
    const loomc_host_size_t command_count = materialization_range.command_count;
    if (command_count >
        LOOMC_HOST_SIZE_MAX / 2 / sizeof(*hal_program->recorded_operations)) {
      status = loomc_make_status(
          LOOMC_STATUS_RESOURCE_EXHAUSTED,
          "recorded command operation map exceeds the host address domain");
    } else {
      operation_map.capacity = command_count * 2;
      if (operation_map.capacity != 0) {
        status = loomc_allocator_malloc(
            allocator,
            operation_map.capacity * sizeof(*hal_program->recorded_operations),
            (void**)&hal_program->recorded_operations);
        operation_map.entries = hal_program->recorded_operations;
      }
      if (loomc_status_is_ok(status)) operation_map_ptr = &operation_map;
    }
  }

  iree_hal_command_buffer_t* command_buffer = NULL;
  if (loomc_status_is_ok(status)) {
    if (range_options != NULL) {
      status =
          loomc_status_from_iree(loom_cmd_iree_hal_materialize_program_range(
              parsed, materialization_range, first_barrier_wave_ordinal,
              &inputs, package->device, options->command_buffer_mode,
              options->queue_affinity, operation_map_ptr, &command_buffer,
              iree_allocator_from_loomc(allocator)));
    } else {
      status = loomc_status_from_iree(loom_cmd_iree_hal_materialize_program(
          parsed, &inputs, package->device, options->command_buffer_mode,
          options->queue_affinity, operation_map_ptr, &command_buffer,
          iree_allocator_from_loomc(allocator)));
    }
  }
  if (loomc_status_is_ok(status)) {
    iree_atomic_ref_count_init(&hal_program->ref_count);
    hal_program->package = package;
    loomc_cmd_iree_hal_package_retain(package);
    hal_program->command_program = command_program;
    loomc_cmd_program_retain(command_program);
    hal_program->command_buffer = command_buffer;
    hal_program->launch_function = launch_function;
    hal_program->recorded_operation_count = operation_map.count;
    *out_hal_program = hal_program;
  } else {
    iree_hal_command_buffer_release(command_buffer);
    if (hal_program != NULL) {
      loomc_allocator_free(allocator, hal_program->recorded_operations);
      loomc_allocator_free(allocator, hal_program);
    }
  }
  return status;
}

void loomc_cmd_iree_hal_program_retain(
    loomc_cmd_iree_hal_program_t* hal_program) {
  if (hal_program == NULL) return;
  iree_atomic_ref_count_inc(&hal_program->ref_count);
}

void loomc_cmd_iree_hal_program_release(
    loomc_cmd_iree_hal_program_t* hal_program) {
  if (hal_program == NULL) return;
  if (iree_atomic_ref_count_dec(&hal_program->ref_count) == 1) {
    loomc_cmd_iree_hal_program_destroy(hal_program);
  }
}

iree_hal_command_buffer_t* loomc_cmd_iree_hal_program_command_buffer(
    const loomc_cmd_iree_hal_program_t* hal_program) {
  return hal_program ? hal_program->command_buffer : NULL;
}

loomc_launch_config_module_t* loomc_cmd_iree_hal_program_launch_module(
    const loomc_cmd_iree_hal_program_t* hal_program) {
  return hal_program ? hal_program->package->launch_module : NULL;
}

loomc_launch_config_function_t loomc_cmd_iree_hal_program_launch_function(
    const loomc_cmd_iree_hal_program_t* hal_program) {
  return hal_program ? hal_program->launch_function
                     : loomc_launch_config_function_invalid();
}

loomc_host_size_t loomc_cmd_iree_hal_program_recorded_operation_count(
    const loomc_cmd_iree_hal_program_t* hal_program) {
  return hal_program ? hal_program->recorded_operation_count : 0;
}

loomc_status_t loomc_cmd_iree_hal_program_recorded_operation_info(
    const loomc_cmd_iree_hal_program_t* hal_program,
    loomc_host_size_t operation_index,
    loomc_cmd_iree_hal_recorded_operation_info_t* out_info) {
  if (out_info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  if (out_info->type != LOOMC_STRUCTURE_TYPE_NONE &&
      out_info->type !=
          LOOMC_STRUCTURE_TYPE_CMD_IREE_HAL_RECORDED_OPERATION_INFO) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL recorded operation output has an unknown structure type");
  }
  if (out_info->structure_size != 0 &&
      out_info->structure_size < sizeof(*out_info)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "IREE HAL recorded operation output structure_size is too small");
  }
  if (out_info->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "IREE HAL recorded operation output extensions are not supported");
  }
  if (hal_program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "hal_program must not be NULL");
  }
  if (operation_index >= hal_program->recorded_operation_count) {
    return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                             "recorded operation index is out of range");
  }

  const loom_cmd_iree_hal_operation_map_entry_t* operation =
      &hal_program->recorded_operations[operation_index];
  *out_info = (loomc_cmd_iree_hal_recorded_operation_info_t){
      .type = LOOMC_STRUCTURE_TYPE_CMD_IREE_HAL_RECORDED_OPERATION_INFO,
      .structure_size = sizeof(*out_info),
      .command_ordinal = operation->command_ordinal,
      .barrier_wave_ordinal = operation->barrier_wave_ordinal,
      .phase = (loomc_cmd_iree_hal_recorded_operation_phase_t)operation->phase,
  };
  return loomc_ok_status();
}
