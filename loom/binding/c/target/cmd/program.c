// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/target/cmd/program.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "loomc/iree.h"

struct loomc_cmd_program_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release this selected-root handle.
  loomc_allocator_t allocator;

  // Generic compiled package retained for artifact storage and dependencies.
  loomc_program_t* package;

  // Public export name borrowed from |package|.
  loomc_string_view_t name;

  // Verified portable command program borrowing artifact bytes from |package|.
  loom_cmd_program_t parsed;
};

typedef struct loomc_cmd_program_info_prefix_t {
  // Structure type identifying the output descriptor.
  loomc_structure_type_t type;

  // Size of the complete output descriptor in bytes.
  loomc_host_size_t structure_size;

  // Reserved extension chain.
  void* next;
} loomc_cmd_program_info_prefix_t;

static loomc_status_t loomc_cmd_program_validate_output_info(
    const void* info, loomc_structure_type_t expected_type,
    loomc_host_size_t expected_size) {
  if (info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  const loomc_cmd_program_info_prefix_t* prefix =
      (const loomc_cmd_program_info_prefix_t*)info;
  if (prefix->type != LOOMC_STRUCTURE_TYPE_NONE &&
      prefix->type != expected_type) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command program output has an unknown structure type");
  }
  if (prefix->structure_size != 0 && prefix->structure_size < expected_size) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command program output structure_size is too small");
  }
  if (prefix->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "command program output extensions are not supported");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_cmd_program_find_root_artifact(
    const loomc_program_t* package, loomc_string_view_t name,
    const loomc_artifact_t** out_artifact) {
  *out_artifact = NULL;
  const loomc_host_size_t artifact_count =
      loomc_program_artifact_count(package);
  for (loomc_host_size_t i = 0; i < artifact_count; ++i) {
    const loomc_artifact_t* artifact = loomc_program_artifact_at(package, i);
    if (artifact->kind != LOOMC_ARTIFACT_KIND_EXECUTABLE ||
        !loomc_string_view_equal(
            artifact->format,
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM)) ||
        !loomc_string_view_equal(artifact->identifier, name)) {
      continue;
    }
    if (*out_artifact != NULL) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "program export owns duplicate portable command artifacts");
    }
    *out_artifact = artifact;
  }
  if (*out_artifact == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program export does not own a portable command artifact");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_cmd_program_create_from_export(
    loomc_program_t* program, loomc_program_export_t export_token,
    loomc_allocator_t allocator, loomc_cmd_program_t** out_command_program) {
  if (out_command_program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_command_program must not be NULL");
  }
  *out_command_program = NULL;
  if (program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program must not be NULL");
  }

  loomc_program_export_info_t export_info = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_program_export_info(program, export_token, &export_info));
  const loomc_artifact_t* artifact = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_cmd_program_find_root_artifact(
      program, export_info.name, &artifact));
  loom_cmd_program_t parsed = {0};
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(loom_cmd_program_parse(
      iree_const_byte_span_from_loomc(artifact->contents), &parsed)));

  loomc_cmd_program_t* command_program = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(
      allocator, sizeof(*command_program), (void**)&command_program));
  memset(command_program, 0, sizeof(*command_program));
  iree_atomic_ref_count_init(&command_program->ref_count);
  command_program->allocator = allocator;
  command_program->package = program;
  loomc_program_retain(program);
  command_program->name = export_info.name;
  command_program->parsed = parsed;
  *out_command_program = command_program;
  return loomc_ok_status();
}

void loomc_cmd_program_retain(loomc_cmd_program_t* command_program) {
  if (command_program == NULL) return;
  iree_atomic_ref_count_inc(&command_program->ref_count);
}

void loomc_cmd_program_release(loomc_cmd_program_t* command_program) {
  if (command_program == NULL) return;
  if (iree_atomic_ref_count_dec(&command_program->ref_count) == 1) {
    loomc_allocator_t allocator = command_program->allocator;
    loomc_program_release(command_program->package);
    loomc_allocator_free(allocator, command_program);
  }
}

loomc_status_t loomc_cmd_program_info(
    const loomc_cmd_program_t* command_program,
    loomc_cmd_program_info_t* out_info) {
  LOOMC_RETURN_IF_ERROR(loomc_cmd_program_validate_output_info(
      out_info, LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO, sizeof(*out_info)));
  if (command_program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "command_program must not be NULL");
  }
  const loom_cmd_program_requirements_t* requirements =
      &command_program->parsed.requirements;
  *out_info = (loomc_cmd_program_info_t){
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO,
      .structure_size = sizeof(*out_info),
      .name = command_program->name,
      .fixed_buffer_count = requirements->fixed_buffer_count,
      .rebindable_binding_count = requirements->rebindable_binding_count,
      .parameter_root_count = command_program->parsed.parameter_roots.count,
      .parameter_count = command_program->parsed.parameters.count,
      .transient =
          {
              .binding_index = requirements->transient.binding_index,
              .required_byte_length =
                  requirements->transient.required_byte_length,
              .minimum_alignment = requirements->transient.minimum_alignment,
          },
      .launch_counts =
          {
              .binding_index = requirements->launch_counts.binding_index,
              .required_byte_length =
                  requirements->launch_counts.required_byte_length,
              .minimum_alignment =
                  requirements->launch_counts.minimum_alignment,
          },
  };
  return loomc_ok_status();
}

loomc_status_t loomc_cmd_program_parameter_root_info(
    const loomc_cmd_program_t* command_program, loomc_host_size_t index,
    loomc_cmd_program_parameter_root_info_t* out_info) {
  LOOMC_RETURN_IF_ERROR(loomc_cmd_program_validate_output_info(
      out_info, LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      sizeof(*out_info)));
  if (command_program == NULL ||
      index >= command_program->parsed.parameter_roots.count) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command program parameter-root index is out of range");
  }
  const loom_cmd_program_parameter_root_t root =
      loom_cmd_program_parameter_root_at(&command_program->parsed,
                                         (uint32_t)index);
  *out_info = (loomc_cmd_program_parameter_root_info_t){
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      .structure_size = sizeof(*out_info),
      .fixed_buffer_index = root.fixed_buffer_index,
      .required_byte_length = root.required_byte_length,
      .minimum_alignment = root.minimum_alignment,
  };
  return loomc_ok_status();
}

loomc_status_t loomc_cmd_program_parameter_info(
    const loomc_cmd_program_t* command_program, loomc_host_size_t index,
    loomc_cmd_program_parameter_info_t* out_info) {
  LOOMC_RETURN_IF_ERROR(loomc_cmd_program_validate_output_info(
      out_info, LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_INFO,
      sizeof(*out_info)));
  if (command_program == NULL ||
      index >= command_program->parsed.parameters.count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "command program parameter index is out of range");
  }
  const loom_cmd_program_parameter_t parameter =
      loom_cmd_program_parameter_at(&command_program->parsed, (uint32_t)index);
  *out_info = (loomc_cmd_program_parameter_info_t){
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_INFO,
      .structure_size = sizeof(*out_info),
      .key = loomc_string_view_from_iree(parameter.key),
      .fixed_buffer_index = parameter.fixed_buffer_index,
      .byte_offset = parameter.byte_offset,
      .byte_length = parameter.byte_length,
      .minimum_alignment = parameter.minimum_alignment,
  };
  return loomc_ok_status();
}

const loomc_program_t* loomc_cmd_program_package(
    const loomc_cmd_program_t* command_program) {
  return command_program->package;
}

const loom_cmd_program_t* loomc_cmd_program_parsed(
    const loomc_cmd_program_t* command_program) {
  return &command_program->parsed;
}
