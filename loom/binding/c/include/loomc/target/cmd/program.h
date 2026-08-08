// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_PROGRAM_H_
#define LOOMC_TARGET_CMD_PROGRAM_H_

#include "loomc/program.h"

/// @file
/// Selected command-program roots and their runtime ABI requirements.

#ifdef __cplusplus
extern "C" {
#endif

/// Immutable selected command-program root.
///
/// A selected root retains its generic compiled program and parses the root's
/// portable command artifact once. Queries are constant-time projections of
/// that verified representation and never reparse artifact storage.
///
/// @thread_safety
/// Selected roots are immutable after creation. Retained handles may be shared
/// and queried concurrently across threads.
typedef struct loomc_cmd_program_t loomc_cmd_program_t;

/// Invalid fixed-buffer or issue-time binding index.
#define LOOMC_CMD_PROGRAM_BINDING_INVALID UINT32_MAX

/// One aggregate buffer requirement in a selected command-program ABI.
typedef struct loomc_cmd_program_buffer_requirement_t {
  /// Dense fixed-buffer or issue-time binding index, or
  /// `LOOMC_CMD_PROGRAM_BINDING_INVALID` when the storage is absent.
  uint32_t binding_index;

  /// Minimum required byte length of the supplied buffer range.
  uint64_t required_byte_length;

  /// Minimum required alignment of the supplied buffer range.
  uint64_t minimum_alignment;
} loomc_cmd_program_buffer_requirement_t;

/// Immutable ABI metadata for one selected command-program root.
typedef struct loomc_cmd_program_info_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Public export name borrowed from the selected command program.
  loomc_string_view_t name;

  /// Number of materialization-time fixed buffer slots.
  ///
  /// Parameter roots select slots from this broader fixed-buffer table. A
  /// program lowered from source normally uses fixed slots for parameters,
  /// while low command ISA producers may bake other immutable resources.
  uint32_t fixed_buffer_count;

  /// Number of issue-time rebindable buffer slots.
  uint32_t rebindable_binding_count;

  /// Number of fixed parameter-buffer root requirements.
  loomc_host_size_t parameter_root_count;

  /// Number of concrete immutable parameter requirements.
  loomc_host_size_t parameter_count;

  /// Packed issue-time transient slab requirement.
  loomc_cmd_program_buffer_requirement_t transient;

  /// Host-produced static-indirect launch-count table requirement.
  loomc_cmd_program_buffer_requirement_t launch_counts;
} loomc_cmd_program_info_t;

/// Immutable metadata for one fixed parameter-buffer root.
typedef struct loomc_cmd_program_parameter_root_info_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Dense materialization-time fixed-buffer index populated by this root.
  uint32_t fixed_buffer_index;

  /// Minimum byte length required by all parameters assigned to the root.
  uint64_t required_byte_length;

  /// Minimum required alignment of the supplied fixed-buffer range.
  uint64_t minimum_alignment;
} loomc_cmd_program_parameter_root_info_t;

/// Immutable metadata for one concrete command-program parameter.
typedef struct loomc_cmd_program_parameter_info_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_INFO` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Fully substituted parameter key borrowed from the selected root.
  loomc_string_view_t key;

  /// Dense fixed-buffer index containing the parameter.
  uint32_t fixed_buffer_index;

  /// Root-relative byte offset of the parameter payload.
  uint64_t byte_offset;

  /// Exact byte length of the parameter payload.
  uint64_t byte_length;

  /// Minimum required alignment of the parameter payload.
  uint64_t minimum_alignment;
} loomc_cmd_program_parameter_info_t;

/// Selects and verifies one command-program export from a compiled program.
///
/// @param program Generic compiled program containing the selected export.
/// @param export_token Program-local export token to select.
/// @param allocator Host allocator used for the selected-root handle.
/// @param out_command_program Receives one retained selected root on success.
/// @return OK when the export owns one valid portable command artifact.
///
/// @ownership
/// The returned handle retains `program` and does not copy artifact bytes. The
/// caller releases it with `loomc_cmd_program_release`.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_create_from_export(
    loomc_program_t* program, loomc_program_export_t export_token,
    loomc_allocator_t allocator, loomc_cmd_program_t** out_command_program);

/// Retains `command_program` for another owner.
LOOMC_API_EXPORT void loomc_cmd_program_retain(
    loomc_cmd_program_t* command_program);

/// Releases `command_program` from one owner. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_cmd_program_release(
    loomc_cmd_program_t* command_program);

/// Returns immutable ABI metadata for `command_program`.
LOOMC_API_EXPORT loomc_status_t
loomc_cmd_program_info(const loomc_cmd_program_t* command_program,
                       loomc_cmd_program_info_t* out_info);

/// Returns one fixed parameter-buffer root requirement by dense index.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_parameter_root_info(
    const loomc_cmd_program_t* command_program, loomc_host_size_t index,
    loomc_cmd_program_parameter_root_info_t* out_info);

/// Returns one concrete immutable parameter requirement by dense index.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_program_parameter_info(
    const loomc_cmd_program_t* command_program, loomc_host_size_t index,
    loomc_cmd_program_parameter_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CMD_PROGRAM_H_
