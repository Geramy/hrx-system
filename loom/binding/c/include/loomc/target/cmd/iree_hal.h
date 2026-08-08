// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_IREE_HAL_H_
#define LOOMC_TARGET_CMD_IREE_HAL_H_

#include "iree/hal/api.h"
#include "loomc/launch_config_module.h"
#include "loomc/target/cmd/program.h"

/// @file
/// IREE HAL materialization for compiled command-program packages.
///
/// A package loads the ancillary executables and shared host launch module from
/// one generic compiled program. Many selected roots from that program may then
/// materialize reusable command buffers without reloading shared executables.

#ifdef __cplusplus
extern "C" {
#endif

/// Device-loaded resources shared by command-program roots in one package.
///
/// @thread_safety
/// Packages are immutable after creation. Retained handles may be shared while
/// roots are materialized concurrently when the HAL device supports concurrent
/// command-buffer creation.
typedef struct loomc_cmd_iree_hal_package_t loomc_cmd_iree_hal_package_t;

/// One reusable IREE HAL command buffer materialized from a selected root.
///
/// The object retains the loaded package and selected root. Its command buffer
/// owns directly referenced resources according to the requested HAL command
/// buffer mode; issue-time bindings remain supplied by each queue execution.
typedef struct loomc_cmd_iree_hal_program_t loomc_cmd_iree_hal_program_t;

/// Device-loading options for a compiled command-program package.
typedef struct loomc_cmd_iree_hal_package_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_CMD_IREE_HAL_PACKAGE_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  const void* next;

  /// IREE HAL device receiving every ancillary executable.
  iree_hal_device_t* device;

  /// Queue affinity used while loading ancillary executables.
  iree_hal_queue_affinity_t executable_queue_affinity;

  /// Device executable target selection for ancillary artifacts.
  iree_hal_executable_target_selection_t target_selection;

  /// Exact artifact format expected from every ancillary program.
  ///
  /// This must match the dependency format supplied while preparing the
  /// command-program compilation plan.
  loomc_string_view_t executable_artifact_format;
} loomc_cmd_iree_hal_package_options_t;

/// Materialization options for one selected command-program root.
typedef struct loomc_cmd_iree_hal_program_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_CMD_IREE_HAL_PROGRAM_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  const void* next;

  /// HAL command-buffer mode used for the reusable recording.
  iree_hal_command_buffer_mode_t command_buffer_mode;

  /// Queue affinity encoded in the reusable command buffer.
  iree_hal_queue_affinity_t queue_affinity;

  /// Direct buffer ranges whose identity is baked into fixed-buffer slots.
  const iree_hal_buffer_ref_t* fixed_buffers;

  /// Number of entries in `fixed_buffers`.
  loomc_host_size_t fixed_buffer_count;
} loomc_cmd_iree_hal_program_options_t;

/// Loads the shared device and host resources in a compiled program package.
///
/// Every ancillary program must contain one executable artifact in the exact
/// format requested by `options`, and each executable must provide reflected
/// entry zero. The initial command compiler produces this shape by
/// independently compiling one kernel unit per dependency slot.
///
/// @ownership
/// The returned package retains `program` and `options->device`. Loaded
/// executables and the host launch module are owned until the final package
/// release.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_iree_hal_package_create(
    loomc_program_t* program,
    const loomc_cmd_iree_hal_package_options_t* options,
    loomc_allocator_t allocator, loomc_cmd_iree_hal_package_t** out_package);

/// Retains `package` for another owner.
LOOMC_API_EXPORT void loomc_cmd_iree_hal_package_retain(
    loomc_cmd_iree_hal_package_t* package);

/// Releases `package` from one owner. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_cmd_iree_hal_package_release(
    loomc_cmd_iree_hal_package_t* package);

/// Materializes one selected root as a reusable IREE HAL command buffer.
///
/// `command_program` must have been selected from the same generic program used
/// to create `package`. Fixed buffers are captured while recording; transient,
/// request, output, and launch-count bindings remain unresolved until queue
/// execution.
///
/// @ownership
/// The returned object retains `package` and `command_program`. Fixed buffer
/// lifetime follows `options->command_buffer_mode`; an unretained HAL command
/// buffer requires the caller to keep those buffers alive.
LOOMC_API_EXPORT loomc_status_t loomc_cmd_iree_hal_program_create(
    loomc_cmd_iree_hal_package_t* package, loomc_cmd_program_t* command_program,
    const loomc_cmd_iree_hal_program_options_t* options,
    loomc_allocator_t allocator,
    loomc_cmd_iree_hal_program_t** out_hal_program);

/// Retains `hal_program` for another owner.
LOOMC_API_EXPORT void loomc_cmd_iree_hal_program_retain(
    loomc_cmd_iree_hal_program_t* hal_program);

/// Releases `hal_program` from one owner. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_cmd_iree_hal_program_release(
    loomc_cmd_iree_hal_program_t* hal_program);

/// Returns the reusable command buffer borrowed from `hal_program`.
LOOMC_API_EXPORT iree_hal_command_buffer_t*
loomc_cmd_iree_hal_program_command_buffer(
    const loomc_cmd_iree_hal_program_t* hal_program);

/// Returns the immutable launch module borrowed from `hal_program`.
///
/// Evaluation contexts created from this module retain it independently.
LOOMC_API_EXPORT loomc_launch_config_module_t*
loomc_cmd_iree_hal_program_launch_module(
    const loomc_cmd_iree_hal_program_t* hal_program);

/// Returns the launch function matching this materialized root.
LOOMC_API_EXPORT loomc_launch_config_function_t
loomc_cmd_iree_hal_program_launch_function(
    const loomc_cmd_iree_hal_program_t* hal_program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CMD_IREE_HAL_H_
