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

/// Materialized command-program behavior flag bits.
typedef enum loomc_cmd_iree_hal_program_flag_bits_e {
  /// Retains the exact relation between recorded HAL operations and portable
  /// command ordinals for diagnostics and profile correlation.
  ///
  /// This adds one compact host record per HAL command-buffer operation during
  /// materialization. Programs without this flag allocate no map storage.
  LOOMC_CMD_IREE_HAL_PROGRAM_FLAG_RETAIN_RECORDED_OPERATIONS = 1u << 0,
} loomc_cmd_iree_hal_program_flag_bits_t;

/// Bitmask of `loomc_cmd_iree_hal_program_flag_bits_t` values.
typedef uint32_t loomc_cmd_iree_hal_program_flags_t;

/// Relationship between a recorded HAL operation and its portable command.
typedef enum loomc_cmd_iree_hal_recorded_operation_phase_e {
  /// Full execution barrier preceding a payload or emitted by a standalone
  /// barrier command.
  LOOMC_CMD_IREE_HAL_RECORDED_OPERATION_PHASE_BARRIER = 0,

  /// Fill, copy, or dispatch payload emitted for a portable command.
  LOOMC_CMD_IREE_HAL_RECORDED_OPERATION_PHASE_PAYLOAD = 1,
} loomc_cmd_iree_hal_recorded_operation_phase_t;

/// Immutable identity of one HAL operation in a materialized command buffer.
///
/// Records are indexed by the zero-based command index reported by retained
/// HAL command-buffer metadata and dispatch profiling. A portable command with
/// a folded leading barrier has one barrier record followed by one payload
/// record; a standalone barrier has only a barrier record.
typedef struct loomc_cmd_iree_hal_recorded_operation_info_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_CMD_IREE_HAL_RECORDED_OPERATION_INFO` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Zero-based ordinal in the canonical portable command table.
  uint32_t command_ordinal;

  /// Zero-based barrier wave containing this operation.
  uint32_t barrier_wave_ordinal;

  /// Relationship between this HAL operation and its portable command.
  loomc_cmd_iree_hal_recorded_operation_phase_t phase;
} loomc_cmd_iree_hal_recorded_operation_info_t;

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

  /// Flags controlling materialized-program behavior and retained metadata.
  loomc_cmd_iree_hal_program_flags_t flags;
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

/// Returns the number of retained recorded-operation records.
///
/// Returns zero when
/// `LOOMC_CMD_IREE_HAL_PROGRAM_FLAG_RETAIN_RECORDED_OPERATIONS` was not set
/// during materialization.
LOOMC_API_EXPORT loomc_host_size_t
loomc_cmd_iree_hal_program_recorded_operation_count(
    const loomc_cmd_iree_hal_program_t* hal_program);

/// Returns one retained recorded-operation identity by HAL command index.
///
/// Fails with `LOOMC_STATUS_OUT_OF_RANGE` when `operation_index` is not less
/// than `loomc_cmd_iree_hal_program_recorded_operation_count`.
LOOMC_API_EXPORT loomc_status_t
loomc_cmd_iree_hal_program_recorded_operation_info(
    const loomc_cmd_iree_hal_program_t* hal_program,
    loomc_host_size_t operation_index,
    loomc_cmd_iree_hal_recorded_operation_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CMD_IREE_HAL_H_
