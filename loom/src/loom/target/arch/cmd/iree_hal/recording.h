// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE HAL materialization for portable command-machine low functions.

#ifndef LOOM_TARGET_ARCH_CMD_IREE_HAL_RECORDING_H_
#define LOOM_TARGET_ARCH_CMD_IREE_HAL_RECORDING_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Associates an executable-local function token with its package executable.
//
// Command low IR imports executable and entry ordinals through separate dense
// tables. Keeping the association here allows several entries to share one
// executable while preserving the invariant that an entry token is only used
// with the executable that produced it.
typedef struct loom_cmd_iree_hal_entry_t {
  // Ordinal of the executable in the materialization input table.
  uint32_t executable_index;
  // Executable-local function token used by HAL dispatch commands.
  iree_hal_executable_function_t function;
  // Reflected function layout used to pack logical command arguments.
  iree_hal_executable_function_info_t info;
  // Reflected logical parameter table with |info.parameter_count| entries.
  const iree_hal_executable_function_parameter_t* parameters;
} loom_cmd_iree_hal_entry_t;

// Dense package inputs consumed while recording one command program.
//
// Every array is borrowed for the duration of materialization. The resulting
// command buffer retains directly referenced HAL resources according to its
// mode. Binding ordinals remain unresolved until queue execution and therefore
// do not retain issue-time buffers.
typedef struct loom_cmd_iree_hal_inputs_t {
  // Number of issue-time buffer slots available to the command program.
  iree_host_size_t binding_count;
  // Number of directly referenced ranges in |fixed_buffers|.
  iree_host_size_t fixed_buffer_count;
  // Direct buffer ranges baked into the recorded command buffer.
  const iree_hal_buffer_ref_t* fixed_buffers;
  // Number of loaded executables in |executables|.
  iree_host_size_t executable_count;
  // Loaded executable table indexed by reg<cmd.executable> resources.
  iree_hal_executable_t* const* executables;
  // Number of executable-local entry records in |entries|.
  iree_host_size_t entry_count;
  // Entry table indexed by reg<cmd.entry> resources.
  const loom_cmd_iree_hal_entry_t* entries;
} loom_cmd_iree_hal_inputs_t;

// Phase of one HAL operation emitted for a portable command.
typedef enum loom_cmd_iree_hal_operation_phase_e {
  // Full execution barrier emitted before a payload or by a standalone
  // barrier command.
  LOOM_CMD_IREE_HAL_OPERATION_PHASE_BARRIER = 0,
  // Fill, copy, or dispatch payload emitted for a portable command.
  LOOM_CMD_IREE_HAL_OPERATION_PHASE_PAYLOAD = 1,
} loom_cmd_iree_hal_operation_phase_t;

// Identity of one HAL operation emitted while recording a portable program.
//
// Entries are stored in recorder-relative HAL command-index order. A
// materializer recording into a newly created empty command buffer therefore
// produces the absolute command indices reported by HAL profiling.
typedef struct loom_cmd_iree_hal_operation_map_entry_t {
  // Zero-based ordinal in the canonical portable command table.
  uint32_t command_ordinal;
  // Zero-based wave containing the operation after applying leading barriers.
  uint32_t barrier_wave_ordinal;
  // Relationship between this HAL operation and its portable command.
  loom_cmd_iree_hal_operation_phase_t phase;
} loom_cmd_iree_hal_operation_map_entry_t;

// Caller-owned optional operation identity output for program recording.
//
// |entries| must have capacity for twice the portable command count, the
// maximum produced when every command carries a leading barrier. Recording
// resets |count| and appends one entry after each successful HAL operation.
typedef struct loom_cmd_iree_hal_operation_map_t {
  // Caller-owned output storage.
  loom_cmd_iree_hal_operation_map_entry_t* entries;
  // Number of entries available in |entries|.
  iree_host_size_t capacity;
  // Number of entries populated by the recorder.
  iree_host_size_t count;
} loom_cmd_iree_hal_operation_map_t;

// Records one verified cmd.core low function into a begun command buffer.
//
// The function must use the command_program ABI and contain a single
// straight-line block in the closed cmd low ISA. Package input bounds and
// function argument compatibility are checked during recording. All temporary
// interpreter storage is obtained from one |host_allocator| allocation. No
// per-command allocation, executable query, or symbol lookup is performed.
iree_status_t loom_cmd_iree_hal_record_function(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_cmd_iree_hal_inputs_t* inputs,
    iree_hal_command_buffer_t* command_buffer, iree_allocator_t host_allocator);

// Creates and records a reusable command buffer for one cmd.core low function.
//
// The returned command buffer has |inputs->binding_count| issue-time binding
// slots. Unless ONE_SHOT is present in |mode| it may be replayed with different
// binding tables as long as the package executables and fixed buffers remain
// compatible with the retained command-buffer resources.
iree_status_t loom_cmd_iree_hal_materialize_function(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_cmd_iree_hal_inputs_t* inputs, iree_hal_device_t* device,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer,
    iree_allocator_t host_allocator);

// Records one parsed portable command program into a begun command buffer.
//
// Package input counts must exactly match the program requirements. All
// buffer references and reflected kernel arguments are resolved through one
// temporary |host_allocator| allocation. Recording performs no per-command
// allocation, executable query, symbol lookup, or compiler IR traversal.
iree_status_t loom_cmd_iree_hal_record_program(
    const loom_cmd_program_t* program, const loom_cmd_iree_hal_inputs_t* inputs,
    iree_hal_command_buffer_t* command_buffer,
    loom_cmd_iree_hal_operation_map_t* operation_map,
    iree_allocator_t host_allocator);

// Records a trusted canonical command range into a begun command buffer.
//
// |barrier_wave_ordinal| is the canonical wave containing the first command.
// The range may begin at a folded or standalone barrier, in which case the
// recorder emits that barrier before entering the supplied wave. A range may
// continue across later barriers. |range| must be within the parsed program;
// use loom_cmd_program_barrier_wave_iterator_t to produce barrier-aligned
// ranges without reconstructing schedule semantics.
iree_status_t loom_cmd_iree_hal_record_program_range(
    const loom_cmd_program_t* program, loom_cmd_program_command_range_t range,
    uint32_t barrier_wave_ordinal, const loom_cmd_iree_hal_inputs_t* inputs,
    iree_hal_command_buffer_t* command_buffer,
    loom_cmd_iree_hal_operation_map_t* operation_map,
    iree_allocator_t host_allocator);

// Creates and records a reusable command buffer for one parsed program.
//
// The returned command buffer has the rebindable slot count declared by the
// program artifact. Unless ONE_SHOT is present in |mode| it may be replayed
// with different binding tables while retaining the package executables and
// fixed buffers supplied during materialization.
iree_status_t loom_cmd_iree_hal_materialize_program(
    const loom_cmd_program_t* program, const loom_cmd_iree_hal_inputs_t* inputs,
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_queue_affinity_t queue_affinity,
    loom_cmd_iree_hal_operation_map_t* operation_map,
    iree_hal_command_buffer_t** out_command_buffer,
    iree_allocator_t host_allocator);

// Creates and records a reusable command buffer for a trusted command range.
//
// The range and |barrier_wave_ordinal| follow the recording contract above.
// Fixed resources and rebindable binding ordinals remain those of the complete
// program so separately materialized ranges share one issue-time binding ABI.
iree_status_t loom_cmd_iree_hal_materialize_program_range(
    const loom_cmd_program_t* program, loom_cmd_program_command_range_t range,
    uint32_t barrier_wave_ordinal, const loom_cmd_iree_hal_inputs_t* inputs,
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_queue_affinity_t queue_affinity,
    loom_cmd_iree_hal_operation_map_t* operation_map,
    iree_hal_command_buffer_t** out_command_buffer,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_IREE_HAL_RECORDING_H_
