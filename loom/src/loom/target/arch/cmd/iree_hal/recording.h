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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_IREE_HAL_RECORDING_H_
