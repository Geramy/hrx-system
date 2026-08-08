// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-owned command root and dependency-unit preparation.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/cmd/lower/kernel_unit.h"
#include "loom/target/arch/cmd/lower/launch_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// One prepared command root within a program plan.
//
// The lowered command function addresses the plan-wide dependency table. Its
// launch function in the plan's shared host module evaluates this root's
// dynamic launch counts. Both remain valid after the source module is released.
typedef struct loom_cmd_program_root_t {
  // Lowered command root in the plan's shared root module.
  loom_op_t* function_op;

  // Host launch-count function in the plan's shared launch module.
  loom_op_t* launch_function_op;

  // Number of unique dynamic xyz tuples returned by |launch_function_op|.
  uint32_t launch_tuple_count;
} loom_cmd_program_root_t;

// Immutable command roots and their union dependency graph.
//
// The root module contains every selected command symbol lowered to the
// portable cmd low ISA. The launch module contains one pure host function per
// selected root. Equivalent dependency launch sites across all roots share one
// selectively linked and specialized kernel unit. No compilation or artifact
// emission occurs while preparing the plan.
typedef struct loom_cmd_program_plan_t {
  // Owned module containing all lowered command roots.
  loom_module_t* root_module;

  // Owned module containing all root launch-count functions.
  loom_module_t* launch_module;

  // Selected roots in caller order.
  loom_cmd_program_root_t* roots;

  // Number of entries in |roots|.
  iree_host_size_t root_count;

  // Unique independently owned dependencies in executable-slot order.
  loom_cmd_kernel_unit_t* dependency_units;

  // Number of entries in |dependency_units|.
  iree_host_size_t dependency_count;

  // Host allocator used for the root and dependency-unit tables.
  iree_allocator_t host_allocator;
} loom_cmd_program_plan_t;

// Prepares targeted command-program roots for independent compilation.
//
// |source_program_ops| must contain unique linked module-boundary
// command.program.def operations with selected targets. Preparation selectively
// links their union dependency closure into one module, interns equivalent
// launch sites across roots into private dependency units, materializes one
// launch-count program per root, assigns plan-wide dense dependency slots, and
// lowers every command root. The source module is unchanged and need not
// outlive the returned plan.
//
// On success |out_plan| owns every module it references and must be
// deinitialized. On failure |out_plan| is empty.
iree_status_t loom_cmd_program_plan_prepare(
    const loom_module_t* source_module,
    const loom_op_t* const* source_program_ops,
    iree_host_size_t source_program_count, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator, loom_cmd_program_plan_t* out_plan);

// Releases all storage owned by |plan| and resets it to empty.
void loom_cmd_program_plan_deinitialize(loom_cmd_program_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_
