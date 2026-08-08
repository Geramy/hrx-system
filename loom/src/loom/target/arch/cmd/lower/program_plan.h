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

// One immutable command root and its independently compilable dependencies.
//
// The root module is selectively linked from the source command symbol and
// lowered to the portable cmd low ISA. The launch graph is an independently
// owned pure host module. Each dependency unit owns a selectively linked and
// specialized kernel module. No compilation or artifact emission occurs while
// preparing the plan.
typedef struct loom_cmd_program_plan_t {
  // Owned module containing the lowered command root.
  loom_module_t* root_module;

  // Lowered command root in |root_module|.
  loom_op_t* root_function_op;

  // Owned module containing the aggregate host launch-count program.
  loom_module_t* launch_module;

  // Aggregate host launch-count function in |launch_module|.
  loom_op_t* launch_function_op;

  // Number of unique dynamic xyz tuples returned by |launch_function_op|.
  uint32_t launch_tuple_count;

  // Independently owned kernel dependency units in executable-slot order.
  loom_cmd_kernel_unit_t* dependency_units;

  // Number of entries in |dependency_units|.
  iree_host_size_t dependency_count;

  // Host allocator used for the dependency-unit table.
  iree_allocator_t host_allocator;
} loom_cmd_program_plan_t;

// Prepares one targeted command-program root for independent compilation.
//
// |source_program_op| must be a linked module-boundary command.program.def with
// a selected target. Preparation selectively links its complete dependency
// closure into a new root module, derives one private kernel unit for each
// scheduled launch, materializes the aggregate launch-count program, assigns
// dense dependency slots, and lowers the command root. The source module is
// unchanged and need not outlive the returned plan.
//
// On success |out_plan| owns every module it references and must be
// deinitialized. On failure |out_plan| is empty.
iree_status_t loom_cmd_program_plan_prepare(const loom_module_t* source_module,
                                            const loom_op_t* source_program_op,
                                            iree_arena_block_pool_t* block_pool,
                                            iree_allocator_t host_allocator,
                                            loom_cmd_program_plan_t* out_plan);

// Releases all storage owned by |plan| and resets it to empty.
void loom_cmd_program_plan_deinitialize(loom_cmd_program_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_PROGRAM_PLAN_H_
