// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Closed source command-program conversion to the portable cmd low ISA.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_LOWER_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_LOWER_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materialization role of one source command-program buffer root.
typedef enum loom_cmd_lower_buffer_role_e {
  // The buffer range is fixed while the command program remains materialized.
  LOOM_CMD_LOWER_BUFFER_ROLE_FIXED = 1,
  // The buffer range is supplied through the issue-time binding table.
  LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE = 2,
} loom_cmd_lower_buffer_role_t;

// Resolved low-ABI placement of one source command-program buffer argument.
typedef struct loom_cmd_lower_binding_t {
  // Materialization role selecting the fixed or rebindable resource table.
  loom_cmd_lower_buffer_role_t role;
  // Dense index in the table selected by |role|.
  uint32_t resource_index;
  // Root-relative byte offset of the range passed to kernel launches.
  uint64_t byte_offset;
  // Byte length of the range, or UINT64_MAX for the remaining root buffer.
  uint64_t byte_length;
} loom_cmd_lower_binding_t;

// Resolved direct-dispatch metadata for one source kernel.launch operation.
typedef struct loom_cmd_lower_direct_launch_t {
  // Source launch represented by this row.
  const loom_op_t* source_op;
  // Dense executable-table index selected for the launch.
  uint32_t executable_index;
  // Dense program entry-table index selecting an executable-local token.
  uint32_t entry_index;
  // Exact workgroup counts recorded into the command program.
  loom_target_dispatch_workgroup_count_t workgroup_count;
} loom_cmd_lower_direct_launch_t;

// Compiler-owned facts consumed by closed command-program conversion.
//
// Rows are already resolved by source specialization, kernel-product
// extraction, binding placement, and aggregate launch analysis. Conversion
// preserves these facts; it does not rediscover kernel identity or launch
// arithmetic from the source module.
typedef struct loom_cmd_lower_plan_t {
  // Derived cmd.core target referenced by the resulting low function.
  loom_symbol_ref_t command_target;
  // Source binding rows in command-program signature order.
  const loom_cmd_lower_binding_t* bindings;
  // Number of source binding rows.
  iree_host_size_t binding_count;
  // Number of dense fixed-buffer ABI resources.
  uint32_t fixed_buffer_count;
  // Number of dense issue-time binding ABI resources.
  uint32_t rebindable_binding_count;
  // Number of dense executable ABI resources.
  uint32_t executable_count;
  // Number of dense executable-local entry ABI resources.
  uint32_t entry_count;
  // Direct-launch rows in portable schedule traversal order.
  const loom_cmd_lower_direct_launch_t* launches;
  // Number of direct-launch rows.
  iree_host_size_t launch_count;
} loom_cmd_lower_plan_t;

// Replaces one specialized command.program.def with a zero-signature
// command_program low.func.def using the cmd.core representation contract.
//
// The first closed slice accepts buffer-only launch arguments and exact direct
// workgroup counts supplied by |plan|. Unsupported residual source semantics
// fail without changing the source program. On success the replacement keeps
// the source symbol identity and is returned in |out_low_function|.
iree_status_t loom_cmd_lower_program_to_low(loom_module_t* module,
                                            loom_op_t* program_op,
                                            const loom_cmd_lower_plan_t* plan,
                                            loom_op_t** out_low_function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_LOWER_H_
