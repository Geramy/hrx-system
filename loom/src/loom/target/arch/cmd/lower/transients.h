// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Command-program transient allocation planning.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_TRANSIENTS_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_TRANSIENTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/cmd/lower/lower.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Aggregate issue-time storage required by command-program allocas.
typedef struct loom_cmd_transient_requirement_t {
  // Dense rebindable resource index, or UINT32_MAX when no slab is required.
  uint32_t binding_index;

  // Minimum byte length of the supplied transient slab.
  uint64_t required_byte_length;

  // Minimum required alignment of the supplied transient slab.
  uint64_t minimum_alignment;
} loom_cmd_transient_requirement_t;

// Scratch-owned transient placement consumed by command low conversion.
typedef struct loom_cmd_transient_layout_t {
  // Aggregate issue-time slab requirement.
  loom_cmd_transient_requirement_t requirement;

  // Source values mapped to packed ranges within the transient slab.
  const loom_cmd_lower_buffer_range_t* buffer_ranges;

  // Number of entries in |buffer_ranges|.
  iree_host_size_t buffer_range_count;
} loom_cmd_transient_layout_t;

// Packs device-global buffer.alloca roots into one issue-time slab.
//
// Allocation roots and statically resolved derived views are assigned
// source-order ranges honoring each root's declared base alignment. Allocation
// lengths must have finite nonnegative maxima after source specialization. The
// resulting ranges and requirement remain valid until |scratch_arena| resets.
iree_status_t loom_cmd_transient_layout_build(
    const loom_module_t* module, loom_func_like_t program,
    const loom_value_fact_table_t* fact_table, uint32_t binding_index,
    iree_arena_allocator_t* scratch_arena,
    loom_cmd_transient_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_TRANSIENTS_H_
