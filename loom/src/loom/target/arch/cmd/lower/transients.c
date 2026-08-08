// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/transients.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/util/walk.h"

typedef struct loom_cmd_transient_allocation_t {
  // Source buffer.alloca result identifying the storage root.
  loom_value_id_t root_value;

  // Packed byte offset within the aggregate transient slab.
  uint64_t byte_offset;

  // Reserved byte length of the allocation root.
  uint64_t byte_length;
} loom_cmd_transient_allocation_t;

typedef struct loom_cmd_transient_build_t {
  // Complete value facts for the source program.
  const loom_value_fact_table_t* fact_table;

  // Dense issue-time binding index assigned to the aggregate slab.
  uint32_t binding_index;

  // Scratch storage for allocation and range rows.
  iree_arena_allocator_t* scratch_arena;

  // Source-order allocation roots and their packed placements.
  loom_cmd_transient_allocation_t* allocations;

  // Number of populated entries in |allocations|.
  iree_host_size_t allocation_count;

  // Number of allocated entries in |allocations|.
  iree_host_size_t allocation_capacity;

  // Source values mapped to packed transient ranges.
  loom_cmd_lower_buffer_range_t* buffer_ranges;

  // Number of populated entries in |buffer_ranges|.
  iree_host_size_t buffer_range_count;

  // Number of allocated entries in |buffer_ranges|.
  iree_host_size_t buffer_range_capacity;

  // Next unreserved byte offset in the aggregate slab.
  uint64_t placement_cursor;

  // Maximum base alignment required by any allocation root.
  uint64_t minimum_alignment;
} loom_cmd_transient_build_t;

static bool loom_cmd_transient_align_offset(uint64_t value, uint64_t alignment,
                                            uint64_t* out_value) {
  IREE_ASSERT(iree_is_power_of_two_uint64(alignment));
  const uint64_t mask = alignment - 1;
  if (value > UINT64_MAX - mask) return false;
  *out_value = (value + mask) & ~mask;
  return true;
}

static iree_status_t loom_cmd_transient_reserve_allocation(
    loom_cmd_transient_build_t* build) {
  if (build->allocation_count < build->allocation_capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(
      build->scratch_arena, build->allocation_count,
      iree_max(build->allocation_count + 1, 8u), sizeof(*build->allocations),
      &build->allocation_capacity, (void**)&build->allocations);
}

static iree_status_t loom_cmd_transient_reserve_buffer_range(
    loom_cmd_transient_build_t* build) {
  if (build->buffer_range_count < build->buffer_range_capacity) {
    return iree_ok_status();
  }
  return iree_arena_grow_array(build->scratch_arena, build->buffer_range_count,
                               iree_max(build->buffer_range_count + 1, 16u),
                               sizeof(*build->buffer_ranges),
                               &build->buffer_range_capacity,
                               (void**)&build->buffer_ranges);
}

static const loom_cmd_transient_allocation_t*
loom_cmd_transient_find_allocation(const loom_cmd_transient_build_t* build,
                                   loom_value_id_t root_value) {
  for (iree_host_size_t i = 0; i < build->allocation_count; ++i) {
    if (build->allocations[i].root_value == root_value) {
      return &build->allocations[i];
    }
  }
  return NULL;
}

static iree_status_t loom_cmd_transient_append_range(
    loom_cmd_transient_build_t* build, loom_value_id_t source_value,
    const loom_cmd_transient_allocation_t* allocation, uint64_t byte_offset,
    uint64_t byte_length) {
  if (byte_offset > allocation->byte_length ||
      byte_length > allocation->byte_length - byte_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command transient range [0x%" PRIx64 ", 0x%" PRIx64
                            ") exceeds its 0x%" PRIx64 "-byte allocation",
                            byte_offset, byte_offset + byte_length,
                            allocation->byte_length);
  }
  IREE_RETURN_IF_ERROR(loom_cmd_transient_reserve_buffer_range(build));
  build->buffer_ranges[build->buffer_range_count++] =
      (loom_cmd_lower_buffer_range_t){
          .source_value = source_value,
          .role = LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE,
          .resource_index = build->binding_index,
          .byte_offset = allocation->byte_offset + byte_offset,
          .byte_length = byte_length,
      };
  return iree_ok_status();
}

static iree_status_t loom_cmd_transient_append_allocation(
    loom_cmd_transient_build_t* build, const loom_op_t* op) {
  const loom_value_fact_memory_space_t memory_space =
      loom_buffer_alloca_memory_space(op);
  if (memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "portable command programs require buffer.alloca in global memory; "
        "memory space %u is not representable",
        (unsigned)memory_space);
  }

  int64_t byte_length_i64 = 0;
  if (!loom_value_facts_as_non_negative_i64_maximum(
          loom_value_fact_table_lookup(build->fact_table,
                                       loom_buffer_alloca_byte_length(op)),
          &byte_length_i64) ||
      byte_length_i64 <= 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "command buffer.alloca must have a finite positive byte-length "
        "maximum before program preparation");
  }
  const uint64_t byte_length = (uint64_t)byte_length_i64;
  const uint64_t base_alignment =
      (uint64_t)loom_buffer_alloca_base_alignment(op);
  IREE_ASSERT(iree_is_power_of_two_uint64(base_alignment));

  uint64_t byte_offset = 0;
  if (!loom_cmd_transient_align_offset(build->placement_cursor, base_alignment,
                                       &byte_offset) ||
      byte_length > UINT64_MAX - byte_offset) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command transient slab exceeds 64-bit offsets");
  }

  IREE_RETURN_IF_ERROR(loom_cmd_transient_reserve_allocation(build));
  loom_cmd_transient_allocation_t* allocation =
      &build->allocations[build->allocation_count++];
  *allocation = (loom_cmd_transient_allocation_t){
      .root_value = loom_buffer_alloca_result(op),
      .byte_offset = byte_offset,
      .byte_length = byte_length,
  };
  build->placement_cursor = byte_offset + byte_length;
  build->minimum_alignment = iree_max(build->minimum_alignment, base_alignment);
  return loom_cmd_transient_append_range(build, allocation->root_value,
                                         allocation, /*byte_offset=*/0,
                                         allocation->byte_length);
}

static iree_status_t loom_cmd_transient_append_result_range(
    loom_cmd_transient_build_t* build, loom_value_id_t result) {
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(build->fact_table, result);
  loom_value_fact_view_reference_t view_reference = {0};
  if (loom_value_facts_query_view_reference(&build->fact_table->context, facts,
                                            &view_reference)) {
    const loom_cmd_transient_allocation_t* allocation =
        loom_cmd_transient_find_allocation(build, view_reference.root_value_id);
    if (!allocation) return iree_ok_status();
    int64_t byte_offset = 0;
    int64_t byte_length = 0;
    if (!loom_value_facts_as_exact_i64(view_reference.base_byte_offset,
                                       &byte_offset) ||
        !loom_value_facts_as_exact_i64(view_reference.footprint_byte_length,
                                       &byte_length) ||
        byte_offset < 0 || byte_length < 0) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "command transient views require exact nonnegative byte ranges "
          "before program preparation");
    }
    return loom_cmd_transient_append_range(build, result, allocation,
                                           (uint64_t)byte_offset,
                                           (uint64_t)byte_length);
  }

  loom_value_fact_buffer_reference_t buffer_reference = {0};
  if (!loom_value_facts_query_buffer_reference(&build->fact_table->context,
                                               facts, &buffer_reference)) {
    return iree_ok_status();
  }
  const loom_value_id_t root_value =
      loom_value_fact_buffer_reference_resolve_root_value(buffer_reference,
                                                          result);
  const loom_cmd_transient_allocation_t* allocation =
      loom_cmd_transient_find_allocation(build, root_value);
  if (!allocation) return iree_ok_status();
  return loom_cmd_transient_append_range(
      build, result, allocation, /*byte_offset=*/0, allocation->byte_length);
}

static iree_status_t loom_cmd_transient_visit(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_cmd_transient_build_t* build = (loom_cmd_transient_build_t*)user_data;
  if (loom_buffer_alloca_isa(op)) {
    return loom_cmd_transient_append_allocation(build, op);
  }

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_cmd_transient_append_result_range(build, results[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_cmd_transient_layout_build(
    const loom_module_t* module, loom_func_like_t program,
    const loom_value_fact_table_t* fact_table, uint32_t binding_index,
    iree_arena_allocator_t* scratch_arena,
    loom_cmd_transient_layout_t* out_layout) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT(loom_func_like_isa(program));
  IREE_ASSERT_ARGUMENT(fact_table);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_layout);
  memset(out_layout, 0, sizeof(*out_layout));

  loom_cmd_transient_build_t build = {
      .fact_table = fact_table,
      .binding_index = binding_index,
      .scratch_arena = scratch_arena,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_function(
      module, program, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_cmd_transient_visit, &build}, scratch_arena,
      &walk_result));
  IREE_ASSERT_EQ(walk_result, LOOM_WALK_CONTINUE);

  *out_layout = (loom_cmd_transient_layout_t){
      .requirement =
          {
              .binding_index =
                  build.allocation_count != 0 ? binding_index : UINT32_MAX,
              .required_byte_length = build.placement_cursor,
              .minimum_alignment = build.minimum_alignment,
          },
      .buffer_ranges = build.buffer_ranges,
      .buffer_range_count = build.buffer_range_count,
  };
  return iree_ok_status();
}
