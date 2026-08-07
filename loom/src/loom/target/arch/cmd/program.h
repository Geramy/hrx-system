// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable target-neutral portable command programs.

#ifndef LOOM_TARGET_ARCH_CMD_PROGRAM_H_
#define LOOM_TARGET_ARCH_CMD_PROGRAM_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materialization role of a serialized buffer root.
typedef enum loom_cmd_program_buffer_role_e {
  // A concrete buffer is fixed while the live program remains materialized.
  LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED = 1,
  // A stable binding slot receives its concrete buffer at issue time.
  LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE = 2,
} loom_cmd_program_buffer_role_t;

// One resolved range within a fixed or rebindable buffer root.
typedef struct loom_cmd_program_buffer_ref_t {
  // Root materialization role.
  loom_cmd_program_buffer_role_t role;
  // Dense root index in the table selected by |role|.
  uint32_t root_index;
  // Root-relative byte offset.
  uint64_t byte_offset;
  // Byte length, or UINT64_MAX for the remaining root range.
  uint64_t byte_length;
} loom_cmd_program_buffer_ref_t;

// Kind of one flattened logical kernel argument.
typedef enum loom_cmd_program_argument_kind_e {
  // An unsigned 32-bit scalar stored in the low bits of |payload|.
  LOOM_CMD_PROGRAM_ARGUMENT_KIND_U32 = 1,
  // An unsigned 64-bit scalar stored directly in |payload|.
  LOOM_CMD_PROGRAM_ARGUMENT_KIND_U64 = 2,
  // An index into the program buffer-reference table.
  LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER_REF = 3,
} loom_cmd_program_argument_kind_t;

// One flattened logical kernel argument.
typedef struct loom_cmd_program_argument_t {
  // Interpretation of |payload|.
  loom_cmd_program_argument_kind_t kind;
  // Immediate scalar value or buffer-reference table index.
  uint64_t payload;
} loom_cmd_program_argument_t;

// Kind of one command recorded by a portable command program.
typedef enum loom_cmd_program_command_kind_e {
  // Fill one buffer range with a repeated scalar pattern.
  LOOM_CMD_PROGRAM_COMMAND_KIND_FILL = 1,
  // Copy one buffer range into another.
  LOOM_CMD_PROGRAM_COMMAND_KIND_COPY = 2,
  // Dispatch with exact workgroup counts embedded in the program.
  LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT = 3,
  // Dispatch with indirect counts stable before command execution begins.
  LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC = 4,
  // Dispatch with indirect counts produced within the command program.
  LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC = 5,
  // Order all earlier commands before all later commands.
  LOOM_CMD_PROGRAM_COMMAND_KIND_EXECUTION_BARRIER = 6,
} loom_cmd_program_command_kind_t;

// One decoded portable command.
typedef struct loom_cmd_program_command_t {
  // Command payload selector.
  loom_cmd_program_command_kind_t kind;
  // First logical argument in the flattened program argument table.
  uint32_t argument_offset;
  // Number of logical arguments consumed by the command.
  uint32_t argument_count;
  // Command-specific payload.
  union {
    // Payload for LOOM_CMD_PROGRAM_COMMAND_KIND_FILL.
    struct {
      // Target buffer-reference table index.
      uint32_t target_buffer_ref;
      // Repeated fill pattern bits.
      uint32_t pattern;
      // Number of low pattern bytes to repeat.
      uint32_t pattern_length;
    } fill;
    // Payload for LOOM_CMD_PROGRAM_COMMAND_KIND_COPY.
    struct {
      // Source buffer-reference table index.
      uint32_t source_buffer_ref;
      // Target buffer-reference table index.
      uint32_t target_buffer_ref;
    } copy;
    // Payload for LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT.
    struct {
      // Dense executable requirement index.
      uint32_t executable_index;
      // Dense program entry requirement index.
      uint32_t entry_index;
      // Exact X workgroup count.
      uint32_t workgroup_count_x;
      // Exact Y workgroup count.
      uint32_t workgroup_count_y;
      // Exact Z workgroup count.
      uint32_t workgroup_count_z;
    } dispatch_direct;
    // Payload for either indirect dispatch command kind.
    struct {
      // Dense executable requirement index.
      uint32_t executable_index;
      // Dense program entry requirement index.
      uint32_t entry_index;
      // Buffer-reference table index containing XYZ workgroup counts.
      uint32_t workgroup_count_buffer_ref;
    } dispatch_indirect;
  } payload;
} loom_cmd_program_command_t;

// External resource counts required to materialize one command program.
typedef struct loom_cmd_program_requirements_t {
  // Number of fixed buffer roots supplied during materialization.
  uint32_t fixed_buffer_count;
  // Number of rebindable buffer roots supplied during issue.
  uint32_t rebindable_binding_count;
  // Number of loaded executables supplied during materialization.
  uint32_t executable_count;
  // Number of program-local executable entry tokens.
  uint32_t entry_count;
} loom_cmd_program_requirements_t;

// Borrowed table within a parsed command-program artifact.
typedef struct loom_cmd_program_table_t {
  // First byte of the canonical fixed-size records.
  const uint8_t* data;
  // Number of records in the table.
  uint32_t count;
} loom_cmd_program_table_t;

// Validated zero-allocation view of one serialized command program.
//
// All storage is borrowed from |data| passed to loom_cmd_program_parse and
// must remain live while the view is used. Fields are read-only after parsing.
typedef struct loom_cmd_program_t {
  // Complete canonical artifact storage.
  iree_const_byte_span_t storage;
  // External resources required by the program.
  loom_cmd_program_requirements_t requirements;
  // Resolved buffer-range table.
  loom_cmd_program_table_t buffer_refs;
  // Flattened logical kernel argument table.
  loom_cmd_program_table_t arguments;
  // Ordered command table.
  loom_cmd_program_table_t commands;
} loom_cmd_program_t;

// Parses and validates one complete command-program artifact.
//
// This is the untrusted byte boundary. Successful parsing guarantees that all
// table ranges, enum values, indices, slices, and reserved fields satisfy the
// canonical format. |out_program| borrows |data| without allocating.
iree_status_t loom_cmd_program_parse(iree_const_byte_span_t data,
                                     loom_cmd_program_t* out_program);

// Returns one validated buffer-reference table entry.
loom_cmd_program_buffer_ref_t loom_cmd_program_buffer_ref_at(
    const loom_cmd_program_t* program, uint32_t index);

// Returns one validated flattened logical argument.
loom_cmd_program_argument_t loom_cmd_program_argument_at(
    const loom_cmd_program_t* program, uint32_t index);

// Returns one validated command table entry.
loom_cmd_program_command_t loom_cmd_program_command_at(
    const loom_cmd_program_t* program, uint32_t index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_PROGRAM_H_
