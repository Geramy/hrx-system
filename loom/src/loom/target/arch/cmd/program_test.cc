// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/program.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/cmd/format.h"

namespace loom {
namespace {

static void StoreBufferRef(std::vector<uint8_t>& data, uint32_t table_offset,
                           uint32_t index, loom_cmd_program_buffer_role_t role,
                           uint32_t root_index, uint64_t byte_offset,
                           uint64_t byte_length) {
  uint8_t* record =
      data.data() + table_offset + index * LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
  iree_unaligned_store_le_u32(record + LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET,
                              role);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET, root_index);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET, byte_offset);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET, byte_length);
}

static void StoreArgument(std::vector<uint8_t>& data, uint32_t table_offset,
                          uint32_t index, loom_cmd_program_argument_kind_t kind,
                          uint64_t payload) {
  uint8_t* record =
      data.data() + table_offset + index * LOOM_CMD_PROGRAM_ARGUMENT_SIZE;
  iree_unaligned_store_le_u32(record + LOOM_CMD_PROGRAM_ARGUMENT_KIND_OFFSET,
                              kind);
  iree_unaligned_store_le_u64(record + LOOM_CMD_PROGRAM_ARGUMENT_PAYLOAD_OFFSET,
                              payload);
}

static void StoreCommand(std::vector<uint8_t>& data, uint32_t table_offset,
                         uint32_t index, loom_cmd_program_command_kind_t kind,
                         uint32_t argument_offset, uint32_t argument_count,
                         uint32_t operand_0 = 0, uint32_t operand_1 = 0,
                         uint32_t operand_2 = 0, uint32_t operand_3 = 0,
                         uint32_t operand_4 = 0) {
  uint8_t* record =
      data.data() + table_offset + index * LOOM_CMD_PROGRAM_COMMAND_SIZE;
  iree_unaligned_store_le_u32(record + LOOM_CMD_PROGRAM_COMMAND_KIND_OFFSET,
                              kind);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_OFFSET_OFFSET,
      argument_offset);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_COUNT_OFFSET, argument_count);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_0_OFFSET, operand_0);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_1_OFFSET, operand_1);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_2_OFFSET, operand_2);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_3_OFFSET, operand_3);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_4_OFFSET, operand_4);
}

static std::vector<uint8_t> BuildValidProgram() {
  static constexpr uint32_t kBufferRefCount = 2;
  static constexpr uint32_t kArgumentCount = 3;
  static constexpr uint32_t kCommandCount = 6;
  loom_cmd_program_format_layout_t layout = {};
  IREE_CHECK_OK(loom_cmd_program_format_calculate_layout(
      kBufferRefCount, kArgumentCount, kCommandCount, &layout));
  std::vector<uint8_t> data(layout.total_length, 0);

  memcpy(data.data() + LOOM_CMD_PROGRAM_HEADER_MAGIC_OFFSET,
         LOOM_CMD_PROGRAM_FORMAT_MAGIC, LOOM_CMD_PROGRAM_FORMAT_MAGIC_LENGTH);
  iree_unaligned_store_le_u16(
      data.data() + LOOM_CMD_PROGRAM_HEADER_VERSION_OFFSET,
      LOOM_CMD_PROGRAM_FORMAT_VERSION);
  iree_unaligned_store_le_u16(data.data() + LOOM_CMD_PROGRAM_HEADER_SIZE_OFFSET,
                              LOOM_CMD_PROGRAM_HEADER_SIZE);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_TOTAL_LENGTH_OFFSET,
      layout.total_length);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_FIXED_BUFFER_COUNT_OFFSET, 1);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BINDING_COUNT_OFFSET, 1);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_EXECUTABLE_COUNT_OFFSET, 1);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ENTRY_COUNT_OFFSET, 1);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_COUNT_OFFSET,
      kBufferRefCount);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_COUNT_OFFSET,
      kArgumentCount);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_COUNT_OFFSET,
      kCommandCount);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET,
      layout.buffer_ref_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_TABLE_OFFSET,
      layout.argument_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET,
      layout.command_offset);

  StoreBufferRef(data, layout.buffer_ref_offset, 0,
                 LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED, 0, 0, 256);
  StoreBufferRef(data, layout.buffer_ref_offset, 1,
                 LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE, 0, 16, 64);
  StoreArgument(data, layout.argument_offset, 0,
                LOOM_CMD_PROGRAM_ARGUMENT_KIND_U32, 7);
  StoreArgument(data, layout.argument_offset, 1,
                LOOM_CMD_PROGRAM_ARGUMENT_KIND_U64, UINT64_C(0x123456789));
  StoreArgument(data, layout.argument_offset, 2,
                LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER_REF, 1);
  StoreCommand(data, layout.command_offset, 0,
               LOOM_CMD_PROGRAM_COMMAND_KIND_FILL, 0, 0, 0, 0x12345678, 4);
  StoreCommand(data, layout.command_offset, 1,
               LOOM_CMD_PROGRAM_COMMAND_KIND_COPY, 0, 0, 0, 1);
  StoreCommand(data, layout.command_offset, 2,
               LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT, 0, 3, 0, 0, 1, 2,
               3);
  StoreCommand(data, layout.command_offset, 3,
               LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC, 0, 3, 0,
               0, 1);
  StoreCommand(data, layout.command_offset, 4,
               LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC, 0, 3, 0,
               0, 1);
  StoreCommand(data, layout.command_offset, 5,
               LOOM_CMD_PROGRAM_COMMAND_KIND_EXECUTION_BARRIER, 0, 0);
  return data;
}

static iree_const_byte_span_t AsByteSpan(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

TEST(CmdProgramTest, ParsesCanonicalProgram) {
  const std::vector<uint8_t> data = BuildValidProgram();
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(AsByteSpan(data), &program));

  EXPECT_EQ(program.requirements.fixed_buffer_count, 1u);
  EXPECT_EQ(program.requirements.rebindable_binding_count, 1u);
  EXPECT_EQ(program.requirements.executable_count, 1u);
  EXPECT_EQ(program.requirements.entry_count, 1u);
  EXPECT_EQ(program.buffer_refs.count, 2u);
  EXPECT_EQ(program.arguments.count, 3u);
  EXPECT_EQ(program.commands.count, 6u);
  const loom_cmd_program_buffer_ref_t binding =
      loom_cmd_program_buffer_ref_at(&program, 1);
  EXPECT_EQ(binding.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
  EXPECT_EQ(binding.root_index, 0u);
  EXPECT_EQ(binding.byte_offset, 16u);
  EXPECT_EQ(binding.byte_length, 64u);
  const loom_cmd_program_argument_t argument =
      loom_cmd_program_argument_at(&program, 1);
  EXPECT_EQ(argument.kind, LOOM_CMD_PROGRAM_ARGUMENT_KIND_U64);
  EXPECT_EQ(argument.payload, UINT64_C(0x123456789));
  const loom_cmd_program_command_t command =
      loom_cmd_program_command_at(&program, 2);
  EXPECT_EQ(command.kind, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  EXPECT_EQ(command.argument_count, 3u);
  EXPECT_EQ(command.payload.dispatch_direct.workgroup_count_x, 1u);
  EXPECT_EQ(command.payload.dispatch_direct.workgroup_count_y, 2u);
  EXPECT_EQ(command.payload.dispatch_direct.workgroup_count_z, 3u);
}

TEST(CmdProgramTest, RejectsMalformedHeader) {
  std::vector<uint8_t> data = BuildValidProgram();
  data[LOOM_CMD_PROGRAM_HEADER_MAGIC_OFFSET] ^= 0xFF;
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsMalformedBufferReference) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET);
  iree_unaligned_store_le_u32(data.data() + table_offset +
                                  LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET,
                              1);
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsMalformedArgument) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_TABLE_OFFSET);
  iree_unaligned_store_le_u64(data.data() + table_offset +
                                  2 * LOOM_CMD_PROGRAM_ARGUMENT_SIZE +
                                  LOOM_CMD_PROGRAM_ARGUMENT_PAYLOAD_OFFSET,
                              2);
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsMalformedCommand) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET);
  iree_unaligned_store_le_u32(
      data.data() + table_offset + 2 * LOOM_CMD_PROGRAM_COMMAND_SIZE +
          LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_COUNT_OFFSET,
      4);
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

}  // namespace
}  // namespace loom
