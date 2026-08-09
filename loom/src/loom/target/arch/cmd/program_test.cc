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

static void StoreParameterRoot(std::vector<uint8_t>& data,
                               uint32_t table_offset, uint32_t index,
                               uint32_t fixed_buffer_index,
                               uint64_t required_byte_length,
                               uint64_t minimum_alignment) {
  uint8_t* record =
      data.data() + table_offset + index * LOOM_CMD_PROGRAM_PARAMETER_ROOT_SIZE;
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_FIXED_BUFFER_INDEX_OFFSET,
      fixed_buffer_index);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_REQUIRED_BYTE_LENGTH_OFFSET,
      required_byte_length);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_MINIMUM_ALIGNMENT_OFFSET,
      minimum_alignment);
}

static void StoreParameter(std::vector<uint8_t>& data, uint32_t table_offset,
                           uint32_t index, uint32_t key_offset,
                           uint32_t key_length, uint32_t fixed_buffer_index,
                           uint64_t byte_offset, uint64_t byte_length,
                           uint64_t minimum_alignment) {
  uint8_t* record =
      data.data() + table_offset + index * LOOM_CMD_PROGRAM_PARAMETER_SIZE;
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_PARAMETER_KEY_OFFSET_OFFSET, key_offset);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_PARAMETER_KEY_LENGTH_OFFSET, key_length);
  iree_unaligned_store_le_u32(
      record + LOOM_CMD_PROGRAM_PARAMETER_FIXED_BUFFER_INDEX_OFFSET,
      fixed_buffer_index);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_PARAMETER_BYTE_OFFSET_OFFSET, byte_offset);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_PARAMETER_BYTE_LENGTH_OFFSET, byte_length);
  iree_unaligned_store_le_u64(
      record + LOOM_CMD_PROGRAM_PARAMETER_MINIMUM_ALIGNMENT_OFFSET,
      minimum_alignment);
}

static std::vector<uint8_t> BuildValidProgram() {
  static constexpr uint32_t kBufferRefCount = 2;
  static constexpr uint32_t kArgumentCount = 3;
  static constexpr uint32_t kCommandCount = 6;
  static constexpr uint32_t kParameterRootCount = 1;
  static constexpr uint32_t kParameterCount = 2;
  static constexpr char kFirstKey[] = "alpha";
  static constexpr char kSecondKey[] = "blk.3.weight";
  static constexpr uint32_t kFirstKeyLength = sizeof(kFirstKey) - 1;
  static constexpr uint32_t kSecondKeyLength = sizeof(kSecondKey) - 1;
  static constexpr uint32_t kParameterKeyLength =
      kFirstKeyLength + kSecondKeyLength;
  loom_cmd_program_format_layout_t layout = {};
  IREE_CHECK_OK(loom_cmd_program_format_calculate_layout(
      kBufferRefCount, kArgumentCount, kCommandCount, kParameterRootCount,
      kParameterCount, kParameterKeyLength, &layout));
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
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_COUNT_OFFSET,
      kParameterRootCount);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_COUNT_OFFSET,
      kParameterCount);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_KEY_LENGTH_OFFSET,
      kParameterKeyLength);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET,
      layout.buffer_ref_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_TABLE_OFFSET,
      layout.argument_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET,
      layout.command_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_TABLE_OFFSET,
      layout.parameter_root_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_TABLE_OFFSET,
      layout.parameter_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_KEY_TABLE_OFFSET,
      layout.parameter_key_offset);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BINDING_INDEX_OFFSET,
      UINT32_MAX);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BINDING_INDEX_OFFSET,
      0);
  iree_unaligned_store_le_u64(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BYTE_LENGTH_OFFSET,
      12);
  iree_unaligned_store_le_u64(
      data.data() +
          LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_MINIMUM_ALIGNMENT_OFFSET,
      4);

  StoreBufferRef(data, layout.buffer_ref_offset, 0,
                 LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED, 0, 0, 256);
  StoreBufferRef(data, layout.buffer_ref_offset, 1,
                 LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE, 0, 0, 12);
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
               LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION, 0, 0);
  StoreParameterRoot(data, layout.parameter_root_offset, 0,
                     /*fixed_buffer_index=*/0, /*required_byte_length=*/512,
                     /*minimum_alignment=*/256);
  StoreParameter(data, layout.parameter_offset, 0, /*key_offset=*/0,
                 kFirstKeyLength, /*fixed_buffer_index=*/0, /*byte_offset=*/0,
                 /*byte_length=*/64, /*minimum_alignment=*/256);
  StoreParameter(data, layout.parameter_offset, 1,
                 /*key_offset=*/kFirstKeyLength, kSecondKeyLength,
                 /*fixed_buffer_index=*/0, /*byte_offset=*/256,
                 /*byte_length=*/128, /*minimum_alignment=*/256);
  memcpy(data.data() + layout.parameter_key_offset, kFirstKey, kFirstKeyLength);
  memcpy(data.data() + layout.parameter_key_offset + kFirstKeyLength,
         kSecondKey, kSecondKeyLength);
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
  EXPECT_EQ(program.requirements.transient.binding_index, UINT32_MAX);
  EXPECT_EQ(program.requirements.transient.required_byte_length, 0u);
  EXPECT_EQ(program.requirements.transient.minimum_alignment, 0u);
  EXPECT_EQ(program.requirements.launch_counts.binding_index, 0u);
  EXPECT_EQ(program.requirements.launch_counts.required_byte_length, 12u);
  EXPECT_EQ(program.requirements.launch_counts.minimum_alignment, 4u);
  EXPECT_EQ(program.requirements.executable_count, 1u);
  EXPECT_EQ(program.requirements.entry_count, 1u);
  EXPECT_EQ(program.buffer_refs.count, 2u);
  EXPECT_EQ(program.arguments.count, 3u);
  EXPECT_EQ(program.commands.count, 6u);
  ASSERT_EQ(program.parameter_roots.count, 1u);
  ASSERT_EQ(program.parameters.count, 2u);
  const loom_cmd_program_buffer_ref_t binding =
      loom_cmd_program_buffer_ref_at(&program, 1);
  EXPECT_EQ(binding.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
  EXPECT_EQ(binding.root_index, 0u);
  EXPECT_EQ(binding.byte_offset, 0u);
  EXPECT_EQ(binding.byte_length, 12u);
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
  const loom_cmd_program_parameter_root_t parameter_root =
      loom_cmd_program_parameter_root_at(&program, 0);
  EXPECT_EQ(parameter_root.fixed_buffer_index, 0u);
  EXPECT_EQ(parameter_root.required_byte_length, 512u);
  EXPECT_EQ(parameter_root.minimum_alignment, 256u);
  const loom_cmd_program_parameter_t parameter =
      loom_cmd_program_parameter_at(&program, 1);
  EXPECT_TRUE(iree_string_view_equal(parameter.key, IREE_SV("blk.3.weight")));
  EXPECT_EQ(parameter.fixed_buffer_index, 0u);
  EXPECT_EQ(parameter.byte_offset, 256u);
  EXPECT_EQ(parameter.byte_length, 128u);
  EXPECT_EQ(parameter.minimum_alignment, 256u);
}

TEST(CmdProgramTest, ParsesTransientRequirement) {
  std::vector<uint8_t> data = BuildValidProgram();
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BINDING_COUNT_OFFSET, 2);
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BINDING_INDEX_OFFSET, 1);
  iree_unaligned_store_le_u64(
      data.data() + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BYTE_LENGTH_OFFSET, 4096);
  iree_unaligned_store_le_u64(
      data.data() + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_MINIMUM_ALIGNMENT_OFFSET,
      256);

  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(AsByteSpan(data), &program));
  EXPECT_EQ(program.requirements.transient.binding_index, 1u);
  EXPECT_EQ(program.requirements.transient.required_byte_length, 4096u);
  EXPECT_EQ(program.requirements.transient.minimum_alignment, 256u);
}

TEST(CmdProgramTest, RejectsMalformedLaunchCountRequirement) {
  std::vector<uint8_t> data = BuildValidProgram();
  iree_unaligned_store_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BINDING_INDEX_OFFSET,
      1);

  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsMalformedLaunchCountTuple) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET);
  iree_unaligned_store_le_u64(
      data.data() + table_offset + LOOM_CMD_PROGRAM_BUFFER_REF_SIZE +
          LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET,
      8);

  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RelocatesDependencyIndices) {
  const std::vector<uint8_t> data = BuildValidProgram();
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(AsByteSpan(data), &program));

  const uint32_t executable_indices[] = {3};
  const uint32_t entry_indices[] = {4};
  const loom_cmd_program_dependency_relocation_t relocation = {
      /*.executable_indices=*/executable_indices,
      /*.executable_count=*/5,
      /*.entry_indices=*/entry_indices,
      /*.entry_count=*/6,
  };
  iree_byte_span_t relocated_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_relocate_dependencies(
      &program, &relocation, &relocated_data, iree_allocator_system()));

  loom_cmd_program_t relocated_program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(relocated_data.data,
                                relocated_data.data_length),
      &relocated_program));
  EXPECT_EQ(relocated_program.requirements.executable_count, 5u);
  EXPECT_EQ(relocated_program.requirements.entry_count, 6u);
  for (uint32_t command_index = 2; command_index <= 4; ++command_index) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(&relocated_program, command_index);
    const bool is_direct =
        command.kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT;
    EXPECT_EQ(is_direct ? command.payload.dispatch_direct.executable_index
                        : command.payload.dispatch_indirect.executable_index,
              3u);
    EXPECT_EQ(is_direct ? command.payload.dispatch_direct.entry_index
                        : command.payload.dispatch_indirect.entry_index,
              4u);
  }
  EXPECT_EQ(loom_cmd_program_command_at(&relocated_program, 0).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_FILL);
  EXPECT_EQ(loom_cmd_program_command_at(&relocated_program, 5).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION);
  EXPECT_TRUE(iree_string_view_equal(
      loom_cmd_program_parameter_at(&relocated_program, 1).key,
      IREE_SV("blk.3.weight")));

  iree_allocator_free(iree_allocator_system(), relocated_data.data);
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

TEST(CmdProgramTest, RejectsMalformedParameterRoot) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_TABLE_OFFSET);
  iree_unaligned_store_le_u32(
      data.data() + table_offset +
          LOOM_CMD_PROGRAM_PARAMETER_ROOT_FIXED_BUFFER_INDEX_OFFSET,
      1);
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

TEST(CmdProgramTest, RejectsMalformedParameter) {
  std::vector<uint8_t> data = BuildValidProgram();
  const uint32_t table_offset = iree_unaligned_load_le_u32(
      data.data() + LOOM_CMD_PROGRAM_HEADER_PARAMETER_TABLE_OFFSET);
  iree_unaligned_store_le_u32(
      data.data() + table_offset + LOOM_CMD_PROGRAM_PARAMETER_KEY_OFFSET_OFFSET,
      1);
  loom_cmd_program_t program = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_cmd_program_parse(AsByteSpan(data), &program));
}

}  // namespace
}  // namespace loom
