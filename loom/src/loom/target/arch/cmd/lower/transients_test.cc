// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/cmd/format.h"
#include "loom/target/arch/cmd/lower/program_plan.h"
#include "loom/target/arch/cmd/lower/serialize.h"
#include "loom/target/arch/cmd/program.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;
using ::loom::testing::DiagnosticCapture;

class CmdTransientsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseAndVerify(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    DiagnosticCapture capture;
    parse_options.diagnostic_sink = capture.sink();
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("cmd_transients_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    ModulePtr module_ptr(module);

    loom_verify_options_t verify_options = {};
    verify_options.max_errors = 20;
    verify_options.sink = capture.sink();
    loom_verify_result_t result = {};
    IREE_CHECK_OK(loom_verify_module(module, &verify_options, &result));
    for (const auto& diagnostic : capture.diagnostics) {
      ADD_FAILURE() << diagnostic.error->summary << " at line "
                    << diagnostic.origin_line << ", column "
                    << diagnostic.origin_column;
    }
    EXPECT_EQ(result.error_count, 0u);
    return module_ptr;
  }

  loom_op_t* FindSymbol(loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return module->symbols.entries[symbol_id].defining_op;
  }

  static loom_cmd_program_buffer_ref_t DispatchBufferRef(
      const loom_cmd_program_t* program, uint32_t command_index,
      uint32_t argument_index) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(program, command_index);
    const loom_cmd_program_entry_schema_t schema =
        loom_cmd_program_entry_schema_at(program,
                                         command.argument_schema_index);
    const iree_const_byte_span_t data =
        loom_cmd_program_command_argument_data(program, &command);
    const uint8_t* cursor = data.data;
    for (uint32_t i = 0; i < argument_index; ++i) {
      switch (loom_cmd_program_entry_schema_kind_at(program, &schema, i)) {
        case LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER:
          cursor += LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
          break;
        case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B8:
          cursor += 1;
          break;
        case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B16:
          cursor += 2;
          break;
        case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B32:
          cursor += 4;
          break;
        case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B64:
          cursor += 8;
          break;
      }
    }
    EXPECT_EQ(
        loom_cmd_program_entry_schema_kind_at(program, &schema, argument_index),
        LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER);
    return (loom_cmd_program_buffer_ref_t){
        /*.role=*/(loom_cmd_program_buffer_role_t)iree_unaligned_load_le_u32(
            cursor + LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET),
        /*.root_index=*/
        iree_unaligned_load_le_u32(
            cursor + LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET),
        /*.byte_offset=*/
        iree_unaligned_load_le_u64(
            cursor + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET),
        /*.byte_length=*/
        iree_unaligned_load_le_u64(
            cursor + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET),
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(CmdTransientsTest, AliasesDisjointWavesAndSeparatesConcurrentWaves) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.def @copy_one() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  %zero = index.constant 0 : offset
  %element = index.constant 0 : index
  %source_view = buffer.view %source[%zero] : buffer -> view<1xi32, #dense>
  %target_view = buffer.view %target[%zero] : buffer -> view<1xi32, #dense>
  %value = view.load %source_view[%element] : view<1xi32, #dense> -> i32
  view.store %value, %target_view[%element] : i32, view<1xi32, #dense>
  kernel.return
}

kernel.def @sum_three() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%a: buffer, %b: buffer, %c: buffer, %target: buffer) {
  %zero = index.constant 0 : offset
  %element = index.constant 0 : index
  %a_view = buffer.view %a[%zero] : buffer -> view<1xi32, #dense>
  %b_view = buffer.view %b[%zero] : buffer -> view<1xi32, #dense>
  %c_view = buffer.view %c[%zero] : buffer -> view<1xi32, #dense>
  %target_view = buffer.view %target[%zero] : buffer -> view<1xi32, #dense>
  %a_value = view.load %a_view[%element] : view<1xi32, #dense> -> i32
  %b_value = view.load %b_view[%element] : view<1xi32, #dense> -> i32
  %c_value = view.load %c_view[%element] : view<1xi32, #dense> -> i32
  %ab = scalar.addi %a_value, %b_value : i32
  %abc = scalar.addi %ab, %c_value : i32
  view.store %abc, %target_view[%element] : i32, view<1xi32, #dense>
  kernel.return
}

command.program.def public @attention_wave() launch(%source: buffer, %target: buffer) {
  %branch_bytes = index.constant 64 : offset
  %post_bytes = index.constant 256 : offset
  %unused_bytes = index.constant 4096 : offset
  %query = buffer.alloca %branch_bytes {base_alignment = 64, memory_space = global} : buffer
  %key = buffer.alloca %branch_bytes {base_alignment = 256, memory_space = global} : buffer
  %value = buffer.alloca %branch_bytes {base_alignment = 64, memory_space = global} : buffer
  %post = buffer.alloca %post_bytes {base_alignment = 256, memory_space = global} : buffer
  %unused = buffer.alloca %unused_bytes {base_alignment = 4096, memory_space = global} : buffer
  command.concurrent {
    kernel.launch @copy_one[](%source, %query) : [](buffer, buffer)
    kernel.launch @copy_one[](%source, %key) : [](buffer, buffer)
    kernel.launch @copy_one[](%source, %value) : [](buffer, buffer)
  }
  kernel.launch @sum_three[](%query, %key, %value, %target) : [](buffer, buffer, buffer, buffer)
  kernel.launch @copy_one[](%source, %post) : [](buffer, buffer)
  kernel.launch @copy_one[](%post, %target) : [](buffer, buffer)
  command.return
}
)");

  const loom_op_t* roots[] = {
      FindSymbol(source_module.get(), IREE_SV("attention_wave")),
  };
  loom_cmd_program_plan_t plan = {};
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(
      source_module.get(), roots, IREE_ARRAYSIZE(roots), &block_pool_,
      iree_allocator_system(), &plan));
  source_module.reset();

  ASSERT_EQ(plan.root_count, 1u);
  const loom_cmd_program_root_t& root = plan.roots[0];
  EXPECT_EQ(root.transient.binding_index, 2u);
  EXPECT_EQ(root.transient.required_byte_length, 320u);
  EXPECT_EQ(root.transient.minimum_alignment, 256u);

  iree_byte_span_t program_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      plan.root_module, root.function_op, &root.parameters, &root.transient,
      &program_data, iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(program_data.data, program_data.data_length),
      &program));
  EXPECT_EQ(program.requirements.rebindable_binding_count, 3u);
  EXPECT_EQ(program.requirements.transient.binding_index, 2u);
  EXPECT_EQ(program.requirements.transient.required_byte_length, 320u);
  EXPECT_EQ(program.requirements.launch_counts.binding_index, UINT32_MAX);
  EXPECT_EQ(program.requirements.launch_counts.required_byte_length, 0u);
  ASSERT_EQ(program.commands.count, 9u);

  const std::array<loom_cmd_program_buffer_ref_t, 3> branches = {
      DispatchBufferRef(&program, 0, 1),
      DispatchBufferRef(&program, 1, 1),
      DispatchBufferRef(&program, 2, 1),
  };
  for (const loom_cmd_program_buffer_ref_t branch : branches) {
    EXPECT_EQ(branch.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
    EXPECT_EQ(branch.root_index, 2u);
    EXPECT_EQ(branch.byte_length, 64u);
  }
  EXPECT_NE(branches[0].byte_offset, branches[1].byte_offset);
  EXPECT_NE(branches[0].byte_offset, branches[2].byte_offset);
  EXPECT_NE(branches[1].byte_offset, branches[2].byte_offset);

  const loom_cmd_program_buffer_ref_t post = DispatchBufferRef(&program, 6, 1);
  EXPECT_EQ(post.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
  EXPECT_EQ(post.root_index, 2u);
  EXPECT_EQ(post.byte_offset, 0u);
  EXPECT_EQ(post.byte_length, 256u);

  iree_allocator_free(iree_allocator_system(), program_data.data);
  loom_cmd_program_plan_deinitialize(&plan);
}

}  // namespace
}  // namespace loom
