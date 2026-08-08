// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_plan.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/cmd/descriptors/low_registry.h"
#include "loom/target/arch/cmd/lower/launch_artifact.h"
#include "loom/target/arch/cmd/lower/serialize.h"
#include "loom/target/arch/cmd/program.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticCapture;
using ::loom::testing::DiagnosticEmissionCapture;
using ModulePtr = ::loom::testing::ModulePtr;

class CmdProgramPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_cmd_low_descriptor_registry_initialize(&low_registry_);
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
        iree_make_cstring_view(source), IREE_SV("cmd_program_plan_test.loom"),
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

  bool HasSymbol(loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    return name_id != LOOM_STRING_ID_INVALID &&
           loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID;
  }

  void VerifyLowModule(loom_module_t* module) {
    DiagnosticEmissionCapture capture;
    loom_low_verify_options_t verify_options = {};
    verify_options.descriptor_registry = &low_registry_.registry;
    verify_options.emitter = capture.emitter();
    verify_options.max_errors = 20;
    loom_low_verify_scratch_t scratch =
        loom_low_verify_scratch_for_module(module);
    loom_low_verify_result_t result = {};
    IREE_ASSERT_OK(
        loom_low_verify_module(module, &verify_options, &scratch, &result));
    EXPECT_TRUE(capture.emissions.empty());
    IREE_ASSERT_EQ(result.error_count, 0u);
  }

  // Shared arena block pool backing source and prepared modules.
  iree_arena_block_pool_t block_pool_;
  // Source dialect context used by parsing, linking, and lowering.
  loom_context_t context_;
  // Portable command descriptor registry used by low verification.
  loom_target_low_descriptor_registry_t low_registry_ = {};
};

TEST_F(CmdProgramPlanTest, PreparesOwnedRootAndIndependentDependencies) {
  ModulePtr source_module = ParseAndVerify(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def @add_one(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer, %unused: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %one = scalar.constant 1 : i32
  %source_view = buffer.view %source[%base] : buffer -> view<128xi32, #dense>
  %target_view = buffer.view %target[%base] : buffer -> view<128xi32, #dense>
  %value = view.load %source_view[%workgroup] : view<128xi32, #dense> -> i32
  %result = scalar.addi %value, %one : i32
  view.store %result, %target_view[%workgroup] : i32, view<128xi32, #dense>
  kernel.return
}

kernel.def @add_two(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %two = scalar.constant 2 : i32
  %source_view = buffer.view %source[%base] : buffer -> view<128xi32, #dense>
  %target_view = buffer.view %target[%base] : buffer -> view<128xi32, #dense>
  %value = view.load %source_view[%workgroup] : view<128xi32, #dense> -> i32
  %result = scalar.addi %value, %two : i32
  view.store %result, %target_view[%workgroup] : i32, view<128xi32, #dense>
  kernel.return
}

command.program.def public target(@command_target) @pipeline(%element_count: index) launch(%source: buffer, %scratch: buffer, %intermediate: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  kernel.launch @add_one[%element_count](%source, %intermediate, %scratch) : [index](buffer, buffer, buffer)
  kernel.launch @add_two[%element_count](%scratch, %target) : [index](buffer, buffer)
  command.return
}
)");
  ASSERT_NE(source_module.get(), nullptr);
  loom_op_t* source_program =
      FindSymbol(source_module.get(), IREE_SV("pipeline"));

  loom_cmd_program_plan_t plan = {};
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(source_module.get(),
                                               source_program, &block_pool_,
                                               iree_allocator_system(), &plan));

  EXPECT_TRUE(loom_command_program_def_isa(source_program));
  EXPECT_EQ(FindSymbol(source_module.get(), IREE_SV("pipeline")),
            source_program);
  source_module.reset();

  ASSERT_NE(plan.root_module, nullptr);
  ASSERT_NE(plan.root_function_op, nullptr);
  EXPECT_TRUE(loom_low_func_def_isa(plan.root_function_op));
  EXPECT_EQ(FindSymbol(plan.root_module, IREE_SV("pipeline")),
            plan.root_function_op);
  EXPECT_FALSE(HasSymbol(plan.root_module, IREE_SV("add_one")));
  EXPECT_FALSE(HasSymbol(plan.root_module, IREE_SV("add_two")));
  VerifyLowModule(plan.root_module);

  ASSERT_EQ(plan.dependency_count, 2u);
  ASSERT_NE(plan.dependency_units, nullptr);
  const loom_cmd_kernel_unit_t& add_one = plan.dependency_units[0];
  const loom_cmd_kernel_unit_t& add_two = plan.dependency_units[1];
  ASSERT_NE(add_one.module, nullptr);
  ASSERT_NE(add_two.module, nullptr);
  EXPECT_NE(add_one.module, add_two.module);
  EXPECT_EQ(add_one.source_argument_count, 3u);
  ASSERT_EQ(add_one.argument_count, 2u);
  EXPECT_EQ(add_one.source_argument_ordinals[0], 0u);
  EXPECT_EQ(add_one.source_argument_ordinals[1], 1u);
  EXPECT_EQ(add_two.source_argument_count, 2u);
  EXPECT_EQ(add_two.argument_count, 2u);

  ASSERT_NE(plan.launch_module, nullptr);
  ASSERT_NE(plan.launch_function_op, nullptr);
  EXPECT_EQ(plan.launch_tuple_count, 1u);

  iree_byte_span_t launch_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_launch_program_serialize(
      plan.launch_module, &block_pool_, iree_allocator_system(), &launch_data));
  EXPECT_GT(launch_data.data_length, 0u);

  iree_byte_span_t program_data = iree_byte_span_empty();
  IREE_ASSERT_OK(
      loom_cmd_program_serialize_low(plan.root_module, plan.root_function_op,
                                     &program_data, iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(program_data.data, program_data.data_length),
      &program));
  EXPECT_EQ(program.requirements.fixed_buffer_count, 0u);
  EXPECT_EQ(program.requirements.rebindable_binding_count, 5u);
  EXPECT_EQ(program.requirements.executable_count, 2u);
  EXPECT_EQ(program.requirements.entry_count, 2u);
  ASSERT_EQ(program.commands.count, 3u);
  const loom_cmd_program_command_t first_dispatch =
      loom_cmd_program_command_at(&program, 0);
  EXPECT_EQ(first_dispatch.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  EXPECT_EQ(first_dispatch.payload.dispatch_indirect.executable_index, 0u);
  EXPECT_EQ(first_dispatch.payload.dispatch_indirect.entry_index, 0u);
  EXPECT_EQ(first_dispatch.argument_count, 2u);
  EXPECT_EQ(loom_cmd_program_command_at(&program, 1).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_EXECUTION_BARRIER);
  const loom_cmd_program_command_t second_dispatch =
      loom_cmd_program_command_at(&program, 2);
  EXPECT_EQ(second_dispatch.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  EXPECT_EQ(second_dispatch.payload.dispatch_indirect.executable_index, 1u);
  EXPECT_EQ(second_dispatch.payload.dispatch_indirect.entry_index, 1u);
  EXPECT_EQ(second_dispatch.argument_count, 2u);

  iree_allocator_free(iree_allocator_system(), program_data.data);
  iree_allocator_free(iree_allocator_system(), launch_data.data);
  loom_cmd_program_plan_deinitialize(&plan);
  EXPECT_EQ(plan.root_module, nullptr);
}

}  // namespace
}  // namespace loom
