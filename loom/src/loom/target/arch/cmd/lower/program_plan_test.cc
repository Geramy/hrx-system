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
  const loom_op_t* source_programs[] = {source_program};

  loom_cmd_program_plan_t plan = {};
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(
      source_module.get(), source_programs, IREE_ARRAYSIZE(source_programs),
      &block_pool_, iree_allocator_system(), &plan));

  EXPECT_TRUE(loom_command_program_def_isa(source_program));
  EXPECT_EQ(FindSymbol(source_module.get(), IREE_SV("pipeline")),
            source_program);
  source_module.reset();

  ASSERT_NE(plan.root_module, nullptr);
  ASSERT_EQ(plan.root_count, 1u);
  ASSERT_NE(plan.roots, nullptr);
  const loom_cmd_program_root_t& root = plan.roots[0];
  ASSERT_NE(root.function_op, nullptr);
  EXPECT_TRUE(loom_low_func_def_isa(root.function_op));
  EXPECT_EQ(FindSymbol(plan.root_module, IREE_SV("pipeline")),
            root.function_op);
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
  ASSERT_NE(root.launch_function_op, nullptr);
  EXPECT_EQ(root.launch_tuple_count, 1u);

  iree_byte_span_t launch_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_launch_program_serialize(
      plan.launch_module, &block_pool_, iree_allocator_system(), &launch_data));
  EXPECT_GT(launch_data.data_length, 0u);

  iree_byte_span_t program_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      plan.root_module, root.function_op, &root.parameters, &root.transient,
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

TEST_F(CmdProgramPlanTest, PlacesParametersInFixedSourceRoots) {
  ModulePtr source_module = ParseAndVerify(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def @combine() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%lhs: view<3xi32, #dense>, %rhs: view<4xi32, #dense>, %target: buffer) {
  %zero = index.constant 0 : index
  %base = index.constant 0 : offset
  %lhs_value = view.load %lhs[%zero] : view<3xi32, #dense> -> i32
  %rhs_value = view.load %rhs[%zero] : view<4xi32, #dense> -> i32
  %sum = scalar.addi %lhs_value, %rhs_value : i32
  %target_view = buffer.view %target[%base] : buffer -> view<1xi32, #dense>
  view.store %sum, %target_view[%zero] : i32, view<1xi32, #dense>
  kernel.return
}

command.program.def public target(@command_target) @parameterized() launch(%parameters: buffer, %target: buffer) {
  %layer = index.constant 3 : index
  %lhs = command.parameter %parameters, "blk.{}.lhs"[%layer] : view<3xi32, #dense>
  %rhs = command.parameter %parameters, "shared.rhs"[] : view<4xi32, #dense>
  kernel.launch @combine[](%lhs, %rhs, %target) : [](view<3xi32, #dense>, view<4xi32, #dense>, buffer)
  command.return
}
)");
  ASSERT_NE(source_module.get(), nullptr);
  loom_op_t* source_program =
      FindSymbol(source_module.get(), IREE_SV("parameterized"));
  const loom_op_t* source_programs[] = {source_program};

  loom_cmd_program_plan_t plan = {};
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(
      source_module.get(), source_programs, IREE_ARRAYSIZE(source_programs),
      &block_pool_, iree_allocator_system(), &plan));
  source_module.reset();

  ASSERT_EQ(plan.root_count, 1u);
  const loom_cmd_program_root_t& root = plan.roots[0];
  ASSERT_EQ(root.parameters.root_count, 1u);
  ASSERT_NE(root.parameters.roots, nullptr);
  EXPECT_EQ(root.parameters.roots[0].source_binding_ordinal, 0u);
  EXPECT_EQ(root.parameters.roots[0].fixed_buffer_index, 0u);
  EXPECT_EQ(root.parameters.roots[0].required_byte_length, 272u);
  EXPECT_EQ(root.parameters.roots[0].minimum_alignment, 256u);

  ASSERT_EQ(root.parameters.count, 2u);
  ASSERT_NE(root.parameters.entries, nullptr);
  const loom_cmd_parameter_requirement_t& lhs = root.parameters.entries[0];
  EXPECT_TRUE(iree_string_view_equal(lhs.key, IREE_SV("blk.3.lhs")));
  EXPECT_EQ(lhs.source_binding_ordinal, 0u);
  EXPECT_EQ(lhs.fixed_buffer_index, 0u);
  EXPECT_EQ(lhs.byte_offset, 0u);
  EXPECT_EQ(lhs.byte_length, 12u);
  EXPECT_EQ(lhs.minimum_alignment, 256u);
  const loom_cmd_parameter_requirement_t& rhs = root.parameters.entries[1];
  EXPECT_TRUE(iree_string_view_equal(rhs.key, IREE_SV("shared.rhs")));
  EXPECT_EQ(rhs.source_binding_ordinal, 0u);
  EXPECT_EQ(rhs.fixed_buffer_index, 0u);
  EXPECT_EQ(rhs.byte_offset, 256u);
  EXPECT_EQ(rhs.byte_length, 16u);
  EXPECT_EQ(rhs.minimum_alignment, 256u);

  iree_byte_span_t program_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      plan.root_module, root.function_op, &root.parameters, &root.transient,
      &program_data, iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(program_data.data, program_data.data_length),
      &program));
  EXPECT_EQ(program.requirements.fixed_buffer_count, 1u);
  EXPECT_EQ(program.requirements.rebindable_binding_count, 1u);
  EXPECT_EQ(program.requirements.executable_count, 1u);
  EXPECT_EQ(program.requirements.entry_count, 1u);
  ASSERT_EQ(program.parameter_roots.count, 1u);
  const loom_cmd_program_parameter_root_t serialized_root =
      loom_cmd_program_parameter_root_at(&program, 0);
  EXPECT_EQ(serialized_root.fixed_buffer_index, 0u);
  EXPECT_EQ(serialized_root.required_byte_length, 272u);
  EXPECT_EQ(serialized_root.minimum_alignment, 256u);
  ASSERT_EQ(program.parameters.count, 2u);
  const loom_cmd_program_parameter_t serialized_lhs =
      loom_cmd_program_parameter_at(&program, 0);
  EXPECT_TRUE(iree_string_view_equal(serialized_lhs.key, IREE_SV("blk.3.lhs")));
  EXPECT_EQ(serialized_lhs.fixed_buffer_index, 0u);
  EXPECT_EQ(serialized_lhs.byte_offset, 0u);
  EXPECT_EQ(serialized_lhs.byte_length, 12u);
  EXPECT_EQ(serialized_lhs.minimum_alignment, 256u);
  const loom_cmd_program_parameter_t serialized_rhs =
      loom_cmd_program_parameter_at(&program, 1);
  EXPECT_TRUE(
      iree_string_view_equal(serialized_rhs.key, IREE_SV("shared.rhs")));
  EXPECT_EQ(serialized_rhs.fixed_buffer_index, 0u);
  EXPECT_EQ(serialized_rhs.byte_offset, 256u);
  EXPECT_EQ(serialized_rhs.byte_length, 16u);
  EXPECT_EQ(serialized_rhs.minimum_alignment, 256u);
  ASSERT_EQ(program.commands.count, 1u);
  const loom_cmd_program_command_t dispatch =
      loom_cmd_program_command_at(&program, 0);
  ASSERT_EQ(dispatch.kind, LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  ASSERT_EQ(dispatch.argument_count, 3u);

  const loom_cmd_program_argument_t lhs_argument =
      loom_cmd_program_argument_at(&program, dispatch.argument_offset + 0);
  const loom_cmd_program_argument_t rhs_argument =
      loom_cmd_program_argument_at(&program, dispatch.argument_offset + 1);
  const loom_cmd_program_argument_t target_argument =
      loom_cmd_program_argument_at(&program, dispatch.argument_offset + 2);
  ASSERT_EQ(lhs_argument.kind, LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER_REF);
  ASSERT_EQ(rhs_argument.kind, LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER_REF);
  ASSERT_EQ(target_argument.kind, LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER_REF);

  const loom_cmd_program_buffer_ref_t lhs_ref =
      loom_cmd_program_buffer_ref_at(&program, (uint32_t)lhs_argument.payload);
  EXPECT_EQ(lhs_ref.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED);
  EXPECT_EQ(lhs_ref.root_index, 0u);
  EXPECT_EQ(lhs_ref.byte_offset, 0u);
  EXPECT_EQ(lhs_ref.byte_length, 12u);
  const loom_cmd_program_buffer_ref_t rhs_ref =
      loom_cmd_program_buffer_ref_at(&program, (uint32_t)rhs_argument.payload);
  EXPECT_EQ(rhs_ref.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED);
  EXPECT_EQ(rhs_ref.root_index, 0u);
  EXPECT_EQ(rhs_ref.byte_offset, 256u);
  EXPECT_EQ(rhs_ref.byte_length, 16u);
  const loom_cmd_program_buffer_ref_t target_ref =
      loom_cmd_program_buffer_ref_at(&program,
                                     (uint32_t)target_argument.payload);
  EXPECT_EQ(target_ref.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
  EXPECT_EQ(target_ref.root_index, 0u);
  EXPECT_EQ(target_ref.byte_offset, 0u);
  EXPECT_EQ(target_ref.byte_length, UINT64_MAX);

  iree_allocator_free(iree_allocator_system(), program_data.data);
  loom_cmd_program_plan_deinitialize(&plan);
}

TEST_F(CmdProgramPlanTest, InternsEquivalentDependencySpecializations) {
  ModulePtr source_module = ParseAndVerify(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def @add_bias(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer, %bias: i32) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %source_view = buffer.view %source[%base] : buffer -> view<128xi32, #dense>
  %target_view = buffer.view %target[%base] : buffer -> view<128xi32, #dense>
  %value = view.load %source_view[%workgroup] : view<128xi32, #dense> -> i32
  %result = scalar.addi %value, %bias : i32
  view.store %result, %target_view[%workgroup] : i32, view<128xi32, #dense>
  kernel.return
}

command.program.def public target(@command_target) @pipeline(%element_count: index) launch(%source: buffer, %first: buffer, %second: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  %one_a = scalar.constant 1 : i32
  %one_b = scalar.constant 1 : i32
  %two = scalar.constant 2 : i32
  kernel.launch @add_bias[%element_count](%source, %first, %one_a) : [index](buffer, buffer, i32)
  kernel.launch @add_bias[%element_count](%first, %second, %one_b) : [index](buffer, buffer, i32)
  kernel.launch @add_bias[%element_count](%second, %target, %two) : [index](buffer, buffer, i32)
  command.return
}
)");
  ASSERT_NE(source_module.get(), nullptr);
  loom_op_t* source_program =
      FindSymbol(source_module.get(), IREE_SV("pipeline"));
  const loom_op_t* source_programs[] = {source_program};

  loom_cmd_program_plan_t plan = {};
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(
      source_module.get(), source_programs, IREE_ARRAYSIZE(source_programs),
      &block_pool_, iree_allocator_system(), &plan));
  source_module.reset();

  ASSERT_EQ(plan.dependency_count, 2u);
  ASSERT_NE(plan.dependency_units, nullptr);
  EXPECT_EQ(plan.dependency_units[0].source_argument_count, 3u);
  EXPECT_EQ(plan.dependency_units[0].argument_count, 2u);
  EXPECT_EQ(plan.dependency_units[1].source_argument_count, 3u);
  EXPECT_EQ(plan.dependency_units[1].argument_count, 2u);

  iree_byte_span_t program_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      plan.root_module, plan.roots[0].function_op, &plan.roots[0].parameters,
      &plan.roots[0].transient, &program_data, iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(program_data.data, program_data.data_length),
      &program));
  EXPECT_EQ(program.requirements.executable_count, 2u);
  EXPECT_EQ(program.requirements.entry_count, 2u);
  ASSERT_EQ(program.commands.count, 5u);

  const loom_cmd_program_command_t first_dispatch =
      loom_cmd_program_command_at(&program, 0);
  ASSERT_EQ(first_dispatch.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  EXPECT_EQ(first_dispatch.payload.dispatch_indirect.executable_index, 0u);
  EXPECT_EQ(first_dispatch.payload.dispatch_indirect.entry_index, 0u);
  EXPECT_EQ(first_dispatch.argument_count, 2u);
  EXPECT_EQ(loom_cmd_program_command_at(&program, 1).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_EXECUTION_BARRIER);
  const loom_cmd_program_command_t second_dispatch =
      loom_cmd_program_command_at(&program, 2);
  ASSERT_EQ(second_dispatch.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  EXPECT_EQ(second_dispatch.payload.dispatch_indirect.executable_index, 0u);
  EXPECT_EQ(second_dispatch.payload.dispatch_indirect.entry_index, 0u);
  EXPECT_EQ(second_dispatch.argument_count, 2u);
  EXPECT_EQ(loom_cmd_program_command_at(&program, 3).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_EXECUTION_BARRIER);
  const loom_cmd_program_command_t third_dispatch =
      loom_cmd_program_command_at(&program, 4);
  ASSERT_EQ(third_dispatch.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  EXPECT_EQ(third_dispatch.payload.dispatch_indirect.executable_index, 1u);
  EXPECT_EQ(third_dispatch.payload.dispatch_indirect.entry_index, 1u);
  EXPECT_EQ(third_dispatch.argument_count, 2u);

  iree_allocator_free(iree_allocator_system(), program_data.data);
  loom_cmd_program_plan_deinitialize(&plan);
}

TEST_F(CmdProgramPlanTest, PreparesMultipleRootsWithSharedDependencies) {
  ModulePtr source_module = ParseAndVerify(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def @increment(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
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

kernel.def @double(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %two = scalar.constant 2 : i32
  %source_view = buffer.view %source[%base] : buffer -> view<128xi32, #dense>
  %target_view = buffer.view %target[%base] : buffer -> view<128xi32, #dense>
  %value = view.load %source_view[%workgroup] : view<128xi32, #dense> -> i32
  %result = scalar.muli %value, %two : i32
  view.store %result, %target_view[%workgroup] : i32, view<128xi32, #dense>
  kernel.return
}

command.program.def public target(@command_target) @increment_then_double(%element_count: index) launch(%source: buffer, %scratch: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  kernel.launch @increment[%element_count](%source, %scratch) : [index](buffer, buffer)
  kernel.launch @double[%element_count](%scratch, %target) : [index](buffer, buffer)
  command.return
}

command.program.def public target(@command_target) @increment_twice(%element_count: index) launch(%source: buffer, %scratch: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  kernel.launch @increment[%element_count](%source, %scratch) : [index](buffer, buffer)
  kernel.launch @increment[%element_count](%scratch, %target) : [index](buffer, buffer)
  command.return
}
)");
  ASSERT_NE(source_module.get(), nullptr);
  const loom_op_t* source_programs[] = {
      FindSymbol(source_module.get(), IREE_SV("increment_twice")),
      FindSymbol(source_module.get(), IREE_SV("increment_then_double")),
  };

  loom_cmd_program_plan_t plan = {};
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(
      source_module.get(), source_programs, IREE_ARRAYSIZE(source_programs),
      &block_pool_, iree_allocator_system(), &plan));
  source_module.reset();

  ASSERT_EQ(plan.root_count, 2u);
  ASSERT_NE(plan.roots, nullptr);
  ASSERT_NE(plan.root_module, nullptr);
  VerifyLowModule(plan.root_module);
  EXPECT_FALSE(HasSymbol(plan.root_module, IREE_SV("increment")));

  const loom_cmd_program_root_t& twice = plan.roots[0];
  const loom_cmd_program_root_t& mixed = plan.roots[1];
  EXPECT_EQ(FindSymbol(plan.root_module, IREE_SV("increment_twice")),
            twice.function_op);
  EXPECT_EQ(FindSymbol(plan.root_module, IREE_SV("increment_then_double")),
            mixed.function_op);
  EXPECT_TRUE(loom_low_func_def_isa(twice.function_op));
  EXPECT_TRUE(loom_low_func_def_isa(mixed.function_op));
  ASSERT_NE(plan.launch_module, nullptr);
  ASSERT_NE(twice.launch_function_op, nullptr);
  ASSERT_NE(mixed.launch_function_op, nullptr);
  EXPECT_EQ(FindSymbol(plan.launch_module, IREE_SV("increment_twice")),
            twice.launch_function_op);
  EXPECT_EQ(FindSymbol(plan.launch_module, IREE_SV("increment_then_double")),
            mixed.launch_function_op);
  EXPECT_EQ(twice.launch_tuple_count, 1u);
  EXPECT_EQ(mixed.launch_tuple_count, 1u);

  ASSERT_EQ(plan.dependency_count, 2u);
  ASSERT_NE(plan.dependency_units, nullptr);
  EXPECT_EQ(plan.dependency_units[0].source_argument_count, 2u);
  EXPECT_EQ(plan.dependency_units[0].argument_count, 2u);
  EXPECT_EQ(plan.dependency_units[1].source_argument_count, 2u);
  EXPECT_EQ(plan.dependency_units[1].argument_count, 2u);
  ASSERT_EQ(twice.dependency_count, 1u);
  ASSERT_NE(twice.dependency_unit_indices, nullptr);
  EXPECT_EQ(twice.dependency_unit_indices[0], 0u);
  ASSERT_EQ(mixed.dependency_count, 2u);
  ASSERT_NE(mixed.dependency_unit_indices, nullptr);
  EXPECT_EQ(mixed.dependency_unit_indices[0], 0u);
  EXPECT_EQ(mixed.dependency_unit_indices[1], 1u);

  iree_byte_span_t launch_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_launch_program_serialize(
      plan.launch_module, &block_pool_, iree_allocator_system(), &launch_data));
  EXPECT_GT(launch_data.data_length, 0u);

  iree_byte_span_t twice_program_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      plan.root_module, twice.function_op, &twice.parameters, &twice.transient,
      &twice_program_data, iree_allocator_system()));
  loom_cmd_program_t twice_program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(twice_program_data.data,
                                twice_program_data.data_length),
      &twice_program));
  EXPECT_EQ(twice_program.requirements.rebindable_binding_count, 4u);
  EXPECT_EQ(twice_program.requirements.executable_count, 1u);
  EXPECT_EQ(twice_program.requirements.entry_count, 1u);
  ASSERT_EQ(twice_program.commands.count, 3u);
  EXPECT_EQ(loom_cmd_program_command_at(&twice_program, 0)
                .payload.dispatch_indirect.executable_index,
            0u);
  EXPECT_EQ(loom_cmd_program_command_at(&twice_program, 2)
                .payload.dispatch_indirect.executable_index,
            0u);

  iree_byte_span_t mixed_program_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      plan.root_module, mixed.function_op, &mixed.parameters, &mixed.transient,
      &mixed_program_data, iree_allocator_system()));
  loom_cmd_program_t mixed_program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(mixed_program_data.data,
                                mixed_program_data.data_length),
      &mixed_program));
  EXPECT_EQ(mixed_program.requirements.rebindable_binding_count, 4u);
  EXPECT_EQ(mixed_program.requirements.executable_count, 2u);
  EXPECT_EQ(mixed_program.requirements.entry_count, 2u);
  ASSERT_EQ(mixed_program.commands.count, 3u);
  EXPECT_EQ(loom_cmd_program_command_at(&mixed_program, 0)
                .payload.dispatch_indirect.executable_index,
            0u);
  EXPECT_EQ(loom_cmd_program_command_at(&mixed_program, 2)
                .payload.dispatch_indirect.executable_index,
            1u);

  iree_allocator_free(iree_allocator_system(), mixed_program_data.data);
  iree_allocator_free(iree_allocator_system(), twice_program_data.data);
  iree_allocator_free(iree_allocator_system(), launch_data.data);
  loom_cmd_program_plan_deinitialize(&plan);
}

TEST_F(CmdProgramPlanTest, FlattensNestedProgramsAcrossSelectedRoots) {
  ModulePtr source_module = ParseAndVerify(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def @copy_one() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  %base = index.constant 0 : offset
  %zero = index.constant 0 : index
  %source_view = buffer.view %source[%base] : buffer -> view<1xi32, #dense>
  %target_view = buffer.view %target[%base] : buffer -> view<1xi32, #dense>
  %value = view.load %source_view[%zero] : view<1xi32, #dense> -> i32
  view.store %value, %target_view[%zero] : i32, view<1xi32, #dense>
  kernel.return
}

command.program.def target(@command_target) @copy_stage() launch(%source: buffer, %target: buffer) {
  kernel.launch @copy_one[](%source, %target) : [](buffer, buffer)
  command.return
}

command.program.def public target(@command_target) @copy_twice() launch(%source: buffer, %scratch: buffer, %target: buffer) {
  command.program.launch @copy_stage[](%source, %scratch) : [](buffer, buffer)
  command.program.launch @copy_stage[](%scratch, %target) : [](buffer, buffer)
  command.return
}

command.program.def public target(@command_target) @copy_once() launch(%source: buffer, %target: buffer) {
  command.program.launch @copy_stage[](%source, %target) : [](buffer, buffer)
  command.return
}
)");
  ASSERT_NE(source_module.get(), nullptr);
  const loom_op_t* source_programs[] = {
      FindSymbol(source_module.get(), IREE_SV("copy_twice")),
      FindSymbol(source_module.get(), IREE_SV("copy_once")),
  };

  loom_cmd_program_plan_t plan = {};
  IREE_ASSERT_OK(loom_cmd_program_plan_prepare(
      source_module.get(), source_programs, IREE_ARRAYSIZE(source_programs),
      &block_pool_, iree_allocator_system(), &plan));
  source_module.reset();

  ASSERT_EQ(plan.root_count, 2u);
  ASSERT_EQ(plan.dependency_count, 1u);
  EXPECT_EQ(plan.dependency_units[0].argument_count, 2u);
  ASSERT_EQ(plan.roots[0].dependency_count, 1u);
  ASSERT_EQ(plan.roots[1].dependency_count, 1u);
  VerifyLowModule(plan.root_module);
  EXPECT_FALSE(HasSymbol(plan.root_module, IREE_SV("copy_stage")));
  EXPECT_FALSE(HasSymbol(plan.root_module, IREE_SV("copy_one")));

  iree_byte_span_t twice_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      plan.root_module, plan.roots[0].function_op, &plan.roots[0].parameters,
      &plan.roots[0].transient, &twice_data, iree_allocator_system()));
  loom_cmd_program_t twice = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(twice_data.data, twice_data.data_length),
      &twice));
  EXPECT_EQ(twice.requirements.rebindable_binding_count, 3u);
  ASSERT_EQ(twice.commands.count, 3u);
  EXPECT_EQ(loom_cmd_program_command_at(&twice, 0).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  EXPECT_EQ(loom_cmd_program_command_at(&twice, 1).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_EXECUTION_BARRIER);
  EXPECT_EQ(loom_cmd_program_command_at(&twice, 2).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  const loom_cmd_program_command_t first_dispatch =
      loom_cmd_program_command_at(&twice, 0);
  const loom_cmd_program_command_t second_dispatch =
      loom_cmd_program_command_at(&twice, 2);
  ASSERT_EQ(first_dispatch.argument_count, 2u);
  ASSERT_EQ(second_dispatch.argument_count, 2u);
  const loom_cmd_program_argument_t first_source =
      loom_cmd_program_argument_at(&twice, first_dispatch.argument_offset);
  const loom_cmd_program_argument_t first_target =
      loom_cmd_program_argument_at(&twice, first_dispatch.argument_offset + 1);
  const loom_cmd_program_argument_t second_source =
      loom_cmd_program_argument_at(&twice, second_dispatch.argument_offset);
  const loom_cmd_program_argument_t second_target =
      loom_cmd_program_argument_at(&twice, second_dispatch.argument_offset + 1);
  ASSERT_EQ(first_source.kind, LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER_REF);
  ASSERT_EQ(first_target.kind, LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER_REF);
  ASSERT_EQ(second_source.kind, LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER_REF);
  ASSERT_EQ(second_target.kind, LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER_REF);
  EXPECT_EQ(
      loom_cmd_program_buffer_ref_at(&twice, (uint32_t)first_source.payload)
          .root_index,
      0u);
  EXPECT_EQ(
      loom_cmd_program_buffer_ref_at(&twice, (uint32_t)first_target.payload)
          .root_index,
      1u);
  EXPECT_EQ(
      loom_cmd_program_buffer_ref_at(&twice, (uint32_t)second_source.payload)
          .root_index,
      1u);
  EXPECT_EQ(
      loom_cmd_program_buffer_ref_at(&twice, (uint32_t)second_target.payload)
          .root_index,
      2u);

  iree_byte_span_t once_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      plan.root_module, plan.roots[1].function_op, &plan.roots[1].parameters,
      &plan.roots[1].transient, &once_data, iree_allocator_system()));
  loom_cmd_program_t once = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(once_data.data, once_data.data_length), &once));
  EXPECT_EQ(once.requirements.rebindable_binding_count, 2u);
  ASSERT_EQ(once.commands.count, 1u);
  EXPECT_EQ(loom_cmd_program_command_at(&once, 0).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);

  iree_allocator_free(iree_allocator_system(), once_data.data);
  iree_allocator_free(iree_allocator_system(), twice_data.data);
  loom_cmd_program_plan_deinitialize(&plan);
}

TEST_F(CmdProgramPlanTest, RejectsRecursiveProgramComposition) {
  ModulePtr source_module = ParseAndVerify(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

command.program.def public target(@command_target) @recursive() launch() {
  command.program.launch @recursive[]() : []()
  command.return
}
)");
  ASSERT_NE(source_module.get(), nullptr);
  const loom_op_t* source_programs[] = {
      FindSymbol(source_module.get(), IREE_SV("recursive")),
  };

  loom_cmd_program_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_cmd_program_plan_prepare(
          source_module.get(), source_programs, IREE_ARRAYSIZE(source_programs),
          &block_pool_, iree_allocator_system(), &plan));
  EXPECT_EQ(plan.root_module, nullptr);
}

}  // namespace
}  // namespace loom
