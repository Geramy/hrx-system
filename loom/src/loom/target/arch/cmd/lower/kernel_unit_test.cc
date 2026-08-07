// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/kernel_unit.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/ops/view/ops.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;
using ::loom::testing::DiagnosticCapture;

class CmdKernelUnitTest : public ::testing::Test {
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
        iree_make_cstring_view(source), IREE_SV("cmd_kernel_unit_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    if (!module) {
      for (const auto& diagnostic : capture.diagnostics) {
        ADD_FAILURE() << diagnostic.error->summary << " at line "
                      << diagnostic.origin_line << ", column "
                      << diagnostic.origin_column;
      }
      return ModulePtr();
    }
    EXPECT_TRUE(capture.diagnostics.empty());
    ModulePtr module_ptr(module);
    Verify(module);
    return module_ptr;
  }

  void Verify(loom_module_t* module) {
    loom_verify_options_t options = {};
    options.max_errors = 20;
    DiagnosticCapture capture;
    options.sink = capture.sink();
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
    for (const auto& diagnostic : capture.diagnostics) {
      ADD_FAILURE() << diagnostic.error->summary << " at line "
                    << diagnostic.origin_line << ", column "
                    << diagnostic.origin_column;
    }
    EXPECT_EQ(result.error_count, 0u);
  }

  loom_op_t* FindSymbol(loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    loom_op_t* defining_op = module->symbols.entries[symbol_id].defining_op;
    IREE_ASSERT_NE(defining_op, nullptr);
    return defining_op;
  }

  loom_op_t* FindLaunch(loom_func_like_t function) {
    loom_block_t* body = loom_region_entry_block(loom_func_like_body(function));
    loom_op_t* op = nullptr;
    loom_block_for_each_op(body, op) {
      if (loom_kernel_launch_isa(op)) return op;
    }
    return nullptr;
  }

  iree_host_size_t CountOpKind(const loom_op_t* op, loom_op_kind_t kind) {
    iree_host_size_t count = op->kind == kind ? 1 : 0;
    loom_region_t* const* regions = loom_op_regions(op);
    for (uint8_t region_index = 0; region_index < op->region_count;
         ++region_index) {
      const loom_region_t* region = regions[region_index];
      if (!region) continue;
      for (uint16_t block_index = 0; block_index < region->block_count;
           ++block_index) {
        const loom_block_t* block =
            loom_region_const_block(region, block_index);
        const loom_op_t* child_op = nullptr;
        loom_block_for_each_op(block, child_op) {
          count += CountOpKind(child_op, kind);
        }
      }
    }
    return count;
  }

  // Shared arena block pool backing source and derived modules.
  iree_arena_block_pool_t block_pool_;
  // Source dialect context used by the text parser and linker.
  loom_context_t context_;
};

TEST_F(CmdKernelUnitTest, SpecializesExactLaunchWithoutMutatingSourceKernel) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.def @project(%row_count: index, %column_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%row_count, %column_count, %one) workgroup_size(%one, %one, %one) : index
} launch(%token_count: index, %input: buffer, %scratch: buffer, %output: buffer) {
  %buffer_base = index.constant 0 : offset
  %one = index.constant 1 : index
  %input_view = buffer.view %input[%buffer_base] : buffer -> view<1xi32, #dense>
  %output_view = buffer.view %output[%buffer_base] : buffer -> view<1xi32, #dense>
  %value = view.load %input_view[0] : view<1xi32, #dense> -> i32
  %is_decode = index.cmp eq, %token_count, %one : index
  scf.if %is_decode {
    view.store %value, %output_view[0] : i32, view<1xi32, #dense>
  } else {
    %scratch_view = buffer.view %scratch[%buffer_base] : buffer -> view<1xi32, #dense>
    view.store %value, %scratch_view[0] : i32, view<1xi32, #dense>
    %scratch_value = view.load %scratch_view[0] : view<1xi32, #dense> -> i32
    view.store %scratch_value, %output_view[0] : i32, view<1xi32, #dense>
  }
  kernel.return
}

command.program.def @decode(%column_count: index) launch(%input: buffer, %scratch: buffer, %output: buffer) where [range(%column_count, 1, 4096)] {
  %one = index.constant 1 : index
  kernel.launch @project[%one, %column_count](%one, %input, %scratch, %output) : [index, index](index, buffer, buffer, buffer)
  command.return
}
)");
  ASSERT_NE(source_module.get(), nullptr);

  loom_op_t* source_kernel =
      FindSymbol(source_module.get(), IREE_SV("project"));
  loom_func_like_t source_kernel_like =
      loom_func_like_cast(source_module.get(), source_kernel);
  loom_func_like_t source_program = loom_func_like_cast(
      source_module.get(), FindSymbol(source_module.get(), IREE_SV("decode")));
  loom_op_t* source_launch = FindLaunch(source_program);
  ASSERT_NE(source_launch, nullptr);

  iree_arena_allocator_t fact_arena;
  iree_arena_initialize(&block_pool_, &fact_arena);
  loom_value_fact_table_t source_facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source_facts, &fact_arena,
                                                  source_module->values.count));
  loom_type_registry_configure_fact_context(&source_facts.context);
  IREE_ASSERT_OK(loom_value_fact_table_compute(
      &source_facts, source_module.get(), source_program));

  loom_cmd_kernel_unit_t unit = {};
  IREE_ASSERT_OK(loom_cmd_kernel_unit_materialize(
      source_module.get(), source_launch, &source_facts, &block_pool_,
      iree_allocator_system(), &unit));
  iree_arena_deinitialize(&fact_arena);

  ASSERT_NE(unit.module, nullptr);
  ASSERT_NE(unit.kernel_op, nullptr);
  Verify(unit.module);
  EXPECT_NE(unit.module, source_module.get());
  EXPECT_NE(unit.kernel_op, source_kernel);

  EXPECT_EQ(unit.source_workload_count, 2u);
  ASSERT_EQ(unit.workload_count, 1u);
  ASSERT_NE(unit.source_workload_ordinals, nullptr);
  EXPECT_EQ(unit.source_workload_ordinals[0], 1u);
  EXPECT_EQ(unit.source_argument_count, 4u);
  ASSERT_EQ(unit.argument_count, 2u);
  ASSERT_NE(unit.source_argument_ordinals, nullptr);
  EXPECT_EQ(unit.source_argument_ordinals[0], 1u);
  EXPECT_EQ(unit.source_argument_ordinals[1], 3u);
  EXPECT_EQ(CountOpKind(unit.kernel_op, LOOM_OP_SCF_IF), 0u);
  EXPECT_EQ(CountOpKind(unit.kernel_op, LOOM_OP_BUFFER_VIEW), 2u);
  EXPECT_EQ(CountOpKind(unit.kernel_op, LOOM_OP_VIEW_LOAD), 1u);
  EXPECT_EQ(CountOpKind(unit.kernel_op, LOOM_OP_VIEW_STORE), 1u);

  uint16_t source_argument_count = 0;
  loom_func_like_arg_ids(source_kernel_like, &source_argument_count);
  EXPECT_EQ(source_argument_count, 4u);
  EXPECT_EQ(
      loom_kernel_workload_arg_ids(source_module.get(), source_kernel).count,
      2u);
  EXPECT_EQ(CountOpKind(source_kernel, LOOM_OP_SCF_IF), 1u);
  const loom_string_id_t decode_name =
      loom_module_lookup_string(unit.module, IREE_SV("decode"));
  if (decode_name != LOOM_STRING_ID_INVALID) {
    EXPECT_EQ(loom_module_find_symbol(unit.module, decode_name),
              LOOM_SYMBOL_ID_INVALID);
  }

  loom_cmd_kernel_unit_deinitialize(&unit);
  EXPECT_EQ(unit.module, nullptr);
}

}  // namespace
}  // namespace loom
