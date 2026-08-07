// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/exact_function.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;
using ::iree::testing::status::StatusIs;

class ExactFunctionTest : public ::testing::Test {
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
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("exact_function_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    EXPECT_NE(module, nullptr);
    if (module == nullptr) return ModulePtr();
    loom_verify_options_t verify_options = {};
    loom_verify_result_t verify_result = {};
    IREE_EXPECT_OK(loom_verify_module(module, &verify_options, &verify_result));
    EXPECT_EQ(verify_result.error_count, 0u);
    return ModulePtr(module);
  }

  loom_op_t* FindSymbol(loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    EXPECT_NE(name_id, LOOM_STRING_ID_INVALID);
    if (name_id == LOOM_STRING_ID_INVALID) return nullptr;
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    EXPECT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    if (symbol_id == LOOM_SYMBOL_ID_INVALID) return nullptr;
    return module->symbols.entries[symbol_id].defining_op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(ExactFunctionTest, ReusesContextAndWritesCountTableDirectly) {
  ModulePtr module = ParseAndVerify(R"(
func.def public pure @prefill_counts(%token_count: index) -> (index, index, index, index, index, index) where [range(%token_count, 1, 512)] {
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %row0 = index.add %token_count, %one : index
  %row1 = index.mul %token_count, %two : index
  func.return %row0, %one, %one, %row1, %one, %one : index, index, index, index, index, index
}
)");

  loom_exact_function_t function = {};
  IREE_ASSERT_OK(loom_exact_function_bind(
      module.get(), FindSymbol(module.get(), IREE_SV("prefill_counts")),
      &function));
  EXPECT_TRUE(iree_string_view_equal(function.name, IREE_SV("prefill_counts")));
  EXPECT_EQ(function.argument_count, 1u);
  EXPECT_EQ(function.result_count, 6u);

  loom_exact_function_context_t context = {};
  loom_exact_function_context_initialize(module.get(), &block_pool_, &context);
  std::array<uint32_t, 6> count_table = {};
  const int64_t first_arguments[] = {1};
  IREE_ASSERT_OK(loom_exact_function_evaluate_u32(
      &context, &function, first_arguments, IREE_ARRAYSIZE(first_arguments),
      count_table.data(), count_table.size()));
  EXPECT_EQ(count_table, (std::array<uint32_t, 6>{2, 1, 1, 2, 1, 1}));

  const int64_t second_arguments[] = {127};
  IREE_ASSERT_OK(loom_exact_function_evaluate_u32(
      &context, &function, second_arguments, IREE_ARRAYSIZE(second_arguments),
      count_table.data(), count_table.size()));
  EXPECT_EQ(count_table, (std::array<uint32_t, 6>{128, 1, 1, 254, 1, 1}));
  loom_exact_function_context_deinitialize(&context);
}

TEST_F(ExactFunctionTest, EnforcesInvocationPredicates) {
  ModulePtr module = ParseAndVerify(R"(
func.def pure @bounded(%token_count: index) -> (index) where [range(%token_count, 1, 512), pow2(%token_count)] {
  func.return %token_count : index
}
)");
  loom_exact_function_t function = {};
  IREE_ASSERT_OK(loom_exact_function_bind(
      module.get(), FindSymbol(module.get(), IREE_SV("bounded")), &function));
  loom_exact_function_context_t context = {};
  loom_exact_function_context_initialize(module.get(), &block_pool_, &context);

  int64_t output = 0;
  const int64_t valid_arguments[] = {128};
  IREE_EXPECT_OK(loom_exact_function_evaluate_i64(
      &context, &function, valid_arguments, IREE_ARRAYSIZE(valid_arguments),
      &output, 1));
  EXPECT_EQ(output, 128);

  const int64_t invalid_arguments[] = {127};
  iree::Status status(loom_exact_function_evaluate_i64(
      &context, &function, invalid_arguments, IREE_ARRAYSIZE(invalid_arguments),
      &output, 1));
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  loom_exact_function_context_deinitialize(&context);
}

TEST_F(ExactFunctionTest, RejectsNonEvaluableFunctionAtBind) {
  ModulePtr module = ParseAndVerify(R"(
func.def pure @identity(%value: index) -> (index) {
  func.return %value : index
}

func.def public pure @calls(%value: index) -> (index) {
  %result = func.call pure @identity(%value) : (index) -> (index)
  func.return %result : index
}
)");
  loom_exact_function_t function = {};
  iree::Status status(loom_exact_function_bind(
      module.get(), FindSymbol(module.get(), IREE_SV("calls")), &function));
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kUnimplemented));
}

TEST_F(ExactFunctionTest, RejectsResultOutsideCountDomain) {
  ModulePtr module = ParseAndVerify(R"(
func.def pure @negative() -> (index) {
  %negative = index.constant -1 : index
  func.return %negative : index
}
)");
  loom_exact_function_t function = {};
  IREE_ASSERT_OK(loom_exact_function_bind(
      module.get(), FindSymbol(module.get(), IREE_SV("negative")), &function));
  loom_exact_function_context_t context = {};
  loom_exact_function_context_initialize(module.get(), &block_pool_, &context);
  uint32_t output = 0;
  iree::Status status(loom_exact_function_evaluate_u32(&context, &function,
                                                       nullptr, 0, &output, 1));
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kOutOfRange));
  loom_exact_function_context_deinitialize(&context);
}

}  // namespace
}  // namespace loom
