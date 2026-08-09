// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/kernel_unit.h"

#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
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

  loom_op_t* FindLaunchInRegion(loom_region_t* region) {
    if (!region) return nullptr;
    loom_block_t* block = nullptr;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (loom_kernel_launch_isa(op)) return op;
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          if (loom_op_t* launch = FindLaunchInRegion(regions[i])) {
            return launch;
          }
        }
      }
    }
    return nullptr;
  }

  loom_op_t* FindLaunch(loom_func_like_t function) {
    return FindLaunchInRegion(loom_func_like_body(function));
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

  std::vector<uint8_t> WriteCanonicalModule(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    const loom_bytecode_write_options_t options = {
        /*.producer=*/{},
        /*.location_mode=*/LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS,
        /*.low_repr_environment=*/{},
    };
    IREE_CHECK_OK(
        loom_bytecode_write_module(module, stream, &options, &block_pool_));

    const iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
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
} launch(%token_count: index, %column_count: index, %input: buffer, %scratch: buffer, %output: buffer) where [range(%token_count, 1, 1024), range(%column_count, 8, 4096)] {
  %buffer_base = index.constant 0 : offset
  %one = index.constant 1 : index
  %input_view = buffer.view %input[%buffer_base] : buffer -> view<4097xi32, #dense>
  %output_view = buffer.view %output[%buffer_base] : buffer -> view<4097xi32, #dense>
  %value = view.load %input_view[%column_count] : view<4097xi32, #dense> -> i32
  %is_decode = index.cmp eq, %token_count, %one : index
  scf.if %is_decode {
    view.store %value, %output_view[%column_count] : i32, view<4097xi32, #dense>
  } else {
    %scratch_view = buffer.view %scratch[%buffer_base] : buffer -> view<1xi32, #dense>
    view.store %value, %scratch_view[0] : i32, view<1xi32, #dense>
    %scratch_value = view.load %scratch_view[0] : view<1xi32, #dense> -> i32
    view.store %scratch_value, %output_view[%column_count] : i32, view<4097xi32, #dense>
  }
  kernel.return
}

command.program.def @decode(%column_count: index) launch(%input: buffer, %scratch: buffer, %output: buffer) where [range(%column_count, 8, 4096), mul(%column_count, 8), pow2(%column_count)] {
  %one = index.constant 1 : index
  kernel.launch.serial {
    kernel.launch.concurrent {
      kernel.launch @project[%one, %column_count](%one, %column_count, %input, %scratch, %output) : [index, index](index, index, buffer, buffer, buffer)
    }
  }
  command.return
}

command.program.def @prefill(%token_count: index, %column_count: index) launch(%input: buffer, %scratch: buffer, %output: buffer) where [range(%token_count, 1, 1024), pow2(%token_count), range(%column_count, 8, 4096), mul(%column_count, 8), pow2(%column_count)] {
  kernel.launch @project[%token_count, %column_count](%token_count, %column_count, %input, %scratch, %output) : [index, index](index, index, buffer, buffer, buffer)
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
  loom_func_like_t source_prefill_program = loom_func_like_cast(
      source_module.get(), FindSymbol(source_module.get(), IREE_SV("prefill")));
  loom_op_t* source_launch = FindLaunch(source_program);
  loom_op_t* source_prefill_launch = FindLaunch(source_prefill_program);
  ASSERT_NE(source_launch, nullptr);
  ASSERT_NE(source_prefill_launch, nullptr);

  iree_arena_allocator_t fact_arena;
  iree_arena_initialize(&block_pool_, &fact_arena);
  loom_value_fact_table_t source_facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&source_facts, &fact_arena,
                                                  source_module->values.count));
  loom_type_registry_configure_fact_context(&source_facts.context);
  IREE_ASSERT_OK(loom_value_fact_table_compute(
      &source_facts, source_module.get(), source_program));
  loom_value_fact_table_t source_prefill_facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(
      &source_prefill_facts, &fact_arena, source_module->values.count));
  loom_type_registry_configure_fact_context(&source_prefill_facts.context);
  IREE_ASSERT_OK(loom_value_fact_table_compute(
      &source_prefill_facts, source_module.get(), source_prefill_program));

  loom_cmd_kernel_unit_t unit = {};
  loom_cmd_kernel_unit_t prefill_unit = {};
  IREE_ASSERT_OK(loom_cmd_kernel_unit_materialize(
      source_module.get(), source_launch, &source_facts, &block_pool_,
      iree_allocator_system(), &unit));
  IREE_ASSERT_OK(loom_cmd_kernel_unit_materialize(
      source_module.get(), source_prefill_launch, &source_prefill_facts,
      &block_pool_, iree_allocator_system(), &prefill_unit));
  iree_arena_deinitialize(&fact_arena);

  ASSERT_NE(unit.module, nullptr);
  ASSERT_NE(unit.kernel_op, nullptr);
  Verify(unit.module);
  EXPECT_NE(unit.module, source_module.get());
  EXPECT_NE(unit.kernel_op, source_kernel);

  EXPECT_EQ(unit.source_workload_count, 2u);
  ASSERT_EQ(unit.workload_count, 2u);
  ASSERT_NE(unit.source_workload_ordinals, nullptr);
  EXPECT_EQ(unit.source_workload_ordinals[0], 0u);
  EXPECT_EQ(unit.source_workload_ordinals[1], 1u);
  EXPECT_EQ(unit.source_argument_count, 5u);
  ASSERT_EQ(unit.argument_count, 3u);
  ASSERT_NE(unit.source_argument_ordinals, nullptr);
  EXPECT_EQ(unit.source_argument_ordinals[0], 1u);
  EXPECT_EQ(unit.source_argument_ordinals[1], 2u);
  EXPECT_EQ(unit.source_argument_ordinals[2], 4u);
  uint16_t unit_predicate_count = 0;
  loom_func_like_predicates(loom_func_like_cast(unit.module, unit.kernel_op),
                            &unit_predicate_count);
  EXPECT_EQ(unit_predicate_count, 1u);
  EXPECT_EQ(CountOpKind(unit.kernel_op, LOOM_OP_SCF_IF), 0u);
  EXPECT_EQ(CountOpKind(unit.kernel_op, LOOM_OP_BUFFER_VIEW), 2u);
  EXPECT_EQ(CountOpKind(unit.kernel_op, LOOM_OP_INDEX_ASSUME), 1u);
  EXPECT_EQ(CountOpKind(unit.kernel_op, LOOM_OP_VIEW_LOAD), 1u);
  EXPECT_EQ(CountOpKind(unit.kernel_op, LOOM_OP_VIEW_STORE), 1u);
  ASSERT_NE(prefill_unit.module, nullptr);
  ASSERT_NE(prefill_unit.kernel_op, nullptr);
  Verify(prefill_unit.module);
  EXPECT_EQ(prefill_unit.workload_count, 2u);
  EXPECT_EQ(prefill_unit.argument_count, 5u);
  EXPECT_EQ(CountOpKind(prefill_unit.kernel_op, LOOM_OP_INDEX_ASSUME), 2u);
  EXPECT_EQ(CountOpKind(prefill_unit.kernel_op, LOOM_OP_SCF_IF), 1u);
  EXPECT_NE(WriteCanonicalModule(unit.module),
            WriteCanonicalModule(prefill_unit.module));

  iree_arena_allocator_t unit_fact_arena;
  iree_arena_initialize(&block_pool_, &unit_fact_arena);
  loom_value_fact_table_t unit_facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&unit_facts, &unit_fact_arena,
                                                  unit.module->values.count));
  loom_type_registry_configure_fact_context(&unit_facts.context);
  loom_func_like_t unit_kernel =
      loom_func_like_cast(unit.module, unit.kernel_op);
  IREE_ASSERT_OK(
      loom_value_fact_table_compute(&unit_facts, unit.module, unit_kernel));
  loom_block_t* body_entry =
      loom_region_entry_block(loom_func_like_body(unit_kernel));
  loom_op_t* assume_op = body_entry->first_op;
  ASSERT_TRUE(loom_index_assume_isa(assume_op));
  const loom_value_facts_t column_count_facts = loom_value_fact_table_lookup(
      &unit_facts, loom_index_assume_results(assume_op).values[0]);
  EXPECT_EQ(column_count_facts.range_lo, 8);
  EXPECT_EQ(column_count_facts.range_hi, 4096);
  EXPECT_EQ(column_count_facts.known_divisor, 8);
  EXPECT_TRUE(loom_value_facts_is_power_of_two(column_count_facts));
  iree_arena_deinitialize(&unit_fact_arena);

  uint16_t source_argument_count = 0;
  loom_func_like_arg_ids(source_kernel_like, &source_argument_count);
  EXPECT_EQ(source_argument_count, 5u);
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

  loom_cmd_kernel_unit_deinitialize(&prefill_unit);
  loom_cmd_kernel_unit_deinitialize(&unit);
  EXPECT_EQ(unit.module, nullptr);
}

TEST_F(CmdKernelUnitTest, RewritesBodyPredicatesUsingExactArguments) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.def @scatter(%token_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%token_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%token_count: index, %row: index, %output: buffer) where [range(%token_count, 1, 1024), range(%row, 0, 1023)] {
  %bounded_row, %bounded_token_count = index.assume %row, %token_count [lt(%row, %token_count)] : index, index
  %zero = index.constant 0 : offset
  %output_view = buffer.view %output[%zero] : buffer -> view<1024xi32, #dense>
  %value = view.load %output_view[%bounded_row] : view<1024xi32, #dense> -> i32
  view.store %value, %output_view[%bounded_row] : i32, view<1024xi32, #dense>
  kernel.return
}

command.program.def @root(%row: index) launch(%output: buffer) where [range(%row, 0, 1023)] {
  %token_count = index.constant 512 : index
  kernel.launch @scatter[%token_count](%token_count, %row, %output) : [index](index, index, buffer)
  command.return
}
)");
  ASSERT_NE(source_module.get(), nullptr);

  loom_func_like_t source_program = loom_func_like_cast(
      source_module.get(), FindSymbol(source_module.get(), IREE_SV("root")));
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
  EXPECT_EQ(unit.source_argument_count, 3u);
  ASSERT_EQ(unit.argument_count, 2u);
  ASSERT_NE(unit.source_argument_ordinals, nullptr);
  EXPECT_EQ(unit.source_argument_ordinals[0], 1u);
  EXPECT_EQ(unit.source_argument_ordinals[1], 2u);

  loom_func_like_t unit_kernel =
      loom_func_like_cast(unit.module, unit.kernel_op);
  uint16_t unit_argument_count = 0;
  const loom_value_id_t* unit_arguments =
      loom_func_like_arg_ids(unit_kernel, &unit_argument_count);
  ASSERT_EQ(unit_argument_count, 2u);
  EXPECT_TRUE(loom_module_value_has_predicate_attribute_uses(
      unit.module, unit_arguments[0]));
  EXPECT_EQ(CountOpKind(unit.kernel_op, LOOM_OP_INDEX_ASSUME), 2u);

  loom_cmd_kernel_unit_deinitialize(&unit);
}

TEST_F(CmdKernelUnitTest, MaterializesTypedViewsAtNativeAbiBoundary) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.def @copy(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: view<1xi32, #dense>, %target: buffer) {
  %zero = index.constant 0 : offset
  %target_view = buffer.view %target[%zero] : buffer -> view<1xi32, #dense>
  %value = view.load %source[0] : view<1xi32, #dense> -> i32
  view.store %value, %target_view[0] : i32, view<1xi32, #dense>
  kernel.return
}

command.program.def @decode() launch(%source: buffer, %target: buffer) {
  %zero = index.constant 0 : offset
  %one = index.constant 1 : index
  %source_view = buffer.view %source[%zero] : buffer -> view<1xi32, #dense>
  kernel.launch @copy[%one](%source_view, %target) : [index](view<1xi32, #dense>, buffer)
  command.return
}
)");
  ASSERT_NE(source_module.get(), nullptr);

  loom_op_t* source_kernel = FindSymbol(source_module.get(), IREE_SV("copy"));
  loom_func_like_t source_kernel_like =
      loom_func_like_cast(source_module.get(), source_kernel);
  loom_func_like_t source_program = loom_func_like_cast(
      source_module.get(), FindSymbol(source_module.get(), IREE_SV("decode")));
  loom_op_t* source_launch = FindLaunch(source_program);
  ASSERT_NE(source_launch, nullptr);

  uint16_t source_argument_count = 0;
  const loom_value_id_t* source_arguments =
      loom_func_like_arg_ids(source_kernel_like, &source_argument_count);
  ASSERT_EQ(source_argument_count, 2u);
  EXPECT_TRUE(loom_type_is_view(
      loom_module_value_type(source_module.get(), source_arguments[0])));

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
  loom_func_like_t unit_kernel =
      loom_func_like_cast(unit.module, unit.kernel_op);
  uint16_t unit_argument_count = 0;
  const loom_value_id_t* unit_arguments =
      loom_func_like_arg_ids(unit_kernel, &unit_argument_count);
  ASSERT_EQ(unit_argument_count, 2u);
  EXPECT_TRUE(loom_type_is_buffer(
      loom_module_value_type(unit.module, unit_arguments[0])));
  EXPECT_TRUE(loom_type_is_buffer(
      loom_module_value_type(unit.module, unit_arguments[1])));
  EXPECT_EQ(CountOpKind(unit.kernel_op, LOOM_OP_BUFFER_VIEW), 2u);

  source_arguments =
      loom_func_like_arg_ids(source_kernel_like, &source_argument_count);
  ASSERT_EQ(source_argument_count, 2u);
  EXPECT_TRUE(loom_type_is_view(
      loom_module_value_type(source_module.get(), source_arguments[0])));
  EXPECT_EQ(CountOpKind(source_kernel, LOOM_OP_BUFFER_VIEW), 1u);

  loom_cmd_kernel_unit_deinitialize(&unit);
}

TEST_F(CmdKernelUnitTest, WorkloadOnlyFactsConvergeToIdenticalKernelUnits) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.def @normalize(%token_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%token_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%input: buffer, %output: buffer) {
  %buffer_base = index.constant 0 : offset
  %input_view = buffer.view %input[%buffer_base] : buffer -> view<1xi32, #dense>
  %output_view = buffer.view %output[%buffer_base] : buffer -> view<1xi32, #dense>
  %value = view.load %input_view[0] : view<1xi32, #dense> -> i32
  view.store %value, %output_view[0] : i32, view<1xi32, #dense>
  kernel.return
}

command.program.def @decode() launch(%input: buffer, %output: buffer) {
  %one = index.constant 1 : index
  kernel.launch @normalize[%one](%input, %output) : [index](buffer, buffer)
  command.return
}

command.program.def @prefill(%token_count: index) launch(%input: buffer, %output: buffer) where [range(%token_count, 1, 1024), pow2(%token_count)] {
  kernel.launch @normalize[%token_count](%input, %output) : [index](buffer, buffer)
  command.return
}
)");
  ASSERT_NE(source_module.get(), nullptr);

  loom_func_like_t decode_program = loom_func_like_cast(
      source_module.get(), FindSymbol(source_module.get(), IREE_SV("decode")));
  loom_func_like_t prefill_program = loom_func_like_cast(
      source_module.get(), FindSymbol(source_module.get(), IREE_SV("prefill")));
  loom_op_t* decode_launch = FindLaunch(decode_program);
  loom_op_t* prefill_launch = FindLaunch(prefill_program);
  ASSERT_NE(decode_launch, nullptr);
  ASSERT_NE(prefill_launch, nullptr);

  iree_arena_allocator_t fact_arena;
  iree_arena_initialize(&block_pool_, &fact_arena);
  loom_value_fact_table_t decode_facts = {};
  loom_value_fact_table_t prefill_facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&decode_facts, &fact_arena,
                                                  source_module->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&prefill_facts, &fact_arena,
                                                  source_module->values.count));
  loom_type_registry_configure_fact_context(&decode_facts.context);
  loom_type_registry_configure_fact_context(&prefill_facts.context);
  IREE_ASSERT_OK(loom_value_fact_table_compute(
      &decode_facts, source_module.get(), decode_program));
  IREE_ASSERT_OK(loom_value_fact_table_compute(
      &prefill_facts, source_module.get(), prefill_program));

  const loom_value_slice_t decode_workloads =
      loom_kernel_launch_workloads(decode_launch);
  const loom_value_slice_t prefill_workloads =
      loom_kernel_launch_workloads(prefill_launch);
  ASSERT_EQ(decode_workloads.count, 1u);
  ASSERT_EQ(prefill_workloads.count, 1u);
  EXPECT_TRUE(loom_value_facts_is_exact(
      loom_value_fact_table_lookup(&decode_facts, decode_workloads.values[0])));
  const loom_value_facts_t prefill_token_count_facts =
      loom_value_fact_table_lookup(&prefill_facts, prefill_workloads.values[0]);
  EXPECT_FALSE(loom_value_facts_is_exact(prefill_token_count_facts));
  EXPECT_EQ(prefill_token_count_facts.range_lo, 1);
  EXPECT_EQ(prefill_token_count_facts.range_hi, 1024);
  EXPECT_TRUE(loom_value_facts_is_power_of_two(prefill_token_count_facts));

  loom_cmd_kernel_unit_t decode_unit = {};
  loom_cmd_kernel_unit_t prefill_unit = {};
  IREE_ASSERT_OK(loom_cmd_kernel_unit_materialize(
      source_module.get(), decode_launch, &decode_facts, &block_pool_,
      iree_allocator_system(), &decode_unit));
  IREE_ASSERT_OK(loom_cmd_kernel_unit_materialize(
      source_module.get(), prefill_launch, &prefill_facts, &block_pool_,
      iree_allocator_system(), &prefill_unit));
  iree_arena_deinitialize(&fact_arena);

  ASSERT_NE(decode_unit.module, nullptr);
  ASSERT_NE(prefill_unit.module, nullptr);
  Verify(decode_unit.module);
  Verify(prefill_unit.module);
  EXPECT_EQ(decode_unit.workload_count, 1u);
  EXPECT_EQ(prefill_unit.workload_count, 1u);
  EXPECT_EQ(WriteCanonicalModule(decode_unit.module),
            WriteCanonicalModule(prefill_unit.module));

  loom_cmd_kernel_unit_deinitialize(&prefill_unit);
  loom_cmd_kernel_unit_deinitialize(&decode_unit);
}

}  // namespace
}  // namespace loom
