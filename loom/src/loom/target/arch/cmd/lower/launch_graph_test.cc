// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/launch_graph.h"

#include <array>
#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/exact_function.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/cmd/lower/launch_artifact.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;
using ::loom::testing::DiagnosticCapture;

class CmdLaunchGraphTest : public ::testing::Test {
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
        iree_make_cstring_view(source), IREE_SV("cmd_launch_graph_test.loom"),
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

  ModulePtr ReadAndVerifyModule(const std::vector<uint8_t>& bytes) {
    loom_bytecode_read_options_t options = {};
    options.verify_module = true;
    loom_bytecode_read_result_t result = {};
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_bytecode_read_module(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("command_launch_config.loombc"), &context_, &block_pool_,
        &options, &result, &module, iree_allocator_system()));
    EXPECT_EQ(result.error_count, 0u);
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  // Shared arena block pool backing source and derived modules.
  iree_arena_block_pool_t block_pool_;
  // Source dialect context used by parsing and launch-graph materialization.
  loom_context_t context_;
};

TEST_F(CmdLaunchGraphTest, SharesDynamicTuplesAndElidesDirectTuples) {
  ModulePtr source_module = ParseAndVerify(R"(
kernel.def @project(%row_count: index) {
  %one = index.constant 1 : index
  %row_groups = index.add %row_count, %one : index
  kernel.launch.config workgroups(%row_groups, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%row_count: index, %storage: buffer) where [range(%row_count, 1, 512)] {
  kernel.return
}

command.program.def @prefill(%token_count: index) launch(%storage: buffer) where [range(%token_count, 1, 512)] {
  %one = index.constant 1 : index
  kernel.launch.serial {
    kernel.launch.concurrent {
      kernel.launch @project[%token_count](%token_count, %storage) : [index](index, buffer)
      kernel.launch @project[%token_count](%token_count, %storage) : [index](index, buffer)
    }
    kernel.launch @project[%one](%one, %storage) : [index](index, buffer)
  }
  command.return
}
)");
  ASSERT_NE(source_module.get(), nullptr);
  const std::vector<uint8_t> source_before =
      WriteCanonicalModule(source_module.get());

  loom_op_t* source_program =
      FindSymbol(source_module.get(), IREE_SV("prefill"));
  const loom_func_like_t source_program_like =
      loom_func_like_cast(source_module.get(), source_program);
  iree_arena_allocator_t schedule_arena;
  iree_arena_initialize(&block_pool_, &schedule_arena);
  loom_cmd_schedule_plan_t schedule = {};
  IREE_ASSERT_OK(loom_cmd_schedule_plan_build(
      source_module.get(), loom_func_like_body(source_program_like),
      &schedule_arena, &schedule));

  loom_cmd_launch_graph_t graph = {};
  IREE_ASSERT_OK(loom_cmd_launch_graph_materialize(
      source_module.get(), source_program, &schedule, &block_pool_,
      iree_allocator_system(), &graph));
  iree_arena_deinitialize(&schedule_arena);

  ASSERT_NE(graph.module, nullptr);
  ASSERT_NE(graph.host_function_op, nullptr);
  ASSERT_NE(graph.launches, nullptr);
  Verify(graph.module);
  EXPECT_EQ(graph.launch_count, 3u);
  ASSERT_EQ(graph.wave_count, 2u);
  EXPECT_EQ(graph.waves[0].command_offset, 0u);
  EXPECT_EQ(graph.waves[0].command_count, 2u);
  EXPECT_EQ(graph.waves[1].command_offset, 2u);
  EXPECT_EQ(graph.waves[1].command_count, 1u);
  EXPECT_EQ(graph.host_tuple_count, 1u);

  EXPECT_EQ(graph.launches[0].kind, LOOM_CMD_LAUNCH_COUNT_KIND_HOST);
  EXPECT_EQ(graph.launches[0].payload.host_tuple_ordinal, 0u);
  EXPECT_EQ(graph.launches[1].kind, LOOM_CMD_LAUNCH_COUNT_KIND_HOST);
  EXPECT_EQ(graph.launches[1].payload.host_tuple_ordinal, 0u);
  EXPECT_EQ(graph.launches[2].kind, LOOM_CMD_LAUNCH_COUNT_KIND_DIRECT);
  EXPECT_EQ(graph.launches[2].payload.direct.x, 2u);
  EXPECT_EQ(graph.launches[2].payload.direct.y, 1u);
  EXPECT_EQ(graph.launches[2].payload.direct.z, 1u);

  ASSERT_TRUE(loom_func_def_isa(graph.host_function_op));
  EXPECT_EQ(loom_func_def_results(graph.host_function_op).count,
            LOOM_CMD_LAUNCH_COUNT_DIMENSION_COUNT);
  const loom_func_like_t host_function =
      loom_func_like_cast(graph.module, graph.host_function_op);
  uint16_t host_argument_count = 0;
  const loom_value_id_t* host_arguments =
      loom_func_like_arg_ids(host_function, &host_argument_count);
  ASSERT_EQ(host_argument_count, 1u);
  EXPECT_TRUE(
      loom_type_equal(loom_module_value_type(graph.module, host_arguments[0]),
                      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX)));
  uint16_t predicate_count = 0;
  EXPECT_NE(loom_func_like_predicates(host_function, &predicate_count),
            nullptr);
  EXPECT_EQ(predicate_count, 1u);
  const loom_block_t* host_body =
      loom_region_entry_block(loom_func_like_body(host_function));
  ASSERT_NE(host_body->last_op, nullptr);
  ASSERT_TRUE(loom_func_return_isa(host_body->last_op));
  EXPECT_EQ(loom_func_return_operands(host_body->last_op).count,
            LOOM_CMD_LAUNCH_COUNT_DIMENSION_COUNT);
  EXPECT_EQ(CountOpKind(graph.host_function_op, LOOM_OP_INDEX_ADD), 1u);
  iree_byte_span_t host_artifact = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_launch_graph_serialize(
      &graph, &block_pool_, iree_allocator_system(), &host_artifact));
  EXPECT_NE(host_artifact.data_length, 0u);
  const std::vector<uint8_t> host_bytecode(
      host_artifact.data, host_artifact.data + host_artifact.data_length);
  iree_allocator_free(iree_allocator_system(), host_artifact.data);

  EXPECT_EQ(WriteCanonicalModule(source_module.get()), source_before);
  loom_cmd_launch_graph_deinitialize(&graph);
  source_module.reset();

  ModulePtr loaded_host_module = ReadAndVerifyModule(host_bytecode);
  ASSERT_NE(loaded_host_module.get(), nullptr);
  loom_exact_function_t loaded_host_function = {};
  IREE_ASSERT_OK(loom_exact_function_bind(
      loaded_host_module.get(),
      FindSymbol(loaded_host_module.get(), IREE_SV("prefill")),
      &loaded_host_function));
  EXPECT_EQ(loaded_host_function.argument_count, 1u);
  EXPECT_EQ(loaded_host_function.result_count, 3u);

  loom_exact_function_context_t evaluation_context = {};
  loom_exact_function_context_initialize(loaded_host_module.get(), &block_pool_,
                                         &evaluation_context);
  std::array<uint32_t, 3> count_table = {};
  const int64_t first_arguments[] = {1};
  IREE_ASSERT_OK(loom_exact_function_evaluate_u32(
      &evaluation_context, &loaded_host_function, first_arguments,
      IREE_ARRAYSIZE(first_arguments), count_table.data(), count_table.size()));
  EXPECT_EQ(count_table, (std::array<uint32_t, 3>{2, 1, 1}));

  const int64_t second_arguments[] = {127};
  IREE_ASSERT_OK(loom_exact_function_evaluate_u32(
      &evaluation_context, &loaded_host_function, second_arguments,
      IREE_ARRAYSIZE(second_arguments), count_table.data(),
      count_table.size()));
  EXPECT_EQ(count_table, (std::array<uint32_t, 3>{128, 1, 1}));
  loom_exact_function_context_deinitialize(&evaluation_context);
}

}  // namespace
}  // namespace loom
