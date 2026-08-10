// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/schedule.h"

#include <array>
#include <string_view>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class CmdScheduleTest : public ::testing::Test {
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

  ModulePtr Parse(const char* source) {
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("cmd_schedule_test.loom"), &context_,
                                  &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_func_like_t FindProgram(loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return loom_func_like_cast(module,
                               module->symbols.entries[symbol_id].defining_op);
  }

  std::string_view CalleeName(const loom_module_t* module,
                              const loom_op_t* launch_op) {
    const loom_symbol_ref_t callee = loom_kernel_launch_callee(launch_op);
    return std::string_view(
        module->strings
            .entries[module->symbols.entries[callee.symbol_id].name_id]
            .data,
        module->strings
            .entries[module->symbols.entries[callee.symbol_id].name_id]
            .size);
  }

  // Shared arena block pool backing each parsed test module.
  iree_arena_block_pool_t block_pool_;
  // Source dialect context used by the text parser.
  loom_context_t context_;
};

TEST_F(CmdScheduleTest, PlansPrepareThenConcurrentQkv) {
  ModulePtr module = Parse(R"(
kernel.decl @prepare() launch(%parameters: buffer, %input: buffer, %scratch: buffer)
kernel.decl @query() launch(%parameters: buffer, %scratch: buffer, %output: buffer)
kernel.decl @key() launch(%parameters: buffer, %scratch: buffer, %output: buffer)
kernel.decl @value() launch(%parameters: buffer, %scratch: buffer, %output: buffer)

command.program.def @attention() launch(%parameters: buffer, %input: buffer, %scratch: buffer, %query_output: buffer, %key_output: buffer, %value_output: buffer) {
  kernel.launch @prepare[](%parameters, %input, %scratch) : [](buffer, buffer, buffer)
  command.concurrent {
    kernel.launch @query[](%parameters, %scratch, %query_output) : [](buffer, buffer, buffer)
    kernel.launch @key[](%parameters, %scratch, %key_output) : [](buffer, buffer, buffer)
    kernel.launch @value[](%parameters, %scratch, %value_output) : [](buffer, buffer, buffer)
  }
  command.return
}
)");

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  const loom_func_like_t program =
      FindProgram(module.get(), IREE_SV("attention"));
  loom_cmd_schedule_plan_t plan = {};
  IREE_ASSERT_OK(loom_cmd_schedule_plan_build(
      module.get(), loom_func_like_body(program), &arena, &plan));

  ASSERT_EQ(plan.wave_count, 2u);
  ASSERT_EQ(plan.command_count, 4u);
  EXPECT_EQ(plan.waves[0].command_offset, 0u);
  EXPECT_EQ(plan.waves[0].command_count, 1u);
  EXPECT_EQ(plan.waves[1].command_offset, 1u);
  EXPECT_EQ(plan.waves[1].command_count, 3u);
  EXPECT_EQ(CalleeName(module.get(), plan.commands[0]), "prepare");
  EXPECT_EQ(CalleeName(module.get(), plan.commands[1]), "query");
  EXPECT_EQ(CalleeName(module.get(), plan.commands[2]), "key");
  EXPECT_EQ(CalleeName(module.get(), plan.commands[3]), "value");

  iree_arena_deinitialize(&arena);
}

TEST_F(CmdScheduleTest, AlignsNestedSerialSpansWithinConcurrency) {
  ModulePtr module = Parse(R"(
kernel.decl @a() launch()
kernel.decl @b() launch()
kernel.decl @c() launch()
kernel.decl @d() launch()
kernel.decl @e() launch()

command.program.def @nested() launch() {
  command.concurrent {
    command.serial {
      kernel.launch @a[]() : []()
      kernel.launch @b[]() : []()
    }
    command.serial {
      kernel.launch @c[]() : []()
      kernel.launch @d[]() : []()
      kernel.launch @e[]() : []()
    }
  }
  command.return
}
)");

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  const loom_func_like_t program = FindProgram(module.get(), IREE_SV("nested"));
  loom_cmd_schedule_plan_t plan = {};
  IREE_ASSERT_OK(loom_cmd_schedule_plan_build(
      module.get(), loom_func_like_body(program), &arena, &plan));

  ASSERT_EQ(plan.wave_count, 3u);
  ASSERT_EQ(plan.command_count, 5u);
  const std::array<std::string_view, 5> expected = {"a", "c", "b", "d", "e"};
  for (iree_host_size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(CalleeName(module.get(), plan.commands[i]), expected[i]);
  }
  EXPECT_EQ(plan.waves[0].command_count, 2u);
  EXPECT_EQ(plan.waves[1].command_count, 2u);
  EXPECT_EQ(plan.waves[2].command_count, 1u);

  iree_arena_deinitialize(&arena);
}

TEST_F(CmdScheduleTest, PreservesKernelLaunchSchedulingForms) {
  ModulePtr module = Parse(R"(
kernel.decl @a(%workgroup_count: index) launch(%storage: buffer)
kernel.decl @b(%workgroup_count: index) launch(%storage: buffer)

command.program.def @kernel_schedule(%workgroup_count: index) launch(%storage: buffer) {
  %one = index.constant 1 : index
  kernel.launch.serial {
    kernel.launch.concurrent {
      kernel.launch @a[%workgroup_count](%storage) : [index](buffer)
      kernel.launch @b[%workgroup_count](%storage) : [index](buffer)
    }
    kernel.launch @a[%one](%storage) : [index](buffer)
  }
  command.return
}
)");

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  const loom_func_like_t program =
      FindProgram(module.get(), IREE_SV("kernel_schedule"));
  loom_cmd_schedule_plan_t plan = {};
  IREE_ASSERT_OK(loom_cmd_schedule_plan_build(
      module.get(), loom_func_like_body(program), &arena, &plan));

  ASSERT_EQ(plan.wave_count, 2u);
  ASSERT_EQ(plan.command_count, 3u);
  EXPECT_EQ(plan.waves[0].command_count, 2u);
  EXPECT_EQ(plan.waves[1].command_count, 1u);
  EXPECT_EQ(CalleeName(module.get(), plan.commands[0]), "a");
  EXPECT_EQ(CalleeName(module.get(), plan.commands[1]), "b");
  EXPECT_EQ(CalleeName(module.get(), plan.commands[2]), "a");

  iree_arena_deinitialize(&arena);
}

TEST_F(CmdScheduleTest, RecordsAllocationDefinitionWaves) {
  ModulePtr module = Parse(R"(
kernel.decl @use() launch(%storage: buffer)

command.program.def @allocation_waves() launch() {
  %bytes = index.constant 64 : offset
  %early = buffer.alloca %bytes {base_alignment = 64, memory_space = global} : buffer
  kernel.launch @use[](%early) : [](buffer)
  %later = buffer.alloca %bytes {base_alignment = 64, memory_space = global} : buffer
  kernel.launch @use[](%later) : [](buffer)
  command.return
}
)");

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  const loom_func_like_t program =
      FindProgram(module.get(), IREE_SV("allocation_waves"));
  loom_cmd_schedule_plan_t plan = {};
  IREE_ASSERT_OK(loom_cmd_schedule_plan_build(
      module.get(), loom_func_like_body(program), &arena, &plan));

  ASSERT_EQ(plan.allocation_count, 2u);
  EXPECT_TRUE(loom_buffer_alloca_isa(plan.allocations[0].op));
  EXPECT_EQ(plan.allocations[0].definition_wave, 0u);
  EXPECT_TRUE(loom_buffer_alloca_isa(plan.allocations[1].op));
  EXPECT_EQ(plan.allocations[1].definition_wave, 1u);

  iree_arena_deinitialize(&arena);
}

TEST_F(CmdScheduleTest, RejectsResidualCommandOperations) {
  ModulePtr module = Parse(R"(
command.program.def @leaf() launch() {
  command.return
}

command.program.def @residual() launch() {
  command.program.launch @leaf[]() : []()
  command.return
}
)");

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  const loom_func_like_t program =
      FindProgram(module.get(), IREE_SV("residual"));
  loom_cmd_schedule_plan_t plan = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_cmd_schedule_plan_build(module.get(), loom_func_like_body(program),
                                   &arena, &plan));
  EXPECT_EQ(plan.command_count, 0u);
  EXPECT_EQ(plan.wave_count, 0u);

  iree_arena_deinitialize(&arena);
}

}  // namespace
}  // namespace loom
