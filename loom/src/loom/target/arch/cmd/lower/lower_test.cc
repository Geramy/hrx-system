// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/lower.h"

#include <array>
#include <cstdint>
#include <vector>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/cmd/descriptors/low_registry.h"
#include "loom/target/arch/cmd/iree_hal/recording.h"
#include "loom/target/arch/cmd/lower/schedule.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticEmissionCapture;
using ModulePtr = ::loom::testing::ModulePtr;

enum class CapturedCommandKind {
  // Full execution barrier between portable schedule waves.
  kBarrier,
  // Direct kernel dispatch within one portable schedule wave.
  kDispatch,
};

struct CapturedCommand {
  // Kind selecting the populated command payload.
  CapturedCommandKind kind;
  // Executable passed to a captured dispatch.
  iree_hal_executable_t* executable = nullptr;
  // Executable-local function token passed to a captured dispatch.
  iree_hal_executable_function_t function = {};
  // Workgroup counts passed to a captured dispatch.
  iree_hal_dispatch_config_t dispatch_config = {};
  // Buffer references passed to a captured dispatch.
  std::vector<iree_hal_buffer_ref_t> bindings;
};

struct CaptureCommandBuffer {
  // Base HAL command buffer implemented by the capture vtable.
  iree_hal_command_buffer_t base;
  // Number of observed begin operations.
  uint32_t begin_count = 0;
  // Number of observed end operations.
  uint32_t end_count = 0;
  // Recorded barriers and dispatches in invocation order.
  std::vector<CapturedCommand> commands;
};

static CaptureCommandBuffer* CastCommandBuffer(
    iree_hal_command_buffer_t* base_command_buffer) {
  return reinterpret_cast<CaptureCommandBuffer*>(base_command_buffer);
}

static void DestroyCommandBuffer(iree_hal_command_buffer_t*) {}

static iree_status_t BeginCommandBuffer(
    iree_hal_command_buffer_t* base_command_buffer) {
  ++CastCommandBuffer(base_command_buffer)->begin_count;
  return iree_ok_status();
}

static iree_status_t EndCommandBuffer(
    iree_hal_command_buffer_t* base_command_buffer) {
  ++CastCommandBuffer(base_command_buffer)->end_count;
  return iree_ok_status();
}

static iree_status_t CaptureExecutionBarrier(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_execution_stage_t,
    iree_hal_execution_stage_t, iree_hal_execution_barrier_flags_t,
    iree_host_size_t, const iree_hal_memory_barrier_t*, iree_host_size_t,
    const iree_hal_buffer_barrier_t*) {
  CastCommandBuffer(base_command_buffer)
      ->commands.push_back({/*.kind=*/CapturedCommandKind::kBarrier});
  return iree_ok_status();
}

static iree_status_t CaptureDispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  IREE_ASSERT(iree_const_byte_span_is_empty(constants));
  IREE_ASSERT_EQ(flags, IREE_HAL_DISPATCH_FLAG_NONE);
  CapturedCommand command = {
      /*.kind=*/CapturedCommandKind::kDispatch,
      /*.executable=*/executable,
      /*.function=*/function,
      /*.dispatch_config=*/config,
  };
  command.bindings.assign(bindings.values, bindings.values + bindings.count);
  CastCommandBuffer(base_command_buffer)
      ->commands.push_back(std::move(command));
  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t kCaptureCommandBufferVtable = {
    /*.destroy=*/DestroyCommandBuffer,
    /*.begin=*/BeginCommandBuffer,
    /*.end=*/EndCommandBuffer,
    /*.begin_debug_group=*/nullptr,
    /*.end_debug_group=*/nullptr,
    /*.execution_barrier=*/CaptureExecutionBarrier,
    /*.signal_event=*/nullptr,
    /*.reset_event=*/nullptr,
    /*.wait_events=*/nullptr,
    /*.advise_buffer=*/nullptr,
    /*.fill_buffer=*/nullptr,
    /*.update_buffer=*/nullptr,
    /*.copy_buffer=*/nullptr,
    /*.collective=*/nullptr,
    /*.dispatch=*/CaptureDispatch,
};

static void InitializeCommandBuffer(iree_host_size_t binding_count,
                                    CaptureCommandBuffer* command_buffer) {
  iree_hal_command_buffer_initialize(
      /*device_allocator=*/nullptr,
      IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED |
          IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
      IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY, binding_count,
      /*validation_state=*/nullptr, &kCaptureCommandBufferVtable,
      &command_buffer->base);
}

class CmdLowerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_cmd_low_descriptor_registry_initialize(&registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseAndVerifySource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("cmd_lower_test.loom"), &context_,
                                  &block_pool_, &parse_options, &module));
    ModulePtr module_ptr(module);

    loom_verify_options_t verify_options = {};
    verify_options.max_errors = 20;
    loom_verify_result_t result = {};
    IREE_CHECK_OK(loom_verify_module(module, &verify_options, &result));
    IREE_ASSERT_EQ(result.error_count, 0u);
    return module_ptr;
  }

  void VerifyLowModule(loom_module_t* module) {
    DiagnosticEmissionCapture capture;
    loom_low_verify_options_t verify_options = {};
    verify_options.descriptor_registry = &registry_.registry;
    verify_options.emitter = capture.emitter();
    verify_options.max_errors = 20;
    loom_low_verify_scratch_t scratch =
        loom_low_verify_scratch_for_module(module);
    loom_low_verify_result_t result = {};
    IREE_ASSERT_OK(
        loom_low_verify_module(module, &verify_options, &scratch, &result));
    IREE_ASSERT_EQ(result.error_count, 0u);
    IREE_ASSERT(capture.emissions.empty());
  }

  loom_op_t* FindSymbol(loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return module->symbols.entries[symbol_id].defining_op;
  }

  loom_symbol_ref_t FindSymbolRef(loom_module_t* module,
                                  iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return loom_symbol_ref_t{
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
  }

  // Shared arena block pool backing each parsed test module.
  iree_arena_block_pool_t block_pool_;
  // Source dialect context used by the text parser and verifier.
  loom_context_t context_;
  // Portable command descriptor registry used by low verification.
  loom_target_low_descriptor_registry_t registry_ = {};
};

TEST_F(CmdLowerTest, LowersAndRecordsPrepareThenConcurrentQkv) {
  ModulePtr module = ParseAndVerifySource(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.decl @prepare() launch(%parameters: buffer, %input: buffer, %scratch: buffer)
kernel.decl @query() launch(%parameters: buffer, %scratch: buffer, %output: buffer)
kernel.decl @key() launch(%parameters: buffer, %scratch: buffer, %output: buffer)
kernel.decl @value() launch(%parameters: buffer, %scratch: buffer, %output: buffer)

command.program.def public @attention() launch(%parameters: buffer, %input: buffer, %scratch: buffer, %query_output: buffer, %key_output: buffer, %value_output: buffer) {
  kernel.launch @prepare[](%parameters, %input, %scratch) : [](buffer, buffer, buffer)
  command.concurrent {
    kernel.launch @query[](%parameters, %scratch, %query_output) : [](buffer, buffer, buffer)
    kernel.launch @key[](%parameters, %scratch, %key_output) : [](buffer, buffer, buffer)
    kernel.launch @value[](%parameters, %scratch, %value_output) : [](buffer, buffer, buffer)
  }
  command.return
}
)");

  loom_op_t* source_program = FindSymbol(module.get(), IREE_SV("attention"));
  loom_func_like_t source_program_like =
      loom_func_like_cast(module.get(), source_program);
  iree_arena_allocator_t schedule_arena;
  iree_arena_initialize(&block_pool_, &schedule_arena);
  loom_cmd_schedule_plan_t schedule = {};
  IREE_ASSERT_OK(loom_cmd_schedule_plan_build(
      module.get(), loom_func_like_body(source_program_like), &schedule_arena,
      &schedule));
  ASSERT_EQ(schedule.command_count, 4u);

  static constexpr uint64_t kBufferLength = 4096;
  const std::array<loom_cmd_lower_binding_t, 6> binding_plan = {{
      {LOOM_CMD_LOWER_BUFFER_ROLE_FIXED, 0, 0, kBufferLength},
      {LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE, 0, 0, kBufferLength},
      {LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE, 1, 0, kBufferLength},
      {LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE, 2, 0, kBufferLength},
      {LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE, 3, 0, kBufferLength},
      {LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE, 4, 0, kBufferLength},
  }};
  std::array<loom_cmd_lower_direct_launch_t, 4> launch_plan = {{
      {schedule.commands[0], 0, 0, {16, 1, 1}},
      {schedule.commands[1], 1, 1, {32, 1, 1}},
      {schedule.commands[2], 1, 2, {32, 1, 1}},
      {schedule.commands[3], 1, 3, {32, 1, 1}},
  }};
  const loom_cmd_lower_plan_t plan = {
      /*.command_target=*/FindSymbolRef(module.get(),
                                        IREE_SV("command_target")),
      /*.bindings=*/binding_plan.data(),
      /*.binding_count=*/binding_plan.size(),
      /*.fixed_buffer_count=*/1,
      /*.rebindable_binding_count=*/5,
      /*.executable_count=*/2,
      /*.entry_count=*/4,
      /*.launches=*/launch_plan.data(),
      /*.launch_count=*/launch_plan.size(),
  };
  loom_op_t* low_function = nullptr;
  IREE_ASSERT_OK(loom_cmd_lower_program_to_low(module.get(), source_program,
                                               &plan, &low_function));
  iree_arena_deinitialize(&schedule_arena);

  ASSERT_NE(low_function, nullptr);
  EXPECT_TRUE(loom_low_func_def_isa(low_function));
  EXPECT_EQ(loom_low_func_def_abi(low_function),
            LOOM_TARGET_ABI_COMMAND_PROGRAM);
  EXPECT_EQ(FindSymbol(module.get(), IREE_SV("attention")), low_function);
  VerifyLowModule(module.get());

  std::array<iree_hal_executable_function_parameter_t, 3> parameters = {};
  for (uint32_t i = 0; i < parameters.size(); ++i) {
    parameters[i].type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING;
    parameters[i].offset = i;
  }
  std::array<iree_hal_resource_t, 2> executable_storage = {};
  const std::array<iree_hal_executable_t*, 2> executables = {
      reinterpret_cast<iree_hal_executable_t*>(&executable_storage[0]),
      reinterpret_cast<iree_hal_executable_t*>(&executable_storage[1]),
  };
  std::array<loom_cmd_iree_hal_entry_t, 4> entries = {};
  for (uint32_t i = 0; i < entries.size(); ++i) {
    entries[i].executable_index = i == 0 ? 0 : 1;
    entries[i].function = iree_hal_executable_function_from_value(100 + i);
    entries[i].info.binding_count = parameters.size();
    entries[i].info.parameter_count = parameters.size();
    entries[i].parameters = parameters.data();
  }
  iree_hal_resource_t fixed_buffer_storage = {};
  const iree_hal_buffer_ref_t fixed_buffer = iree_hal_make_buffer_ref(
      reinterpret_cast<iree_hal_buffer_t*>(&fixed_buffer_storage), 64,
      kBufferLength);
  const loom_cmd_iree_hal_inputs_t inputs = {
      /*.binding_count=*/5,
      /*.fixed_buffer_count=*/1,
      /*.fixed_buffers=*/&fixed_buffer,
      /*.executable_count=*/executables.size(),
      /*.executables=*/executables.data(),
      /*.entry_count=*/entries.size(),
      /*.entries=*/entries.data(),
  };

  CaptureCommandBuffer command_buffer = {};
  InitializeCommandBuffer(inputs.binding_count, &command_buffer);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(&command_buffer.base));
  IREE_ASSERT_OK(loom_cmd_iree_hal_record_function(
      module.get(), low_function, &inputs, &command_buffer.base,
      iree_allocator_system()));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(&command_buffer.base));

  EXPECT_EQ(command_buffer.begin_count, 1u);
  EXPECT_EQ(command_buffer.end_count, 1u);
  ASSERT_EQ(command_buffer.commands.size(), 5u);
  EXPECT_EQ(command_buffer.commands[1].kind, CapturedCommandKind::kBarrier);
  const std::array<iree_host_size_t, 4> command_indices = {0, 2, 3, 4};
  for (iree_host_size_t i = 0; i < command_indices.size(); ++i) {
    const CapturedCommand& dispatch =
        command_buffer.commands[command_indices[i]];
    ASSERT_EQ(dispatch.kind, CapturedCommandKind::kDispatch);
    EXPECT_EQ(dispatch.executable, executables[i == 0 ? 0 : 1]);
    EXPECT_EQ(dispatch.function.value, 100u + i);
    EXPECT_EQ(dispatch.dispatch_config.workgroup_count[0], i == 0 ? 16u : 32u);
    EXPECT_EQ(dispatch.dispatch_config.workgroup_count[1], 1u);
    EXPECT_EQ(dispatch.dispatch_config.workgroup_count[2], 1u);
    ASSERT_EQ(dispatch.bindings.size(), 3u);
    EXPECT_EQ(dispatch.bindings[0].buffer, fixed_buffer.buffer);
    EXPECT_EQ(dispatch.bindings[0].offset, fixed_buffer.offset);
    EXPECT_EQ(dispatch.bindings[0].length, kBufferLength);
    EXPECT_EQ(dispatch.bindings[1].buffer, nullptr);
    EXPECT_EQ(dispatch.bindings[1].buffer_slot, i == 0 ? 0u : 1u);
    EXPECT_EQ(dispatch.bindings[1].offset, 0u);
    EXPECT_EQ(dispatch.bindings[1].length, kBufferLength);
    EXPECT_EQ(dispatch.bindings[2].buffer, nullptr);
    EXPECT_EQ(dispatch.bindings[2].buffer_slot, i == 0 ? 1u : 1u + i);
    EXPECT_EQ(dispatch.bindings[2].offset, 0u);
    EXPECT_EQ(dispatch.bindings[2].length, kBufferLength);
  }
}

TEST_F(CmdLowerTest, RejectsResidualSourceOperationWithoutMutation) {
  ModulePtr module = ParseAndVerifySource(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

command.program.def @residual() launch() {
  %unit = index.constant 1 : index
  command.return
}
)");

  loom_op_t* source_program = FindSymbol(module.get(), IREE_SV("residual"));
  const loom_cmd_lower_plan_t plan = {
      /*.command_target=*/FindSymbolRef(module.get(),
                                        IREE_SV("command_target")),
  };
  loom_op_t* low_function = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_cmd_lower_program_to_low(module.get(), source_program, &plan,
                                    &low_function));

  EXPECT_EQ(low_function, nullptr);
  EXPECT_EQ(FindSymbol(module.get(), IREE_SV("residual")), source_program);
  EXPECT_FALSE(iree_any_bit_set(source_program->flags, LOOM_OP_FLAG_DEAD));
}

}  // namespace
}  // namespace loom
