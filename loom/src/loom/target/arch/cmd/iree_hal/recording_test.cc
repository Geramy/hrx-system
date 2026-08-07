// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/iree_hal/recording.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/cmd/descriptors/low_registry.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticEmissionCapture;
using ModulePtr = ::loom::testing::ModulePtr;

enum class CapturedCommandKind {
  kFill,
  kCopy,
  kBarrier,
  kDispatch,
};

struct CapturedCommand {
  CapturedCommandKind kind;
  iree_hal_buffer_ref_t source_ref = {};
  iree_hal_buffer_ref_t target_ref = {};
  uint32_t fill_pattern = 0;
  iree_host_size_t fill_pattern_length = 0;
  iree_hal_execution_stage_t source_stage_mask = 0;
  iree_hal_execution_stage_t target_stage_mask = 0;
  std::vector<iree_hal_memory_barrier_t> memory_barriers;
  iree_hal_executable_t* executable = nullptr;
  iree_hal_executable_function_t function = {};
  iree_hal_dispatch_config_t dispatch_config = {};
  iree_hal_dispatch_flags_t dispatch_flags = 0;
  std::vector<uint8_t> constants;
  std::vector<iree_hal_buffer_ref_t> bindings;
};

struct CaptureCommandBuffer {
  iree_hal_command_buffer_t base;
  uint32_t begin_count = 0;
  uint32_t end_count = 0;
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
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t, iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers, iree_host_size_t,
    const iree_hal_buffer_barrier_t*) {
  CapturedCommand command;
  command.kind = CapturedCommandKind::kBarrier;
  command.source_stage_mask = source_stage_mask;
  command.target_stage_mask = target_stage_mask;
  command.memory_barriers.assign(memory_barriers,
                                 memory_barriers + memory_barrier_count);
  CastCommandBuffer(base_command_buffer)
      ->commands.push_back(std::move(command));
  return iree_ok_status();
}

static iree_status_t CaptureFill(iree_hal_command_buffer_t* base_command_buffer,
                                 iree_hal_buffer_ref_t target_ref,
                                 const void* pattern,
                                 iree_host_size_t pattern_length,
                                 iree_hal_fill_flags_t) {
  CapturedCommand command;
  command.kind = CapturedCommandKind::kFill;
  command.target_ref = target_ref;
  command.fill_pattern_length = pattern_length;
  std::memcpy(&command.fill_pattern, pattern, pattern_length);
  CastCommandBuffer(base_command_buffer)
      ->commands.push_back(std::move(command));
  return iree_ok_status();
}

static iree_status_t CaptureCopy(iree_hal_command_buffer_t* base_command_buffer,
                                 iree_hal_buffer_ref_t source_ref,
                                 iree_hal_buffer_ref_t target_ref,
                                 iree_hal_copy_flags_t) {
  CapturedCommand command;
  command.kind = CapturedCommandKind::kCopy;
  command.source_ref = source_ref;
  command.target_ref = target_ref;
  CastCommandBuffer(base_command_buffer)
      ->commands.push_back(std::move(command));
  return iree_ok_status();
}

static iree_status_t CaptureDispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  CapturedCommand command;
  command.kind = CapturedCommandKind::kDispatch;
  command.executable = executable;
  command.function = function;
  command.dispatch_config = config;
  command.dispatch_flags = flags;
  command.constants.assign(constants.data,
                           constants.data + constants.data_length);
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
    /*.fill_buffer=*/CaptureFill,
    /*.update_buffer=*/nullptr,
    /*.copy_buffer=*/CaptureCopy,
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

class CmdIreeHalRecordingTest : public ::testing::Test {
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

  ModulePtr ParseAndVerify(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_.registry, &parse_options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("cmd_recording_test.loom"), &context_,
                                  &block_pool_, &parse_options, &module));
    ModulePtr module_ptr(module);

    DiagnosticEmissionCapture capture;
    loom_low_verify_options_t verify_options = {};
    verify_options.descriptor_registry = &registry_.registry;
    verify_options.emitter = capture.emitter();
    verify_options.max_errors = 20;
    loom_low_verify_scratch_t scratch =
        loom_low_verify_scratch_for_module(module);
    loom_low_verify_result_t result = {};
    IREE_CHECK_OK(
        loom_low_verify_module(module, &verify_options, &scratch, &result));
    IREE_ASSERT(result.error_count == 0u);
    IREE_ASSERT(capture.emissions.empty());
    return module_ptr;
  }

  const loom_op_t* FindFunction(const loom_module_t* module,
                                iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return module->symbols.entries[symbol_id].defining_op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
};

TEST_F(CmdIreeHalRecordingTest, RecordsQkvAsTwoExplicitWaves) {
  ModulePtr module = ParseAndVerify(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}
low.func.def target<cmd.core>(@command_target) abi(command_program) @qkv_two_wave() {
  %parameters = low.resource<command_input> {index = 0, source_type = buffer} : reg<cmd.buffer>
  %input = low.resource<command_input> {index = 0, source_type = buffer} : reg<cmd.binding>
  %scratch = low.resource<command_input> {index = 1, source_type = buffer} : reg<cmd.binding>
  %query_output = low.resource<command_input> {index = 2, source_type = buffer} : reg<cmd.binding>
  %key_output = low.resource<command_input> {index = 3, source_type = buffer} : reg<cmd.binding>
  %value_output = low.resource<command_input> {index = 4, source_type = buffer} : reg<cmd.binding>
  %query_executable = low.resource<command_input> {index = 0, source_type = index} : reg<cmd.executable>
  %query_entry = low.resource<command_input> {index = 0, source_type = index} : reg<cmd.entry>
  %key_executable = low.resource<command_input> {index = 1, source_type = index} : reg<cmd.executable>
  %key_entry = low.resource<command_input> {index = 1, source_type = index} : reg<cmd.entry>
  %value_executable = low.resource<command_input> {index = 2, source_type = index} : reg<cmd.executable>
  %value_entry = low.resource<command_input> {index = 2, source_type = index} : reg<cmd.entry>
  %zero_u32 = low.const<cmd.constant.u32> {value = 0} : reg<cmd.u32>
  %one_u32 = low.const<cmd.constant.u32> {value = 1} : reg<cmd.u32>
  %workgroup_count_x = low.const<cmd.constant.u32> {value = 32} : reg<cmd.u32>
  %zero_u64 = low.const<cmd.constant.u64> {value = 0} : reg<cmd.u64>
  %buffer_length = low.const<cmd.constant.u64> {value = 4096} : reg<cmd.u64>
  %parameters_ref = low.op<cmd.buffer.ref.direct>(%parameters, %zero_u64, %buffer_length) : (reg<cmd.buffer>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  %input_ref = low.op<cmd.buffer.ref.binding>(%input, %zero_u64, %buffer_length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  %scratch_ref = low.op<cmd.buffer.ref.binding>(%scratch, %zero_u64, %buffer_length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  %query_output_ref = low.op<cmd.buffer.ref.binding>(%query_output, %zero_u64, %buffer_length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  %key_output_ref = low.op<cmd.buffer.ref.binding>(%key_output, %zero_u64, %buffer_length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  %value_output_ref = low.op<cmd.buffer.ref.binding>(%value_output, %zero_u64, %buffer_length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  low.op<cmd.fill>(%scratch_ref, %zero_u32, %one_u32) : (reg<cmd.buffer_ref>, reg<cmd.u32>, reg<cmd.u32>)
  low.op<cmd.execution.barrier>() : ()
  %query_args0 = low.op<cmd.arguments.empty>() : () -> reg<cmd.arguments>
  %query_args1 = low.op<cmd.arguments.append.buffer_ref>(%query_args0, %parameters_ref) : (reg<cmd.arguments>, reg<cmd.buffer_ref>) -> reg<cmd.arguments>
  %query_args2 = low.op<cmd.arguments.append.buffer_ref>(%query_args1, %input_ref) : (reg<cmd.arguments>, reg<cmd.buffer_ref>) -> reg<cmd.arguments>
  %query_args3 = low.op<cmd.arguments.append.buffer_ref>(%query_args2, %query_output_ref) : (reg<cmd.arguments>, reg<cmd.buffer_ref>) -> reg<cmd.arguments>
  low.op<cmd.dispatch.direct>(%query_executable, %query_entry, %workgroup_count_x, %one_u32, %one_u32, %query_args3) : (reg<cmd.executable>, reg<cmd.entry>, reg<cmd.u32>, reg<cmd.u32>, reg<cmd.u32>, reg<cmd.arguments>)
  %key_args0 = low.op<cmd.arguments.empty>() : () -> reg<cmd.arguments>
  %key_args1 = low.op<cmd.arguments.append.buffer_ref>(%key_args0, %parameters_ref) : (reg<cmd.arguments>, reg<cmd.buffer_ref>) -> reg<cmd.arguments>
  %key_args2 = low.op<cmd.arguments.append.buffer_ref>(%key_args1, %input_ref) : (reg<cmd.arguments>, reg<cmd.buffer_ref>) -> reg<cmd.arguments>
  %key_args3 = low.op<cmd.arguments.append.buffer_ref>(%key_args2, %key_output_ref) : (reg<cmd.arguments>, reg<cmd.buffer_ref>) -> reg<cmd.arguments>
  low.op<cmd.dispatch.direct>(%key_executable, %key_entry, %workgroup_count_x, %one_u32, %one_u32, %key_args3) : (reg<cmd.executable>, reg<cmd.entry>, reg<cmd.u32>, reg<cmd.u32>, reg<cmd.u32>, reg<cmd.arguments>)
  %value_args0 = low.op<cmd.arguments.empty>() : () -> reg<cmd.arguments>
  %value_args1 = low.op<cmd.arguments.append.buffer_ref>(%value_args0, %parameters_ref) : (reg<cmd.arguments>, reg<cmd.buffer_ref>) -> reg<cmd.arguments>
  %value_args2 = low.op<cmd.arguments.append.buffer_ref>(%value_args1, %input_ref) : (reg<cmd.arguments>, reg<cmd.buffer_ref>) -> reg<cmd.arguments>
  %value_args3 = low.op<cmd.arguments.append.buffer_ref>(%value_args2, %value_output_ref) : (reg<cmd.arguments>, reg<cmd.buffer_ref>) -> reg<cmd.arguments>
  low.op<cmd.dispatch.direct>(%value_executable, %value_entry, %workgroup_count_x, %one_u32, %one_u32, %value_args3) : (reg<cmd.executable>, reg<cmd.entry>, reg<cmd.u32>, reg<cmd.u32>, reg<cmd.u32>, reg<cmd.arguments>)
  low.op<cmd.execution.barrier>() : ()
  low.return
}
)");

  std::array<iree_hal_executable_function_parameter_t, 3> parameters = {};
  parameters[0].type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING;
  parameters[0].offset = 0;
  parameters[1].type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING;
  parameters[1].offset = 1;
  parameters[2].type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING;
  parameters[2].offset = 2;
  std::array<iree_hal_resource_t, 3> executable_storage = {};
  std::array<iree_hal_executable_t*, 3> executables = {
      reinterpret_cast<iree_hal_executable_t*>(&executable_storage[0]),
      reinterpret_cast<iree_hal_executable_t*>(&executable_storage[1]),
      reinterpret_cast<iree_hal_executable_t*>(&executable_storage[2]),
  };
  std::array<loom_cmd_iree_hal_entry_t, 3> entries = {};
  for (uint32_t i = 0; i < entries.size(); ++i) {
    entries[i].executable_index = i;
    entries[i].function = iree_hal_executable_function_from_value(100 + i);
    entries[i].info.binding_count = 3;
    entries[i].info.parameter_count = 3;
    entries[i].parameters = parameters.data();
  }
  iree_hal_resource_t fixed_buffer_storage = {};
  const iree_hal_buffer_ref_t fixed_buffer = iree_hal_make_buffer_ref(
      reinterpret_cast<iree_hal_buffer_t*>(&fixed_buffer_storage), 64, 4096);
  loom_cmd_iree_hal_inputs_t inputs = {};
  inputs.binding_count = 5;
  inputs.fixed_buffer_count = 1;
  inputs.fixed_buffers = &fixed_buffer;
  inputs.executable_count = executables.size();
  inputs.executables = executables.data();
  inputs.entry_count = entries.size();
  inputs.entries = entries.data();

  CaptureCommandBuffer command_buffer = {};
  InitializeCommandBuffer(inputs.binding_count, &command_buffer);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(&command_buffer.base));
  IREE_ASSERT_OK(loom_cmd_iree_hal_record_function(
      module.get(), FindFunction(module.get(), IREE_SV("qkv_two_wave")),
      &inputs, &command_buffer.base, iree_allocator_system()));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(&command_buffer.base));

  EXPECT_EQ(command_buffer.begin_count, 1u);
  EXPECT_EQ(command_buffer.end_count, 1u);
  ASSERT_EQ(command_buffer.commands.size(), 6u);
  EXPECT_EQ(command_buffer.commands[0].kind, CapturedCommandKind::kFill);
  EXPECT_EQ(command_buffer.commands[0].target_ref.buffer, nullptr);
  EXPECT_EQ(command_buffer.commands[0].target_ref.buffer_slot, 1u);
  EXPECT_EQ(command_buffer.commands[0].target_ref.length, 4096u);
  EXPECT_EQ(command_buffer.commands[0].fill_pattern, 0u);
  EXPECT_EQ(command_buffer.commands[0].fill_pattern_length, 1u);
  EXPECT_EQ(command_buffer.commands[1].kind, CapturedCommandKind::kBarrier);
  EXPECT_EQ(command_buffer.commands[5].kind, CapturedCommandKind::kBarrier);
  EXPECT_EQ(command_buffer.commands[1].memory_barriers.size(), 1u);

  for (iree_host_size_t i = 0; i < 3; ++i) {
    const CapturedCommand& dispatch = command_buffer.commands[2 + i];
    ASSERT_EQ(dispatch.kind, CapturedCommandKind::kDispatch);
    EXPECT_EQ(dispatch.executable, executables[i]);
    EXPECT_EQ(dispatch.function.value, 100u + i);
    EXPECT_EQ(dispatch.dispatch_config.workgroup_count[0], 32u);
    EXPECT_EQ(dispatch.dispatch_config.workgroup_count[1], 1u);
    EXPECT_EQ(dispatch.dispatch_config.workgroup_count[2], 1u);
    EXPECT_EQ(dispatch.dispatch_flags, IREE_HAL_DISPATCH_FLAG_NONE);
    EXPECT_TRUE(dispatch.constants.empty());
    ASSERT_EQ(dispatch.bindings.size(), 3u);
    EXPECT_EQ(dispatch.bindings[0].buffer, fixed_buffer.buffer);
    EXPECT_EQ(dispatch.bindings[0].offset, 64u);
    EXPECT_EQ(dispatch.bindings[0].length, 4096u);
    EXPECT_EQ(dispatch.bindings[1].buffer, nullptr);
    EXPECT_EQ(dispatch.bindings[1].buffer_slot, 0u);
    EXPECT_EQ(dispatch.bindings[2].buffer, nullptr);
    EXPECT_EQ(dispatch.bindings[2].buffer_slot, 2u + i);
  }
}

TEST_F(CmdIreeHalRecordingTest, PacksLogicalArgumentsByEntryMetadata) {
  ModulePtr module = ParseAndVerify(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}
low.func.def target<cmd.core>(@command_target) abi(command_program) @packing() {
  %buffer = low.resource<command_input> {index = 0, source_type = buffer} : reg<cmd.binding>
  %executable = low.resource<command_input> {index = 0, source_type = index} : reg<cmd.executable>
  %entry = low.resource<command_input> {index = 0, source_type = index} : reg<cmd.entry>
  %u32 = low.const<cmd.constant.u32> {value = 287454020} : reg<cmd.u32>
  %u64 = low.const<cmd.constant.u64> {value = 72623859790382856} : reg<cmd.u64>
  %one = low.const<cmd.constant.u32> {value = 1} : reg<cmd.u32>
  %zero = low.const<cmd.constant.u64> {value = 0} : reg<cmd.u64>
  %length = low.const<cmd.constant.u64> {value = 64} : reg<cmd.u64>
  %buffer_ref = low.op<cmd.buffer.ref.binding>(%buffer, %zero, %length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  %args0 = low.op<cmd.arguments.empty>() : () -> reg<cmd.arguments>
  %args1 = low.op<cmd.arguments.append.u32>(%args0, %u32) : (reg<cmd.arguments>, reg<cmd.u32>) -> reg<cmd.arguments>
  %args2 = low.op<cmd.arguments.append.buffer_ref>(%args1, %buffer_ref) : (reg<cmd.arguments>, reg<cmd.buffer_ref>) -> reg<cmd.arguments>
  %args3 = low.op<cmd.arguments.append.u64>(%args2, %u64) : (reg<cmd.arguments>, reg<cmd.u64>) -> reg<cmd.arguments>
  low.op<cmd.dispatch.direct>(%executable, %entry, %one, %one, %one, %args3) : (reg<cmd.executable>, reg<cmd.entry>, reg<cmd.u32>, reg<cmd.u32>, reg<cmd.u32>, reg<cmd.arguments>)
  low.return
}
)");

  std::array<iree_hal_executable_function_parameter_t, 3> parameters = {};
  parameters[0].type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT;
  parameters[0].size = sizeof(uint32_t);
  parameters[0].offset = 0;
  parameters[1].type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING;
  parameters[1].offset = 0;
  parameters[2].type = IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT;
  parameters[2].size = sizeof(uint64_t);
  parameters[2].offset = 8;
  iree_hal_resource_t executable_storage = {};
  iree_hal_executable_t* executable =
      reinterpret_cast<iree_hal_executable_t*>(&executable_storage);
  loom_cmd_iree_hal_entry_t entry = {};
  entry.executable_index = 0;
  entry.function = iree_hal_executable_function_from_index(0);
  entry.info.constant_byte_length = 16;
  entry.info.binding_count = 1;
  entry.info.parameter_count = 3;
  entry.parameters = parameters.data();
  loom_cmd_iree_hal_inputs_t inputs = {};
  inputs.binding_count = 1;
  inputs.executable_count = 1;
  inputs.executables = &executable;
  inputs.entry_count = 1;
  inputs.entries = &entry;

  CaptureCommandBuffer command_buffer = {};
  InitializeCommandBuffer(inputs.binding_count, &command_buffer);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(&command_buffer.base));
  IREE_ASSERT_OK(loom_cmd_iree_hal_record_function(
      module.get(), FindFunction(module.get(), IREE_SV("packing")), &inputs,
      &command_buffer.base, iree_allocator_system()));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(&command_buffer.base));

  ASSERT_EQ(command_buffer.commands.size(), 1u);
  const CapturedCommand& dispatch = command_buffer.commands[0];
  ASSERT_EQ(dispatch.kind, CapturedCommandKind::kDispatch);
  ASSERT_EQ(dispatch.constants.size(), 16u);
  uint32_t u32 = 0;
  uint64_t u64 = 0;
  std::memcpy(&u32, dispatch.constants.data(), sizeof(u32));
  std::memcpy(&u64, dispatch.constants.data() + 8, sizeof(u64));
  EXPECT_EQ(u32, UINT32_C(0x11223344));
  EXPECT_EQ(u64, UINT64_C(0x0102030405060708));
  EXPECT_EQ(dispatch.constants[4], 0u);
  ASSERT_EQ(dispatch.bindings.size(), 1u);
  EXPECT_EQ(dispatch.bindings[0].buffer, nullptr);
  EXPECT_EQ(dispatch.bindings[0].buffer_slot, 0u);
  EXPECT_EQ(dispatch.bindings[0].length, 64u);
}

TEST_F(CmdIreeHalRecordingTest, PreservesStaticAndDynamicIndirectModes) {
  ModulePtr module = ParseAndVerify(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}
low.func.def target<cmd.core>(@command_target) abi(command_program) @indirect_modes() {
  %launch_counts = low.resource<command_input> {index = 0, source_type = buffer} : reg<cmd.binding>
  %executable = low.resource<command_input> {index = 0, source_type = index} : reg<cmd.executable>
  %entry = low.resource<command_input> {index = 0, source_type = index} : reg<cmd.entry>
  %zero = low.const<cmd.constant.u64> {value = 0} : reg<cmd.u64>
  %count_length = low.const<cmd.constant.u64> {value = 12} : reg<cmd.u64>
  %count_ref = low.op<cmd.buffer.ref.binding>(%launch_counts, %zero, %count_length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  %arguments = low.op<cmd.arguments.empty>() : () -> reg<cmd.arguments>
  low.op<cmd.dispatch.indirect.static>(%executable, %entry, %count_ref, %arguments) : (reg<cmd.executable>, reg<cmd.entry>, reg<cmd.buffer_ref>, reg<cmd.arguments>)
  low.op<cmd.execution.barrier>() : ()
  low.op<cmd.dispatch.indirect.dynamic>(%executable, %entry, %count_ref, %arguments) : (reg<cmd.executable>, reg<cmd.entry>, reg<cmd.buffer_ref>, reg<cmd.arguments>)
  low.return
}
)");

  iree_hal_resource_t executable_storage = {};
  iree_hal_executable_t* executable =
      reinterpret_cast<iree_hal_executable_t*>(&executable_storage);
  loom_cmd_iree_hal_entry_t entry = {};
  entry.executable_index = 0;
  entry.function = iree_hal_executable_function_from_index(0);
  loom_cmd_iree_hal_inputs_t inputs = {};
  inputs.binding_count = 1;
  inputs.executable_count = 1;
  inputs.executables = &executable;
  inputs.entry_count = 1;
  inputs.entries = &entry;

  CaptureCommandBuffer command_buffer = {};
  InitializeCommandBuffer(inputs.binding_count, &command_buffer);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(&command_buffer.base));
  IREE_ASSERT_OK(loom_cmd_iree_hal_record_function(
      module.get(), FindFunction(module.get(), IREE_SV("indirect_modes")),
      &inputs, &command_buffer.base, iree_allocator_system()));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(&command_buffer.base));

  ASSERT_EQ(command_buffer.commands.size(), 3u);
  const CapturedCommand& static_dispatch = command_buffer.commands[0];
  const CapturedCommand& dynamic_dispatch = command_buffer.commands[2];
  ASSERT_EQ(static_dispatch.kind, CapturedCommandKind::kDispatch);
  ASSERT_EQ(command_buffer.commands[1].kind, CapturedCommandKind::kBarrier);
  ASSERT_EQ(dynamic_dispatch.kind, CapturedCommandKind::kDispatch);
  EXPECT_EQ(static_dispatch.dispatch_flags,
            IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS);
  EXPECT_EQ(dynamic_dispatch.dispatch_flags,
            IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS);
  EXPECT_EQ(static_dispatch.dispatch_config.workgroup_count_ref.buffer,
            nullptr);
  EXPECT_EQ(static_dispatch.dispatch_config.workgroup_count_ref.buffer_slot,
            0u);
  EXPECT_EQ(static_dispatch.dispatch_config.workgroup_count_ref.length, 12u);
  EXPECT_EQ(dynamic_dispatch.dispatch_config.workgroup_count_ref.buffer_slot,
            0u);
}

}  // namespace
}  // namespace loom
