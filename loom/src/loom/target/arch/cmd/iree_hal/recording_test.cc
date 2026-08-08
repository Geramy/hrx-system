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

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/cmd/descriptors/low_registry.h"
#include "loom/target/arch/cmd/iree_hal/recording_test_executable.h"
#include "loom/target/arch/cmd/lower/serialize.h"
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

static iree_hal_device_t* CreateSyncDevice() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactor_pool));

  iree_hal_allocator_t* device_allocator = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_create_heap(
      IREE_SV("command-program-test"), iree_allocator_system(),
      iree_allocator_system(), &device_allocator));

  iree_hal_sync_device_params_t sync_params;
  iree_hal_sync_device_params_initialize(&sync_params);
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;

  const iree_hal_executable_library_query_fn_t library_queries[] = {
      loom_cmd_recording_test_executable_query,
  };
  iree_hal_executable_loader_t* executable_loader = nullptr;
  IREE_CHECK_OK(iree_hal_static_library_loader_create(
      IREE_ARRAYSIZE(library_queries), library_queries,
      iree_hal_executable_import_provider_null(), iree_allocator_system(),
      &executable_loader));

  iree_hal_device_t* device = nullptr;
  iree_status_t status = iree_hal_sync_device_create(
      IREE_SV("command-program-test"), &sync_params, &create_params,
      /*loader_count=*/1, &executable_loader, device_allocator,
      iree_allocator_system(), &device);
  iree_hal_executable_loader_release(executable_loader);
  iree_hal_allocator_release(device_allocator);
  iree_async_proactor_pool_release(proactor_pool);
  IREE_CHECK_OK(status);
  return device;
}

static iree_hal_device_group_t* CreateSyncDeviceGroup() {
  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
  iree_async_frontier_tracker_release(frontier_tracker);

  iree_hal_device_t* device = CreateSyncDevice();
  IREE_CHECK_OK(iree_hal_device_group_builder_add_device(&builder, device));

  iree_hal_device_group_t* device_group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &device_group));
  iree_hal_device_release(device);
  return device_group;
}

static iree_hal_buffer_t* CreateTransferBuffer(iree_hal_device_t* device,
                                               iree_device_size_t byte_length) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), params, byte_length, &buffer));
  return buffer;
}

static iree_status_t LoadTestExecutable(
    iree_hal_device_t* device, iree_hal_executable_t** out_executable) {
  *out_executable = nullptr;
  const iree_hal_executable_target_selection_t target_selection = {
      /*.family=*/IREE_SV("cpu"),
      /*.target_key=*/iree_string_view_empty(),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      /*.physical_device_affinity=*/0,
  };
  const iree_hal_executable_target_selection_result_t target_result =
      iree_hal_device_spec_select_executable_target(
          iree_hal_device_spec(device), &target_selection);
  if (target_result.outcome !=
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "local device has no exact CPU target");
  }
  static const char kExecutableName[] = "loom_cmd_recording_test";
  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.executable_data = iree_make_const_byte_span(
      kExecutableName, IREE_ARRAYSIZE(kExecutableName));
  return iree_hal_device_load_executable(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                         target_result.target, &load_params,
                                         out_executable);
}

static iree_status_t SubmitAndWait(
    iree_hal_device_t* device, iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));
  uint64_t target_value = 1;
  const iree_hal_semaphore_list_t signal_semaphores = {
      /*.count=*/1,
      /*.semaphores=*/&semaphore,
      /*.payload_values=*/&target_value,
  };
  iree_status_t status = iree_hal_device_queue_execute(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      signal_semaphores, command_buffer, binding_table,
      IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(semaphore, target_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_semaphore_release(semaphore);
  return status;
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

  iree_byte_span_t program_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      module.get(), FindFunction(module.get(), IREE_SV("packing")),
      /*parameter_requirements=*/nullptr, /*transient_requirement=*/nullptr,
      &program_data, iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(program_data.data, program_data.data_length),
      &program));
  module.reset();

  CaptureCommandBuffer command_buffer = {};
  InitializeCommandBuffer(inputs.binding_count, &command_buffer);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(&command_buffer.base));
  IREE_ASSERT_OK(loom_cmd_iree_hal_record_program(
      &program, &inputs, &command_buffer.base, iree_allocator_system()));
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
  iree_allocator_free(iree_allocator_system(), program_data.data);
}

TEST_F(CmdIreeHalRecordingTest, DispatchesWithReflectedEntryAbi) {
  ModulePtr module = ParseAndVerify(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}
low.func.def target<cmd.core>(@command_target) abi(command_program) @add_u32() {
  %source = low.resource<command_input> {index = 0, source_type = buffer} : reg<cmd.binding>
  %target = low.resource<command_input> {index = 1, source_type = buffer} : reg<cmd.binding>
  %executable = low.resource<command_input> {index = 0, source_type = index} : reg<cmd.executable>
  %entry = low.resource<command_input> {index = 0, source_type = index} : reg<cmd.entry>
  %addend = low.const<cmd.constant.u32> {value = 7} : reg<cmd.u32>
  %workgroup_count = low.const<cmd.constant.u32> {value = 4} : reg<cmd.u32>
  %one = low.const<cmd.constant.u32> {value = 1} : reg<cmd.u32>
  %zero = low.const<cmd.constant.u64> {value = 0} : reg<cmd.u64>
  %length = low.const<cmd.constant.u64> {value = 16} : reg<cmd.u64>
  %source_ref = low.op<cmd.buffer.ref.binding>(%source, %zero, %length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  %target_ref = low.op<cmd.buffer.ref.binding>(%target, %zero, %length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  %args0 = low.op<cmd.arguments.empty>() : () -> reg<cmd.arguments>
  %args1 = low.op<cmd.arguments.append.u32>(%args0, %addend) : (reg<cmd.arguments>, reg<cmd.u32>) -> reg<cmd.arguments>
  %args2 = low.op<cmd.arguments.append.buffer_ref>(%args1, %source_ref) : (reg<cmd.arguments>, reg<cmd.buffer_ref>) -> reg<cmd.arguments>
  %args3 = low.op<cmd.arguments.append.buffer_ref>(%args2, %target_ref) : (reg<cmd.arguments>, reg<cmd.buffer_ref>) -> reg<cmd.arguments>
  low.op<cmd.dispatch.direct>(%executable, %entry, %workgroup_count, %one, %one, %args3) : (reg<cmd.executable>, reg<cmd.entry>, reg<cmd.u32>, reg<cmd.u32>, reg<cmd.u32>, reg<cmd.arguments>)
  low.return
}
)");

  iree_byte_span_t program_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      module.get(), FindFunction(module.get(), IREE_SV("add_u32")),
      /*parameter_requirements=*/nullptr, /*transient_requirement=*/nullptr,
      &program_data, iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(program_data.data, program_data.data_length),
      &program));
  module.reset();

  static constexpr iree_device_size_t kByteLength = 4 * sizeof(uint32_t);
  iree_hal_device_group_t* device_group = CreateSyncDeviceGroup();
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(LoadTestExecutable(device, &executable));
  const iree_hal_executable_function_t function =
      iree_hal_executable_function_from_index(0);
  loom_cmd_iree_hal_entry_t entry = {};
  entry.executable_index = 0;
  entry.function = function;
  IREE_ASSERT_OK(
      iree_hal_executable_function_info(executable, function, &entry.info));
  std::vector<iree_hal_executable_function_parameter_t> parameters(
      entry.info.parameter_count);
  IREE_ASSERT_OK(iree_hal_executable_function_parameters(
      executable, function, parameters.size(), parameters.data()));
  entry.parameters = parameters.data();
  loom_cmd_iree_hal_inputs_t inputs = {};
  inputs.binding_count = 2;
  inputs.executable_count = 1;
  inputs.executables = &executable;
  inputs.entry_count = 1;
  inputs.entries = &entry;

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(loom_cmd_iree_hal_materialize_program(
      &program, &inputs, device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_QUEUE_AFFINITY_ANY, &command_buffer, iree_allocator_system()));
  iree_allocator_free(iree_allocator_system(), program_data.data);
  iree_hal_executable_release(executable);

  const std::array<std::array<uint32_t, 4>, 2> source_values = {{
      {{1, 2, 3, 4}},
      {{10, 20, 30, 40}},
  }};
  for (const std::array<uint32_t, 4>& source_value : source_values) {
    iree_hal_buffer_t* source_buffer =
        CreateTransferBuffer(device, kByteLength);
    iree_hal_buffer_t* target_buffer =
        CreateTransferBuffer(device, kByteLength);
    IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
        device, source_value.data(), source_buffer, 0, kByteLength,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
    const iree_hal_buffer_binding_t bindings[] = {
        /*source=*/{source_buffer, 0, kByteLength},
        /*target=*/{target_buffer, 0, kByteLength},
    };
    IREE_ASSERT_OK(SubmitAndWait(device, command_buffer,
                                 {/*.count=*/IREE_ARRAYSIZE(bindings),
                                  /*.bindings=*/bindings}));

    std::array<uint32_t, 4> actual = {};
    IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
        device, target_buffer, 0, actual.data(), kByteLength,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
    for (iree_host_size_t i = 0; i < actual.size(); ++i) {
      EXPECT_EQ(actual[i], source_value[i] + 7);
    }
    iree_hal_buffer_release(target_buffer);
    iree_hal_buffer_release(source_buffer);
  }

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_device_group_release(device_group);
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

  iree_byte_span_t program_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      module.get(), FindFunction(module.get(), IREE_SV("indirect_modes")),
      /*parameter_requirements=*/nullptr, /*transient_requirement=*/nullptr,
      &program_data, iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(program_data.data, program_data.data_length),
      &program));
  module.reset();

  CaptureCommandBuffer command_buffer = {};
  InitializeCommandBuffer(inputs.binding_count, &command_buffer);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(&command_buffer.base));
  IREE_ASSERT_OK(loom_cmd_iree_hal_record_program(
      &program, &inputs, &command_buffer.base, iree_allocator_system()));
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
  iree_allocator_free(iree_allocator_system(), program_data.data);
}

TEST_F(CmdIreeHalRecordingTest, ReplaysWithDifferentBindingTables) {
  ModulePtr module = ParseAndVerify(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}
low.func.def target<cmd.core>(@command_target) abi(command_program) @rebindable_copy() {
  %fixed = low.resource<command_input> {index = 0, source_type = buffer} : reg<cmd.buffer>
  %source = low.resource<command_input> {index = 0, source_type = buffer} : reg<cmd.binding>
  %target = low.resource<command_input> {index = 1, source_type = buffer} : reg<cmd.binding>
  %pattern = low.const<cmd.constant.u32> {value = 287454020} : reg<cmd.u32>
  %pattern_length = low.const<cmd.constant.u32> {value = 4} : reg<cmd.u32>
  %zero = low.const<cmd.constant.u64> {value = 0} : reg<cmd.u64>
  %length = low.const<cmd.constant.u64> {value = 64} : reg<cmd.u64>
  %fixed_ref = low.op<cmd.buffer.ref.direct>(%fixed, %zero, %length) : (reg<cmd.buffer>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  %source_ref = low.op<cmd.buffer.ref.binding>(%source, %zero, %length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  %target_ref = low.op<cmd.buffer.ref.binding>(%target, %zero, %length) : (reg<cmd.binding>, reg<cmd.u64>, reg<cmd.u64>) -> reg<cmd.buffer_ref>
  low.op<cmd.fill>(%source_ref, %pattern, %pattern_length) : (reg<cmd.buffer_ref>, reg<cmd.u32>, reg<cmd.u32>)
  low.op<cmd.execution.barrier>() : ()
  low.op<cmd.copy>(%source_ref, %fixed_ref) : (reg<cmd.buffer_ref>, reg<cmd.buffer_ref>)
  low.op<cmd.execution.barrier>() : ()
  low.op<cmd.copy>(%fixed_ref, %target_ref) : (reg<cmd.buffer_ref>, reg<cmd.buffer_ref>)
  low.return
}
)");

  iree_byte_span_t program_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      module.get(), FindFunction(module.get(), IREE_SV("rebindable_copy")),
      /*parameter_requirements=*/nullptr, /*transient_requirement=*/nullptr,
      &program_data, iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(program_data.data, program_data.data_length),
      &program));
  module.reset();

  static constexpr iree_device_size_t kByteLength = 64;
  static constexpr uint32_t kPattern = UINT32_C(0x11223344);
  iree_hal_device_group_t* device_group = CreateSyncDeviceGroup();
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_buffer_t* fixed_buffer = CreateTransferBuffer(device, kByteLength);
  const iree_hal_buffer_ref_t fixed_ref =
      iree_hal_make_buffer_ref(fixed_buffer, 0, kByteLength);
  loom_cmd_iree_hal_inputs_t inputs = {};
  inputs.binding_count = 2;
  inputs.fixed_buffer_count = 1;
  inputs.fixed_buffers = &fixed_ref;

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(loom_cmd_iree_hal_materialize_program(
      &program, &inputs, device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_QUEUE_AFFINITY_ANY, &command_buffer, iree_allocator_system()));
  iree_allocator_free(iree_allocator_system(), program_data.data);
  iree_hal_buffer_release(fixed_buffer);

  std::array<iree_hal_buffer_t*, 2> source_buffers = {
      CreateTransferBuffer(device, kByteLength),
      CreateTransferBuffer(device, kByteLength),
  };
  std::array<iree_hal_buffer_t*, 2> target_buffers = {
      CreateTransferBuffer(device, kByteLength),
      CreateTransferBuffer(device, kByteLength),
  };
  for (iree_host_size_t i = 0; i < target_buffers.size(); ++i) {
    const std::array<uint32_t, kByteLength / sizeof(uint32_t)> sentinel = {};
    IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
        device, sentinel.data(), target_buffers[i], 0, kByteLength,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
    const iree_hal_buffer_binding_t bindings[] = {
        /*source=*/{source_buffers[i], 0, kByteLength},
        /*target=*/{target_buffers[i], 0, kByteLength},
    };
    IREE_ASSERT_OK(SubmitAndWait(device, command_buffer,
                                 {/*.count=*/IREE_ARRAYSIZE(bindings),
                                  /*.bindings=*/bindings}));
  }

  for (iree_hal_buffer_t* target_buffer : target_buffers) {
    std::array<uint32_t, kByteLength / sizeof(uint32_t)> actual = {};
    IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
        device, target_buffer, 0, actual.data(), kByteLength,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
    for (uint32_t value : actual) EXPECT_EQ(value, kPattern);
  }

  for (iree_hal_buffer_t* target_buffer : target_buffers) {
    iree_hal_buffer_release(target_buffer);
  }
  for (iree_hal_buffer_t* source_buffer : source_buffers) {
    iree_hal_buffer_release(source_buffer);
  }
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_device_group_release(device_group);
}

}  // namespace
}  // namespace loom
