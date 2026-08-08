// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/type_registry.h"
#include "loom/target/arch/cmd/descriptors/low_registry.h"
#include "loom/target/arch/cmd/iree_hal/recording.h"
#include "loom/target/arch/cmd/iree_hal/recording_test_executable.h"
#include "loom/target/arch/cmd/lower/kernel_unit.h"
#include "loom/target/arch/cmd/lower/launch_artifact.h"
#include "loom/target/arch/cmd/lower/lower.h"
#include "loom/target/arch/cmd/lower/schedule.h"
#include "loom/target/arch/cmd/lower/serialize.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"
#include "loomc/launch_config_module.h"
#include "test/util.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticEmissionCapture;
using InternalModulePtr = ::loom::testing::ModulePtr;
using LaunchModulePtr =
    loomc::testing::HandlePtr<loomc_launch_config_module_t,
                              loomc_launch_config_module_release>;
using LaunchContextPtr =
    loomc::testing::HandlePtr<loomc_launch_config_context_t,
                              loomc_launch_config_context_release>;

struct CommandArtifacts {
  // Evaluation-ready launch-configuration Loombc.
  std::vector<uint8_t> launch_config;
  // Closed portable command-program bytes.
  std::vector<uint8_t> command_program;
};

static iree_hal_device_t* CreateSyncDevice() {
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      1, /*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
      iree_allocator_system(), &proactor_pool));

  iree_hal_allocator_t* device_allocator = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_create_heap(
      IREE_SV("command-lifecycle-test"), iree_allocator_system(),
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
      IREE_SV("command-lifecycle-test"), &sync_params, &create_params,
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

static iree_hal_buffer_t* CreateLaunchCountBuffer(
    iree_hal_device_t* device, iree_device_size_t byte_length) {
  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
                 IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), params, byte_length, &buffer));
  return buffer;
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

class CommandLifecycleTest : public ::testing::Test {
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

  InternalModulePtr ParseAndVerifySource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("command_lifecycle_test.loom"),
        &context_, &block_pool_, &parse_options, &module));
    InternalModulePtr module_ptr(module);

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

  CommandArtifacts CompileArtifacts() {
    InternalModulePtr module = ParseAndVerifySource(R"(
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
  %incremented = scalar.addi %value, %one : i32
  view.store %incremented, %target_view[%workgroup] : i32, view<128xi32, #dense>
  kernel.return
}

command.program.def public @increment_elements(%element_count: index) launch(%source: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  kernel.launch @increment[%element_count](%source, %target) : [index](buffer, buffer)
  command.return
}
)");

    loom_op_t* source_program =
        FindSymbol(module.get(), IREE_SV("increment_elements"));
    const loom_func_like_t source_program_like =
        loom_func_like_cast(module.get(), source_program);
    iree_arena_allocator_t schedule_arena;
    iree_arena_initialize(&block_pool_, &schedule_arena);
    loom_cmd_schedule_plan_t schedule = {};
    IREE_CHECK_OK(loom_cmd_schedule_plan_build(
        module.get(), loom_func_like_body(source_program_like), &schedule_arena,
        &schedule));

    loom_value_fact_table_t source_facts = {};
    IREE_CHECK_OK(loom_value_fact_table_initialize(
        &source_facts, &schedule_arena, module->values.count));
    loom_type_registry_configure_fact_context(&source_facts.context);
    IREE_CHECK_OK(loom_value_fact_table_compute(&source_facts, module.get(),
                                                source_program_like));
    loom_cmd_kernel_unit_t kernel_unit = {};
    IREE_CHECK_OK(loom_cmd_kernel_unit_materialize(
        module.get(), schedule.commands[0], &source_facts, &block_pool_,
        iree_allocator_system(), &kernel_unit));
    IREE_ASSERT_EQ(kernel_unit.argument_count, 2u);

    loom_cmd_launch_graph_t launch_graph = {};
    IREE_CHECK_OK(loom_cmd_launch_graph_materialize(
        module.get(), source_program, &schedule, &block_pool_,
        iree_allocator_system(), &launch_graph));
    iree_arena_deinitialize(&schedule_arena);
    IREE_ASSERT_EQ(launch_graph.host_tuple_count, 1u);

    iree_byte_span_t launch_config_data = iree_byte_span_empty();
    IREE_CHECK_OK(loom_cmd_launch_program_serialize(
        launch_graph.module, &block_pool_, iree_allocator_system(),
        &launch_config_data));

    static constexpr uint64_t kBufferByteLength = 128 * sizeof(uint32_t);
    const std::array<loom_cmd_lower_binding_t, 2> binding_plan = {{
        {LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE, 0, 0, kBufferByteLength},
        {LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE, 1, 0, kBufferByteLength},
    }};
    const loom_cmd_lower_launch_t launch_plan = {
        /*.executable_index=*/0,
        /*.entry_index=*/0,
        /*.argument_count=*/kernel_unit.argument_count,
        /*.source_argument_ordinals=*/kernel_unit.source_argument_ordinals,
    };
    const loom_cmd_lower_plan_t plan = {
        /*.command_target=*/FindSymbolRef(module.get(),
                                          IREE_SV("command_target")),
        /*.bindings=*/binding_plan.data(),
        /*.binding_count=*/binding_plan.size(),
        /*.fixed_buffer_count=*/0,
        /*.rebindable_binding_count=*/3,
        /*.executable_count=*/1,
        /*.entry_count=*/1,
        /*.launch_graph=*/&launch_graph,
        /*.launch_count_binding=*/{2, 0},
        /*.launches=*/&launch_plan,
    };
    loom_op_t* low_function = nullptr;
    IREE_CHECK_OK(loom_cmd_lower_program_to_low(module.get(), source_program,
                                                &plan, &low_function));
    loom_cmd_kernel_unit_deinitialize(&kernel_unit);
    loom_cmd_launch_graph_deinitialize(&launch_graph);
    VerifyLowModule(module.get());

    iree_byte_span_t command_program_data = iree_byte_span_empty();
    IREE_CHECK_OK(loom_cmd_program_serialize_low(module.get(), low_function,
                                                 &command_program_data,
                                                 iree_allocator_system()));

    CommandArtifacts artifacts;
    artifacts.launch_config.assign(
        launch_config_data.data,
        launch_config_data.data + launch_config_data.data_length);
    artifacts.command_program.assign(
        command_program_data.data,
        command_program_data.data + command_program_data.data_length);
    iree_allocator_free(iree_allocator_system(), launch_config_data.data);
    iree_allocator_free(iree_allocator_system(), command_program_data.data);
    module.reset();
    return artifacts;
  }

  // Shared arena block pool backing compiler-owned modules.
  iree_arena_block_pool_t block_pool_;
  // Source dialect context used by parsing, verification, and lowering.
  loom_context_t context_;
  // Portable command descriptor registry used by low verification.
  loom_target_low_descriptor_registry_t registry_ = {};
};

TEST_F(CommandLifecycleTest, ReplaysOneCommandBufferWithEvaluatedWorkloads) {
  CommandArtifacts artifacts = CompileArtifacts();
  ASSERT_FALSE(artifacts.launch_config.empty());
  ASSERT_FALSE(artifacts.command_program.empty());

  const loomc_artifact_t launch_artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      /*.format=*/loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
      /*.identifier=*/loomc_make_cstring_view("increment_elements.loombc"),
      /*.contents=*/
      loomc_make_byte_span(artifacts.launch_config.data(),
                           artifacts.launch_config.size()),
  };
  loomc_launch_config_module_t* raw_launch_module = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_module_load(
      &launch_artifact, /*options=*/nullptr, loomc_allocator_system(),
      &raw_launch_module));
  LaunchModulePtr launch_module(raw_launch_module);
  artifacts.launch_config.clear();
  loomc_launch_config_function_t launch_function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_module_lookup_function_by_name(
      launch_module.get(), loomc_make_cstring_view("increment_elements"),
      &launch_function));
  loomc_launch_config_function_info_t launch_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_FUNCTION_INFO,
      /*.structure_size=*/sizeof(launch_info),
  };
  LOOMC_ASSERT_OK(loomc_launch_config_module_function_info(
      launch_module.get(), launch_function, &launch_info));
  ASSERT_EQ(launch_info.workload_argument_count, 1u);
  ASSERT_EQ(launch_info.result_count, 1u);
  ASSERT_EQ(launch_info.output_byte_length,
            LOOM_CMD_LAUNCH_COUNT_TUPLE_BYTE_LENGTH);
  loomc_launch_config_context_t* raw_launch_context = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_context_create(
      launch_module.get(), /*options=*/nullptr, loomc_allocator_system(),
      &raw_launch_context));
  LaunchContextPtr launch_context(raw_launch_context);

  loom_cmd_program_t command_program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(artifacts.command_program.data(),
                                artifacts.command_program.size()),
      &command_program));
  ASSERT_EQ(command_program.requirements.fixed_buffer_count, 0u);
  ASSERT_EQ(command_program.requirements.rebindable_binding_count, 3u);
  ASSERT_EQ(command_program.requirements.executable_count, 1u);
  ASSERT_EQ(command_program.requirements.entry_count, 1u);
  ASSERT_EQ(command_program.commands.count, 1u);
  EXPECT_EQ(loom_cmd_program_command_at(&command_program, 0).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);

  iree_hal_device_group_t* device_group = CreateSyncDeviceGroup();
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(LoadTestExecutable(device, &executable));
  const iree_hal_executable_function_t function =
      iree_hal_executable_function_from_index(1);
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
  const loom_cmd_iree_hal_inputs_t inputs = {
      /*.binding_count=*/3,
      /*.fixed_buffer_count=*/0,
      /*.fixed_buffers=*/nullptr,
      /*.executable_count=*/1,
      /*.executables=*/&executable,
      /*.entry_count=*/1,
      /*.entries=*/&entry,
  };
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(loom_cmd_iree_hal_materialize_program(
      &command_program, &inputs, device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_QUEUE_AFFINITY_ANY, &command_buffer, iree_allocator_system()));
  artifacts.command_program.clear();
  iree_hal_executable_release(executable);

  static constexpr iree_host_size_t kElementCount = 128;
  static constexpr iree_device_size_t kBufferByteLength =
      kElementCount * sizeof(uint32_t);
  static constexpr iree_device_size_t kLaunchCountByteLength =
      LOOM_CMD_LAUNCH_COUNT_TUPLE_BYTE_LENGTH;
  iree_hal_buffer_t* source_buffer =
      CreateTransferBuffer(device, kBufferByteLength);
  iree_hal_buffer_t* target_buffer =
      CreateTransferBuffer(device, kBufferByteLength);
  iree_hal_buffer_t* launch_count_buffer =
      CreateLaunchCountBuffer(device, kLaunchCountByteLength);
  iree_hal_buffer_mapping_t launch_count_mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      launch_count_buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_WRITE, /*byte_offset=*/0, kLaunchCountByteLength,
      &launch_count_mapping));

  std::array<uint32_t, kElementCount> source_values = {};
  for (iree_host_size_t i = 0; i < source_values.size(); ++i) {
    source_values[i] = 1000u + static_cast<uint32_t>(i);
  }
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device, source_values.data(), source_buffer, 0, kBufferByteLength,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  const std::array<int64_t, 2> workloads = {1, 127};
  for (int64_t workload : workloads) {
    const std::array<uint32_t, kElementCount> zero_values = {};
    IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
        device, zero_values.data(), target_buffer, 0, kBufferByteLength,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

    const loomc_launch_config_arguments_t arguments = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_ARGUMENTS,
        /*.structure_size=*/sizeof(arguments),
        /*.next=*/nullptr,
        /*.workload_arguments=*/&workload,
        /*.workload_argument_count=*/1,
    };
    loomc_launch_config_outputs_t outputs = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_OUTPUTS,
        /*.structure_size=*/sizeof(outputs),
        /*.next=*/nullptr,
        /*.storage=*/launch_count_mapping.contents.data,
        /*.storage_length=*/launch_count_mapping.contents.data_length,
    };
    LOOMC_ASSERT_OK(loomc_launch_config_context_evaluate(
        launch_context.get(), launch_function, &arguments, &outputs));
    IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(
        &launch_count_mapping, /*byte_offset=*/0, kLaunchCountByteLength));

    const iree_hal_buffer_binding_t bindings[] = {
        /*source=*/{source_buffer, 0, kBufferByteLength},
        /*target=*/{target_buffer, 0, kBufferByteLength},
        /*launch_count=*/{launch_count_buffer, 0, kLaunchCountByteLength},
    };
    IREE_ASSERT_OK(SubmitAndWait(
        device, command_buffer,
        {/*.count=*/IREE_ARRAYSIZE(bindings), /*.bindings=*/bindings}));

    std::array<uint32_t, kElementCount> actual = {};
    IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
        device, target_buffer, 0, actual.data(), kBufferByteLength,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
    for (iree_host_size_t i = 0; i < actual.size(); ++i) {
      const uint32_t expected = i < static_cast<iree_host_size_t>(workload)
                                    ? source_values[i] + 7
                                    : 0;
      EXPECT_EQ(actual[i], expected)
          << "element " << i << " at workload " << workload;
    }
  }

  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&launch_count_mapping));
  iree_hal_buffer_release(launch_count_buffer);
  iree_hal_buffer_release(target_buffer);
  iree_hal_buffer_release(source_buffer);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_device_group_release(device_group);
}

}  // namespace
}  // namespace loom
