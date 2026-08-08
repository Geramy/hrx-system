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
#include "loom/binding/c/src/context.h"
#include "loom/binding/c/src/module.h"
#include "loom/binding/c/src/workspace.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/cmd/descriptors/low_registry.h"
#include "loom/target/arch/cmd/iree_hal/recording.h"
#include "loom/target/arch/cmd/iree_hal/recording_test_executable.h"
#include "loom/target/arch/cmd/lower/launch_artifact.h"
#include "loom/target/arch/cmd/lower/program_plan.h"
#include "loom/target/arch/cmd/lower/serialize.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"
#include "loomc/compile.h"
#include "loomc/launch_config_module.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/workspace.h"
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
using LoomContextPtr =
    loomc::testing::HandlePtr<loomc_context_t, loomc_context_release>;
using WorkspacePtr =
    loomc::testing::HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using ModulePtr =
    loomc::testing::HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr =
    loomc::testing::HandlePtr<loomc_result_t, loomc_result_release>;
using CompilerPtr =
    loomc::testing::HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using PassProgramPtr =
    loomc::testing::HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;

struct CommandArtifacts {
  // Evaluation-ready launch-configuration Loombc.
  std::vector<uint8_t> launch_config;
  // Closed portable command-program bytes.
  std::vector<uint8_t> command_program;
  // Independently compiled dependency-unit Loombc artifacts.
  std::vector<std::vector<uint8_t>> dependency_modules;
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
    loomc_context_t* context = nullptr;
    LOOMC_ASSERT_OK(loomc_context_create(
        /*options=*/nullptr, loomc_allocator_system(), &context));
    context_handle_.reset(context);
    loomc_workspace_t* workspace = nullptr;
    LOOMC_ASSERT_OK(loomc_workspace_create(
        /*options=*/nullptr, loomc_allocator_system(), &workspace));
    coordinator_workspace_.reset(workspace);
    context_ = loomc_context_loom_context(context_handle_.get());
    block_pool_ = loomc_workspace_block_pool(coordinator_workspace_.get());
    loom_cmd_low_descriptor_registry_initialize(&registry_);
  }

  void TearDown() override {
    block_pool_ = nullptr;
    context_ = nullptr;
    coordinator_workspace_.reset();
    context_handle_.reset();
  }

  InternalModulePtr ParseAndVerifySource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(
        iree_make_cstring_view(source), IREE_SV("command_lifecycle_test.loom"),
        context_, block_pool_, &parse_options, &module));
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

  void CompileDependencyModules(loom_cmd_program_plan_t* plan,
                                CommandArtifacts* artifacts) {
    loomc_compiler_t* compiler = nullptr;
    LOOMC_ASSERT_OK(loomc_compiler_create(context_handle_.get(),
                                          /*options=*/nullptr,
                                          loomc_allocator_system(), &compiler));
    CompilerPtr compiler_ptr(compiler);

    loomc_pass_program_t* pass_program = nullptr;
    LOOMC_ASSERT_OK(loomc_pass_program_create_empty(
        context_handle_.get(), /*options=*/nullptr, loomc_allocator_system(),
        &pass_program));
    PassProgramPtr pass_program_ptr(pass_program);

    artifacts->dependency_modules.resize(plan->dependency_count);
    for (iree_host_size_t i = 0; i < plan->dependency_count; ++i) {
      loomc_module_t* module = nullptr;
      LOOMC_ASSERT_OK(loomc_module_create_empty(
          context_handle_.get(), coordinator_workspace_.get(),
          loomc_allocator_system(), &module));
      ModulePtr module_ptr(module);
      LOOMC_ASSERT_OK(loomc_module_set_loom_module(
          module_ptr.get(), plan->dependency_units[i].module));
      plan->dependency_units[i].module = nullptr;

      loomc_workspace_t* worker_workspace = nullptr;
      LOOMC_ASSERT_OK(loomc_workspace_create(
          /*options=*/nullptr, loomc_allocator_system(), &worker_workspace));
      WorkspacePtr worker_workspace_ptr(worker_workspace);
      const loomc_compile_options_t compile_options = {
          /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
          /*.structure_size=*/sizeof(compile_options),
          /*.next=*/nullptr,
          /*.module_name=*/loomc_make_cstring_view("dependency"),
          /*.artifact_flags=*/LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE,
      };
      loomc_result_t* result = nullptr;
      LOOMC_ASSERT_OK(loomc_compile_module(
          compiler_ptr.get(), worker_workspace_ptr.get(),
          pass_program_ptr.get(), module_ptr.get(), &compile_options,
          loomc_allocator_system(), &result));
      ResultPtr result_ptr(result);
      ASSERT_TRUE(loomc_result_succeeded(result_ptr.get()));
      ASSERT_EQ(loomc_result_artifact_count(result_ptr.get()), 1u);
      const loomc_artifact_t* artifact =
          loomc_result_artifact_at(result_ptr.get(), 0);
      ASSERT_NE(artifact, nullptr);
      EXPECT_EQ(artifact->kind, LOOMC_ARTIFACT_KIND_MODULE);
      EXPECT_TRUE(loomc_string_view_equal(
          artifact->format,
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE)));
      artifacts->dependency_modules[i].assign(
          artifact->contents.data,
          artifact->contents.data + artifact->contents.data_length);
    }
  }

  CommandArtifacts CompileArtifacts() {
    InternalModulePtr module = ParseAndVerifySource(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def @increment_first(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %seven = scalar.constant 7 : i32
  %source_view = buffer.view %source[%base] : buffer -> view<128xi32, #dense>
  %target_view = buffer.view %target[%base] : buffer -> view<128xi32, #dense>
  %value = view.load %source_view[%workgroup] : view<128xi32, #dense> -> i32
  %incremented = scalar.addi %value, %seven : i32
  view.store %incremented, %target_view[%workgroup] : i32, view<128xi32, #dense>
  kernel.return
}

kernel.def @increment_second(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %seven = scalar.constant 7 : i32
  %source_view = buffer.view %source[%base] : buffer -> view<128xi32, #dense>
  %target_view = buffer.view %target[%base] : buffer -> view<128xi32, #dense>
  %value = view.load %source_view[%workgroup] : view<128xi32, #dense> -> i32
  %incremented = scalar.addi %value, %seven : i32
  view.store %incremented, %target_view[%workgroup] : i32, view<128xi32, #dense>
  kernel.return
}

command.program.def public target(@command_target) @increment_elements(%element_count: index) launch(%source: buffer, %intermediate: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  kernel.launch @increment_first[%element_count](%source, %intermediate) : [index](buffer, buffer)
  kernel.launch @increment_second[%element_count](%intermediate, %target) : [index](buffer, buffer)
  command.return
}
)");

    loom_op_t* source_program =
        FindSymbol(module.get(), IREE_SV("increment_elements"));
    const loom_op_t* source_programs[] = {source_program};
    loom_cmd_program_plan_t plan = {};
    IREE_CHECK_OK(loom_cmd_program_plan_prepare(
        module.get(), source_programs, IREE_ARRAYSIZE(source_programs),
        block_pool_, iree_allocator_system(), &plan));
    module.reset();
    IREE_ASSERT_EQ(plan.root_count, 1u);
    IREE_ASSERT_EQ(plan.dependency_count, 2u);
    const loom_cmd_program_root_t& root = plan.roots[0];
    IREE_ASSERT_EQ(root.launch_tuple_count, 1u);
    VerifyLowModule(plan.root_module);

    iree_byte_span_t launch_config_data = iree_byte_span_empty();
    IREE_CHECK_OK(loom_cmd_launch_program_serialize(
        root.launch_module, block_pool_, iree_allocator_system(),
        &launch_config_data));

    iree_byte_span_t command_program_data = iree_byte_span_empty();
    IREE_CHECK_OK(loom_cmd_program_serialize_low(
        plan.root_module, root.function_op, &command_program_data,
        iree_allocator_system()));

    CommandArtifacts artifacts;
    CompileDependencyModules(&plan, &artifacts);
    artifacts.launch_config.assign(
        launch_config_data.data,
        launch_config_data.data + launch_config_data.data_length);
    artifacts.command_program.assign(
        command_program_data.data,
        command_program_data.data + command_program_data.data_length);
    iree_allocator_free(iree_allocator_system(), launch_config_data.data);
    iree_allocator_free(iree_allocator_system(), command_program_data.data);
    loom_cmd_program_plan_deinitialize(&plan);
    return artifacts;
  }

  // Public context retaining the internal source dialect context.
  LoomContextPtr context_handle_;
  // Workspace retaining the arena block pool backing prepared modules.
  WorkspacePtr coordinator_workspace_;
  // Internal dialect context borrowed from |context_handle_|.
  loom_context_t* context_ = nullptr;
  // Arena block pool borrowed from |coordinator_workspace_|.
  iree_arena_block_pool_t* block_pool_ = nullptr;
  // Portable command descriptor registry used by low verification.
  loom_target_low_descriptor_registry_t registry_ = {};
};

TEST_F(CommandLifecycleTest, CompilesDependenciesAndReplaysAssembledRoot) {
  CommandArtifacts artifacts = CompileArtifacts();
  ASSERT_FALSE(artifacts.launch_config.empty());
  ASSERT_FALSE(artifacts.command_program.empty());
  ASSERT_EQ(artifacts.dependency_modules.size(), 2u);
  EXPECT_FALSE(artifacts.dependency_modules[0].empty());
  EXPECT_FALSE(artifacts.dependency_modules[1].empty());

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
  ASSERT_EQ(command_program.requirements.rebindable_binding_count, 4u);
  ASSERT_EQ(command_program.requirements.executable_count, 2u);
  ASSERT_EQ(command_program.requirements.entry_count, 2u);
  ASSERT_EQ(command_program.commands.count, 3u);
  EXPECT_EQ(loom_cmd_program_command_at(&command_program, 0).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  EXPECT_EQ(loom_cmd_program_command_at(&command_program, 1).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_EXECUTION_BARRIER);
  EXPECT_EQ(loom_cmd_program_command_at(&command_program, 2).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);

  iree_hal_device_group_t* device_group = CreateSyncDeviceGroup();
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_executable_t* executable = nullptr;
  IREE_ASSERT_OK(LoadTestExecutable(device, &executable));
  const iree_hal_executable_function_t function =
      iree_hal_executable_function_from_index(1);
  iree_hal_executable_function_info_t function_info = {};
  IREE_ASSERT_OK(
      iree_hal_executable_function_info(executable, function, &function_info));
  std::vector<iree_hal_executable_function_parameter_t> parameters(
      function_info.parameter_count);
  IREE_ASSERT_OK(iree_hal_executable_function_parameters(
      executable, function, parameters.size(), parameters.data()));
  const std::array<iree_hal_executable_t*, 2> executables = {
      executable,
      executable,
  };
  const std::array<loom_cmd_iree_hal_entry_t, 2> entries = {{
      {
          /*.executable_index=*/0,
          /*.function=*/function,
          /*.info=*/function_info,
          /*.parameters=*/parameters.data(),
      },
      {
          /*.executable_index=*/1,
          /*.function=*/function,
          /*.info=*/function_info,
          /*.parameters=*/parameters.data(),
      },
  }};
  const loom_cmd_iree_hal_inputs_t inputs = {
      /*.binding_count=*/4,
      /*.fixed_buffer_count=*/0,
      /*.fixed_buffers=*/nullptr,
      /*.executable_count=*/executables.size(),
      /*.executables=*/executables.data(),
      /*.entry_count=*/entries.size(),
      /*.entries=*/entries.data(),
  };
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(loom_cmd_iree_hal_materialize_program(
      &command_program, &inputs, device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_QUEUE_AFFINITY_ANY, &command_buffer, iree_allocator_system()));
  artifacts.command_program.clear();
  artifacts.dependency_modules.clear();
  iree_hal_executable_release(executable);

  static constexpr iree_host_size_t kElementCount = 128;
  static constexpr iree_device_size_t kBufferByteLength =
      kElementCount * sizeof(uint32_t);
  static constexpr iree_device_size_t kLaunchCountByteLength =
      LOOM_CMD_LAUNCH_COUNT_TUPLE_BYTE_LENGTH;
  iree_hal_buffer_t* source_buffer =
      CreateTransferBuffer(device, kBufferByteLength);
  iree_hal_buffer_t* intermediate_buffer =
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
        device, zero_values.data(), intermediate_buffer, 0, kBufferByteLength,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
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
        /*source=*/{source_buffer, 0, IREE_HAL_WHOLE_BUFFER},
        /*intermediate=*/{intermediate_buffer, 0, IREE_HAL_WHOLE_BUFFER},
        /*target=*/{target_buffer, 0, IREE_HAL_WHOLE_BUFFER},
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
                                    ? source_values[i] + 14
                                    : 0;
      EXPECT_EQ(actual[i], expected)
          << "element " << i << " at workload " << workload;
    }
  }

  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&launch_count_mapping));
  iree_hal_buffer_release(launch_count_buffer);
  iree_hal_buffer_release(target_buffer);
  iree_hal_buffer_release(intermediate_buffer);
  iree_hal_buffer_release(source_buffer);
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_device_group_release(device_group);
}

}  // namespace
}  // namespace loom
