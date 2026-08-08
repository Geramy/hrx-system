// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/cmd/iree_hal/recording.h"
#include "loom/target/arch/cmd/iree_hal/recording_test_executable.h"
#include "loom/target/arch/cmd/lower/launch_graph.h"
#include "loom/target/arch/cmd/program.h"
#include "loomc/compile.h"
#include "loomc/launch_config_module.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/program.h"
#include "loomc/program_plan.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target/cmd.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace loom {
namespace {

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
using PlanPtr =
    loomc::testing::HandlePtr<loomc_program_plan_t, loomc_program_plan_release>;
using ProgramEnvironmentPtr =
    loomc::testing::HandlePtr<loomc_program_environment_t,
                              loomc_program_environment_release>;
using ProgramPtr =
    loomc::testing::HandlePtr<loomc_program_t, loomc_program_release>;
using SourcePtr =
    loomc::testing::HandlePtr<loomc_source_t, loomc_source_release>;

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

static const loomc_artifact_t* FindArtifact(const loomc_program_t* program,
                                            loomc_artifact_kind_t kind,
                                            const char* format,
                                            const char* identifier) {
  const loomc_artifact_t* found = nullptr;
  for (loomc_host_size_t i = 0; i < loomc_program_artifact_count(program);
       ++i) {
    const loomc_artifact_t* artifact = loomc_program_artifact_at(program, i);
    if (artifact->kind != kind ||
        !loomc_string_view_equal(artifact->format,
                                 loomc_make_cstring_view(format)) ||
        !loomc_string_view_equal(artifact->identifier,
                                 loomc_make_cstring_view(identifier))) {
      continue;
    }
    EXPECT_EQ(found, nullptr);
    found = artifact;
  }
  return found;
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
  }

  void TearDown() override {
    coordinator_workspace_.reset();
    context_handle_.reset();
  }

  void CompileProgram(ProgramPtr* out_program) {
    static const char kSource[] = R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def @increment(%element_count: index) {
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

command.program.def public target(@command_target) @increment_once(%element_count: index) launch(%source: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  kernel.launch @increment[%element_count](%source, %target) : [index](buffer, buffer)
  command.return
}

command.program.def public target(@command_target) @increment_twice(%element_count: index) launch(%source: buffer, %intermediate: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  kernel.launch @increment[%element_count](%source, %intermediate) : [index](buffer, buffer)
  kernel.launch @increment[%element_count](%intermediate, %target) : [index](buffer, buffer)
  command.return
}
)";
    const loomc_source_options_t source_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        /*.structure_size=*/sizeof(source_options),
        /*.next=*/nullptr,
        /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
        /*.identifier=*/loomc_make_cstring_view("command_lifecycle_test.loom"),
        /*.contents=*/loomc_make_byte_span(kSource, std::strlen(kSource)),
        /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
    };
    loomc_source_t* raw_source = nullptr;
    LOOMC_ASSERT_OK(loomc_source_create(&source_options,
                                        loomc_allocator_system(), &raw_source));
    SourcePtr source(raw_source);

    loomc_module_t* raw_module = nullptr;
    loomc_result_t* raw_parse_result = nullptr;
    LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
        context_handle_.get(), coordinator_workspace_.get(), source.get(),
        /*options=*/nullptr, loomc_allocator_system(), &raw_module,
        &raw_parse_result));
    ModulePtr module(raw_module);
    ResultPtr parse_result(raw_parse_result);
    ASSERT_TRUE(loomc_result_succeeded(parse_result.get()));

    loomc_compiler_t* raw_compiler = nullptr;
    loomc_program_environment_t* raw_program_environment = nullptr;
    LOOMC_ASSERT_OK(loomc_program_environment_create_command(
        loomc_allocator_system(), &raw_program_environment));
    ProgramEnvironmentPtr program_environment(raw_program_environment);
    const loomc_compiler_program_options_t program_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILER_PROGRAM_OPTIONS,
        /*.structure_size=*/sizeof(program_options),
        /*.next=*/nullptr,
        /*.program_environment=*/program_environment.get(),
    };
    const loomc_compiler_options_t compiler_options = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILER_OPTIONS,
        /*.structure_size=*/sizeof(compiler_options),
        /*.next=*/&program_options,
    };
    LOOMC_ASSERT_OK(
        loomc_compiler_create(context_handle_.get(), &compiler_options,
                              loomc_allocator_system(), &raw_compiler));
    CompilerPtr compiler(raw_compiler);
    loomc_pass_program_t* raw_pass_program = nullptr;
    LOOMC_ASSERT_OK(loomc_pass_program_create_empty(
        context_handle_.get(), /*options=*/nullptr, loomc_allocator_system(),
        &raw_pass_program));
    PassProgramPtr pass_program(raw_pass_program);

    loomc_program_plan_t* raw_plan = nullptr;
    loomc_result_t* raw_prepare_result = nullptr;
    LOOMC_ASSERT_OK(loomc_prepare_programs(
        compiler.get(), coordinator_workspace_.get(), pass_program.get(),
        pass_program.get(), module.get(), /*options=*/nullptr,
        loomc_allocator_system(), &raw_plan, &raw_prepare_result));
    PlanPtr plan(raw_plan);
    ResultPtr prepare_result(raw_prepare_result);
    ASSERT_TRUE(loomc_result_succeeded(prepare_result.get()));
    module.reset();
    source.reset();
    parse_result.reset();
    prepare_result.reset();
    pass_program.reset();
    compiler.reset();

    ASSERT_EQ(loomc_program_plan_root_count(plan.get()), 2u);
    ASSERT_EQ(loomc_program_plan_unit_count(plan.get()), 2u);
    loomc_program_plan_root_t once_root = loomc_program_plan_root_invalid();
    LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
        plan.get(), loomc_make_cstring_view("increment_once"), &once_root));
    loomc_program_plan_root_t twice_root = loomc_program_plan_root_invalid();
    LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
        plan.get(), loomc_make_cstring_view("increment_twice"), &twice_root));

    const loomc_host_size_t unit_count =
        loomc_program_plan_unit_count(plan.get());
    std::vector<ProgramPtr> unit_programs(unit_count);
    std::vector<loomc_program_t*> unit_table_values(unit_count);
    for (uint32_t i = 0; i < unit_count; ++i) {
      loomc_workspace_t* raw_worker_workspace = nullptr;
      LOOMC_ASSERT_OK(loomc_workspace_create(
          /*options=*/nullptr, loomc_allocator_system(),
          &raw_worker_workspace));
      WorkspacePtr worker_workspace(raw_worker_workspace);
      loomc_program_t* raw_unit_program = nullptr;
      loomc_result_t* raw_compile_result = nullptr;
      LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
          plan.get(), worker_workspace.get(),
          loomc_program_plan_unit_from_index(i), /*options=*/nullptr,
          loomc_allocator_system(), &raw_unit_program, &raw_compile_result));
      unit_programs[i].reset(raw_unit_program);
      ResultPtr compile_result(raw_compile_result);
      ASSERT_TRUE(loomc_result_succeeded(compile_result.get()));
      unit_table_values[i] = unit_programs[i].get();
    }

    const loomc_program_plan_unit_table_t unit_table = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_TABLE,
        /*.structure_size=*/sizeof(unit_table),
        /*.next=*/nullptr,
        /*.programs=*/unit_table_values.data(),
        /*.program_count=*/unit_table_values.size(),
    };
    const loomc_program_plan_root_t selected_roots[] = {once_root, twice_root};
    loomc_program_t* raw_program = nullptr;
    loomc_result_t* raw_assemble_result = nullptr;
    LOOMC_ASSERT_OK(loomc_program_plan_assemble(
        plan.get(), coordinator_workspace_.get(), selected_roots,
        IREE_ARRAYSIZE(selected_roots), &unit_table, /*options=*/nullptr,
        loomc_allocator_system(), &raw_program, &raw_assemble_result));
    ResultPtr assemble_result(raw_assemble_result);
    ASSERT_TRUE(loomc_result_succeeded(assemble_result.get()));
    out_program->reset(raw_program);

    plan.reset();
    assemble_result.reset();
    for (ProgramPtr& unit_program : unit_programs) unit_program.reset();
  }

  // Public context shared by source, plan, and compilation operations.
  LoomContextPtr context_handle_;
  // Coordinator scratch used for plan preparation and assembly.
  WorkspacePtr coordinator_workspace_;
};

TEST_F(CommandLifecycleTest,
       CompilesSharedDependenciesAndReplaysAssembledRoots) {
  ProgramPtr program;
  CompileProgram(&program);
  ASSERT_EQ(loomc_program_export_count(program.get()), 2u);
  ASSERT_EQ(loomc_program_dependency_count(program.get()), 1u);
  loomc_program_dependency_info_t dependency_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_DEPENDENCY_INFO,
      /*.structure_size=*/sizeof(dependency_info),
  };
  LOOMC_ASSERT_OK(
      loomc_program_dependency_info(program.get(), 0, &dependency_info));
  EXPECT_EQ(dependency_info.slot, 0u);
  ASSERT_NE(dependency_info.program, nullptr);
  ASSERT_EQ(loomc_program_artifact_count(dependency_info.program), 1u);
  const loomc_artifact_t* dependency_artifact =
      loomc_program_artifact_at(dependency_info.program, 0);
  ASSERT_NE(dependency_artifact, nullptr);
  EXPECT_EQ(dependency_artifact->kind, LOOMC_ARTIFACT_KIND_MODULE);
  EXPECT_TRUE(loomc_string_view_equal(
      dependency_artifact->format,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE)));

  const loomc_artifact_t* launch_artifact = FindArtifact(
      program.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE, "command-program-launch.loombc");
  ASSERT_NE(launch_artifact, nullptr);
  loomc_launch_config_module_t* raw_launch_module = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_module_load(
      launch_artifact, /*options=*/nullptr, loomc_allocator_system(),
      &raw_launch_module));
  LaunchModulePtr launch_module(raw_launch_module);

  loomc_launch_config_function_t once_launch_function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_module_lookup_function_by_name(
      launch_module.get(), loomc_make_cstring_view("increment_once"),
      &once_launch_function));
  loomc_launch_config_function_info_t once_launch_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_FUNCTION_INFO,
      /*.structure_size=*/sizeof(once_launch_info),
  };
  LOOMC_ASSERT_OK(loomc_launch_config_module_function_info(
      launch_module.get(), once_launch_function, &once_launch_info));
  ASSERT_EQ(once_launch_info.workload_argument_count, 1u);
  ASSERT_EQ(once_launch_info.result_count, 1u);
  ASSERT_EQ(once_launch_info.output_byte_length,
            LOOM_CMD_LAUNCH_COUNT_TUPLE_BYTE_LENGTH);
  loomc_launch_config_function_t twice_launch_function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_module_lookup_function_by_name(
      launch_module.get(), loomc_make_cstring_view("increment_twice"),
      &twice_launch_function));
  loomc_launch_config_function_info_t twice_launch_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_FUNCTION_INFO,
      /*.structure_size=*/sizeof(twice_launch_info),
  };
  LOOMC_ASSERT_OK(loomc_launch_config_module_function_info(
      launch_module.get(), twice_launch_function, &twice_launch_info));
  ASSERT_EQ(twice_launch_info.workload_argument_count, 1u);
  ASSERT_EQ(twice_launch_info.result_count, 1u);
  ASSERT_EQ(twice_launch_info.output_byte_length,
            LOOM_CMD_LAUNCH_COUNT_TUPLE_BYTE_LENGTH);
  loomc_launch_config_context_t* raw_launch_context = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_context_create(
      launch_module.get(), /*options=*/nullptr, loomc_allocator_system(),
      &raw_launch_context));
  LaunchContextPtr launch_context(raw_launch_context);

  const loomc_artifact_t* once_artifact =
      FindArtifact(program.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM, "increment_once");
  ASSERT_NE(once_artifact, nullptr);
  loom_cmd_program_t once_program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(once_artifact->contents.data,
                                once_artifact->contents.data_length),
      &once_program));
  ASSERT_EQ(once_program.requirements.fixed_buffer_count, 0u);
  ASSERT_EQ(once_program.requirements.rebindable_binding_count, 3u);
  ASSERT_EQ(once_program.requirements.executable_count, 1u);
  ASSERT_EQ(once_program.requirements.entry_count, 1u);
  ASSERT_EQ(once_program.commands.count, 1u);
  EXPECT_EQ(loom_cmd_program_command_at(&once_program, 0).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);

  const loomc_artifact_t* twice_artifact =
      FindArtifact(program.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM, "increment_twice");
  ASSERT_NE(twice_artifact, nullptr);
  loom_cmd_program_t twice_program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(twice_artifact->contents.data,
                                twice_artifact->contents.data_length),
      &twice_program));
  ASSERT_EQ(twice_program.requirements.fixed_buffer_count, 0u);
  ASSERT_EQ(twice_program.requirements.rebindable_binding_count, 4u);
  ASSERT_EQ(twice_program.requirements.executable_count, 1u);
  ASSERT_EQ(twice_program.requirements.entry_count, 1u);
  ASSERT_EQ(twice_program.commands.count, 3u);
  EXPECT_EQ(loom_cmd_program_command_at(&twice_program, 0).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  EXPECT_EQ(loom_cmd_program_command_at(&twice_program, 1).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_EXECUTION_BARRIER);
  EXPECT_EQ(loom_cmd_program_command_at(&twice_program, 2).kind,
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
  const std::array<iree_hal_executable_t*, 1> executables = {executable};
  const std::array<loom_cmd_iree_hal_entry_t, 1> entries = {{
      {
          /*.executable_index=*/0,
          /*.function=*/function,
          /*.info=*/function_info,
          /*.parameters=*/parameters.data(),
      },
  }};
  const loom_cmd_iree_hal_inputs_t once_inputs = {
      /*.binding_count=*/3,
      /*.fixed_buffer_count=*/0,
      /*.fixed_buffers=*/nullptr,
      /*.executable_count=*/executables.size(),
      /*.executables=*/executables.data(),
      /*.entry_count=*/entries.size(),
      /*.entries=*/entries.data(),
  };
  iree_hal_command_buffer_t* once_command_buffer = nullptr;
  IREE_ASSERT_OK(loom_cmd_iree_hal_materialize_program(
      &once_program, &once_inputs, device, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_QUEUE_AFFINITY_ANY, &once_command_buffer,
      iree_allocator_system()));
  const loom_cmd_iree_hal_inputs_t twice_inputs = {
      /*.binding_count=*/4,
      /*.fixed_buffer_count=*/0,
      /*.fixed_buffers=*/nullptr,
      /*.executable_count=*/executables.size(),
      /*.executables=*/executables.data(),
      /*.entry_count=*/entries.size(),
      /*.entries=*/entries.data(),
  };
  iree_hal_command_buffer_t* twice_command_buffer = nullptr;
  IREE_ASSERT_OK(loom_cmd_iree_hal_materialize_program(
      &twice_program, &twice_inputs, device,
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_QUEUE_AFFINITY_ANY,
      &twice_command_buffer, iree_allocator_system()));
  program.reset();
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
  iree_hal_buffer_t* once_target_buffer =
      CreateTransferBuffer(device, kBufferByteLength);
  iree_hal_buffer_t* twice_target_buffer =
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
        device, zero_values.data(), once_target_buffer, 0, kBufferByteLength,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
    IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
        device, zero_values.data(), twice_target_buffer, 0, kBufferByteLength,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

    const loomc_launch_config_arguments_t arguments = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_ARGUMENTS,
        /*.structure_size=*/sizeof(arguments),
        /*.next=*/nullptr,
        /*.workload_arguments=*/&workload,
        /*.workload_argument_count=*/1,
    };
    loomc_launch_config_outputs_t once_outputs = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_OUTPUTS,
        /*.structure_size=*/sizeof(once_outputs),
        /*.next=*/nullptr,
        /*.storage=*/launch_count_mapping.contents.data,
        /*.storage_length=*/launch_count_mapping.contents.data_length,
    };
    LOOMC_ASSERT_OK(loomc_launch_config_context_evaluate(
        launch_context.get(), once_launch_function, &arguments, &once_outputs));
    IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(
        &launch_count_mapping, /*byte_offset=*/0, kLaunchCountByteLength));

    const iree_hal_buffer_binding_t once_bindings[] = {
        /*source=*/{source_buffer, 0, IREE_HAL_WHOLE_BUFFER},
        /*target=*/{once_target_buffer, 0, IREE_HAL_WHOLE_BUFFER},
        /*launch_count=*/{launch_count_buffer, 0, kLaunchCountByteLength},
    };
    IREE_ASSERT_OK(SubmitAndWait(device, once_command_buffer,
                                 {/*.count=*/IREE_ARRAYSIZE(once_bindings),
                                  /*.bindings=*/once_bindings}));

    loomc_launch_config_outputs_t twice_outputs = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_OUTPUTS,
        /*.structure_size=*/sizeof(twice_outputs),
        /*.next=*/nullptr,
        /*.storage=*/launch_count_mapping.contents.data,
        /*.storage_length=*/launch_count_mapping.contents.data_length,
    };
    LOOMC_ASSERT_OK(loomc_launch_config_context_evaluate(
        launch_context.get(), twice_launch_function, &arguments,
        &twice_outputs));
    IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(
        &launch_count_mapping, /*byte_offset=*/0, kLaunchCountByteLength));

    const iree_hal_buffer_binding_t twice_bindings[] = {
        /*source=*/{source_buffer, 0, IREE_HAL_WHOLE_BUFFER},
        /*intermediate=*/{intermediate_buffer, 0, IREE_HAL_WHOLE_BUFFER},
        /*target=*/{twice_target_buffer, 0, IREE_HAL_WHOLE_BUFFER},
        /*launch_count=*/{launch_count_buffer, 0, kLaunchCountByteLength},
    };
    IREE_ASSERT_OK(SubmitAndWait(device, twice_command_buffer,
                                 {/*.count=*/IREE_ARRAYSIZE(twice_bindings),
                                  /*.bindings=*/twice_bindings}));

    std::array<uint32_t, kElementCount> once_actual = {};
    IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
        device, once_target_buffer, 0, once_actual.data(), kBufferByteLength,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
    std::array<uint32_t, kElementCount> twice_actual = {};
    IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
        device, twice_target_buffer, 0, twice_actual.data(), kBufferByteLength,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
    for (iree_host_size_t i = 0; i < once_actual.size(); ++i) {
      const bool is_active = i < static_cast<iree_host_size_t>(workload);
      const uint32_t once_expected = is_active ? source_values[i] + 7 : 0;
      EXPECT_EQ(once_actual[i], once_expected)
          << "element " << i << " at workload " << workload;
      const uint32_t twice_expected = is_active ? source_values[i] + 14 : 0;
      EXPECT_EQ(twice_actual[i], twice_expected)
          << "element " << i << " at workload " << workload;
    }
  }

  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&launch_count_mapping));
  iree_hal_buffer_release(launch_count_buffer);
  iree_hal_buffer_release(twice_target_buffer);
  iree_hal_buffer_release(once_target_buffer);
  iree_hal_buffer_release(intermediate_buffer);
  iree_hal_buffer_release(source_buffer);
  iree_hal_command_buffer_release(twice_command_buffer);
  iree_hal_command_buffer_release(once_command_buffer);
  iree_hal_device_group_release(device_group);
}

}  // namespace
}  // namespace loom
