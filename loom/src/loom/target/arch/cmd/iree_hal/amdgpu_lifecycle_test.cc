// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/cmd/iree_hal/recording.h"
#include "loom/target/arch/cmd/program.h"
#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/amdgpu/iree_hal.h"
#include "loomc/target/cmd.h"
#include "loomc/target/iree_hal.h"
#include "test/util.h"

namespace loom {
namespace {

using loomc::testing::HandlePtr;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ExecutablePtr =
    HandlePtr<iree_hal_executable_t, iree_hal_executable_release>;
using FrontierTrackerPtr = HandlePtr<iree_async_frontier_tracker_t,
                                     iree_async_frontier_tracker_release>;
using HalBufferPtr = HandlePtr<iree_hal_buffer_t, iree_hal_buffer_release>;
using HalCommandBufferPtr =
    HandlePtr<iree_hal_command_buffer_t, iree_hal_command_buffer_release>;
using HalDevicePtr = HandlePtr<iree_hal_device_t, iree_hal_device_release>;
using HalDeviceGroupPtr =
    HandlePtr<iree_hal_device_group_t, iree_hal_device_group_release>;
using LaunchContextPtr = HandlePtr<loomc_launch_config_context_t,
                                   loomc_launch_config_context_release>;
using LaunchModulePtr =
    HandlePtr<loomc_launch_config_module_t, loomc_launch_config_module_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using PlanPtr = HandlePtr<loomc_program_plan_t, loomc_program_plan_release>;
using ProactorPoolPtr =
    HandlePtr<iree_async_proactor_pool_t, iree_async_proactor_pool_release>;
using ProgramEnvironmentPtr =
    HandlePtr<loomc_program_environment_t, loomc_program_environment_release>;
using ProgramPtr = HandlePtr<loomc_program_t, loomc_program_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

static constexpr char kSourceText[] = R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def @add_seven(%element_count: index) {
  %one = index.constant 1 : index
  %bounded_count = index.assume %element_count [range(%element_count, 1, 128)] : index
  kernel.launch.config workgroups(%bounded_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: view<128xi32, #dense>, %target: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %seven = scalar.constant 7 : i32
  %target_aligned = buffer.assume.alignment %target {minimum_alignment = 4} : buffer
  %target_view = buffer.view %target_aligned[%base] : buffer -> view<128xi32, #dense>
  %value = view.load %source[%workgroup] : view<128xi32, #dense> -> i32
  %result = scalar.addi %value, %seven : i32
  view.store %result, %target_view[%workgroup] : i32, view<128xi32, #dense>
  kernel.return
}

command.program.def public target(@command_target) @add_seven_once(%element_count: index) launch(%parameters: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  %source = command.parameter %parameters, "source_values"[] : view<128xi32, #dense>
  kernel.launch @add_seven[%element_count](%source, %target) : [index](view<128xi32, #dense>, buffer)
  command.return
}

command.program.def public target(@command_target) @add_seven_twice(%element_count: index) launch(%parameters: buffer, %intermediate: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  %source = command.parameter %parameters, "source_values"[] : view<128xi32, #dense>
  kernel.launch @add_seven[%element_count](%source, %intermediate) : [index](view<128xi32, #dense>, buffer)
  %zero = index.constant 0 : offset
  %intermediate_view = buffer.view %intermediate[%zero] : buffer -> view<128xi32, #dense>
  kernel.launch @add_seven[%element_count](%intermediate_view, %target) : [index](view<128xi32, #dense>, buffer)
  command.return
}
)";

static void PrintResultDiagnostics(const loomc_result_t* result) {
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    std::fprintf(stderr, "%.*s: %.*s\n",
                 static_cast<int>(diagnostic->code.size), diagnostic->code.data,
                 static_cast<int>(diagnostic->message.size),
                 diagnostic->message.data);
  }
}

static bool ResultSucceeded(const loomc_result_t* result) {
  if (result == nullptr || !loomc_result_succeeded(result)) {
    if (result != nullptr) PrintResultDiagnostics(result);
    return false;
  }
  return true;
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

static iree_status_t CreateAmdgpuDevice(
    ProactorPoolPtr* out_proactor_pool,
    FrontierTrackerPtr* out_frontier_tracker,
    HalDeviceGroupPtr* out_device_group, HalDevicePtr* out_device) {
  IREE_RETURN_IF_ERROR(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));

  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      &proactor_pool));
  out_proactor_pool->reset(proactor_pool);

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  iree_hal_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_create_device(
      iree_hal_driver_registry_default(), IREE_SV("amdgpu"), &create_params,
      iree_allocator_system(), &device));
  out_device->reset(device);

  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));
  out_frontier_tracker->reset(frontier_tracker);

  iree_hal_device_group_t* device_group = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
      device, frontier_tracker, iree_allocator_system(), &device_group));
  out_device_group->reset(device_group);
  return iree_ok_status();
}

static iree_status_t AllocateStorageBuffer(iree_hal_device_t* device,
                                           iree_device_size_t byte_length,
                                           iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.usage =
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
}

static iree_status_t AllocateLaunchCountBuffer(iree_hal_device_t* device,
                                               iree_device_size_t byte_length,
                                               iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_params_t params = {0};
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS |
                 IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  return iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device),
                                            params, byte_length, out_buffer);
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

TEST(CommandAmdgpuLifecycleTest,
     CompilesAssemblesMaterializesAndReplaysNativePrograms) {
  ProactorPoolPtr proactor_pool;
  FrontierTrackerPtr frontier_tracker;
  HalDeviceGroupPtr device_group;
  HalDevicePtr device;
  iree_status_t device_status = CreateAmdgpuDevice(
      &proactor_pool, &frontier_tracker, &device_group, &device);
  if (!iree_status_is_ok(device_status) && device.get() == nullptr) {
    const iree_status_code_t code = iree_status_code(device_status);
    if (code == IREE_STATUS_NOT_FOUND || code == IREE_STATUS_UNAVAILABLE ||
        code == IREE_STATUS_FAILED_PRECONDITION) {
      iree_status_free(device_status);
      GTEST_SKIP() << "no live AMDGPU HAL device is available";
    }
  }
  IREE_ASSERT_OK(device_status);

  loomc_target_environment_t* raw_target_environment = nullptr;
  LOOMC_ASSERT_OK(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &raw_target_environment));
  TargetEnvironmentPtr target_environment(raw_target_environment);
  const loomc_context_target_options_t context_target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(context_target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment.get(),
  };
  const loomc_context_options_t context_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(context_options),
      /*.next=*/&context_target_options,
  };
  loomc_context_t* raw_context = nullptr;
  LOOMC_ASSERT_OK(loomc_context_create(&context_options,
                                       loomc_allocator_system(), &raw_context));
  ContextPtr context(raw_context);

  loomc_workspace_t* raw_coordinator_workspace = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(
      /*options=*/nullptr, loomc_allocator_system(),
      &raw_coordinator_workspace));
  WorkspacePtr coordinator_workspace(raw_coordinator_workspace);

  const loomc_source_options_t source_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(source_options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/
      loomc_make_cstring_view("command_amdgpu_lifecycle_test.loom"),
      /*.contents=*/
      loomc_make_byte_span(kSourceText, std::strlen(kSourceText)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_BORROWED,
  };
  loomc_source_t* raw_source = nullptr;
  LOOMC_ASSERT_OK(loomc_source_create(&source_options, loomc_allocator_system(),
                                      &raw_source));
  SourcePtr source(raw_source);

  loomc_module_t* raw_module = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
      context.get(), coordinator_workspace.get(), source.get(),
      /*options=*/nullptr, loomc_allocator_system(), &raw_module, &raw_result));
  ModulePtr module(raw_module);
  ResultPtr result(raw_result);
  ASSERT_TRUE(ResultSucceeded(result.get()));
  result.reset();

  const loomc_iree_hal_profile_provider_t* profile_providers[] = {
      loomc_amdgpu_iree_hal_profile_provider(),
  };
  const loomc_iree_hal_profile_options_t profile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(profile_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("live-amdgpu-command"),
      /*.device=*/device.get(),
      /*.physical_device_affinity=*/0,
      /*.providers=*/profile_providers,
      /*.provider_count=*/IREE_ARRAYSIZE(profile_providers),
  };
  loomc_target_profile_t* raw_target_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_create_iree_hal(
      target_environment.get(), &profile_options, loomc_allocator_system(),
      &raw_target_profile, &raw_result));
  TargetProfilePtr target_profile(raw_target_profile);
  result.reset(raw_result);
  ASSERT_TRUE(ResultSucceeded(result.get()));
  result.reset();

  loomc_program_environment_t* raw_program_environment = nullptr;
  LOOMC_ASSERT_OK(loomc_program_environment_create_command(
      loomc_allocator_system(), &raw_program_environment));
  ProgramEnvironmentPtr program_environment(raw_program_environment);
  const loomc_compiler_program_options_t compiler_program_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILER_PROGRAM_OPTIONS,
      /*.structure_size=*/sizeof(compiler_program_options),
      /*.next=*/nullptr,
      /*.program_environment=*/program_environment.get(),
  };
  const loomc_compiler_options_t compiler_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILER_OPTIONS,
      /*.structure_size=*/sizeof(compiler_options),
      /*.next=*/&compiler_program_options,
  };
  loomc_compiler_t* raw_compiler = nullptr;
  LOOMC_ASSERT_OK(loomc_compiler_create(context.get(), &compiler_options,
                                        loomc_allocator_system(),
                                        &raw_compiler));
  CompilerPtr compiler(raw_compiler);

  loomc_pass_program_t* raw_preparation_pass_program = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_empty(
      context.get(), /*options=*/nullptr, loomc_allocator_system(),
      &raw_preparation_pass_program));
  PassProgramPtr preparation_pass_program(raw_preparation_pass_program);
  const loomc_target_pipeline_options_t target_pipeline_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(target_pipeline_options),
      /*.next=*/nullptr,
      /*.identifier=*/
      loomc_make_cstring_view("command-amdgpu-prepared-low"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/20,
  };
  loomc_pass_program_t* raw_unit_pass_program = nullptr;
  LOOMC_ASSERT_OK(loomc_pass_program_create_from_target_pipeline(
      context.get(), &target_pipeline_options, loomc_allocator_system(),
      &raw_unit_pass_program, &raw_result));
  PassProgramPtr unit_pass_program(raw_unit_pass_program);
  result.reset(raw_result);
  ASSERT_TRUE(ResultSucceeded(result.get()));
  result.reset();

  const loomc_target_specialization_t specialization = {
      /*.function_symbol=*/loomc_make_cstring_view("@add_seven"),
      /*.target_profile=*/target_profile.get(),
  };
  const loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/&specialization,
      /*.specialization_count=*/1,
  };
  const loomc_cmd_program_plan_options_t command_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(command_options),
      /*.next=*/nullptr,
      /*.dependency_artifact_format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
  };
  const loomc_program_plan_options_t plan_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(plan_options),
      /*.next=*/&command_options,
      /*.config=*/{},
      /*.target_specialization=*/&target_options,
  };

  loomc_program_plan_t* raw_plan = nullptr;
  LOOMC_ASSERT_OK(loomc_prepare_programs(
      compiler.get(), coordinator_workspace.get(),
      preparation_pass_program.get(), unit_pass_program.get(), module.get(),
      &plan_options, loomc_allocator_system(), &raw_plan, &raw_result));
  PlanPtr plan(raw_plan);
  result.reset(raw_result);
  ASSERT_TRUE(ResultSucceeded(result.get()));
  result.reset();
  ASSERT_EQ(loomc_program_plan_root_count(plan.get()), 2u);
  ASSERT_EQ(loomc_program_plan_unit_count(plan.get()), 2u);

  std::array<loomc_program_plan_root_t, 2> roots = {
      loomc_program_plan_root_invalid(),
      loomc_program_plan_root_invalid(),
  };
  LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
      plan.get(), loomc_make_cstring_view("add_seven_once"), &roots[0]));
  LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
      plan.get(), loomc_make_cstring_view("add_seven_twice"), &roots[1]));

  target_profile.reset();
  unit_pass_program.reset();
  preparation_pass_program.reset();
  compiler.reset();
  program_environment.reset();
  module.reset();
  source.reset();
  context.reset();
  target_environment.reset();

  std::array<ProgramPtr, 2> unit_programs;
  std::array<loomc_program_t*, 2> unit_table_values = {};
  for (uint32_t i = 0; i < unit_programs.size(); ++i) {
    loomc_workspace_t* raw_worker_workspace = nullptr;
    LOOMC_ASSERT_OK(loomc_workspace_create(
        /*options=*/nullptr, loomc_allocator_system(), &raw_worker_workspace));
    WorkspacePtr worker_workspace(raw_worker_workspace);
    loomc_program_t* raw_unit_program = nullptr;
    LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
        plan.get(), worker_workspace.get(),
        loomc_program_plan_unit_from_index(i), /*options=*/nullptr,
        loomc_allocator_system(), &raw_unit_program, &raw_result));
    unit_programs[i].reset(raw_unit_program);
    result.reset(raw_result);
    ASSERT_TRUE(ResultSucceeded(result.get()));
    result.reset();
    unit_table_values[i] = unit_programs[i].get();
  }

  const loomc_program_plan_unit_table_t unit_table = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_TABLE,
      /*.structure_size=*/sizeof(unit_table),
      /*.next=*/nullptr,
      /*.programs=*/unit_table_values.data(),
      /*.program_count=*/unit_table_values.size(),
  };
  loomc_program_t* raw_program = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_assemble(
      plan.get(), coordinator_workspace.get(), roots.data(), roots.size(),
      &unit_table, /*options=*/nullptr, loomc_allocator_system(), &raw_program,
      &raw_result));
  ProgramPtr program(raw_program);
  result.reset(raw_result);
  ASSERT_TRUE(ResultSucceeded(result.get()));
  result.reset();

  ASSERT_EQ(loomc_program_export_count(program.get()), 2u);
  ASSERT_EQ(loomc_program_dependency_count(program.get()), 1u);
  loomc_program_dependency_info_t dependency_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_DEPENDENCY_INFO,
      /*.structure_size=*/sizeof(dependency_info),
  };
  LOOMC_ASSERT_OK(
      loomc_program_dependency_info(program.get(), 0, &dependency_info));
  ASSERT_NE(dependency_info.program, nullptr);
  ASSERT_EQ(loomc_program_artifact_count(dependency_info.program), 1u);
  const loomc_artifact_t* executable_artifact =
      loomc_program_artifact_at(dependency_info.program, 0);
  ASSERT_NE(executable_artifact, nullptr);
  ASSERT_EQ(executable_artifact->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  ASSERT_TRUE(loomc_string_view_equal(
      executable_artifact->format,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO)));

  const iree_hal_executable_target_selection_t target_selection = {
      /*.family=*/IREE_SV("amdgpu"),
      /*.target_key=*/iree_string_view_empty(),
      /*.kind_flags=*/IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT,
      /*.physical_device_affinity=*/0,
  };
  const iree_hal_executable_target_selection_result_t target_result =
      iree_hal_device_spec_select_executable_target(
          iree_hal_device_spec(device.get()), &target_selection);
  ASSERT_EQ(target_result.outcome,
            IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED);
  iree_hal_executable_load_params_t load_params;
  iree_hal_executable_load_params_initialize(&load_params);
  load_params.executable_data =
      iree_make_const_byte_span(executable_artifact->contents.data,
                                executable_artifact->contents.data_length);
  iree_hal_executable_t* raw_executable = nullptr;
  IREE_ASSERT_OK(iree_hal_device_load_executable(
      device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, target_result.target,
      &load_params, &raw_executable));
  ExecutablePtr executable(raw_executable);

  const iree_hal_executable_function_t function =
      iree_hal_executable_function_from_index(0);
  iree_hal_executable_function_info_t function_info = {};
  IREE_ASSERT_OK(iree_hal_executable_function_info(executable.get(), function,
                                                   &function_info));
  std::vector<iree_hal_executable_function_parameter_t> parameters(
      function_info.parameter_count);
  IREE_ASSERT_OK(iree_hal_executable_function_parameters(
      executable.get(), function, parameters.size(), parameters.data()));

  const loomc_artifact_t* once_command_artifact =
      FindArtifact(program.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM, "add_seven_once");
  ASSERT_NE(once_command_artifact, nullptr);
  loom_cmd_program_t once_command_program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(once_command_artifact->contents.data,
                                once_command_artifact->contents.data_length),
      &once_command_program));
  ASSERT_EQ(once_command_program.requirements.fixed_buffer_count, 1u);
  ASSERT_EQ(once_command_program.requirements.rebindable_binding_count, 2u);
  ASSERT_EQ(once_command_program.requirements.executable_count, 1u);
  ASSERT_EQ(once_command_program.requirements.entry_count, 1u);
  ASSERT_EQ(once_command_program.parameter_roots.count, 1u);
  const loom_cmd_program_parameter_root_t once_parameter_root =
      loom_cmd_program_parameter_root_at(&once_command_program, 0);
  EXPECT_EQ(once_parameter_root.fixed_buffer_index, 0u);
  EXPECT_EQ(once_parameter_root.required_byte_length, 512u);
  EXPECT_EQ(once_parameter_root.minimum_alignment, 256u);
  ASSERT_EQ(once_command_program.parameters.count, 1u);
  const loom_cmd_program_parameter_t once_parameter =
      loom_cmd_program_parameter_at(&once_command_program, 0);
  EXPECT_TRUE(
      iree_string_view_equal(once_parameter.key, IREE_SV("source_values")));
  EXPECT_EQ(once_parameter.fixed_buffer_index, 0u);
  EXPECT_EQ(once_parameter.byte_offset, 0u);
  EXPECT_EQ(once_parameter.byte_length, 512u);
  EXPECT_EQ(once_parameter.minimum_alignment, 256u);
  ASSERT_EQ(once_command_program.commands.count, 1u);
  EXPECT_EQ(loom_cmd_program_command_at(&once_command_program, 0).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);

  const loomc_artifact_t* twice_command_artifact =
      FindArtifact(program.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM, "add_seven_twice");
  ASSERT_NE(twice_command_artifact, nullptr);
  loom_cmd_program_t twice_command_program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(twice_command_artifact->contents.data,
                                twice_command_artifact->contents.data_length),
      &twice_command_program));
  ASSERT_EQ(twice_command_program.requirements.fixed_buffer_count, 1u);
  ASSERT_EQ(twice_command_program.requirements.rebindable_binding_count, 3u);
  ASSERT_EQ(twice_command_program.requirements.executable_count, 1u);
  ASSERT_EQ(twice_command_program.requirements.entry_count, 1u);
  ASSERT_EQ(twice_command_program.parameter_roots.count, 1u);
  ASSERT_EQ(twice_command_program.parameters.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      loom_cmd_program_parameter_at(&twice_command_program, 0).key,
      IREE_SV("source_values")));
  ASSERT_EQ(twice_command_program.commands.count, 3u);
  EXPECT_EQ(loom_cmd_program_command_at(&twice_command_program, 0).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  EXPECT_EQ(loom_cmd_program_command_at(&twice_command_program, 1).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_EXECUTION_BARRIER);
  EXPECT_EQ(loom_cmd_program_command_at(&twice_command_program, 2).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);

  const std::array<iree_hal_executable_t*, 1> executables = {
      executable.get(),
  };
  const std::array<loom_cmd_iree_hal_entry_t, 1> entries = {{
      {
          /*.executable_index=*/0,
          /*.function=*/function,
          /*.info=*/function_info,
          /*.parameters=*/parameters.data(),
      },
  }};
  const loomc_artifact_t* launch_artifact = FindArtifact(
      program.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE, "command-program-launch.loombc");
  ASSERT_NE(launch_artifact, nullptr);
  loomc_launch_config_module_t* raw_launch_module = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_module_load(
      launch_artifact, /*options=*/nullptr, loomc_allocator_system(),
      &raw_launch_module));
  LaunchModulePtr launch_module(raw_launch_module);
  loomc_launch_config_context_t* raw_launch_context = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_context_create(
      launch_module.get(), /*options=*/nullptr, loomc_allocator_system(),
      &raw_launch_context));
  LaunchContextPtr launch_context(raw_launch_context);
  ASSERT_EQ(loomc_launch_config_module_function_count(launch_module.get()), 2u);
  loomc_launch_config_function_t once_launch_function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_module_lookup_function_by_name(
      launch_module.get(), loomc_make_cstring_view("add_seven_once"),
      &once_launch_function));
  loomc_launch_config_function_info_t once_launch_function_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_FUNCTION_INFO,
      /*.structure_size=*/sizeof(once_launch_function_info),
  };
  LOOMC_ASSERT_OK(loomc_launch_config_module_function_info(
      launch_module.get(), once_launch_function, &once_launch_function_info));
  ASSERT_EQ(once_launch_function_info.workload_argument_count, 1u);
  ASSERT_EQ(once_launch_function_info.result_count, 1u);
  ASSERT_GT(once_launch_function_info.output_byte_length, 0u);

  loomc_launch_config_function_t twice_launch_function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_module_lookup_function_by_name(
      launch_module.get(), loomc_make_cstring_view("add_seven_twice"),
      &twice_launch_function));
  loomc_launch_config_function_info_t twice_launch_function_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_FUNCTION_INFO,
      /*.structure_size=*/sizeof(twice_launch_function_info),
  };
  LOOMC_ASSERT_OK(loomc_launch_config_module_function_info(
      launch_module.get(), twice_launch_function, &twice_launch_function_info));
  ASSERT_EQ(twice_launch_function_info.workload_argument_count, 1u);
  ASSERT_EQ(twice_launch_function_info.result_count, 1u);
  ASSERT_EQ(twice_launch_function_info.output_byte_length,
            once_launch_function_info.output_byte_length);

  static constexpr iree_host_size_t kElementCount = 128;
  static constexpr iree_host_size_t kWorkload = 73;
  const iree_device_size_t kBufferByteLength =
      once_parameter_root.required_byte_length;
  const iree_device_size_t launch_count_byte_length =
      once_launch_function_info.output_byte_length;
  iree_hal_buffer_t* raw_source_buffer = nullptr;
  IREE_ASSERT_OK(AllocateStorageBuffer(device.get(), kBufferByteLength,
                                       &raw_source_buffer));
  HalBufferPtr source_buffer(raw_source_buffer);
  const iree_hal_buffer_ref_t fixed_source =
      iree_hal_make_buffer_ref(source_buffer.get(), 0, IREE_HAL_WHOLE_BUFFER);
  const loom_cmd_iree_hal_inputs_t once_materialization_inputs = {
      /*.binding_count=*/2,
      /*.fixed_buffer_count=*/1,
      /*.fixed_buffers=*/&fixed_source,
      /*.executable_count=*/executables.size(),
      /*.executables=*/executables.data(),
      /*.entry_count=*/entries.size(),
      /*.entries=*/entries.data(),
  };
  iree_hal_command_buffer_t* raw_once_command_buffer = nullptr;
  IREE_ASSERT_OK(loom_cmd_iree_hal_materialize_program(
      &once_command_program, &once_materialization_inputs, device.get(),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_QUEUE_AFFINITY_ANY,
      &raw_once_command_buffer, iree_allocator_system()));
  HalCommandBufferPtr once_command_buffer(raw_once_command_buffer);

  const loom_cmd_iree_hal_inputs_t twice_materialization_inputs = {
      /*.binding_count=*/3,
      /*.fixed_buffer_count=*/1,
      /*.fixed_buffers=*/&fixed_source,
      /*.executable_count=*/executables.size(),
      /*.executables=*/executables.data(),
      /*.entry_count=*/entries.size(),
      /*.entries=*/entries.data(),
  };
  iree_hal_command_buffer_t* raw_twice_command_buffer = nullptr;
  IREE_ASSERT_OK(loom_cmd_iree_hal_materialize_program(
      &twice_command_program, &twice_materialization_inputs, device.get(),
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_QUEUE_AFFINITY_ANY,
      &raw_twice_command_buffer, iree_allocator_system()));
  HalCommandBufferPtr twice_command_buffer(raw_twice_command_buffer);

  program.reset();
  for (ProgramPtr& unit_program : unit_programs) unit_program.reset();
  plan.reset();
  coordinator_workspace.reset();
  executable.reset();

  iree_hal_buffer_t* raw_intermediate_buffer = nullptr;
  IREE_ASSERT_OK(AllocateStorageBuffer(device.get(), kBufferByteLength,
                                       &raw_intermediate_buffer));
  HalBufferPtr intermediate_buffer(raw_intermediate_buffer);
  iree_hal_buffer_t* raw_once_target_buffer = nullptr;
  IREE_ASSERT_OK(AllocateStorageBuffer(device.get(), kBufferByteLength,
                                       &raw_once_target_buffer));
  HalBufferPtr once_target_buffer(raw_once_target_buffer);
  iree_hal_buffer_t* raw_twice_target_buffer = nullptr;
  IREE_ASSERT_OK(AllocateStorageBuffer(device.get(), kBufferByteLength,
                                       &raw_twice_target_buffer));
  HalBufferPtr twice_target_buffer(raw_twice_target_buffer);
  iree_hal_buffer_t* raw_launch_count_buffer = nullptr;
  IREE_ASSERT_OK(AllocateLaunchCountBuffer(
      device.get(), launch_count_byte_length, &raw_launch_count_buffer));
  HalBufferPtr launch_count_buffer(raw_launch_count_buffer);

  std::array<uint32_t, kElementCount> source_values = {};
  for (iree_host_size_t i = 0; i < source_values.size(); ++i) {
    source_values[i] = 1000u + static_cast<uint32_t>(i);
  }
  const std::array<uint32_t, kElementCount> zero_values = {};
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), source_values.data(), source_buffer.get(), 0,
      kBufferByteLength, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), zero_values.data(), intermediate_buffer.get(), 0,
      kBufferByteLength, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), zero_values.data(), once_target_buffer.get(), 0,
      kBufferByteLength, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  IREE_ASSERT_OK(iree_hal_device_transfer_h2d(
      device.get(), zero_values.data(), twice_target_buffer.get(), 0,
      kBufferByteLength, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

  iree_hal_buffer_mapping_t launch_count_mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      launch_count_buffer.get(), IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_WRITE, /*byte_offset=*/0, launch_count_byte_length,
      &launch_count_mapping));
  const int64_t workload = kWorkload;
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
      &launch_count_mapping, /*byte_offset=*/0, launch_count_byte_length));

  const iree_hal_buffer_binding_t once_bindings[] = {
      /*target=*/{once_target_buffer.get(), 0, IREE_HAL_WHOLE_BUFFER},
      /*launch_count=*/
      {launch_count_buffer.get(), 0, launch_count_byte_length},
  };
  IREE_ASSERT_OK(SubmitAndWait(device.get(), once_command_buffer.get(),
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
      launch_context.get(), twice_launch_function, &arguments, &twice_outputs));
  IREE_ASSERT_OK(iree_hal_buffer_mapping_flush_range(
      &launch_count_mapping, /*byte_offset=*/0, launch_count_byte_length));

  const iree_hal_buffer_binding_t twice_bindings[] = {
      /*intermediate=*/
      {intermediate_buffer.get(), 0, IREE_HAL_WHOLE_BUFFER},
      /*target=*/{twice_target_buffer.get(), 0, IREE_HAL_WHOLE_BUFFER},
      /*launch_count=*/
      {launch_count_buffer.get(), 0, launch_count_byte_length},
  };
  IREE_ASSERT_OK(SubmitAndWait(device.get(), twice_command_buffer.get(),
                               {/*.count=*/IREE_ARRAYSIZE(twice_bindings),
                                /*.bindings=*/twice_bindings}));

  std::array<uint32_t, kElementCount> once_actual_values = {};
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device.get(), once_target_buffer.get(), 0, once_actual_values.data(),
      kBufferByteLength, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  std::array<uint32_t, kElementCount> twice_actual_values = {};
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device.get(), twice_target_buffer.get(), 0, twice_actual_values.data(),
      kBufferByteLength, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  for (iree_host_size_t i = 0; i < once_actual_values.size(); ++i) {
    const uint32_t expected = i < kWorkload ? source_values[i] + 7 : 0;
    EXPECT_EQ(once_actual_values[i], expected) << "element " << i;
    const uint32_t twice_expected = i < kWorkload ? source_values[i] + 14 : 0;
    EXPECT_EQ(twice_actual_values[i], twice_expected) << "element " << i;
  }

  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&launch_count_mapping));
}

}  // namespace
}  // namespace loom
