// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/launch_config_module.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using LaunchModulePtr =
    HandlePtr<loomc_launch_config_module_t, loomc_launch_config_module_release>;
using LaunchContextPtr = HandlePtr<loomc_launch_config_context_t,
                                   loomc_launch_config_context_release>;

struct CountingAllocator {
  loomc_allocator_t system = loomc_allocator_system();
  loomc_host_size_t allocation_count = 0;
};

loomc_status_t CountingAllocatorControl(void* self,
                                        loomc_allocator_command_t command,
                                        const void* params, void** inout_ptr) {
  auto* allocator = static_cast<CountingAllocator*>(self);
  if (command != LOOMC_ALLOCATOR_COMMAND_FREE) {
    ++allocator->allocation_count;
  }
  return allocator->system.ctl(allocator->system.self, command, params,
                               inout_ptr);
}

loomc_allocator_t MakeAllocator(CountingAllocator* allocator) {
  return loomc_allocator_t{
      /*.self=*/allocator,
      /*.ctl=*/CountingAllocatorControl,
  };
}

ContextPtr CreateContext() {
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(
      loomc_context_create(nullptr, loomc_allocator_system(), &context));
  return ContextPtr(context);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  LOOMC_EXPECT_OK(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
  return WorkspacePtr(workspace);
}

SourcePtr CreateTextSource(const char* contents) {
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("launch_config_source.loom"),
      /*.contents=*/loomc_make_byte_span(contents, std::strlen(contents)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

void BuildArtifactBytes(loomc_context_t* context, const char* source_contents,
                        std::vector<uint8_t>* out_bytes) {
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateTextSource(source_contents);

  loomc_module_t* raw_module = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_module_deserialize_from_source(
      context, workspace.get(), source.get(), nullptr, loomc_allocator_system(),
      &raw_module, &raw_result));
  ModulePtr module(raw_module);
  ResultPtr result(raw_result);
  ASSERT_TRUE(loomc_result_succeeded(result.get()));

  loomc_module_serialize_options_t serialize_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS,
      /*.structure_size=*/sizeof(serialize_options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_BYTECODE,
      /*.identifier=*/loomc_make_cstring_view("launch_config.loombc"),
  };
  loomc_source_t* raw_bytecode = nullptr;
  LOOMC_ASSERT_OK(loomc_module_serialize_to_source(
      module.get(), &serialize_options, loomc_allocator_system(),
      &raw_bytecode));
  SourcePtr bytecode(raw_bytecode);
  const loomc_byte_span_t contents = loomc_source_contents(bytecode.get());
  out_bytes->assign(contents.data, contents.data + contents.data_length);
}

void BuildLaunchArtifactBytes(loomc_context_t* context,
                              std::vector<uint8_t>* out_bytes) {
  BuildArtifactBytes(context, R"(
func.def public pure @single(%token_count: index) -> (index, index, index) where [range(%token_count, 1, 512)] {
  %one = index.constant 1 : index
  %groups = index.add %token_count, %one : index
  func.return %groups, %one, %one : index, index, index
}

func.def public pure @aggregate(%token_count: index) -> (index, index, index, index, index, index) where [range(%token_count, 1, 512)] {
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %row0 = index.add %token_count, %one : index
  %row1 = index.mul %token_count, %two : index
  func.return %row0, %one, %one, %row1, %one, %one : index, index, index, index, index, index
}

func.def public pure @empty() {
  func.return
}

func.def @private_helper(%value: index) -> (index) {
  func.return %value : index
}
)",
                     out_bytes);
}

loomc_artifact_t MakeArtifact(const std::vector<uint8_t>& bytes) {
  return loomc_artifact_t{
      /*.kind=*/LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      /*.format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
      /*.identifier=*/loomc_make_cstring_view("launch_config.loombc"),
      /*.contents=*/loomc_make_byte_span(bytes.data(), bytes.size()),
  };
}

LaunchModulePtr LoadModule(const loomc_artifact_t* artifact,
                           loomc_allocator_t allocator) {
  loomc_launch_config_module_t* module = nullptr;
  LOOMC_EXPECT_OK(
      loomc_launch_config_module_load(artifact, nullptr, allocator, &module));
  return LaunchModulePtr(module);
}

loomc_launch_config_function_t LookupFunction(
    const loomc_launch_config_module_t* module, const char* name) {
  loomc_launch_config_function_t function =
      loomc_launch_config_function_invalid();
  LOOMC_EXPECT_OK(loomc_launch_config_module_lookup_function_by_name(
      module, loomc_make_cstring_view(name), &function));
  EXPECT_TRUE(loomc_launch_config_function_is_valid(function));
  return function;
}

TEST(LaunchConfigModuleTest, LoadsOwnsAndEvaluatesMultipleFunctions) {
  ContextPtr context = CreateContext();
  std::vector<uint8_t> bytes;
  BuildLaunchArtifactBytes(context.get(), &bytes);
  const loomc_artifact_t artifact = MakeArtifact(bytes);
  CountingAllocator allocator_state;
  const loomc_allocator_t allocator = MakeAllocator(&allocator_state);
  LaunchModulePtr module = LoadModule(&artifact, allocator);
  ASSERT_NE(module, nullptr);
  EXPECT_EQ(loomc_launch_config_module_function_count(module.get()), 3u);

  const loomc_launch_config_function_t aggregate =
      LookupFunction(module.get(), "aggregate");
  loomc_launch_config_function_info_t info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_FUNCTION_INFO,
      /*.structure_size=*/sizeof(info),
  };
  LOOMC_ASSERT_OK(
      loomc_launch_config_module_function_info(module.get(), aggregate, &info));
  EXPECT_EQ(std::string(info.name.data, info.name.size), "aggregate");
  EXPECT_EQ(info.workload_argument_count, 1u);
  EXPECT_EQ(info.result_count, 2u);
  EXPECT_EQ(info.output_byte_length, 2u * sizeof(loomc_dimension3_t));
  EXPECT_EQ(info.output_alignment, alignof(uint32_t));

  // The loaded module owns materialized IR rather than borrowing artifact
  // bytes.
  std::fill(bytes.begin(), bytes.end(), 0);

  loomc_launch_config_context_t* raw_launch_context = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_context_create(
      module.get(), nullptr, allocator, &raw_launch_context));
  LaunchContextPtr launch_context(raw_launch_context);
  allocator_state.allocation_count = 0;

  const int64_t first_workload[] = {1};
  loomc_launch_config_arguments_t first_arguments = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_ARGUMENTS,
      /*.structure_size=*/sizeof(first_arguments),
      /*.next=*/nullptr,
      /*.workload_arguments=*/first_workload,
      /*.workload_argument_count=*/std::size(first_workload),
  };
  std::array<loomc_dimension3_t, 2> output = {};
  loomc_launch_config_outputs_t outputs = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_OUTPUTS,
      /*.structure_size=*/sizeof(outputs),
      /*.next=*/nullptr,
      /*.storage=*/reinterpret_cast<uint8_t*>(output.data()),
      /*.storage_length=*/sizeof(output),
  };
  LOOMC_ASSERT_OK(loomc_launch_config_context_evaluate(
      launch_context.get(), aggregate, &first_arguments, &outputs));
  EXPECT_EQ(output[0].x, 2u);
  EXPECT_EQ(output[0].y, 1u);
  EXPECT_EQ(output[0].z, 1u);
  EXPECT_EQ(output[1].x, 2u);
  EXPECT_EQ(output[1].y, 1u);
  EXPECT_EQ(output[1].z, 1u);

  const int64_t second_workload[] = {127};
  loomc_launch_config_arguments_t second_arguments = first_arguments;
  second_arguments.workload_arguments = second_workload;
  LOOMC_ASSERT_OK(loomc_launch_config_context_evaluate(
      launch_context.get(), aggregate, &second_arguments, &outputs));
  EXPECT_EQ(output[0].x, 128u);
  EXPECT_EQ(output[1].x, 254u);
  EXPECT_EQ(allocator_state.allocation_count, 0u);

  const loomc_launch_config_function_t single =
      LookupFunction(module.get(), "single");
  loomc_launch_config_t config = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(config),
  };
  LOOMC_ASSERT_OK(loomc_launch_config_context_evaluate_one(
      launch_context.get(), single, &second_arguments, &config));
  EXPECT_EQ(config.fields, LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT);
  EXPECT_EQ(config.workgroup_count.x, 128u);
  EXPECT_EQ(config.workgroup_count.y, 1u);
  EXPECT_EQ(config.workgroup_count.z, 1u);
  EXPECT_EQ(allocator_state.allocation_count, 0u);

  const loomc_launch_config_function_t empty =
      LookupFunction(module.get(), "empty");
  loomc_launch_config_function_info_t empty_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_FUNCTION_INFO,
      /*.structure_size=*/sizeof(empty_info),
  };
  LOOMC_ASSERT_OK(loomc_launch_config_module_function_info(module.get(), empty,
                                                           &empty_info));
  EXPECT_EQ(empty_info.workload_argument_count, 0u);
  EXPECT_EQ(empty_info.result_count, 0u);
  EXPECT_EQ(empty_info.output_byte_length, 0u);
  loomc_launch_config_arguments_t empty_arguments = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_ARGUMENTS,
      /*.structure_size=*/sizeof(empty_arguments),
  };
  loomc_launch_config_outputs_t empty_outputs = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_OUTPUTS,
      /*.structure_size=*/sizeof(empty_outputs),
  };
  LOOMC_ASSERT_OK(loomc_launch_config_context_evaluate(
      launch_context.get(), empty, &empty_arguments, &empty_outputs));
  EXPECT_EQ(allocator_state.allocation_count, 0u);
}

TEST(LaunchConfigModuleTest, RejectsMalformedArtifactAndInvocation) {
  ContextPtr context = CreateContext();
  std::vector<uint8_t> bytes;
  BuildLaunchArtifactBytes(context.get(), &bytes);

  loomc_artifact_t wrong_kind = MakeArtifact(bytes);
  wrong_kind.kind = LOOMC_ARTIFACT_KIND_MODULE;
  loomc_launch_config_module_t* raw_module = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_launch_config_module_load(&wrong_kind, nullptr,
                                      loomc_allocator_system(), &raw_module));
  EXPECT_EQ(raw_module, nullptr);

  const std::vector<uint8_t> malformed_bytes = {0x4C, 0x4F, 0x4F, 0x4D};
  const loomc_artifact_t malformed_artifact = MakeArtifact(malformed_bytes);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_launch_config_module_load(&malformed_artifact, nullptr,
                                      loomc_allocator_system(), &raw_module));
  EXPECT_EQ(raw_module, nullptr);

  const loomc_artifact_t artifact = MakeArtifact(bytes);
  LaunchModulePtr module = LoadModule(&artifact, loomc_allocator_system());
  const loomc_launch_config_function_t aggregate =
      LookupFunction(module.get(), "aggregate");
  loomc_launch_config_context_t* raw_context = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_context_create(
      module.get(), nullptr, loomc_allocator_system(), &raw_context));
  LaunchContextPtr launch_context(raw_context);

  const int64_t invalid_workload[] = {0};
  loomc_launch_config_arguments_t arguments = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_ARGUMENTS,
      /*.structure_size=*/sizeof(arguments),
      /*.next=*/nullptr,
      /*.workload_arguments=*/invalid_workload,
      /*.workload_argument_count=*/std::size(invalid_workload),
  };
  std::array<loomc_dimension3_t, 2> output = {};
  loomc_launch_config_outputs_t outputs = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_OUTPUTS,
      /*.structure_size=*/sizeof(outputs),
      /*.next=*/nullptr,
      /*.storage=*/reinterpret_cast<uint8_t*>(output.data()),
      /*.storage_length=*/sizeof(output),
  };
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_launch_config_context_evaluate(launch_context.get(), aggregate,
                                           &arguments, &outputs));

  const int64_t valid_workload[] = {8};
  arguments.workload_arguments = valid_workload;
  outputs.storage_length = sizeof(loomc_dimension3_t);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_OUT_OF_RANGE,
      loomc_launch_config_context_evaluate(launch_context.get(), aggregate,
                                           &arguments, &outputs));

  std::vector<uint8_t> invalid_abi_bytes;
  BuildArtifactBytes(context.get(), R"(
func.def public pure @incomplete(%value: index) -> (index) {
  func.return %value : index
}
)",
                     &invalid_abi_bytes);
  const loomc_artifact_t invalid_abi_artifact = MakeArtifact(invalid_abi_bytes);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_launch_config_module_load(&invalid_abi_artifact, nullptr,
                                      loomc_allocator_system(), &raw_module));
  EXPECT_EQ(raw_module, nullptr);
}

TEST(LaunchConfigModuleTest, EvaluatesIndependentContextsConcurrently) {
  ContextPtr context = CreateContext();
  std::vector<uint8_t> bytes;
  BuildLaunchArtifactBytes(context.get(), &bytes);
  const loomc_artifact_t artifact = MakeArtifact(bytes);
  LaunchModulePtr module = LoadModule(&artifact, loomc_allocator_system());
  const loomc_launch_config_function_t aggregate =
      LookupFunction(module.get(), "aggregate");

  loomc_launch_config_context_t* raw_first_context = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_context_create(
      module.get(), nullptr, loomc_allocator_system(), &raw_first_context));
  LaunchContextPtr first_context(raw_first_context);
  loomc_launch_config_context_t* raw_second_context = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_context_create(
      module.get(), nullptr, loomc_allocator_system(), &raw_second_context));
  LaunchContextPtr second_context(raw_second_context);

  // Each evaluation context retains the immutable module independently.
  module.reset();

  struct ThreadResult {
    loomc_status_code_t status_code = LOOMC_STATUS_UNKNOWN;
    std::array<loomc_dimension3_t, 2> output = {};
  };
  auto evaluate_repeatedly = [aggregate](loomc_launch_config_context_t* context,
                                         int64_t token_count,
                                         ThreadResult* result) {
    loomc_launch_config_arguments_t arguments = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_ARGUMENTS,
        /*.structure_size=*/sizeof(arguments),
        /*.next=*/nullptr,
        /*.workload_arguments=*/&token_count,
        /*.workload_argument_count=*/1,
    };
    loomc_launch_config_outputs_t outputs = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_OUTPUTS,
        /*.structure_size=*/sizeof(outputs),
        /*.next=*/nullptr,
        /*.storage=*/reinterpret_cast<uint8_t*>(result->output.data()),
        /*.storage_length=*/sizeof(result->output),
    };
    for (int i = 0; i < 1000; ++i) {
      loomc_status_t status = loomc_launch_config_context_evaluate(
          context, aggregate, &arguments, &outputs);
      result->status_code = loomc_status_code(status);
      loomc_status_free(status);
      if (result->status_code != LOOMC_STATUS_OK) return;
    }
  };

  ThreadResult first_result;
  ThreadResult second_result;
  std::thread first_thread(evaluate_repeatedly, first_context.get(), 7,
                           &first_result);
  std::thread second_thread(evaluate_repeatedly, second_context.get(), 113,
                            &second_result);
  first_thread.join();
  second_thread.join();

  EXPECT_EQ(first_result.status_code, LOOMC_STATUS_OK);
  EXPECT_EQ(first_result.output[0].x, 8u);
  EXPECT_EQ(first_result.output[1].x, 14u);
  EXPECT_EQ(second_result.status_code, LOOMC_STATUS_OK);
  EXPECT_EQ(second_result.output[0].x, 114u);
  EXPECT_EQ(second_result.output[1].x, 226u);
}

}  // namespace
