// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/all.h"

#include <cstring>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/emit.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/spirv.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

static std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

static void ExpectSucceededResult(const loomc_result_t* result) {
  ASSERT_NE(result, nullptr);
  if (!loomc_result_succeeded(result) &&
      loomc_result_diagnostic_count(result) != 0) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, 0);
    ASSERT_NE(diagnostic, nullptr);
    ADD_FAILURE() << ToString(diagnostic->message);
  }
  EXPECT_TRUE(loomc_result_succeeded(result));
}

static TargetEnvironmentPtr CreateAllTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_all(loomc_allocator_system(),
                                                      &target_environment));
  return TargetEnvironmentPtr(target_environment);
}

static ContextPtr CreateContext(
    loomc_target_environment_t* target_environment) {
  const loomc_context_target_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment,
  };
  const loomc_context_options_t context_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(context_options),
      /*.next=*/&target_options,
  };
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(loomc_context_create(&context_options,
                                       loomc_allocator_system(), &context));
  return ContextPtr(context);
}

static WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  LOOMC_EXPECT_OK(loomc_workspace_create(
      /*options=*/nullptr, loomc_allocator_system(), &workspace));
  return WorkspacePtr(workspace);
}

static ModulePtr ParseModule(loomc_context_t* context,
                             loomc_workspace_t* workspace,
                             const char* identifier, const char* contents) {
  const loomc_source_options_t source_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(source_options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view(identifier),
      /*.contents=*/loomc_make_byte_span(contents, std::strlen(contents)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&source_options, loomc_allocator_system(), &source));
  SourcePtr source_ptr(source);

  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
      context, workspace, source_ptr.get(), /*options=*/nullptr,
      loomc_allocator_system(), &module, &result));
  ResultPtr result_ptr(result);
  ExpectSucceededResult(result_ptr.get());
  return ModulePtr(module);
}

static ResultPtr EmitModule(loomc_target_environment_t* target_environment,
                            loomc_workspace_t* workspace,
                            loomc_module_t* module, const char* format,
                            const char* identifier, const void* next) {
  const loomc_emit_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/next,
      /*.artifact_format=*/loomc_make_cstring_view(format),
      /*.identifier=*/loomc_make_cstring_view(identifier),
      /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_emit_module(target_environment, workspace, module,
                                    &options, loomc_allocator_system(),
                                    &result));
  return ResultPtr(result);
}

TEST(AllTargetTest, ComposesPortableAndHardwareTargetProducts) {
  TargetEnvironmentPtr target_environment = CreateAllTargetEnvironment();
  ContextPtr context = CreateContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();

  loomc_amdgpu_profile_options_t amdgpu_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(amdgpu_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("gfx1100"),
      /*.identity=*/
      {
          /*.target=*/loomc_make_cstring_view("gfx1100"),
      },
  };
  loomc_target_profile_t* amdgpu_profile = nullptr;
  LOOMC_EXPECT_OK(loomc_target_profile_create_amdgpu(
      target_environment.get(), &amdgpu_options, loomc_allocator_system(),
      &amdgpu_profile));
  TargetProfilePtr amdgpu_profile_ptr(amdgpu_profile);

  loomc_spirv_profile_options_t spirv_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(spirv_options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("vulkan1.3"),
      /*.preset=*/LOOMC_SPIRV_PROFILE_PRESET_VULKAN_1_3_BDA,
  };
  loomc_target_profile_t* spirv_profile = nullptr;
  loomc_result_t* spirv_profile_result = nullptr;
  LOOMC_EXPECT_OK(loomc_target_profile_create_spirv(
      target_environment.get(), &spirv_options, loomc_allocator_system(),
      &spirv_profile, &spirv_profile_result));
  TargetProfilePtr spirv_profile_ptr(spirv_profile);
  ResultPtr spirv_profile_result_ptr(spirv_profile_result);
  ExpectSucceededResult(spirv_profile_result_ptr.get());

  ModulePtr command_module =
      ParseModule(context.get(), workspace.get(), "portable_command.loom", R"(
low.func.def target<cmd.core> abi(command_program) @empty() {
  low.return
}
)");
  EXPECT_NE(command_module.get(), nullptr);

  ModulePtr spirv_module =
      ParseModule(context.get(), workspace.get(), "standalone_spirv.loom", R"(
spirv.target<vulkan1_3> @target

low.func.def target<spirv.logical.core>(@target) abi(shader_entry_point) @barrier() asm {
  OpControlBarrier.subgroup.workgroup.acq_rel
  return
}
)");
  const loomc_spirv_emit_options_t spirv_emit_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SPIRV_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(spirv_emit_options),
      /*.next=*/nullptr,
  };
  ResultPtr spirv_result = EmitModule(
      target_environment.get(), workspace.get(), spirv_module.get(),
      LOOMC_ARTIFACT_FORMAT_SPIRV, "standalone.spv", &spirv_emit_options);
  ExpectSucceededResult(spirv_result.get());
  ASSERT_EQ(loomc_result_artifact_count(spirv_result.get()), 1u);
  const loomc_artifact_t* spirv_artifact =
      loomc_result_artifact_at(spirv_result.get(), 0);
  ASSERT_NE(spirv_artifact, nullptr);
  ASSERT_GE(spirv_artifact->contents.data_length, sizeof(uint32_t));
  uint32_t spirv_magic = 0;
  std::memcpy(&spirv_magic, spirv_artifact->contents.data, sizeof(spirv_magic));
  EXPECT_EQ(spirv_magic, 0x07230203u);

  ModulePtr amdgpu_module =
      ParseModule(context.get(), workspace.get(), "standalone_amdgpu.loom", R"(
amdgpu.target<gfx11-generic> @target

low.kernel.def target<amdgpu.gfx11.generic.core>(@target) workgroup_size(64, 1, 1) @kernel() {
  %zero = low.const<amdgpu.v_mov_b32> {imm32 = 0} : reg<amdgpu.vgpr>
  %one = low.const<amdgpu.v_mov_b32> {imm32 = 1} : reg<amdgpu.vgpr>
  %sum = low.op<amdgpu.v_add_u32>(%zero, %one) : (reg<amdgpu.vgpr>, reg<amdgpu.vgpr>) -> reg<amdgpu.vgpr>
  low.return
}
)");
  const loomc_amdgpu_emit_options_t amdgpu_emit_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(amdgpu_emit_options),
      /*.next=*/nullptr,
      /*.runtime_globals=*/LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE,
  };
  ResultPtr amdgpu_result =
      EmitModule(target_environment.get(), workspace.get(), amdgpu_module.get(),
                 LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO, "standalone.hsaco",
                 &amdgpu_emit_options);
  ExpectSucceededResult(amdgpu_result.get());
  ASSERT_EQ(loomc_result_artifact_count(amdgpu_result.get()), 1u);
  const loomc_artifact_t* amdgpu_artifact =
      loomc_result_artifact_at(amdgpu_result.get(), 0);
  ASSERT_NE(amdgpu_artifact, nullptr);
  ASSERT_GE(amdgpu_artifact->contents.data_length, 4u);
  EXPECT_EQ(amdgpu_artifact->contents.data[0], 0x7Fu);
  EXPECT_EQ(amdgpu_artifact->contents.data[1], 'E');
  EXPECT_EQ(amdgpu_artifact->contents.data[2], 'L');
  EXPECT_EQ(amdgpu_artifact->contents.data[3], 'F');
}

}  // namespace
