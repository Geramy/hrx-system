// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/program_plan.h"

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/cmd/program.h"
#include "loomc/artifact.h"
#include "loomc/context.h"
#include "loomc/launch_config_module.h"
#include "loomc/program.h"
#include "loomc/source.h"
#include "loomc/target/cmd.h"
#include "loomc/target/spirv.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using CmdProgramPtr = HandlePtr<loomc_cmd_program_t, loomc_cmd_program_release>;
using LaunchModulePtr =
    HandlePtr<loomc_launch_config_module_t, loomc_launch_config_module_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using PlanPtr = HandlePtr<loomc_program_plan_t, loomc_program_plan_release>;
using ProgramEnvironmentPtr =
    HandlePtr<loomc_program_environment_t, loomc_program_environment_release>;
using ProgramPtr = HandlePtr<loomc_program_t, loomc_program_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

struct OwnedArtifact {
  // Artifact category.
  loomc_artifact_kind_t kind;
  // Owned artifact format.
  std::string format;
  // Owned diagnostic identifier.
  std::string identifier;
  // Owned artifact payload.
  std::vector<uint8_t> contents;
};

static ContextPtr CreateContext() {
  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_spirv(
      loomc_allocator_system(), &target_environment));
  TargetEnvironmentPtr target_environment_ptr(target_environment);
  const loomc_context_target_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment_ptr.get(),
  };
  const loomc_context_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&target_options,
  };
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(
      loomc_context_create(&options, loomc_allocator_system(), &context));
  return ContextPtr(context);
}

static WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  LOOMC_EXPECT_OK(loomc_workspace_create(
      /*options=*/nullptr, loomc_allocator_system(), &workspace));
  return WorkspacePtr(workspace);
}

static SourcePtr CreateSource(const char* contents) {
  const loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view("command_program_plan.loom"),
      /*.contents=*/loomc_make_byte_span(contents, std::strlen(contents)),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

static ModulePtr DeserializeModule(loomc_context_t* context,
                                   loomc_workspace_t* workspace,
                                   const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
      context, workspace, source, /*options=*/nullptr, loomc_allocator_system(),
      &module, &result));
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(module);
}

static CompilerPtr CreateCompiler(loomc_context_t* context) {
  loomc_program_environment_t* program_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_program_environment_create_command(
      loomc_allocator_system(), &program_environment));
  ProgramEnvironmentPtr program_environment_ptr(program_environment);
  const loomc_compiler_program_options_t program_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILER_PROGRAM_OPTIONS,
      /*.structure_size=*/sizeof(program_options),
      /*.next=*/nullptr,
      /*.program_environment=*/program_environment_ptr.get(),
  };
  const loomc_compiler_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILER_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/&program_options,
  };
  loomc_compiler_t* compiler = nullptr;
  LOOMC_EXPECT_OK(loomc_compiler_create(context, &options,
                                        loomc_allocator_system(), &compiler));
  return CompilerPtr(compiler);
}

static CompilerPtr CreateCompilerWithoutProgramEnvironment(
    loomc_context_t* context) {
  loomc_compiler_t* compiler = nullptr;
  LOOMC_EXPECT_OK(loomc_compiler_create(context, /*options=*/nullptr,
                                        loomc_allocator_system(), &compiler));
  return CompilerPtr(compiler);
}

static PassProgramPtr CreateEmptyPassProgram(loomc_context_t* context) {
  loomc_pass_program_t* pass_program = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_empty(
      context, /*options=*/nullptr, loomc_allocator_system(), &pass_program));
  return PassProgramPtr(pass_program);
}

static PassProgramPtr CreateCommandPreparationPassProgram(
    loomc_context_t* context) {
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_from_pipeline_text(
      context, loomc_make_cstring_view("unroll-scf-for,canonicalize,cse"),
      /*options=*/nullptr, loomc_allocator_system(), &pass_program, &result));
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return PassProgramPtr(pass_program);
}

static PassProgramPtr CreateTargetPassProgram(loomc_context_t* context) {
  const loomc_target_pipeline_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("command-dependency-spirv"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_from_target_pipeline(
      context, &options, loomc_allocator_system(), &pass_program, &result));
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return PassProgramPtr(pass_program);
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

static loom_cmd_program_command_t ParseOnlyCommand(
    const loomc_artifact_t* artifact, loom_cmd_program_t* out_program) {
  IREE_EXPECT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(artifact->contents.data,
                                artifact->contents.data_length),
      out_program));
  EXPECT_EQ(out_program->commands.count, 1u);
  return loom_cmd_program_command_at(out_program, 0);
}

static std::vector<OwnedArtifact> CopyArtifacts(
    const loomc_program_t* program) {
  std::vector<OwnedArtifact> copies;
  copies.reserve(loomc_program_artifact_count(program));
  for (loomc_host_size_t i = 0; i < loomc_program_artifact_count(program);
       ++i) {
    const loomc_artifact_t* artifact = loomc_program_artifact_at(program, i);
    copies.push_back(OwnedArtifact{
        /*.kind=*/artifact->kind,
        /*.format=*/std::string(artifact->format.data, artifact->format.size),
        /*.identifier=*/
        std::string(artifact->identifier.data, artifact->identifier.size),
        /*.contents=*/
        std::vector<uint8_t>(
            artifact->contents.data,
            artifact->contents.data + artifact->contents.data_length),
    });
  }
  return copies;
}

static std::vector<loomc_artifact_t> MakeArtifactViews(
    std::vector<OwnedArtifact>* artifacts) {
  std::vector<loomc_artifact_t> views;
  views.reserve(artifacts->size());
  for (OwnedArtifact& artifact : *artifacts) {
    views.push_back(loomc_artifact_t{
        /*.kind=*/artifact.kind,
        /*.format=*/
        loomc_make_string_view(artifact.format.data(), artifact.format.size()),
        /*.identifier=*/
        loomc_make_string_view(artifact.identifier.data(),
                               artifact.identifier.size()),
        /*.contents=*/
        loomc_make_byte_span(artifact.contents.data(),
                             artifact.contents.size()),
    });
  }
  return views;
}

TEST(CmdProgramPlanTest, CompilesLoadsAndAssemblesMultipleRoots) {
  ContextPtr context = CreateContext();
  WorkspacePtr coordinator_workspace = CreateWorkspace();
  SourcePtr source = CreateSource(R"(
spirv.target<vulkan1_3> @kernel_target {abi = hal_kernel}

target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def target(@kernel_target) @add_seven(%element_count: index) {
  %one = index.constant 1 : index
  %bounded_count = index.assume %element_count [range(%element_count, 1, 128)] : index
  kernel.launch.config workgroups(%bounded_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %seven = scalar.constant 7 : i32
  %source_aligned = buffer.assume.alignment %source {minimum_alignment = 4} : buffer
  %target_aligned = buffer.assume.alignment %target {minimum_alignment = 4} : buffer
  %source_view = buffer.view %source_aligned[%base] : buffer -> view<128xi32, #dense>
  %target_view = buffer.view %target_aligned[%base] : buffer -> view<128xi32, #dense>
  %value = view.load %source_view[%workgroup] : view<128xi32, #dense> -> i32
  %result = scalar.addi %value, %seven : i32
  view.store %result, %target_view[%workgroup] : i32, view<128xi32, #dense>
  kernel.return
}

kernel.def target(@kernel_target) @add_eleven(%element_count: index) {
  %one = index.constant 1 : index
  %bounded_count = index.assume %element_count [range(%element_count, 1, 128)] : index
  kernel.launch.config workgroups(%bounded_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %eleven = scalar.constant 11 : i32
  %source_aligned = buffer.assume.alignment %source {minimum_alignment = 4} : buffer
  %target_aligned = buffer.assume.alignment %target {minimum_alignment = 4} : buffer
  %source_view = buffer.view %source_aligned[%base] : buffer -> view<128xi32, #dense>
  %target_view = buffer.view %target_aligned[%base] : buffer -> view<128xi32, #dense>
  %value = view.load %source_view[%workgroup] : view<128xi32, #dense> -> i32
  %result = scalar.addi %value, %eleven : i32
  view.store %result, %target_view[%workgroup] : i32, view<128xi32, #dense>
  kernel.return
}

command.program.decl public target(@command_target) @external_program() launch(%parameters: buffer)

command.program.def public target(@command_target) @add_seven_once(%element_count: index) launch(%source: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  kernel.launch @add_seven[%element_count](%source, %target) : [index](buffer, buffer)
  command.return
}

command.program.def public target(@command_target) @add_eleven_once(%element_count: index) launch(%source: buffer, %target: buffer) where [range(%element_count, 1, 128)] {
  kernel.launch @add_eleven[%element_count](%source, %target) : [index](buffer, buffer)
  command.return
}
)");
  ModulePtr module = DeserializeModule(
      context.get(), coordinator_workspace.get(), source.get());
  PassProgramPtr preparation_pass_program =
      CreateEmptyPassProgram(context.get());
  PassProgramPtr unit_pass_program = CreateTargetPassProgram(context.get());

  CompilerPtr compiler_without_program_environment =
      CreateCompilerWithoutProgramEnvironment(context.get());
  loomc_program_plan_t* rejected_plan = nullptr;
  loomc_result_t* rejected_result = nullptr;
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_FAILED_PRECONDITION,
      loomc_prepare_programs(compiler_without_program_environment.get(),
                             coordinator_workspace.get(),
                             preparation_pass_program.get(),
                             unit_pass_program.get(), module.get(),
                             /*options=*/nullptr, loomc_allocator_system(),
                             &rejected_plan, &rejected_result));
  EXPECT_EQ(rejected_plan, nullptr);
  EXPECT_EQ(rejected_result, nullptr);
  compiler_without_program_environment.reset();

  CompilerPtr compiler = CreateCompiler(context.get());
  loomc_program_plan_t* missing_format_plan = nullptr;
  loomc_result_t* raw_missing_format_result = nullptr;
  LOOMC_ASSERT_OK(loomc_prepare_programs(
      compiler.get(), coordinator_workspace.get(),
      preparation_pass_program.get(), unit_pass_program.get(), module.get(),
      /*options=*/nullptr, loomc_allocator_system(), &missing_format_plan,
      &raw_missing_format_result));
  ResultPtr missing_format_result(raw_missing_format_result);
  EXPECT_EQ(missing_format_plan, nullptr);
  ASSERT_NE(missing_format_result.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(missing_format_result.get()));

  const loomc_cmd_program_plan_options_t command_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(command_options),
      /*.next=*/nullptr,
      /*.dependency_artifact_format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
  };
  const loomc_program_plan_options_t plan_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(plan_options),
      /*.next=*/&command_options,
  };

  loomc_program_plan_t* raw_plan = nullptr;
  loomc_result_t* raw_prepare_result = nullptr;
  LOOMC_ASSERT_OK(loomc_prepare_programs(
      compiler.get(), coordinator_workspace.get(),
      preparation_pass_program.get(), unit_pass_program.get(), module.get(),
      &plan_options, loomc_allocator_system(), &raw_plan, &raw_prepare_result));
  PlanPtr plan(raw_plan);
  ResultPtr prepare_result(raw_prepare_result);
  ASSERT_TRUE(loomc_result_succeeded(prepare_result.get()));

  source.reset();
  module.reset();
  prepare_result.reset();
  ASSERT_EQ(loomc_program_plan_root_count(plan.get()), 2u);
  ASSERT_EQ(loomc_program_plan_unit_count(plan.get()), 3u);

  loomc_program_plan_root_t add_seven_root = loomc_program_plan_root_invalid();
  LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
      plan.get(), loomc_make_cstring_view("add_seven_once"), &add_seven_root));
  loomc_program_plan_root_t add_eleven_root = loomc_program_plan_root_invalid();
  LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
      plan.get(), loomc_make_cstring_view("add_eleven_once"),
      &add_eleven_root));

  loomc_program_plan_dependency_info_t seven_dependency = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_DEPENDENCY_INFO,
      /*.structure_size=*/sizeof(seven_dependency),
  };
  LOOMC_ASSERT_OK(loomc_program_plan_root_dependency_info(
      plan.get(), add_seven_root, 0, &seven_dependency));
  loomc_program_plan_dependency_info_t eleven_dependency = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_DEPENDENCY_INFO,
      /*.structure_size=*/sizeof(eleven_dependency),
  };
  LOOMC_ASSERT_OK(loomc_program_plan_root_dependency_info(
      plan.get(), add_eleven_root, 0, &eleven_dependency));
  EXPECT_EQ(seven_dependency.slot, 0u);
  EXPECT_EQ(eleven_dependency.slot, 0u);
  ASSERT_NE(seven_dependency.unit.value, eleven_dependency.unit.value);

  std::array<ProgramPtr, 3> unit_programs;
  for (uint32_t i = 0; i < unit_programs.size(); ++i) {
    WorkspacePtr worker_workspace = CreateWorkspace();
    loomc_program_t* raw_program = nullptr;
    loomc_result_t* raw_result = nullptr;
    LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
        plan.get(), worker_workspace.get(),
        loomc_program_plan_unit_from_index(i), /*options=*/nullptr,
        loomc_allocator_system(), &raw_program, &raw_result));
    unit_programs[i].reset(raw_program);
    ResultPtr result(raw_result);
    if (!loomc_result_succeeded(result.get())) {
      for (loomc_host_size_t j = 0;
           j < loomc_result_diagnostic_count(result.get()); ++j) {
        const loomc_diagnostic_t* diagnostic =
            loomc_result_diagnostic_at(result.get(), j);
        ADD_FAILURE() << "unit " << i << " "
                      << std::string(diagnostic->code.data,
                                     diagnostic->code.size)
                      << ": "
                      << std::string(diagnostic->message.data,
                                     diagnostic->message.size);
      }
    }
    ASSERT_TRUE(loomc_result_succeeded(result.get()));
  }

  std::vector<OwnedArtifact> cached_root_artifacts =
      CopyArtifacts(unit_programs[0].get());
  std::vector<loomc_artifact_t> cached_root_views =
      MakeArtifactViews(&cached_root_artifacts);
  unit_programs[0].reset();
  loomc_program_t* raw_loaded_root_program = nullptr;
  loomc_result_t* raw_root_load_result = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_load_unit_program(
      plan.get(), loomc_program_plan_unit_from_index(0),
      cached_root_views.data(), cached_root_views.size(),
      loomc_allocator_system(), &raw_loaded_root_program,
      &raw_root_load_result));
  unit_programs[0].reset(raw_loaded_root_program);
  ResultPtr root_load_result(raw_root_load_result);
  ASSERT_TRUE(loomc_result_succeeded(root_load_result.get()));
  cached_root_views.clear();
  cached_root_artifacts.clear();
  root_load_result.reset();

  const uint32_t reloaded_unit_index =
      static_cast<uint32_t>(seven_dependency.unit.value);
  ASSERT_EQ(
      loomc_program_artifact_count(unit_programs[reloaded_unit_index].get()),
      1u);
  const loomc_artifact_t* compiled_artifact =
      loomc_program_artifact_at(unit_programs[reloaded_unit_index].get(), 0);
  ASSERT_NE(compiled_artifact, nullptr);
  EXPECT_EQ(compiled_artifact->kind, LOOMC_ARTIFACT_KIND_EXECUTABLE);
  EXPECT_TRUE(loomc_string_view_equal(
      compiled_artifact->format,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV)));
  std::vector<uint8_t> cached_contents(
      compiled_artifact->contents.data,
      compiled_artifact->contents.data +
          compiled_artifact->contents.data_length);
  loomc_program_plan_unit_info_t reloaded_unit_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_INFO,
      /*.structure_size=*/sizeof(reloaded_unit_info),
  };
  LOOMC_ASSERT_OK(loomc_program_plan_unit_info(
      plan.get(), seven_dependency.unit, &reloaded_unit_info));
  const loomc_artifact_t cached_artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_EXECUTABLE,
      /*.format=*/loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
      /*.identifier=*/reloaded_unit_info.identifier,
      /*.contents=*/
      loomc_make_byte_span(cached_contents.data(), cached_contents.size()),
  };
  unit_programs[reloaded_unit_index].reset();
  loomc_program_t* raw_loaded_program = nullptr;
  loomc_result_t* raw_load_result = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_load_unit_program(
      plan.get(), seven_dependency.unit, &cached_artifact, 1,
      loomc_allocator_system(), &raw_loaded_program, &raw_load_result));
  unit_programs[reloaded_unit_index].reset(raw_loaded_program);
  ResultPtr load_result(raw_load_result);
  ASSERT_TRUE(loomc_result_succeeded(load_result.get()));
  cached_contents.clear();
  load_result.reset();

  const std::array<loomc_program_t*, 3> unit_table_values = {
      unit_programs[0].get(),
      unit_programs[1].get(),
      unit_programs[2].get(),
  };
  const loomc_program_plan_unit_table_t unit_table = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_TABLE,
      /*.structure_size=*/sizeof(unit_table),
      /*.next=*/nullptr,
      /*.programs=*/unit_table_values.data(),
      /*.program_count=*/unit_table_values.size(),
  };
  const loomc_program_plan_root_t selected_roots[] = {
      add_eleven_root,
      add_seven_root,
  };
  loomc_program_t* raw_assembled_program = nullptr;
  loomc_result_t* raw_assemble_result = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_assemble(
      plan.get(), coordinator_workspace.get(), selected_roots,
      IREE_ARRAYSIZE(selected_roots), &unit_table, /*options=*/nullptr,
      loomc_allocator_system(), &raw_assembled_program, &raw_assemble_result));
  ProgramPtr assembled_program(raw_assembled_program);
  ResultPtr assemble_result(raw_assemble_result);
  ASSERT_TRUE(loomc_result_succeeded(assemble_result.get()));

  plan.reset();
  assemble_result.reset();
  for (ProgramPtr& unit_program : unit_programs) unit_program.reset();
  unit_pass_program.reset();
  preparation_pass_program.reset();
  compiler.reset();
  coordinator_workspace.reset();
  context.reset();

  ASSERT_EQ(loomc_program_export_count(assembled_program.get()), 2u);
  ASSERT_EQ(loomc_program_dependency_count(assembled_program.get()), 2u);
  for (uint32_t i = 0; i < 2; ++i) {
    loomc_program_dependency_info_t dependency_info = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_DEPENDENCY_INFO,
        /*.structure_size=*/sizeof(dependency_info),
    };
    LOOMC_ASSERT_OK(loomc_program_dependency_info(assembled_program.get(), i,
                                                  &dependency_info));
    EXPECT_EQ(dependency_info.slot, i);
    ASSERT_NE(dependency_info.program, nullptr);
  }

  const loomc_artifact_t* eleven_artifact =
      FindArtifact(assembled_program.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM, "add_eleven_once");
  ASSERT_NE(eleven_artifact, nullptr);
  loom_cmd_program_t eleven_program = {};
  const loom_cmd_program_command_t eleven_command =
      ParseOnlyCommand(eleven_artifact, &eleven_program);
  ASSERT_EQ(eleven_command.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  EXPECT_EQ(eleven_program.requirements.executable_count, 2u);
  EXPECT_EQ(eleven_program.requirements.entry_count, 2u);
  EXPECT_EQ(eleven_command.payload.dispatch_indirect.executable_index, 0u);
  EXPECT_EQ(eleven_command.payload.dispatch_indirect.entry_index, 0u);

  const loomc_artifact_t* seven_artifact =
      FindArtifact(assembled_program.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM, "add_seven_once");
  ASSERT_NE(seven_artifact, nullptr);
  loom_cmd_program_t seven_program = {};
  const loom_cmd_program_command_t seven_command =
      ParseOnlyCommand(seven_artifact, &seven_program);
  ASSERT_EQ(seven_command.kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
  EXPECT_EQ(seven_program.requirements.executable_count, 2u);
  EXPECT_EQ(seven_program.requirements.entry_count, 2u);
  EXPECT_EQ(seven_command.payload.dispatch_indirect.executable_index, 1u);
  EXPECT_EQ(seven_command.payload.dispatch_indirect.entry_index, 1u);

  const loomc_artifact_t* launch_artifact = FindArtifact(
      assembled_program.get(), LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE, "command-program-launch.loombc");
  ASSERT_NE(launch_artifact, nullptr);
  loomc_launch_config_module_t* raw_launch_module = nullptr;
  LOOMC_ASSERT_OK(loomc_launch_config_module_load(
      launch_artifact, /*options=*/nullptr, loomc_allocator_system(),
      &raw_launch_module));
  LaunchModulePtr launch_module(raw_launch_module);
  loomc_launch_config_function_t launch_function =
      loomc_launch_config_function_invalid();
  LOOMC_ASSERT_OK(loomc_launch_config_module_lookup_function_by_name(
      launch_module.get(), loomc_make_cstring_view("add_eleven_once"),
      &launch_function));
  LOOMC_ASSERT_OK(loomc_launch_config_module_lookup_function_by_name(
      launch_module.get(), loomc_make_cstring_view("add_seven_once"),
      &launch_function));

  loomc_program_export_t eleven_export = loomc_program_export_invalid();
  LOOMC_ASSERT_OK(loomc_program_lookup_export(
      assembled_program.get(), loomc_make_cstring_view("add_eleven_once"),
      &eleven_export));
  loomc_cmd_program_t* raw_eleven_command_program = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_program_create_from_export(
      assembled_program.get(), eleven_export, loomc_allocator_system(),
      &raw_eleven_command_program));
  CmdProgramPtr eleven_command_program(raw_eleven_command_program);

  loomc_cmd_program_info_t command_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO,
      /*.structure_size=*/sizeof(command_info),
  };
  LOOMC_ASSERT_OK(
      loomc_cmd_program_info(eleven_command_program.get(), &command_info));
  EXPECT_TRUE(loomc_string_view_equal(
      command_info.name, loomc_make_cstring_view("add_eleven_once")));
  EXPECT_EQ(command_info.fixed_buffer_count, 0u);
  EXPECT_EQ(command_info.rebindable_binding_count, 3u);
  EXPECT_EQ(command_info.parameter_root_count, 0u);
  EXPECT_EQ(command_info.parameter_count, 0u);
  EXPECT_EQ(command_info.transient.binding_index,
            LOOMC_CMD_PROGRAM_BINDING_INVALID);
  EXPECT_EQ(command_info.launch_counts.binding_index, 2u);
  EXPECT_EQ(command_info.launch_counts.required_byte_length,
            sizeof(loomc_dimension3_t));

  loomc_cmd_program_parameter_root_info_t parameter_root_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      /*.structure_size=*/sizeof(parameter_root_info),
  };
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_cmd_program_parameter_root_info(eleven_command_program.get(), 0,
                                            &parameter_root_info));

  loomc_cmd_program_parameter_info_t parameter_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_INFO,
      /*.structure_size=*/sizeof(parameter_info),
  };
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_cmd_program_parameter_info(
                             eleven_command_program.get(), 0, &parameter_info));

  assembled_program.reset();
  std::memset(&command_info, 0, sizeof(command_info));
  LOOMC_ASSERT_OK(
      loomc_cmd_program_info(eleven_command_program.get(), &command_info));
  EXPECT_TRUE(loomc_string_view_equal(
      command_info.name, loomc_make_cstring_view("add_eleven_once")));
}

TEST(CmdProgramPlanTest, MapsParameterViewToOpaqueKernelBufferAbi) {
  ContextPtr context = CreateContext();
  WorkspacePtr coordinator_workspace = CreateWorkspace();
  SourcePtr source = CreateSource(R"(
spirv.target<vulkan1_3> @kernel_target {abi = hal_kernel}

target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def target(@kernel_target) @consume_parameter() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  %zero_offset = index.constant 0 : offset
  %zero_index = index.constant 0 : index
  %source_aligned = buffer.assume.alignment %source {minimum_alignment = 4} : buffer
  %target_aligned = buffer.assume.alignment %target {minimum_alignment = 4} : buffer
  %source_view = buffer.view %source_aligned[%zero_offset] : buffer -> view<128xi32, #dense>
  %target_view = buffer.view %target_aligned[%zero_offset] : buffer -> view<1xi32, #dense>
  %value = view.load %source_view[%zero_index] : view<128xi32, #dense> -> i32
  view.store %value, %target_view[%zero_index] : i32, view<1xi32, #dense>
  kernel.return
}

command.program.def public target(@command_target) @parameter_root() launch(%parameters: buffer, %target: buffer) {
  %source = command.parameter %parameters, "source_values"[] : view<128xi32, #dense>
  kernel.launch @consume_parameter[](%source, %target) : [](view<128xi32, #dense>, buffer)
  command.return
}
)");
  ModulePtr module = DeserializeModule(
      context.get(), coordinator_workspace.get(), source.get());
  PassProgramPtr preparation_pass_program =
      CreateEmptyPassProgram(context.get());
  PassProgramPtr unit_pass_program = CreateTargetPassProgram(context.get());
  CompilerPtr compiler = CreateCompiler(context.get());
  const loomc_cmd_program_plan_options_t command_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(command_options),
      /*.next=*/nullptr,
      /*.dependency_artifact_format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
  };
  const loomc_program_plan_options_t plan_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(plan_options),
      /*.next=*/&command_options,
  };

  loomc_program_plan_t* raw_plan = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_prepare_programs(
      compiler.get(), coordinator_workspace.get(),
      preparation_pass_program.get(), unit_pass_program.get(), module.get(),
      &plan_options, loomc_allocator_system(), &raw_plan, &raw_result));
  PlanPtr plan(raw_plan);
  ResultPtr result(raw_result);
  ASSERT_TRUE(loomc_result_succeeded(result.get()));
  result.reset();
  ASSERT_EQ(loomc_program_plan_root_count(plan.get()), 1u);
  ASSERT_EQ(loomc_program_plan_unit_count(plan.get()), 2u);

  WorkspacePtr worker_workspace = CreateWorkspace();
  loomc_program_t* raw_root_program = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
      plan.get(), worker_workspace.get(), loomc_program_plan_unit_from_index(0),
      /*options=*/nullptr, loomc_allocator_system(), &raw_root_program,
      &raw_result));
  ProgramPtr root_program(raw_root_program);
  result.reset(raw_result);
  ASSERT_TRUE(loomc_result_succeeded(result.get()));

  loomc_program_export_t root_export = loomc_program_export_invalid();
  LOOMC_ASSERT_OK(loomc_program_lookup_export(
      root_program.get(), loomc_make_cstring_view("parameter_root"),
      &root_export));
  loomc_cmd_program_t* raw_command_program = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_program_create_from_export(
      root_program.get(), root_export, loomc_allocator_system(),
      &raw_command_program));
  CmdProgramPtr command_program(raw_command_program);
  root_program.reset();

  loomc_cmd_program_info_t command_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO,
      /*.structure_size=*/sizeof(command_info),
  };
  LOOMC_ASSERT_OK(loomc_cmd_program_info(command_program.get(), &command_info));
  EXPECT_EQ(command_info.fixed_buffer_count, 1u);
  EXPECT_EQ(command_info.rebindable_binding_count, 1u);
  EXPECT_EQ(command_info.parameter_root_count, 1u);
  EXPECT_EQ(command_info.parameter_count, 1u);
  EXPECT_EQ(command_info.launch_counts.binding_index,
            LOOMC_CMD_PROGRAM_BINDING_INVALID);

  loomc_cmd_program_parameter_root_info_t root_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      /*.structure_size=*/sizeof(root_info),
  };
  LOOMC_ASSERT_OK(loomc_cmd_program_parameter_root_info(command_program.get(),
                                                        0, &root_info));
  EXPECT_EQ(root_info.fixed_buffer_index, 0u);
  EXPECT_EQ(root_info.required_byte_length, 512u);
  EXPECT_EQ(root_info.minimum_alignment, 256u);

  loomc_cmd_program_parameter_info_t parameter_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_INFO,
      /*.structure_size=*/sizeof(parameter_info),
  };
  LOOMC_ASSERT_OK(loomc_cmd_program_parameter_info(command_program.get(), 0,
                                                   &parameter_info));
  EXPECT_TRUE(loomc_string_view_equal(
      parameter_info.key, loomc_make_cstring_view("source_values")));
  EXPECT_EQ(parameter_info.fixed_buffer_index, 0u);
  EXPECT_EQ(parameter_info.byte_offset, 0u);
  EXPECT_EQ(parameter_info.byte_length, 512u);
  EXPECT_EQ(parameter_info.minimum_alignment, 256u);

  loomc_program_t* raw_dependency_program = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
      plan.get(), worker_workspace.get(), loomc_program_plan_unit_from_index(1),
      /*options=*/nullptr, loomc_allocator_system(), &raw_dependency_program,
      &raw_result));
  ProgramPtr dependency_program(raw_dependency_program);
  result.reset(raw_result);
  if (!loomc_result_succeeded(result.get())) {
    for (loomc_host_size_t i = 0;
         i < loomc_result_diagnostic_count(result.get()); ++i) {
      const loomc_diagnostic_t* diagnostic =
          loomc_result_diagnostic_at(result.get(), i);
      ADD_FAILURE() << std::string(diagnostic->code.data, diagnostic->code.size)
                    << ": "
                    << std::string(diagnostic->message.data,
                                   diagnostic->message.size);
    }
  }
  ASSERT_TRUE(loomc_result_succeeded(result.get()));
  EXPECT_EQ(loomc_program_artifact_count(dependency_program.get()), 1u);
}

TEST(CmdProgramPlanTest, LowersModelShapedScheduleAfterPreparation) {
  ContextPtr context = CreateContext();
  WorkspacePtr coordinator_workspace = CreateWorkspace();
  SourcePtr source = CreateSource(R"(
spirv.target<vulkan1_3> @kernel_target {abi = hal_kernel}

target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def target(@kernel_target) @stage(%element_count: index) {
  %one = index.constant 1 : index
  %bounded_count = index.assume %element_count [range(%element_count, 1, 512)] : index
  kernel.launch.config workgroups(%bounded_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%left: view<1xi32, #dense>, %right: view<1xi32, #dense>, %input: buffer, %scratch: buffer, %output: buffer) {
  kernel.return
}

command.program.def public target(@command_target) @decode(%element_count: index) launch(%attention_parameters: buffer, %expert_parameters: buffer, %input: buffer, %output: buffer) where [range(%element_count, 1, 512)] {
  %layer_begin = index.constant 0 : index
  %layer_end = index.constant 48 : index
  %layer_step = index.constant 1 : index
  %scratch_bytes = index.constant 256 : offset
  %scratch = buffer.alloca %scratch_bytes {base_alignment = 256, memory_space = global} : buffer

  %token_embedding = command.parameter %attention_parameters, "token_embd.weight"[] : view<1xi32, #dense>
  %output_norm = command.parameter %attention_parameters, "output_norm.weight"[] : view<1xi32, #dense>
  %output_projection = command.parameter %attention_parameters, "output.weight"[] : view<1xi32, #dense>
  kernel.launch @stage[%element_count](%token_embedding, %token_embedding, %input, %scratch, %output) : [index](view<1xi32, #dense>, view<1xi32, #dense>, buffer, buffer, buffer)
  kernel.launch @stage[%element_count](%output_norm, %output_norm, %input, %scratch, %output) : [index](view<1xi32, #dense>, view<1xi32, #dense>, buffer, buffer, buffer)
  kernel.launch @stage[%element_count](%output_projection, %output_projection, %input, %scratch, %output) : [index](view<1xi32, #dense>, view<1xi32, #dense>, buffer, buffer, buffer)

  scf.for %layer = [%layer_begin to %layer_end step %layer_step] unroll {
    %attention_norm = command.parameter %attention_parameters, "blk.{}.attn_norm.weight"[%layer] : view<1xi32, #dense>
    %query = command.parameter %attention_parameters, "blk.{}.attn_q.weight"[%layer] : view<1xi32, #dense>
    %key = command.parameter %attention_parameters, "blk.{}.attn_k.weight"[%layer] : view<1xi32, #dense>
    %value = command.parameter %attention_parameters, "blk.{}.attn_v.weight"[%layer] : view<1xi32, #dense>
    %attention_output = command.parameter %attention_parameters, "blk.{}.attn_output.weight"[%layer] : view<1xi32, #dense>
    %ffn_norm = command.parameter %attention_parameters, "blk.{}.ffn_norm.weight"[%layer] : view<1xi32, #dense>
    %router = command.parameter %expert_parameters, "blk.{}.ffn_gate_inp.weight"[%layer] : view<1xi32, #dense>
    %gate = command.parameter %expert_parameters, "blk.{}.ffn_gate.weight"[%layer] : view<1xi32, #dense>
    %up = command.parameter %expert_parameters, "blk.{}.ffn_up.weight"[%layer] : view<1xi32, #dense>
    %down = command.parameter %expert_parameters, "blk.{}.ffn_down.weight"[%layer] : view<1xi32, #dense>
    %expert_scale = command.parameter %expert_parameters, "blk.{}.ffn_gate_exps.weight"[%layer] : view<1xi32, #dense>
    %shared_scale = command.parameter %expert_parameters, "blk.{}.ffn_gate_shexp.weight"[%layer] : view<1xi32, #dense>
    command.concurrent {
      kernel.launch @stage[%element_count](%attention_norm, %query, %input, %scratch, %output) : [index](view<1xi32, #dense>, view<1xi32, #dense>, buffer, buffer, buffer)
      kernel.launch @stage[%element_count](%key, %value, %input, %scratch, %output) : [index](view<1xi32, #dense>, view<1xi32, #dense>, buffer, buffer, buffer)
      kernel.launch @stage[%element_count](%attention_output, %ffn_norm, %input, %scratch, %output) : [index](view<1xi32, #dense>, view<1xi32, #dense>, buffer, buffer, buffer)
    }
    kernel.launch @stage[%element_count](%router, %gate, %input, %scratch, %output) : [index](view<1xi32, #dense>, view<1xi32, #dense>, buffer, buffer, buffer)
    kernel.launch @stage[%element_count](%up, %down, %input, %scratch, %output) : [index](view<1xi32, #dense>, view<1xi32, #dense>, buffer, buffer, buffer)
    kernel.launch @stage[%element_count](%expert_scale, %shared_scale, %input, %scratch, %output) : [index](view<1xi32, #dense>, view<1xi32, #dense>, buffer, buffer, buffer)
    scf.yield
  }
  command.return
}
)");
  ModulePtr module = DeserializeModule(
      context.get(), coordinator_workspace.get(), source.get());
  PassProgramPtr preparation_pass_program =
      CreateCommandPreparationPassProgram(context.get());
  PassProgramPtr unit_pass_program = CreateTargetPassProgram(context.get());
  CompilerPtr compiler = CreateCompiler(context.get());
  const loomc_cmd_program_plan_options_t command_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(command_options),
      /*.next=*/nullptr,
      /*.dependency_artifact_format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_SPIRV),
  };
  const loomc_program_plan_options_t plan_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(plan_options),
      /*.next=*/&command_options,
  };

  loomc_program_plan_t* raw_plan = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_prepare_programs(
      compiler.get(), coordinator_workspace.get(),
      preparation_pass_program.get(), unit_pass_program.get(), module.get(),
      &plan_options, loomc_allocator_system(), &raw_plan, &raw_result));
  PlanPtr plan(raw_plan);
  ResultPtr result(raw_result);
  ASSERT_TRUE(loomc_result_succeeded(result.get()));
  ASSERT_EQ(loomc_program_plan_root_count(plan.get()), 1u);
  ASSERT_EQ(loomc_program_plan_unit_count(plan.get()), 2u);

  WorkspacePtr worker_workspace = CreateWorkspace();
  loomc_program_t* raw_root_program = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
      plan.get(), worker_workspace.get(), loomc_program_plan_unit_from_index(0),
      /*options=*/nullptr, loomc_allocator_system(), &raw_root_program,
      &raw_result));
  ProgramPtr root_program(raw_root_program);
  result.reset(raw_result);
  ASSERT_TRUE(loomc_result_succeeded(result.get()));

  loomc_program_export_t root_export = loomc_program_export_invalid();
  LOOMC_ASSERT_OK(loomc_program_lookup_export(
      root_program.get(), loomc_make_cstring_view("decode"), &root_export));
  loomc_cmd_program_t* raw_command_program = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_program_create_from_export(
      root_program.get(), root_export, loomc_allocator_system(),
      &raw_command_program));
  CmdProgramPtr command_program(raw_command_program);

  loomc_cmd_program_info_t info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO,
      /*.structure_size=*/sizeof(info),
  };
  LOOMC_ASSERT_OK(loomc_cmd_program_info(command_program.get(), &info));
  EXPECT_EQ(info.fixed_buffer_count, 2u);
  EXPECT_EQ(info.rebindable_binding_count, 4u);
  EXPECT_EQ(info.parameter_root_count, 2u);
  EXPECT_EQ(info.parameter_count, 579u);
  EXPECT_EQ(info.transient.binding_index, 2u);
  EXPECT_EQ(info.transient.required_byte_length, 256u);
  EXPECT_EQ(info.launch_counts.binding_index, 3u);
  EXPECT_EQ(info.launch_counts.required_byte_length,
            sizeof(loomc_dimension3_t));

  loomc_cmd_program_parameter_root_info_t attention_root = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      /*.structure_size=*/sizeof(attention_root),
  };
  LOOMC_ASSERT_OK(loomc_cmd_program_parameter_root_info(command_program.get(),
                                                        0, &attention_root));
  EXPECT_EQ(attention_root.fixed_buffer_index, 0u);
  EXPECT_EQ(attention_root.required_byte_length, 74244u);
  EXPECT_EQ(attention_root.minimum_alignment, 256u);
  loomc_cmd_program_parameter_root_info_t expert_root = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      /*.structure_size=*/sizeof(expert_root),
  };
  LOOMC_ASSERT_OK(loomc_cmd_program_parameter_root_info(command_program.get(),
                                                        1, &expert_root));
  EXPECT_EQ(expert_root.fixed_buffer_index, 1u);
  EXPECT_EQ(expert_root.required_byte_length, 73476u);
  EXPECT_EQ(expert_root.minimum_alignment, 256u);

  loomc_cmd_program_parameter_info_t final_parameter = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_INFO,
      /*.structure_size=*/sizeof(final_parameter),
  };
  LOOMC_ASSERT_OK(loomc_cmd_program_parameter_info(command_program.get(), 578,
                                                   &final_parameter));
  EXPECT_TRUE(loomc_string_view_equal(
      final_parameter.key,
      loomc_make_cstring_view("blk.47.ffn_gate_shexp.weight")));
  EXPECT_EQ(final_parameter.fixed_buffer_index, 1u);

  const loomc_artifact_t* artifact =
      FindArtifact(root_program.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                   LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM, "decode");
  ASSERT_NE(artifact, nullptr);
  loom_cmd_program_t parsed = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(artifact->contents.data,
                                artifact->contents.data_length),
      &parsed));
  uint32_t dispatch_count = 0;
  for (uint32_t i = 0; i < parsed.commands.count; ++i) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(&parsed, i);
    if (command.kind ==
        LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC) {
      ++dispatch_count;
    }
  }
  EXPECT_EQ(dispatch_count, 291u);
}

}  // namespace
