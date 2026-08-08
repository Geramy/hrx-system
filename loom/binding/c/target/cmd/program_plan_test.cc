// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/target/cmd/program_plan.h"

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
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using LaunchModulePtr =
    HandlePtr<loomc_launch_config_module_t, loomc_launch_config_module_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using PlanPtr = HandlePtr<loomc_program_plan_t, loomc_program_plan_release>;
using ProgramPtr = HandlePtr<loomc_program_t, loomc_program_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
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
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(loomc_context_create(
      /*options=*/nullptr, loomc_allocator_system(), &context));
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
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def @add_seven(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %seven = scalar.constant 7 : i32
  %source_view = buffer.view %source[%base] : buffer -> view<128xi32, #dense>
  %target_view = buffer.view %target[%base] : buffer -> view<128xi32, #dense>
  %value = view.load %source_view[%workgroup] : view<128xi32, #dense> -> i32
  %result = scalar.addi %value, %seven : i32
  view.store %result, %target_view[%workgroup] : i32, view<128xi32, #dense>
  kernel.return
}

kernel.def @add_eleven(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer, %target: buffer) {
  %base = index.constant 0 : offset
  %workgroup = kernel.workgroup.id<x> : index
  %eleven = scalar.constant 11 : i32
  %source_view = buffer.view %source[%base] : buffer -> view<128xi32, #dense>
  %target_view = buffer.view %target[%base] : buffer -> view<128xi32, #dense>
  %value = view.load %source_view[%workgroup] : view<128xi32, #dense> -> i32
  %result = scalar.addi %value, %eleven : i32
  view.store %result, %target_view[%workgroup] : i32, view<128xi32, #dense>
  kernel.return
}

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
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr pass_program = CreateEmptyPassProgram(context.get());

  loomc_program_plan_t* raw_plan = nullptr;
  loomc_result_t* raw_prepare_result = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_program_plan_prepare_module(
      compiler.get(), coordinator_workspace.get(), pass_program.get(),
      module.get(), loomc_allocator_system(), &raw_plan, &raw_prepare_result));
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
      /*.kind=*/LOOMC_ARTIFACT_KIND_MODULE,
      /*.format=*/loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
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
  pass_program.reset();
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
}

}  // namespace
