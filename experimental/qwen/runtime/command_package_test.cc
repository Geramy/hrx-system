// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "experimental/qwen/programs/command_package_source.h"
#include "experimental/qwen/programs/decode_source.h"
#include "experimental/qwen/programs/dense_projection_specialization_test_source.h"
#include "experimental/qwen/programs/prefill_source.h"
#include "iree/testing/gtest.h"
#include "loomc/artifact.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/program.h"
#include "loomc/program_plan.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using CmdProgramPtr = HandlePtr<loomc_cmd_program_t, loomc_cmd_program_release>;
using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
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
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

static std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

static bool ResultSucceeded(const loomc_result_t* result,
                            const std::string& operation) {
  if (result && loomc_result_succeeded(result)) return true;
  if (!result) {
    ADD_FAILURE() << operation << " returned no result";
    return false;
  }
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    ADD_FAILURE() << operation << " " << ToString(diagnostic->code) << ": "
                  << ToString(diagnostic->message);
  }
  return false;
}

static TargetEnvironmentPtr CreateTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &target_environment));
  return TargetEnvironmentPtr(target_environment);
}

static TargetProfilePtr CreateGfx1100TargetProfile(
    loomc_target_environment_t* target_environment) {
  const loomc_amdgpu_profile_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("qwen-test-gfx1100"),
      /*.identity=*/
      {
          /*.target=*/loomc_make_cstring_view("gfx1100"),
      },
  };
  loomc_target_profile_t* target_profile = nullptr;
  LOOMC_EXPECT_OK(loomc_target_profile_create_amdgpu(
      target_environment, &options, loomc_allocator_system(), &target_profile));
  EXPECT_NE(target_profile, nullptr);
  return TargetProfilePtr(target_profile);
}

static std::vector<loomc_target_specialization_t> CreateKernelSpecializations(
    const loomc_module_t* module, loomc_target_profile_t* target_profile) {
  const loomc_module_function_query_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_MODULE_FUNCTION_QUERY_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.function_symbol=*/loomc_string_view_empty(),
      /*.kind=*/LOOMC_MODULE_FUNCTION_KIND_KERNEL,
  };
  loomc_host_size_t function_count = 0;
  loomc_result_t* raw_result = nullptr;
  loomc_status_t status =
      loomc_module_query_functions(module, &options, loomc_allocator_system(),
                                   0, nullptr, &function_count, &raw_result);
  LOOMC_EXPECT_OK(status);
  ResultPtr count_result(raw_result);
  if (!loomc_status_is_ok(status) ||
      !ResultSucceeded(count_result.get(), "kernel count query")) {
    return {};
  }

  std::vector<loomc_module_function_t> functions(function_count);
  raw_result = nullptr;
  status = loomc_module_query_functions(
      module, &options, loomc_allocator_system(), functions.size(),
      functions.data(), &function_count, &raw_result);
  LOOMC_EXPECT_OK(status);
  ResultPtr query_result(raw_result);
  if (!loomc_status_is_ok(status) ||
      !ResultSucceeded(query_result.get(), "kernel query")) {
    return {};
  }
  functions.resize(function_count);

  std::vector<loomc_target_specialization_t> specializations;
  specializations.reserve(functions.size());
  for (const loomc_module_function_t& function : functions) {
    specializations.push_back({
        /*.function_symbol=*/function.symbol_name,
        /*.target_profile=*/target_profile,
    });
  }
  return specializations;
}

static ContextPtr CreateContext(
    loomc_target_environment_t* target_environment) {
  const loomc_context_target_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment,
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

static SourcePtr CreateEmbeddedSource(const iree_file_toc_t* files,
                                      size_t file_count) {
  EXPECT_EQ(file_count, 1u);
  const loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/loomc_make_cstring_view(files[0].name),
      /*.contents=*/loomc_make_byte_span(files[0].data, files[0].size),
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&options, loomc_allocator_system(), &source));
  return SourcePtr(source);
}

static SourcePtr CreateDecodeSource() {
  return CreateEmbeddedSource(qwen_loom_decode_program_source_create(),
                              qwen_loom_decode_program_source_size());
}

static SourcePtr CreateCommandPackageSource() {
  return CreateEmbeddedSource(qwen_loom_command_package_source_create(),
                              qwen_loom_command_package_source_size());
}

static SourcePtr CreatePrefillSource() {
  return CreateEmbeddedSource(qwen_loom_prefill_program_source_create(),
                              qwen_loom_prefill_program_source_size());
}

static SourcePtr CreateDenseProjectionSpecializationSource() {
  return CreateEmbeddedSource(
      qwen_loom_dense_projection_specialization_test_source_create(),
      qwen_loom_dense_projection_specialization_test_source_size());
}

static ModulePtr DeserializeModule(loomc_context_t* context,
                                   loomc_workspace_t* workspace,
                                   const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
      context, workspace, source, /*options=*/nullptr, loomc_allocator_system(),
      &module, &raw_result));
  ResultPtr result(raw_result);
  EXPECT_TRUE(ResultSucceeded(result.get(), "decode archive parse"));
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

static PassProgramPtr CreatePreparationPassProgram(loomc_context_t* context) {
  const loomc_target_pipeline_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("qwen-decode-expanded-source"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_EXPANDED_SOURCE,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/20,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_from_target_pipeline(
      context, &options, loomc_allocator_system(), &pass_program, &raw_result));
  ResultPtr result(raw_result);
  EXPECT_TRUE(ResultSucceeded(result.get(), "preparation pipeline parse"));
  return PassProgramPtr(pass_program);
}

static PassProgramPtr CreateUnitPassProgram(loomc_context_t* context) {
  const loomc_target_pipeline_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("qwen-decode-prepared-low"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/20,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_from_target_pipeline(
      context, &options, loomc_allocator_system(), &pass_program, &raw_result));
  ResultPtr result(raw_result);
  EXPECT_TRUE(ResultSucceeded(result.get(), "AMDGPU unit pipeline creation"));
  return PassProgramPtr(pass_program);
}

static const loomc_artifact_t* FindCommandArtifact(
    const loomc_program_t* program, const char* identifier) {
  for (loomc_host_size_t i = 0; i < loomc_program_artifact_count(program);
       ++i) {
    const loomc_artifact_t* artifact = loomc_program_artifact_at(program, i);
    if (artifact->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE &&
        loomc_string_view_equal(
            artifact->format,
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM)) &&
        loomc_string_view_equal(artifact->identifier,
                                loomc_make_cstring_view(identifier))) {
      return artifact;
    }
  }
  return nullptr;
}

struct PrefillShape {
  // Exact public command-program export.
  const char* export_name;
  // Number of independently compiled dependencies selected by the root.
  loomc_host_size_t dependency_count;
  // Packed transient backing-store requirement.
  uint64_t transient_byte_length;
};

static constexpr PrefillShape kPrefillShapes[] = {
    {"qwen3_30b_prefill_32", 28, 2490368},
    {"qwen3_30b_prefill_64", 28, 4980736},
    {"qwen3_30b_prefill_128", 28, 9961472},
    {"qwen3_30b_prefill_256", 28, 19922944},
    {"qwen3_30b_prefill_512", 27, 39846144},
};

TEST(QwenCommandPackageTest, SpecializesExactDenseProjectionShapes) {
  TargetEnvironmentPtr target_environment = CreateTargetEnvironment();
  ContextPtr context = CreateContext(target_environment.get());
  WorkspacePtr coordinator_workspace = CreateWorkspace();
  SourcePtr source = CreateDenseProjectionSpecializationSource();
  ModulePtr module = DeserializeModule(
      context.get(), coordinator_workspace.get(), source.get());
  TargetProfilePtr target_profile =
      CreateGfx1100TargetProfile(target_environment.get());
  std::vector<loomc_target_specialization_t> specializations =
      CreateKernelSpecializations(module.get(), target_profile.get());
  ASSERT_FALSE(specializations.empty());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr preparation_pass_program =
      CreatePreparationPassProgram(context.get());
  PassProgramPtr unit_pass_program = CreateUnitPassProgram(context.get());

  const loomc_cmd_program_plan_options_t command_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(command_options),
      /*.next=*/nullptr,
      /*.dependency_artifact_format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
  };
  const loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/specializations.data(),
      /*.specialization_count=*/specializations.size(),
  };
  const loomc_program_plan_options_t plan_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(plan_options),
      /*.next=*/&command_options,
      /*.config=*/
      {
          /*.bindings=*/nullptr,
          /*.binding_count=*/0,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
      /*.target_specialization=*/&target_options,
  };
  loomc_program_plan_t* raw_plan = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_prepare_programs(
      compiler.get(), coordinator_workspace.get(),
      preparation_pass_program.get(), unit_pass_program.get(), module.get(),
      &plan_options, loomc_allocator_system(), &raw_plan, &raw_result));
  PlanPtr plan(raw_plan);
  ResultPtr prepare_result(raw_result);
  ASSERT_TRUE(ResultSucceeded(prepare_result.get(),
                              "dense projection plan preparation"));

  ASSERT_EQ(loomc_program_plan_root_count(plan.get()), 1u);
  ASSERT_EQ(loomc_program_plan_unit_count(plan.get()), 5u);
  loomc_program_plan_root_t root = loomc_program_plan_root_invalid();
  LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
      plan.get(),
      loomc_make_cstring_view("qwen3_moe_dense_projection_specialization"),
      &root));
  loomc_program_plan_root_info_t root_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_ROOT_INFO,
      /*.structure_size=*/sizeof(root_info),
  };
  LOOMC_ASSERT_OK(loomc_program_plan_root_info(plan.get(), root, &root_info));
  EXPECT_EQ(root_info.dependency_count, 4u);

  std::vector<uint32_t> dependency_units;
  for (loomc_host_size_t i = 0; i < root_info.dependency_count; ++i) {
    loomc_program_plan_dependency_info_t dependency_info = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_DEPENDENCY_INFO,
        /*.structure_size=*/sizeof(dependency_info),
    };
    LOOMC_ASSERT_OK(loomc_program_plan_root_dependency_info(plan.get(), root, i,
                                                            &dependency_info));
    dependency_units.push_back(dependency_info.unit.value);
  }
  std::sort(dependency_units.begin(), dependency_units.end());
  EXPECT_EQ(std::unique(dependency_units.begin(), dependency_units.end()),
            dependency_units.end());

  for (loomc_host_size_t i = 0; i < loomc_program_plan_unit_count(plan.get());
       ++i) {
    WorkspacePtr worker_workspace = CreateWorkspace();
    loomc_program_t* raw_program = nullptr;
    raw_result = nullptr;
    LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
        plan.get(), worker_workspace.get(),
        loomc_program_plan_unit_from_index(static_cast<uint32_t>(i)),
        /*options=*/nullptr, loomc_allocator_system(), &raw_program,
        &raw_result));
    ProgramPtr program(raw_program);
    ResultPtr result(raw_result);
    ASSERT_TRUE(ResultSucceeded(result.get(),
                                "dense projection unit " + std::to_string(i)));
  }
}

TEST(QwenCommandPackageTest, CompilesCompleteProductionPlan) {
  TargetEnvironmentPtr target_environment = CreateTargetEnvironment();
  ContextPtr context = CreateContext(target_environment.get());
  WorkspacePtr coordinator_workspace = CreateWorkspace();
  SourcePtr source = CreateDecodeSource();
  ModulePtr module = DeserializeModule(
      context.get(), coordinator_workspace.get(), source.get());
  TargetProfilePtr target_profile =
      CreateGfx1100TargetProfile(target_environment.get());
  std::vector<loomc_target_specialization_t> specializations =
      CreateKernelSpecializations(module.get(), target_profile.get());
  ASSERT_FALSE(specializations.empty());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr preparation_pass_program =
      CreatePreparationPassProgram(context.get());
  PassProgramPtr unit_pass_program = CreateUnitPassProgram(context.get());

  const loomc_cmd_program_plan_options_t command_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(command_options),
      /*.next=*/nullptr,
      /*.dependency_artifact_format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
  };
  const loomc_config_binding_t config_bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("qwen3_30b.request.token_capacity"),
          /*.value=*/loomc_make_cstring_view("512"),
      },
  };
  const loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/specializations.data(),
      /*.specialization_count=*/specializations.size(),
  };
  const loomc_program_plan_options_t plan_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(plan_options),
      /*.next=*/&command_options,
      /*.config=*/
      {
          /*.bindings=*/config_bindings,
          /*.binding_count=*/std::size(config_bindings),
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
      /*.target_specialization=*/&target_options,
  };
  loomc_program_plan_t* raw_plan = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_prepare_programs(
      compiler.get(), coordinator_workspace.get(),
      preparation_pass_program.get(), unit_pass_program.get(), module.get(),
      &plan_options, loomc_allocator_system(), &raw_plan, &raw_result));
  PlanPtr plan(raw_plan);
  ResultPtr prepare_result(raw_result);
  ASSERT_TRUE(ResultSucceeded(prepare_result.get(), "decode plan preparation"));

  ASSERT_EQ(loomc_program_plan_root_count(plan.get()), 1u);
  ASSERT_EQ(loomc_program_plan_unit_count(plan.get()), 15u);
  loomc_program_plan_root_t root = loomc_program_plan_root_invalid();
  LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
      plan.get(), loomc_make_cstring_view("qwen3_30b_decode_576"), &root));
  loomc_program_plan_root_info_t root_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_ROOT_INFO,
      /*.structure_size=*/sizeof(root_info),
  };
  LOOMC_ASSERT_OK(loomc_program_plan_root_info(plan.get(), root, &root_info));
  EXPECT_EQ(root_info.dependency_count, 14u);

  std::vector<ProgramPtr> unit_programs;
  std::vector<loomc_program_t*> unit_table_values;
  const loomc_host_size_t unit_count =
      loomc_program_plan_unit_count(plan.get());
  unit_programs.reserve(unit_count);
  unit_table_values.reserve(unit_count);
  for (loomc_host_size_t i = 0; i < unit_count; ++i) {
    const loomc_program_plan_unit_t unit =
        loomc_program_plan_unit_from_index(static_cast<uint32_t>(i));
    loomc_program_plan_unit_info_t unit_info = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_INFO,
        /*.structure_size=*/sizeof(unit_info),
    };
    LOOMC_ASSERT_OK(loomc_program_plan_unit_info(plan.get(), unit, &unit_info));
    WorkspacePtr worker_workspace = CreateWorkspace();
    loomc_program_t* raw_program = nullptr;
    raw_result = nullptr;
    LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
        plan.get(), worker_workspace.get(), unit,
        /*options=*/nullptr, loomc_allocator_system(), &raw_program,
        &raw_result));
    ProgramPtr program(raw_program);
    ResultPtr result(raw_result);
    ASSERT_TRUE(ResultSucceeded(result.get(),
                                "decode unit " + std::to_string(i) + " (" +
                                    ToString(unit_info.identifier) + ")"));
    unit_table_values.push_back(program.get());
    unit_programs.push_back(std::move(program));
  }

  const loomc_program_plan_unit_table_t unit_table = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_TABLE,
      /*.structure_size=*/sizeof(unit_table),
      /*.next=*/nullptr,
      /*.programs=*/unit_table_values.data(),
      /*.program_count=*/unit_table_values.size(),
  };
  loomc_program_t* raw_assembled_program = nullptr;
  loomc_result_t* raw_assemble_result = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_assemble(
      plan.get(), coordinator_workspace.get(), &root, 1, &unit_table,
      /*options=*/nullptr, loomc_allocator_system(), &raw_assembled_program,
      &raw_assemble_result));
  ProgramPtr assembled_program(raw_assembled_program);
  ResultPtr assemble_result(raw_assemble_result);
  ASSERT_TRUE(ResultSucceeded(assemble_result.get(), "decode plan assembly"));
  ASSERT_EQ(loomc_program_export_count(assembled_program.get()), 1u);
  ASSERT_EQ(loomc_program_dependency_count(assembled_program.get()), 14u);

  loomc_program_export_t root_export = loomc_program_export_invalid();
  LOOMC_ASSERT_OK(loomc_program_lookup_export(
      assembled_program.get(), loomc_make_cstring_view("qwen3_30b_decode_576"),
      &root_export));
  loomc_cmd_program_t* raw_command_program = nullptr;
  LOOMC_ASSERT_OK(loomc_cmd_program_create_from_export(
      assembled_program.get(), root_export, loomc_allocator_system(),
      &raw_command_program));
  CmdProgramPtr command_program(raw_command_program);
  loomc_cmd_program_info_t info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO,
      /*.structure_size=*/sizeof(info),
  };
  LOOMC_ASSERT_OK(loomc_cmd_program_info(command_program.get(), &info));
  EXPECT_EQ(info.fixed_buffer_count, 2u);
  EXPECT_EQ(info.rebindable_binding_count, 5u);
  EXPECT_EQ(info.parameter_root_count, 2u);
  EXPECT_EQ(info.parameter_count, 580u);
  EXPECT_EQ(info.transient.binding_index, 4u);
  EXPECT_EQ(info.transient.required_byte_length, 197376u);
  EXPECT_EQ(info.transient.minimum_alignment, 256u);
  EXPECT_EQ(info.launch_counts.binding_index,
            LOOMC_CMD_PROGRAM_BINDING_INVALID);

  loomc_cmd_program_parameter_root_info_t parameter_root = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      /*.structure_size=*/sizeof(parameter_root),
  };
  LOOMC_ASSERT_OK(loomc_cmd_program_parameter_root_info(command_program.get(),
                                                        0, &parameter_root));
  EXPECT_EQ(parameter_root.fixed_buffer_index, 0u);
  EXPECT_EQ(parameter_root.required_byte_length, 18550716416ull);
  EXPECT_EQ(parameter_root.minimum_alignment, 256u);
  LOOMC_ASSERT_OK(loomc_cmd_program_parameter_root_info(command_program.get(),
                                                        1, &parameter_root));
  EXPECT_EQ(parameter_root.fixed_buffer_index, 1u);
  EXPECT_EQ(parameter_root.required_byte_length, 256u);
  EXPECT_EQ(parameter_root.minimum_alignment, 256u);

  const loomc_artifact_t* command_artifact =
      FindCommandArtifact(assembled_program.get(), "qwen3_30b_decode_576");
  ASSERT_NE(command_artifact, nullptr);
  EXPECT_GT(command_artifact->contents.data_length, 0u);
}

TEST(QwenCommandPackageTest, PlansExactPrefillShapeFamily) {
  TargetEnvironmentPtr target_environment = CreateTargetEnvironment();
  ContextPtr context = CreateContext(target_environment.get());
  WorkspacePtr coordinator_workspace = CreateWorkspace();
  SourcePtr source = CreatePrefillSource();
  ModulePtr module = DeserializeModule(
      context.get(), coordinator_workspace.get(), source.get());
  TargetProfilePtr target_profile =
      CreateGfx1100TargetProfile(target_environment.get());
  std::vector<loomc_target_specialization_t> specializations =
      CreateKernelSpecializations(module.get(), target_profile.get());
  ASSERT_FALSE(specializations.empty());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr preparation_pass_program =
      CreatePreparationPassProgram(context.get());
  PassProgramPtr unit_pass_program = CreateUnitPassProgram(context.get());

  const loomc_cmd_program_plan_options_t command_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(command_options),
      /*.next=*/nullptr,
      /*.dependency_artifact_format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
  };
  const loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/specializations.data(),
      /*.specialization_count=*/specializations.size(),
  };
  const loomc_program_plan_options_t plan_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(plan_options),
      /*.next=*/&command_options,
      /*.config=*/
      {
          /*.bindings=*/nullptr,
          /*.binding_count=*/0,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
      /*.target_specialization=*/&target_options,
  };
  loomc_program_plan_t* raw_plan = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_prepare_programs(
      compiler.get(), coordinator_workspace.get(),
      preparation_pass_program.get(), unit_pass_program.get(), module.get(),
      &plan_options, loomc_allocator_system(), &raw_plan, &raw_result));
  PlanPtr plan(raw_plan);
  ResultPtr prepare_result(raw_result);
  ASSERT_TRUE(
      ResultSucceeded(prepare_result.get(), "prefill plan preparation"));

  ASSERT_EQ(loomc_program_plan_root_count(plan.get()),
            std::size(kPrefillShapes));
  for (const PrefillShape& shape : kPrefillShapes) {
    loomc_program_plan_root_t root = loomc_program_plan_root_invalid();
    LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
        plan.get(), loomc_make_cstring_view(shape.export_name), &root));
    loomc_program_plan_root_info_t root_info = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_ROOT_INFO,
        /*.structure_size=*/sizeof(root_info),
    };
    LOOMC_ASSERT_OK(loomc_program_plan_root_info(plan.get(), root, &root_info));
    EXPECT_EQ(root_info.dependency_count, shape.dependency_count)
        << shape.export_name;
  }

  EXPECT_EQ(loomc_program_plan_unit_count(plan.get()), 100u);
}

TEST(QwenCommandPackageTest, CompilesSharedPrefillAndDecodePlan) {
  TargetEnvironmentPtr target_environment = CreateTargetEnvironment();
  ContextPtr context = CreateContext(target_environment.get());
  WorkspacePtr coordinator_workspace = CreateWorkspace();
  SourcePtr source = CreateCommandPackageSource();
  ModulePtr module = DeserializeModule(
      context.get(), coordinator_workspace.get(), source.get());
  TargetProfilePtr target_profile =
      CreateGfx1100TargetProfile(target_environment.get());
  std::vector<loomc_target_specialization_t> specializations =
      CreateKernelSpecializations(module.get(), target_profile.get());
  ASSERT_FALSE(specializations.empty());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr preparation_pass_program =
      CreatePreparationPassProgram(context.get());
  PassProgramPtr unit_pass_program = CreateUnitPassProgram(context.get());

  const loomc_cmd_program_plan_options_t command_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(command_options),
      /*.next=*/nullptr,
      /*.dependency_artifact_format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
  };
  const loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/specializations.data(),
      /*.specialization_count=*/specializations.size(),
  };
  const loomc_program_plan_options_t plan_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS,
      /*.structure_size=*/sizeof(plan_options),
      /*.next=*/&command_options,
      /*.config=*/
      {
          /*.bindings=*/nullptr,
          /*.binding_count=*/0,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
      /*.target_specialization=*/&target_options,
  };
  loomc_program_plan_t* raw_plan = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_ASSERT_OK(loomc_prepare_programs(
      compiler.get(), coordinator_workspace.get(),
      preparation_pass_program.get(), unit_pass_program.get(), module.get(),
      &plan_options, loomc_allocator_system(), &raw_plan, &raw_result));
  PlanPtr plan(raw_plan);
  ResultPtr prepare_result(raw_result);
  ASSERT_TRUE(ResultSucceeded(prepare_result.get(),
                              "shared command-package preparation"));

  ASSERT_EQ(loomc_program_plan_root_count(plan.get()),
            std::size(kPrefillShapes) + 1);
  loomc_program_plan_root_t decode_root = loomc_program_plan_root_invalid();
  LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
      plan.get(), loomc_make_cstring_view("qwen3_30b_decode_576"),
      &decode_root));
  loomc_program_plan_root_info_t decode_root_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_ROOT_INFO,
      /*.structure_size=*/sizeof(decode_root_info),
  };
  LOOMC_ASSERT_OK(
      loomc_program_plan_root_info(plan.get(), decode_root, &decode_root_info));
  EXPECT_EQ(decode_root_info.dependency_count, 14u);

  std::vector<uint64_t> decode_dependency_units;
  for (loomc_host_size_t i = 0; i < decode_root_info.dependency_count; ++i) {
    loomc_program_plan_dependency_info_t dependency_info = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_DEPENDENCY_INFO,
        /*.structure_size=*/sizeof(dependency_info),
    };
    LOOMC_ASSERT_OK(loomc_program_plan_root_dependency_info(
        plan.get(), decode_root, i, &dependency_info));
    decode_dependency_units.push_back(dependency_info.unit.value);
  }
  std::sort(decode_dependency_units.begin(), decode_dependency_units.end());

  std::vector<loomc_program_plan_root_t> roots;
  roots.reserve(std::size(kPrefillShapes) + 1);
  for (const PrefillShape& shape : kPrefillShapes) {
    loomc_program_plan_root_t root = loomc_program_plan_root_invalid();
    LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
        plan.get(), loomc_make_cstring_view(shape.export_name), &root));
    loomc_program_plan_root_info_t root_info = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_ROOT_INFO,
        /*.structure_size=*/sizeof(root_info),
    };
    LOOMC_ASSERT_OK(loomc_program_plan_root_info(plan.get(), root, &root_info));
    EXPECT_EQ(root_info.dependency_count, shape.dependency_count)
        << shape.export_name;

    std::vector<uint64_t> prefill_dependency_units;
    for (loomc_host_size_t i = 0; i < root_info.dependency_count; ++i) {
      loomc_program_plan_dependency_info_t dependency_info = {
          /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_DEPENDENCY_INFO,
          /*.structure_size=*/sizeof(dependency_info),
      };
      LOOMC_ASSERT_OK(loomc_program_plan_root_dependency_info(
          plan.get(), root, i, &dependency_info));
      prefill_dependency_units.push_back(dependency_info.unit.value);
    }
    std::sort(prefill_dependency_units.begin(), prefill_dependency_units.end());
    std::vector<uint64_t> shared_dependency_units;
    std::set_intersection(
        decode_dependency_units.begin(), decode_dependency_units.end(),
        prefill_dependency_units.begin(), prefill_dependency_units.end(),
        std::back_inserter(shared_dependency_units));
    EXPECT_FALSE(shared_dependency_units.empty()) << shape.export_name;
    roots.push_back(root);
  }
  roots.push_back(decode_root);

  const loomc_host_size_t unit_count =
      loomc_program_plan_unit_count(plan.get());
  const loomc_program_plan_unit_compile_options_t unit_compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(unit_compile_options),
      /*.next=*/nullptr,
  };
  std::vector<ProgramPtr> unit_programs;
  std::vector<loomc_program_t*> unit_table_values;
  unit_programs.reserve(unit_count);
  unit_table_values.reserve(unit_count);
  for (loomc_host_size_t i = 0; i < unit_count; ++i) {
    const loomc_program_plan_unit_t unit =
        loomc_program_plan_unit_from_index(static_cast<uint32_t>(i));
    loomc_program_plan_unit_info_t unit_info = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_INFO,
        /*.structure_size=*/sizeof(unit_info),
    };
    LOOMC_ASSERT_OK(loomc_program_plan_unit_info(plan.get(), unit, &unit_info));
    WorkspacePtr worker_workspace = CreateWorkspace();
    loomc_program_t* raw_program = nullptr;
    raw_result = nullptr;
    LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
        plan.get(), worker_workspace.get(), unit, &unit_compile_options,
        loomc_allocator_system(), &raw_program, &raw_result));
    ProgramPtr program(raw_program);
    ResultPtr result(raw_result);
    ASSERT_TRUE(ResultSucceeded(
        result.get(), "shared command-package unit " + std::to_string(i) +
                          " (" + ToString(unit_info.identifier) + ")"));
    unit_table_values.push_back(program.get());
    unit_programs.push_back(std::move(program));
  }

  const loomc_program_plan_unit_table_t unit_table = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_TABLE,
      /*.structure_size=*/sizeof(unit_table),
      /*.next=*/nullptr,
      /*.programs=*/unit_table_values.data(),
      /*.program_count=*/unit_table_values.size(),
  };
  loomc_program_t* raw_assembled_program = nullptr;
  loomc_result_t* raw_assemble_result = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_assemble(
      plan.get(), coordinator_workspace.get(), roots.data(), roots.size(),
      &unit_table, /*options=*/nullptr, loomc_allocator_system(),
      &raw_assembled_program, &raw_assemble_result));
  ProgramPtr assembled_program(raw_assembled_program);
  ResultPtr assemble_result(raw_assemble_result);
  ASSERT_TRUE(ResultSucceeded(assemble_result.get(),
                              "shared command-package assembly"));
  EXPECT_EQ(loomc_program_export_count(assembled_program.get()), roots.size());
  for (const PrefillShape& shape : kPrefillShapes) {
    SCOPED_TRACE(shape.export_name);
    const loomc_artifact_t* command_artifact =
        FindCommandArtifact(assembled_program.get(), shape.export_name);
    ASSERT_NE(command_artifact, nullptr);
    EXPECT_GT(command_artifact->contents.data_length, 0u);

    loomc_program_export_t root_export = loomc_program_export_invalid();
    LOOMC_ASSERT_OK(loomc_program_lookup_export(
        assembled_program.get(), loomc_make_cstring_view(shape.export_name),
        &root_export));
    loomc_cmd_program_t* raw_command_program = nullptr;
    LOOMC_ASSERT_OK(loomc_cmd_program_create_from_export(
        assembled_program.get(), root_export, loomc_allocator_system(),
        &raw_command_program));
    CmdProgramPtr command_program(raw_command_program);
    loomc_cmd_program_info_t info = {
        /*.type=*/LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO,
        /*.structure_size=*/sizeof(info),
    };
    LOOMC_ASSERT_OK(loomc_cmd_program_info(command_program.get(), &info));
    EXPECT_EQ(info.fixed_buffer_count, 2u);
    EXPECT_EQ(info.rebindable_binding_count, 5u);
    EXPECT_EQ(info.parameter_root_count, 2u);
    EXPECT_EQ(info.parameter_count, 580u);
    EXPECT_EQ(info.transient.binding_index, 4u);
    EXPECT_EQ(info.transient.required_byte_length, shape.transient_byte_length);
    EXPECT_EQ(info.transient.minimum_alignment, 256u);
    EXPECT_EQ(info.launch_counts.binding_index,
              LOOMC_CMD_PROGRAM_BINDING_INVALID);
  }
  const loomc_artifact_t* decode_artifact =
      FindCommandArtifact(assembled_program.get(), "qwen3_30b_decode_576");
  ASSERT_NE(decode_artifact, nullptr);
  EXPECT_GT(decode_artifact->contents.data_length, 0u);
}

}  // namespace
