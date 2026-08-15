// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include "experimental/qwen/programs/command_package_source.h"
#include "experimental/qwen/programs/dense_projection_specialization_test_source.h"
#include "iree/testing/gtest.h"
#include "loomc/compile.h"
#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/program_plan.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target/amdgpu.h"
#include "loomc/target/cmd/program.h"
#include "loomc/target/cmd/program_plan.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ProgramPackagePtr =
    HandlePtr<loomc_cmd_program_package_t, loomc_cmd_program_package_release>;
using ProgramPlanPtr =
    HandlePtr<loomc_program_plan_t, loomc_program_plan_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

static constexpr const char* kCommandRootNames[] = {
    "qwen3_30b_prefill_32",  "qwen3_30b_prefill_64",  "qwen3_30b_prefill_128",
    "qwen3_30b_prefill_256", "qwen3_30b_prefill_512", "qwen3_30b_decode",
};

struct PrefillShape {
  // Exact public command-program export.
  const char* export_name;
  // Number of executable entries named by the command program.
  loomc_host_size_t entry_count;
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

static const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                            loomc_artifact_kind_t kind,
                                            const char* format) {
  if (!result) return nullptr;
  const loomc_string_view_t expected_format = loomc_make_cstring_view(format);
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact && artifact->kind == kind &&
        loomc_string_view_equal(artifact->format, expected_format)) {
      return artifact;
    }
  }
  return nullptr;
}

static TargetEnvironmentPtr CreateTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &target_environment));
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

static PassProgramPtr CreatePreparationPassProgram(loomc_context_t* context) {
  const loomc_target_pipeline_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("qwen-command-expanded-source"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_EXPANDED_SOURCE,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/20,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_from_target_pipeline(
      context, &options, loomc_allocator_system(), &pass_program, &raw_result));
  ResultPtr result(raw_result);
  ResultSucceeded(result.get(), "preparation pipeline creation");
  return PassProgramPtr(pass_program);
}

static PassProgramPtr CreateExecutablePassProgram(loomc_context_t* context) {
  const loomc_target_pipeline_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.identifier=*/loomc_make_cstring_view("qwen-command-prepared-low"),
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/20,
  };
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_EXPECT_OK(loomc_pass_program_create_from_target_pipeline(
      context, &options, loomc_allocator_system(), &pass_program, &raw_result));
  ResultPtr result(raw_result);
  ResultSucceeded(result.get(), "executable pipeline creation");
  return PassProgramPtr(pass_program);
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
  return TargetProfilePtr(target_profile);
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

static ModulePtr DeserializeModule(loomc_context_t* context,
                                   loomc_workspace_t* workspace,
                                   const loomc_source_t* source) {
  loomc_module_t* module = nullptr;
  loomc_result_t* raw_result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
      context, workspace, source, /*options=*/nullptr, loomc_allocator_system(),
      &module, &raw_result));
  ResultPtr result(raw_result);
  ResultSucceeded(result.get(), "command package parse");
  return ModulePtr(module);
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

static bool SpecializeModule(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* preparation_pass_program,
    loomc_module_t* module, loomc_target_profile_t* target_profile,
    const loomc_config_binding_t* config_bindings,
    loomc_host_size_t config_binding_count) {
  std::vector<loomc_target_specialization_t> specializations =
      CreateKernelSpecializations(module, target_profile);
  if (specializations.empty()) return false;
  const loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/specializations.data(),
      /*.specialization_count=*/specializations.size(),
  };
  const loomc_compile_options_t compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(compile_options),
      /*.next=*/&target_options,
      /*.module_name=*/loomc_make_cstring_view("qwen-command-test"),
      /*.artifact_flags=*/0,
      /*.config=*/
      {
          /*.bindings=*/config_bindings,
          /*.binding_count=*/config_binding_count,
          /*.json_object=*/loomc_string_view_empty(),
          /*.flags=*/LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      },
  };
  loomc_result_t* raw_result = nullptr;
  loomc_status_t status = loomc_compile_module(
      compiler, workspace, preparation_pass_program, module, &compile_options,
      loomc_allocator_system(), &raw_result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result(raw_result);
  return loomc_status_is_ok(status) &&
         ResultSucceeded(result.get(), "program specialization");
}

static ProgramPlanPtr PreparePlan(
    loomc_workspace_t* workspace, const loomc_module_t* module,
    const std::vector<loomc_string_view_t>& root_names) {
  loomc_program_plan_t* plan = nullptr;
  loomc_result_t* raw_result = nullptr;
  loomc_status_t status = loomc_program_plan_prepare(
      workspace, module, root_names.data(), root_names.size(),
      /*options=*/nullptr, loomc_allocator_system(), &plan, &raw_result);
  LOOMC_EXPECT_OK(status);
  ResultPtr result(raw_result);
  if (!loomc_status_is_ok(status) ||
      !ResultSucceeded(result.get(), "program plan preparation")) {
    loomc_program_plan_release(plan);
    return ProgramPlanPtr(nullptr);
  }
  return ProgramPlanPtr(plan);
}

enum class UnitKind : uint8_t {
  kUnknown = 0,
  kPackage,
  kLaunchConfig,
  kExecutable,
};

static bool MarkUnit(loomc_program_plan_unit_t unit, UnitKind kind,
                     std::vector<UnitKind>& unit_kinds) {
  if (!loomc_program_plan_unit_is_valid(unit) ||
      unit.value >= unit_kinds.size()) {
    ADD_FAILURE() << "command plan contains an invalid unit";
    return false;
  }
  UnitKind& current = unit_kinds[unit.value];
  if (current != UnitKind::kUnknown && current != kind) {
    ADD_FAILURE() << "command plan assigns unit " << unit.value
                  << " conflicting roles";
    return false;
  }
  current = kind;
  return true;
}

struct CompiledPlan {
  // Unit results indexed by the owning plan's dense unit token.
  std::vector<ResultPtr> unit_results;
  // Shared portable package unit.
  loomc_program_plan_unit_t package_unit = loomc_program_plan_unit_invalid();
  // Shared launch-config unit, when any root requires one.
  loomc_program_plan_unit_t launch_config_unit =
      loomc_program_plan_unit_invalid();
};

static bool CompilePlan(loomc_program_plan_t* plan,
                        const std::vector<loomc_program_plan_root_t>& roots,
                        loomc_compiler_t* compiler,
                        const loomc_pass_program_t* empty_pass_program,
                        const loomc_pass_program_t* executable_pass_program,
                        CompiledPlan& out_compiled) {
  const loomc_host_size_t unit_count = loomc_program_plan_unit_count(plan);
  std::vector<UnitKind> unit_kinds(unit_count, UnitKind::kUnknown);
  for (loomc_program_plan_root_t root : roots) {
    loomc_cmd_program_plan_root_info_t info = {};
    loomc_status_t status = loomc_cmd_program_plan_root_info(plan, root, &info);
    LOOMC_EXPECT_OK(status);
    if (!loomc_status_is_ok(status)) return false;
    if (!loomc_program_plan_unit_is_valid(out_compiled.package_unit)) {
      out_compiled.package_unit = info.package_unit;
    } else if (out_compiled.package_unit.value != info.package_unit.value) {
      ADD_FAILURE() << "selected roots do not share one package unit";
      return false;
    }
    if (!MarkUnit(info.package_unit, UnitKind::kPackage, unit_kinds)) {
      return false;
    }
    if (loomc_program_plan_unit_is_valid(info.launch_config_unit)) {
      if (!loomc_program_plan_unit_is_valid(out_compiled.launch_config_unit)) {
        out_compiled.launch_config_unit = info.launch_config_unit;
      } else if (out_compiled.launch_config_unit.value !=
                 info.launch_config_unit.value) {
        ADD_FAILURE() << "selected roots do not share one launch-config unit";
        return false;
      }
      if (!MarkUnit(info.launch_config_unit, UnitKind::kLaunchConfig,
                    unit_kinds)) {
        return false;
      }
    }
    for (loomc_host_size_t i = 0; i < info.executable_requirement_count; ++i) {
      const loomc_cmd_program_plan_executable_requirement_t& requirement =
          info.executable_requirements[i];
      if (!loomc_string_view_is_empty(requirement.import_name)) {
        ADD_FAILURE() << "Qwen production roots unexpectedly require an "
                         "external executable";
        return false;
      }
      if (!MarkUnit(requirement.unit, UnitKind::kExecutable, unit_kinds)) {
        return false;
      }
    }
  }

  out_compiled.unit_results.reserve(unit_count);
  for (loomc_host_size_t i = 0; i < unit_count; ++i) {
    if (unit_kinds[i] == UnitKind::kUnknown) {
      ADD_FAILURE() << "selected root closure left unit " << i
                    << " without a product role";
      return false;
    }
    const loomc_pass_program_t* pass_program =
        unit_kinds[i] == UnitKind::kExecutable ? executable_pass_program
                                               : empty_pass_program;
    WorkspacePtr worker_workspace = CreateWorkspace();
    loomc_result_t* raw_result = nullptr;
    loomc_status_t status = loomc_program_plan_compile_unit(
        plan, compiler, worker_workspace.get(),
        loomc_program_plan_unit_at(plan, i), pass_program,
        /*options=*/nullptr, loomc_allocator_system(), &raw_result);
    LOOMC_EXPECT_OK(status);
    ResultPtr result(raw_result);
    if (!loomc_status_is_ok(status) ||
        !ResultSucceeded(result.get(), "program unit " + std::to_string(i))) {
      return false;
    }
    out_compiled.unit_results.push_back(std::move(result));
  }
  return true;
}

static ProgramPackagePtr LoadPackage(const CompiledPlan& compiled) {
  if (!loomc_program_plan_unit_is_valid(compiled.package_unit) ||
      compiled.package_unit.value >= compiled.unit_results.size()) {
    ADD_FAILURE() << "compiled plan has no package unit";
    return ProgramPackagePtr(nullptr);
  }
  const loomc_artifact_t* artifact = FindArtifact(
      compiled.unit_results[compiled.package_unit.value].get(),
      LOOMC_ARTIFACT_KIND_EXECUTABLE, LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE);
  if (!artifact) {
    ADD_FAILURE() << "package unit emitted no portable command package";
    return ProgramPackagePtr(nullptr);
  }
  loomc_cmd_program_package_t* package = nullptr;
  LOOMC_EXPECT_OK(loomc_cmd_program_package_load(
      artifact, /*release=*/nullptr, /*release_user_data=*/nullptr,
      loomc_allocator_system(), &package));
  return ProgramPackagePtr(package);
}

TEST(QwenCommandPackageTest, SpecializesExactDenseProjectionShapes) {
  TargetEnvironmentPtr target_environment = CreateTargetEnvironment();
  ContextPtr context = CreateContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source = CreateEmbeddedSource(
      qwen_loom_dense_projection_specialization_test_source_create(),
      qwen_loom_dense_projection_specialization_test_source_size());
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  TargetProfilePtr target_profile =
      CreateGfx1100TargetProfile(target_environment.get());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr preparation_pass_program =
      CreatePreparationPassProgram(context.get());
  PassProgramPtr empty_pass_program = CreateEmptyPassProgram(context.get());
  PassProgramPtr executable_pass_program =
      CreateExecutablePassProgram(context.get());
  ASSERT_TRUE(SpecializeModule(compiler.get(), workspace.get(),
                               preparation_pass_program.get(), module.get(),
                               target_profile.get(), nullptr, 0));

  const std::vector<loomc_string_view_t> root_names = {
      loomc_make_cstring_view("qwen3_moe_dense_projection_specialization"),
  };
  ProgramPlanPtr plan = PreparePlan(workspace.get(), module.get(), root_names);
  ASSERT_NE(plan, nullptr);
  ASSERT_EQ(loomc_program_plan_root_count(plan.get()), 1u);
  ASSERT_EQ(loomc_program_plan_unit_count(plan.get()), 2u);
  const loomc_program_plan_root_t root =
      loomc_program_plan_root_at(plan.get(), 0);
  loomc_cmd_program_plan_root_info_t root_info = {};
  LOOMC_ASSERT_OK(
      loomc_cmd_program_plan_root_info(plan.get(), root, &root_info));
  EXPECT_FALSE(loomc_program_plan_unit_is_valid(root_info.launch_config_unit));
  EXPECT_EQ(root_info.executable_requirement_count, 1u);

  CompiledPlan compiled;
  ASSERT_TRUE(CompilePlan(plan.get(), {root}, compiler.get(),
                          empty_pass_program.get(),
                          executable_pass_program.get(), compiled));
  ProgramPackagePtr package = LoadPackage(compiled);
  ASSERT_NE(package, nullptr);
  EXPECT_EQ(loomc_cmd_program_package_export_count(package.get()), 1u);
}

TEST(QwenCommandPackageTest, CompilesSharedPrefillAndDecodeProducts) {
  TargetEnvironmentPtr target_environment = CreateTargetEnvironment();
  ContextPtr context = CreateContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  SourcePtr source =
      CreateEmbeddedSource(qwen_loom_command_package_source_create(),
                           qwen_loom_command_package_source_size());
  ModulePtr module =
      DeserializeModule(context.get(), workspace.get(), source.get());
  TargetProfilePtr target_profile =
      CreateGfx1100TargetProfile(target_environment.get());
  CompilerPtr compiler = CreateCompiler(context.get());
  PassProgramPtr preparation_pass_program =
      CreatePreparationPassProgram(context.get());
  PassProgramPtr empty_pass_program = CreateEmptyPassProgram(context.get());
  PassProgramPtr executable_pass_program =
      CreateExecutablePassProgram(context.get());
  const loomc_config_binding_t config_bindings[] = {
      {
          /*.key=*/loomc_make_cstring_view("qwen3_30b.request.token_capacity"),
          /*.value=*/loomc_make_cstring_view("512"),
      },
      {
          /*.key=*/loomc_make_cstring_view(
              "qwen3_moe.attention.key_value_token_capacity"),
          /*.value=*/loomc_make_cstring_view("576"),
      },
  };
  ASSERT_TRUE(SpecializeModule(compiler.get(), workspace.get(),
                               preparation_pass_program.get(), module.get(),
                               target_profile.get(), config_bindings,
                               std::size(config_bindings)));

  std::vector<loomc_string_view_t> root_names;
  root_names.reserve(std::size(kCommandRootNames));
  for (const char* root_name : kCommandRootNames) {
    root_names.push_back(loomc_make_cstring_view(root_name));
  }
  ProgramPlanPtr plan = PreparePlan(workspace.get(), module.get(), root_names);
  ASSERT_NE(plan, nullptr);
  ASSERT_EQ(loomc_program_plan_root_count(plan.get()), root_names.size());

  std::vector<loomc_program_plan_root_t> roots;
  roots.reserve(root_names.size());
  for (loomc_host_size_t i = 0; i < root_names.size(); ++i) {
    roots.push_back(loomc_program_plan_root_at(plan.get(), i));
  }

  loomc_cmd_program_plan_root_info_t decode_plan_info = {};
  LOOMC_ASSERT_OK(loomc_cmd_program_plan_root_info(plan.get(), roots.back(),
                                                   &decode_plan_info));
  ASSERT_TRUE(
      loomc_program_plan_unit_is_valid(decode_plan_info.launch_config_unit));
  ASSERT_EQ(decode_plan_info.executable_requirement_count, 2u);
  std::vector<uint64_t> decode_units;
  for (loomc_host_size_t i = 0;
       i < decode_plan_info.executable_requirement_count; ++i) {
    decode_units.push_back(
        decode_plan_info.executable_requirements[i].unit.value);
  }
  std::sort(decode_units.begin(), decode_units.end());

  for (iree_host_size_t i = 0; i < std::size(kPrefillShapes); ++i) {
    SCOPED_TRACE(kPrefillShapes[i].export_name);
    loomc_cmd_program_plan_root_info_t prefill_plan_info = {};
    LOOMC_ASSERT_OK(loomc_cmd_program_plan_root_info(plan.get(), roots[i],
                                                     &prefill_plan_info));
    EXPECT_FALSE(
        loomc_program_plan_unit_is_valid(prefill_plan_info.launch_config_unit));
    EXPECT_EQ(prefill_plan_info.executable_requirement_count, 2u);
    std::vector<uint64_t> prefill_units;
    for (loomc_host_size_t j = 0;
         j < prefill_plan_info.executable_requirement_count; ++j) {
      prefill_units.push_back(
          prefill_plan_info.executable_requirements[j].unit.value);
    }
    std::sort(prefill_units.begin(), prefill_units.end());
    std::vector<uint64_t> shared_units;
    std::set_intersection(decode_units.begin(), decode_units.end(),
                          prefill_units.begin(), prefill_units.end(),
                          std::back_inserter(shared_units));
    EXPECT_FALSE(shared_units.empty());
  }

  CompiledPlan compiled;
  ASSERT_TRUE(CompilePlan(plan.get(), roots, compiler.get(),
                          empty_pass_program.get(),
                          executable_pass_program.get(), compiled));
  ASSERT_EQ(compiled.launch_config_unit.value,
            decode_plan_info.launch_config_unit.value);
  ProgramPackagePtr package = LoadPackage(compiled);
  ASSERT_NE(package, nullptr);
  ASSERT_EQ(loomc_cmd_program_package_export_count(package.get()),
            root_names.size());

  for (const PrefillShape& shape : kPrefillShapes) {
    SCOPED_TRACE(shape.export_name);
    loomc_cmd_program_export_t program_export =
        loomc_cmd_program_export_invalid();
    LOOMC_ASSERT_OK(loomc_cmd_program_package_lookup_export(
        package.get(), loomc_make_cstring_view(shape.export_name),
        &program_export));
    loomc_cmd_program_info_t info = {};
    LOOMC_ASSERT_OK(loomc_cmd_program_package_export_info(
        package.get(), program_export, &info));
    EXPECT_EQ(info.fixed_buffer_count, 2u);
    EXPECT_EQ(info.rebindable_binding_count, 5u);
    EXPECT_EQ(info.executable_count, 2u);
    EXPECT_EQ(info.entry_count, shape.entry_count);
    EXPECT_EQ(info.parameter_root_count, 2u);
    EXPECT_EQ(info.parameter_count, 580u);
    EXPECT_EQ(info.transient.binding_index, 4u);
    EXPECT_EQ(info.transient.required_byte_length, shape.transient_byte_length);
    EXPECT_EQ(info.transient.minimum_alignment, 256u);
    EXPECT_EQ(info.config.binding_index, LOOMC_CMD_PROGRAM_BINDING_INVALID);
    EXPECT_EQ(info.config.required_byte_length, 0u);
  }

  loomc_cmd_program_export_t decode_export = loomc_cmd_program_export_invalid();
  LOOMC_ASSERT_OK(loomc_cmd_program_package_lookup_export(
      package.get(), loomc_make_cstring_view("qwen3_30b_decode"),
      &decode_export));
  loomc_cmd_program_info_t decode_info = {};
  LOOMC_ASSERT_OK(loomc_cmd_program_package_export_info(
      package.get(), decode_export, &decode_info));
  EXPECT_EQ(decode_info.fixed_buffer_count, 2u);
  EXPECT_EQ(decode_info.rebindable_binding_count, 6u);
  EXPECT_EQ(decode_info.executable_count, 2u);
  EXPECT_EQ(decode_info.entry_count, 14u);
  EXPECT_EQ(decode_info.parameter_root_count, 2u);
  EXPECT_EQ(decode_info.parameter_count, 580u);
  EXPECT_EQ(decode_info.transient.binding_index, 4u);
  EXPECT_EQ(decode_info.transient.required_byte_length, 197376u);
  EXPECT_EQ(decode_info.transient.minimum_alignment, 256u);
  EXPECT_EQ(decode_info.config.binding_index, 5u);
  EXPECT_EQ(decode_info.config.required_byte_length, 12u);
  EXPECT_EQ(decode_info.config.minimum_alignment, 4u);

  loomc_cmd_program_parameter_root_info_t parameter_root = {};
  LOOMC_ASSERT_OK(loomc_cmd_program_package_parameter_root_info(
      package.get(), decode_export, 0, &parameter_root));
  EXPECT_EQ(parameter_root.fixed_buffer_index, 0u);
  EXPECT_EQ(parameter_root.required_byte_length, 18550716416ull);
  EXPECT_EQ(parameter_root.minimum_alignment, 256u);
  LOOMC_ASSERT_OK(loomc_cmd_program_package_parameter_root_info(
      package.get(), decode_export, 1, &parameter_root));
  EXPECT_EQ(parameter_root.fixed_buffer_index, 1u);
  EXPECT_EQ(parameter_root.required_byte_length, 256u);
  EXPECT_EQ(parameter_root.minimum_alignment, 256u);
}

}  // namespace
