// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "src/program_plan.h"

#include <array>
#include <atomic>
#include <string>

#include "iree/testing/gtest.h"
#include "src/program.h"
#include "src/result.h"
#include "test/util.h"

namespace {

using PlanPtr =
    loomc::testing::HandlePtr<loomc_program_plan_t, loomc_program_plan_release>;
using ProgramPtr =
    loomc::testing::HandlePtr<loomc_program_t, loomc_program_release>;
using ResultPtr =
    loomc::testing::HandlePtr<loomc_result_t, loomc_result_release>;
using WorkspacePtr =
    loomc::testing::HandlePtr<loomc_workspace_t, loomc_workspace_release>;

std::string ToString(loomc_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string ToString(loomc_byte_span_t value) {
  return std::string(reinterpret_cast<const char*>(value.data),
                     value.data_length);
}

struct OperationCounts {
  std::atomic<int> compile = 0;
  std::atomic<int> load = 0;
  std::atomic<int> assemble = 0;
  std::atomic<int> destroy = 0;
};

struct TestPlanStorage {
  OperationCounts* counts;
};

loomc_status_t CreateProgram(
    const loomc_string_view_t* export_names, loomc_host_size_t export_count,
    const loomc_artifact_t* artifacts, loomc_host_size_t artifact_count,
    const loomc_program_dependency_t* dependencies,
    loomc_host_size_t dependency_count, loomc_allocator_t allocator,
    loomc_program_t** out_program, loomc_result_t** out_result) {
  loomc_result_t* result = nullptr;
  loomc_status_t status =
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result);
  for (loomc_host_size_t i = 0;
       i < artifact_count && loomc_status_is_ok(status); ++i) {
    status = loomc_result_add_artifact(result, &artifacts[i]);
  }
  if (loomc_status_is_ok(status)) {
    const loomc_program_create_params_t params = {
        /*.export_names=*/export_names,
        /*.export_count=*/export_count,
        /*.artifact_storage=*/loomc_result_artifact_storage(result),
        /*.dependencies=*/dependencies,
        /*.dependency_count=*/dependency_count,
    };
    status = loomc_program_create(&params, allocator, out_program);
  }
  if (loomc_status_is_ok(status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return status;
}

loomc_status_t CompileUnit(
    const void* storage, loomc_workspace_t* workspace, uint32_t unit_index,
    const loomc_program_plan_unit_compile_options_t* options,
    loomc_allocator_t allocator, loomc_program_t** out_program,
    loomc_result_t** out_result) {
  (void)workspace;
  (void)options;
  const TestPlanStorage* test_storage =
      static_cast<const TestPlanStorage*>(storage);
  ++test_storage->counts->compile;
  const loomc_string_view_t export_name = loomc_make_cstring_view("compiled");
  const uint8_t contents[] = {static_cast<uint8_t>(unit_index)};
  const loomc_artifact_t artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_EXECUTABLE,
      /*.format=*/loomc_make_cstring_view("test-compiled"),
      /*.identifier=*/loomc_make_cstring_view("unit.bin"),
      /*.contents=*/loomc_make_byte_span(contents, sizeof(contents)),
  };
  return CreateProgram(&export_name, 1, &artifact, 1, /*dependencies=*/nullptr,
                       /*dependency_count=*/0, allocator, out_program,
                       out_result);
}

loomc_status_t LoadUnitProgram(const void* storage, uint32_t unit_index,
                               const loomc_artifact_t* artifacts,
                               loomc_host_size_t artifact_count,
                               loomc_allocator_t allocator,
                               loomc_program_t** out_program,
                               loomc_result_t** out_result) {
  (void)unit_index;
  const TestPlanStorage* test_storage =
      static_cast<const TestPlanStorage*>(storage);
  ++test_storage->counts->load;
  const loomc_string_view_t export_name = loomc_make_cstring_view("loaded");
  return CreateProgram(&export_name, 1, artifacts, artifact_count,
                       /*dependencies=*/nullptr, /*dependency_count=*/0,
                       allocator, out_program, out_result);
}

loomc_status_t Assemble(const void* storage, loomc_workspace_t* workspace,
                        const loomc_program_plan_root_t* roots,
                        loomc_host_size_t root_count,
                        const loomc_program_plan_unit_table_t* unit_table,
                        const loomc_program_plan_assembly_options_t* options,
                        loomc_allocator_t allocator,
                        loomc_program_t** out_program,
                        loomc_result_t** out_result) {
  (void)workspace;
  (void)options;
  const TestPlanStorage* test_storage =
      static_cast<const TestPlanStorage*>(storage);
  ++test_storage->counts->assemble;
  std::array<loomc_string_view_t, 2> export_names;
  for (loomc_host_size_t i = 0; i < root_count; ++i) {
    export_names[i] = roots[i].value == 0 ? loomc_make_cstring_view("prefill")
                                          : loomc_make_cstring_view("decode");
  }
  const loomc_program_dependency_t dependencies[] = {
      {
          /*.slot=*/4,
          /*.program=*/unit_table->programs[1],
      },
      {
          /*.slot=*/9,
          /*.program=*/unit_table->programs[2],
      },
  };
  const loomc_artifact_t artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_EXECUTABLE,
      /*.format=*/loomc_make_cstring_view("test-root"),
      /*.identifier=*/loomc_make_cstring_view("root.bin"),
      /*.contents=*/
      loomc_make_byte_span(reinterpret_cast<const uint8_t*>("assembled"), 9),
  };
  return CreateProgram(export_names.data(), root_count, &artifact, 1,
                       dependencies, IREE_ARRAYSIZE(dependencies), allocator,
                       out_program, out_result);
}

void Destroy(void* storage, loomc_allocator_t allocator) {
  TestPlanStorage* test_storage = static_cast<TestPlanStorage*>(storage);
  ++test_storage->counts->destroy;
  loomc_allocator_free(allocator, test_storage);
}

const loomc_program_plan_operations_t kOperations = {
    /*.compile_unit=*/CompileUnit,
    /*.load_unit_program=*/LoadUnitProgram,
    /*.assemble=*/Assemble,
    /*.destroy=*/Destroy,
};

TEST(ProgramPlanTest, OwnsAndDispatchesMultiRootProductionGraph) {
  OperationCounts operation_counts;
  TestPlanStorage* storage = nullptr;
  LOOMC_ASSERT_OK(loomc_allocator_malloc(loomc_allocator_system(),
                                         sizeof(*storage),
                                         reinterpret_cast<void**>(&storage)));
  storage->counts = &operation_counts;

  char prefill_name[] = "prefill";
  char decode_name[] = "decode";
  char root_identifier[] = "command-roots";
  char first_identifier[] = "first-kernel";
  char second_identifier[] = "second-kernel";
  uint8_t root_key[] = {1, 2, 3};
  uint8_t first_key[] = {4, 5};
  uint8_t second_key[] = {6};
  const loomc_program_plan_dependency_t prefill_dependencies[] = {
      {
          /*.slot=*/4,
          /*.unit_index=*/1,
      },
  };
  const loomc_program_plan_dependency_t decode_dependencies[] = {
      {
          /*.slot=*/4,
          /*.unit_index=*/1,
      },
      {
          /*.slot=*/9,
          /*.unit_index=*/2,
      },
  };
  const loomc_program_plan_root_create_params_t roots[] = {
      {
          /*.name=*/loomc_make_cstring_view(prefill_name),
          /*.program_unit_index=*/0,
          /*.dependencies=*/prefill_dependencies,
          /*.dependency_count=*/IREE_ARRAYSIZE(prefill_dependencies),
      },
      {
          /*.name=*/loomc_make_cstring_view(decode_name),
          /*.program_unit_index=*/0,
          /*.dependencies=*/decode_dependencies,
          /*.dependency_count=*/IREE_ARRAYSIZE(decode_dependencies),
      },
  };
  const loomc_program_plan_unit_create_params_t units[] = {
      {
          /*.identifier=*/loomc_make_cstring_view(root_identifier),
          /*.cache_key=*/loomc_make_byte_span(root_key, sizeof(root_key)),
      },
      {
          /*.identifier=*/loomc_make_cstring_view(first_identifier),
          /*.cache_key=*/loomc_make_byte_span(first_key, sizeof(first_key)),
      },
      {
          /*.identifier=*/loomc_make_cstring_view(second_identifier),
          /*.cache_key=*/loomc_make_byte_span(second_key, sizeof(second_key)),
      },
  };
  const loomc_program_plan_create_params_t params = {
      /*.roots=*/roots,
      /*.root_count=*/IREE_ARRAYSIZE(roots),
      /*.units=*/units,
      /*.unit_count=*/IREE_ARRAYSIZE(units),
      /*.operations=*/&kOperations,
      /*.target_storage=*/storage,
  };
  loomc_program_plan_t* plan_raw = nullptr;
  LOOMC_ASSERT_OK(
      loomc_program_plan_create(&params, loomc_allocator_system(), &plan_raw));
  PlanPtr plan(plan_raw);

  prefill_name[0] = 'X';
  decode_name[0] = 'X';
  root_identifier[0] = 'X';
  first_identifier[0] = 'X';
  second_identifier[0] = 'X';
  root_key[0] = 0;
  first_key[0] = 0;
  second_key[0] = 0;

  ASSERT_EQ(loomc_program_plan_root_count(plan.get()), 2u);
  loomc_program_plan_root_t prefill = loomc_program_plan_root_invalid();
  LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
      plan.get(), loomc_make_cstring_view("prefill"), &prefill));
  loomc_program_plan_root_t decode = loomc_program_plan_root_invalid();
  LOOMC_ASSERT_OK(loomc_program_plan_lookup_root(
      plan.get(), loomc_make_cstring_view("decode"), &decode));

  loomc_program_plan_root_info_t root_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_ROOT_INFO,
      /*.structure_size=*/sizeof(root_info),
  };
  LOOMC_ASSERT_OK(loomc_program_plan_root_info(plan.get(), decode, &root_info));
  EXPECT_EQ(ToString(root_info.name), "decode");
  EXPECT_EQ(root_info.program_unit.value, 0u);
  EXPECT_EQ(root_info.dependency_count, 2u);
  loomc_program_plan_dependency_info_t dependency_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_DEPENDENCY_INFO,
      /*.structure_size=*/sizeof(dependency_info),
  };
  LOOMC_ASSERT_OK(loomc_program_plan_root_dependency_info(plan.get(), decode, 1,
                                                          &dependency_info));
  EXPECT_EQ(dependency_info.slot, 9u);
  EXPECT_EQ(dependency_info.unit.value, 2u);

  ASSERT_EQ(loomc_program_plan_unit_count(plan.get()), 3u);
  loomc_program_plan_unit_info_t unit_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_INFO,
      /*.structure_size=*/sizeof(unit_info),
  };
  LOOMC_ASSERT_OK(loomc_program_plan_unit_info(
      plan.get(), loomc_program_plan_unit_from_index(1), &unit_info));
  EXPECT_EQ(ToString(unit_info.identifier), "first-kernel");
  EXPECT_EQ(ToString(unit_info.cache_key), std::string("\x04\x05", 2));

  loomc_workspace_t* workspace_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_workspace_create(
      /*options=*/nullptr, loomc_allocator_system(), &workspace_raw));
  WorkspacePtr workspace(workspace_raw);
  std::array<ProgramPtr, 3> unit_programs;
  for (uint32_t i = 0; i < 2; ++i) {
    loomc_program_t* program_raw = nullptr;
    loomc_result_t* result_raw = nullptr;
    LOOMC_ASSERT_OK(loomc_program_plan_compile_unit(
        plan.get(), workspace.get(), loomc_program_plan_unit_from_index(i),
        /*options=*/nullptr, loomc_allocator_system(), &program_raw,
        &result_raw));
    unit_programs[i].reset(program_raw);
    ResultPtr result(result_raw);
    ASSERT_TRUE(loomc_result_succeeded(result.get()));
  }
  const loomc_artifact_t cached_artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_EXECUTABLE,
      /*.format=*/loomc_make_cstring_view("test-cached"),
      /*.identifier=*/loomc_make_cstring_view("cached.bin"),
      /*.contents=*/
      loomc_make_byte_span(reinterpret_cast<const uint8_t*>("cached"), 6),
  };
  loomc_program_t* loaded_program_raw = nullptr;
  loomc_result_t* loaded_result_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_load_unit_program(
      plan.get(), loomc_program_plan_unit_from_index(2), &cached_artifact, 1,
      loomc_allocator_system(), &loaded_program_raw, &loaded_result_raw));
  unit_programs[2].reset(loaded_program_raw);
  ResultPtr loaded_result(loaded_result_raw);
  ASSERT_TRUE(loomc_result_succeeded(loaded_result.get()));
  loaded_result.reset();

  const std::array<loomc_program_t*, 3> program_table_values = {
      unit_programs[0].get(),
      unit_programs[1].get(),
      unit_programs[2].get(),
  };
  const loomc_program_plan_unit_table_t program_table = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_TABLE,
      /*.structure_size=*/sizeof(program_table),
      /*.next=*/nullptr,
      /*.programs=*/program_table_values.data(),
      /*.program_count=*/program_table_values.size(),
  };
  const loomc_program_plan_root_t selected_roots[] = {prefill, decode};
  loomc_program_t* assembled_program_raw = nullptr;
  loomc_result_t* assembled_result_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_program_plan_assemble(
      plan.get(), workspace.get(), selected_roots,
      IREE_ARRAYSIZE(selected_roots), &program_table, /*options=*/nullptr,
      loomc_allocator_system(), &assembled_program_raw, &assembled_result_raw));
  ProgramPtr assembled_program(assembled_program_raw);
  ResultPtr assembled_result(assembled_result_raw);
  ASSERT_TRUE(loomc_result_succeeded(assembled_result.get()));

  plan.reset();
  assembled_result.reset();
  for (ProgramPtr& unit_program : unit_programs) unit_program.reset();
  EXPECT_EQ(operation_counts.destroy, 1);
  EXPECT_EQ(operation_counts.compile, 2);
  EXPECT_EQ(operation_counts.load, 1);
  EXPECT_EQ(operation_counts.assemble, 1);

  loomc_program_export_t assembled_decode = loomc_program_export_invalid();
  LOOMC_ASSERT_OK(loomc_program_lookup_export(assembled_program.get(),
                                              loomc_make_cstring_view("decode"),
                                              &assembled_decode));
  EXPECT_TRUE(loomc_program_export_is_valid(assembled_decode));
  ASSERT_EQ(loomc_program_dependency_count(assembled_program.get()), 2u);
  loomc_program_dependency_info_t assembled_dependency = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_DEPENDENCY_INFO,
      /*.structure_size=*/sizeof(assembled_dependency),
  };
  LOOMC_ASSERT_OK(loomc_program_dependency_info(assembled_program.get(), 1,
                                                &assembled_dependency));
  EXPECT_EQ(assembled_dependency.slot, 9u);
  ASSERT_NE(assembled_dependency.program, nullptr);
  EXPECT_EQ(
      ToString(
          loomc_program_artifact_at(assembled_dependency.program, 0)->contents),
      "cached");
}

}  // namespace
