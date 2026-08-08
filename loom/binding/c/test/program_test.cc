// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "src/program.h"

#include <string>

#include "iree/testing/gtest.h"
#include "src/result.h"
#include "test/util.h"

namespace {

using ProgramPtr =
    loomc::testing::HandlePtr<loomc_program_t, loomc_program_release>;
using ResultPtr =
    loomc::testing::HandlePtr<loomc_result_t, loomc_result_release>;

std::string ToString(loomc_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string ToString(loomc_byte_span_t value) {
  return std::string(reinterpret_cast<const char*>(value.data),
                     value.data_length);
}

struct TargetStorage {
  int* release_count;
};

void ReleaseTargetStorage(void* storage, loomc_allocator_t allocator) {
  TargetStorage* target_storage = static_cast<TargetStorage*>(storage);
  ++*target_storage->release_count;
  loomc_allocator_free(allocator, target_storage);
}

ProgramPtr CreateProgram(
    loomc_result_t* result, const loomc_string_view_t* export_names,
    loomc_host_size_t export_count,
    const loomc_program_dependency_t* dependencies = nullptr,
    loomc_host_size_t dependency_count = 0) {
  const loomc_program_create_params_t params = {
      /*.export_names=*/export_names,
      /*.export_count=*/export_count,
      /*.artifact_storage=*/loomc_result_artifact_storage(result),
      /*.dependencies=*/dependencies,
      /*.dependency_count=*/dependency_count,
  };
  loomc_program_t* program = nullptr;
  LOOMC_EXPECT_OK(
      loomc_program_create(&params, loomc_allocator_system(), &program));
  return ProgramPtr(program);
}

TEST(ProgramTest, OwnsExportsArtifactsAndDependencies) {
  loomc_result_t* dependency_result_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED,
                                      loomc_allocator_system(),
                                      &dependency_result_raw));
  ResultPtr dependency_result(dependency_result_raw);
  char dependency_contents[] = "executable";
  const loomc_artifact_t dependency_artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_EXECUTABLE,
      /*.format=*/loomc_make_cstring_view("test-executable"),
      /*.identifier=*/loomc_make_cstring_view("dependency.bin"),
      /*.contents=*/
      loomc_make_byte_span(dependency_contents,
                           sizeof(dependency_contents) - 1),
  };
  LOOMC_ASSERT_OK(
      loomc_result_add_artifact(dependency_result.get(), &dependency_artifact));
  char dependency_export_name[] = "kernel";
  const loomc_string_view_t dependency_exports[] = {
      loomc_make_string_view(dependency_export_name,
                             sizeof(dependency_export_name) - 1),
  };
  ProgramPtr dependency =
      CreateProgram(dependency_result.get(), dependency_exports,
                    IREE_ARRAYSIZE(dependency_exports));
  EXPECT_EQ(
      loomc_program_artifact_at(dependency.get(), 0)->contents.data,
      loomc_result_artifact_at(dependency_result.get(), 0)->contents.data);

  loomc_result_t* root_result_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED,
                                      loomc_allocator_system(),
                                      &root_result_raw));
  ResultPtr root_result(root_result_raw);
  char root_contents[] = "command-program";
  const loomc_artifact_t root_artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_EXECUTABLE,
      /*.format=*/loomc_make_cstring_view("cmd"),
      /*.identifier=*/loomc_make_cstring_view("model.cmd"),
      /*.contents=*/
      loomc_make_byte_span(root_contents, sizeof(root_contents) - 1),
  };
  LOOMC_ASSERT_OK(loomc_result_add_artifact(root_result.get(), &root_artifact));
  char prefill_name[] = "prefill";
  char decode_name[] = "decode";
  const loomc_string_view_t root_exports[] = {
      loomc_make_string_view(prefill_name, sizeof(prefill_name) - 1),
      loomc_make_string_view(decode_name, sizeof(decode_name) - 1),
  };
  const loomc_program_dependency_t root_dependencies[] = {
      {
          /*.slot=*/3,
          /*.program=*/dependency.get(),
      },
  };
  ProgramPtr root = CreateProgram(
      root_result.get(), root_exports, IREE_ARRAYSIZE(root_exports),
      root_dependencies, IREE_ARRAYSIZE(root_dependencies));
  EXPECT_EQ(loomc_program_artifact_at(root.get(), 0)->contents.data,
            loomc_result_artifact_at(root_result.get(), 0)->contents.data);

  dependency_export_name[0] = 'X';
  dependency_contents[0] = 'X';
  prefill_name[0] = 'X';
  decode_name[0] = 'X';
  root_contents[0] = 'X';
  dependency_result.reset();
  root_result.reset();
  dependency.reset();

  ASSERT_EQ(loomc_program_export_count(root.get()), 2u);
  loomc_program_export_t decode = loomc_program_export_invalid();
  LOOMC_ASSERT_OK(loomc_program_lookup_export(
      root.get(), loomc_make_cstring_view("decode"), &decode));
  EXPECT_TRUE(loomc_program_export_is_valid(decode));
  EXPECT_EQ(decode.value, 1u);
  loomc_program_export_info_t export_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_EXPORT_INFO,
      /*.structure_size=*/sizeof(export_info),
  };
  LOOMC_ASSERT_OK(loomc_program_export_info(root.get(), decode, &export_info));
  EXPECT_EQ(ToString(export_info.name), "decode");

  ASSERT_EQ(loomc_program_artifact_count(root.get()), 1u);
  const loomc_artifact_t* stored_root_artifact =
      loomc_program_artifact_at(root.get(), 0);
  ASSERT_NE(stored_root_artifact, nullptr);
  EXPECT_EQ(ToString(stored_root_artifact->format), "cmd");
  EXPECT_EQ(ToString(stored_root_artifact->contents), "command-program");

  ASSERT_EQ(loomc_program_dependency_count(root.get()), 1u);
  loomc_program_dependency_info_t dependency_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_DEPENDENCY_INFO,
      /*.structure_size=*/sizeof(dependency_info),
  };
  LOOMC_ASSERT_OK(
      loomc_program_dependency_info(root.get(), 0, &dependency_info));
  EXPECT_EQ(dependency_info.slot, 3u);
  ASSERT_NE(dependency_info.program, nullptr);
  ASSERT_EQ(loomc_program_export_count(dependency_info.program), 1u);
  const loomc_artifact_t* stored_dependency_artifact =
      loomc_program_artifact_at(dependency_info.program, 0);
  ASSERT_NE(stored_dependency_artifact, nullptr);
  EXPECT_EQ(ToString(stored_dependency_artifact->contents), "executable");
}

TEST(ProgramTest, ReportsInvalidQueries) {
  loomc_result_t* result_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED,
                                      loomc_allocator_system(), &result_raw));
  ResultPtr result(result_raw);
  const loomc_string_view_t exports[] = {
      loomc_make_cstring_view("entry"),
  };
  ProgramPtr program =
      CreateProgram(result.get(), exports, IREE_ARRAYSIZE(exports));

  loomc_program_export_t export_token = loomc_program_export_invalid();
  loomc_status_t status = loomc_program_lookup_export(
      program.get(), loomc_make_cstring_view("missing"), &export_token);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_NOT_FOUND, status);
  EXPECT_FALSE(loomc_program_export_is_valid(export_token));

  loomc_program_export_info_t export_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_EXPORT_INFO,
      /*.structure_size=*/sizeof(export_info),
  };
  status = loomc_program_export_info(
      program.get(), loomc_program_export_from_index(1), &export_info);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);

  loomc_program_dependency_info_t dependency_info = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_PROGRAM_DEPENDENCY_INFO,
      /*.structure_size=*/sizeof(dependency_info),
  };
  status = loomc_program_dependency_info(program.get(), 0, &dependency_info);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT, status);
}

TEST(ProgramTest, ReleasesTargetOwnedStorage) {
  loomc_result_t* result_raw = nullptr;
  LOOMC_ASSERT_OK(loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED,
                                      loomc_allocator_system(), &result_raw));
  ResultPtr result(result_raw);
  int release_count = 0;
  TargetStorage* target_storage = nullptr;
  LOOMC_ASSERT_OK(
      loomc_allocator_malloc(loomc_allocator_system(), sizeof(*target_storage),
                             reinterpret_cast<void**>(&target_storage)));
  target_storage->release_count = &release_count;

  const loomc_string_view_t exports[] = {
      loomc_make_cstring_view("entry"),
  };
  const loomc_program_create_params_t params = {
      /*.export_names=*/exports,
      /*.export_count=*/IREE_ARRAYSIZE(exports),
      /*.artifact_storage=*/loomc_result_artifact_storage(result.get()),
      /*.dependencies=*/nullptr,
      /*.dependency_count=*/0,
      /*.target_storage=*/target_storage,
      /*.target_storage_release=*/ReleaseTargetStorage,
  };
  loomc_program_t* program_raw = nullptr;
  LOOMC_ASSERT_OK(
      loomc_program_create(&params, loomc_allocator_system(), &program_raw));
  ProgramPtr program(program_raw);
  result.reset();
  EXPECT_EQ(release_count, 0);
  program.reset();
  EXPECT_EQ(release_count, 1);
}

}  // namespace
