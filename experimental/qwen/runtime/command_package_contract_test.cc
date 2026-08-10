// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/qwen/runtime/command_package.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(QwenCommandPackageContractTest, SelectsExactPrefillPrograms) {
  struct PrefillShape {
    iree_host_size_t token_count;
    qwen_command_program_t program;
  };
  static constexpr PrefillShape kShapes[] = {
      {32, QWEN_COMMAND_PROGRAM_PREFILL_32},
      {64, QWEN_COMMAND_PROGRAM_PREFILL_64},
      {128, QWEN_COMMAND_PROGRAM_PREFILL_128},
      {256, QWEN_COMMAND_PROGRAM_PREFILL_256},
      {512, QWEN_COMMAND_PROGRAM_PREFILL_512},
  };

  for (const PrefillShape& shape : kShapes) {
    SCOPED_TRACE(shape.token_count);
    qwen_command_program_t program = QWEN_COMMAND_PROGRAM_COUNT;
    IREE_EXPECT_OK(
        qwen_command_select_prefill_program(shape.token_count, &program));
    EXPECT_EQ(program, shape.program);
  }
}

TEST(QwenCommandPackageContractTest, RejectsUnsupportedPrefillShape) {
  qwen_command_program_t program = QWEN_COMMAND_PROGRAM_PREFILL_32;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      qwen_command_select_prefill_program(/*token_count=*/33, &program));
  EXPECT_EQ(program, QWEN_COMMAND_PROGRAM_COUNT);
}

}  // namespace
