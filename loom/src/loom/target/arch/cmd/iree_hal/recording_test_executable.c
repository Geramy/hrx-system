// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/iree_hal/recording_test_executable.h"

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"

static int loom_cmd_recording_test_add_u32(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  (void)environment;
  const uint32_t index = workgroup_state->workgroup_id_x;
  if ((index + 1) * sizeof(uint32_t) > dispatch_state->binding_lengths[0] ||
      (index + 1) * sizeof(uint32_t) > dispatch_state->binding_lengths[1]) {
    return 1;
  }
  const uint32_t addend = *(const uint32_t*)dispatch_state->constants.data;
  const uint32_t* source = (const uint32_t*)dispatch_state->binding_ptrs[0];
  uint32_t* target = (uint32_t*)dispatch_state->binding_ptrs[1];
  target[index] = source[index] + addend;
  return 0;
}

static int loom_cmd_recording_test_increment_u32(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {
  (void)environment;
  const uint32_t index = workgroup_state->workgroup_id_x;
  if ((index + 1) * sizeof(uint32_t) > dispatch_state->binding_lengths[0] ||
      (index + 1) * sizeof(uint32_t) > dispatch_state->binding_lengths[1]) {
    return 1;
  }
  const uint32_t* source = (const uint32_t*)dispatch_state->binding_ptrs[0];
  uint32_t* target = (uint32_t*)dispatch_state->binding_ptrs[1];
  target[index] = source[index] + 7;
  return 0;
}

static const iree_hal_executable_library_header_t kLibraryHeader = {
    .version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
    .name = "loom_cmd_recording_test",
    .features = IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE,
    .sanitizer = IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE,
};

static const iree_hal_executable_dispatch_v0_t kEntryPoints[] = {
    loom_cmd_recording_test_add_u32,
    loom_cmd_recording_test_increment_u32,
};

static const iree_hal_executable_dispatch_parameter_v0_t kAddU32Parameters[] = {
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT,
        .size = sizeof(uint32_t),
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 0,
        .offset = 0,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 1,
        .offset = 0,
    },
    {
        .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
        .name = 2,
        .offset = 1,
    },
};

static const iree_hal_executable_dispatch_parameter_v0_t
    kIncrementU32Parameters[] = {
        {
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 1,
            .offset = 0,
        },
        {
            .type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING,
            .flags = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_FLAG_V0_NONE,
            .name = 2,
            .offset = 1,
        },
};

static const iree_hal_executable_dispatch_parameter_v0_t* kEntryParameters[] = {
    kAddU32Parameters,
    kIncrementU32Parameters,
};

static const char* kParameterNames[] = {
    "addend",
    "source",
    "target",
};

static const iree_hal_executable_dispatch_attrs_v0_t kEntryAttributes[] = {
    {
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .constant_byte_length = sizeof(uint32_t),
        .binding_count = 2,
        .workgroup_size_x = 1,
        .workgroup_size_y = 1,
        .workgroup_size_z = 1,
        .parameter_count = IREE_ARRAYSIZE(kAddU32Parameters),
    },
    {
        .flags = IREE_HAL_EXECUTABLE_DISPATCH_FLAG_V0_NONE,
        .constant_byte_length = 0,
        .binding_count = 2,
        .workgroup_size_x = 1,
        .workgroup_size_y = 1,
        .workgroup_size_z = 1,
        .parameter_count = IREE_ARRAYSIZE(kIncrementU32Parameters),
    },
};

static const char* kEntryPointNames[] = {
    "add_u32",
    "increment_u32",
};

static const iree_hal_executable_library_v0_t kLibrary = {
    .header = &kLibraryHeader,
    .imports =
        {
            .count = 0,
            .symbols = NULL,
        },
    .exports =
        {
            .count = IREE_ARRAYSIZE(kEntryPoints),
            .ptrs = kEntryPoints,
            .attrs = kEntryAttributes,
            .params = kEntryParameters,
            .names = kEntryPointNames,
            .parameter_names = kParameterNames,
        },
    .constants =
        {
            .count = 0,
        },
};

const iree_hal_executable_library_header_t* const*
loom_cmd_recording_test_executable_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {
  (void)environment;
  return max_version >= IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST
             ? &kLibrary.header
             : NULL;
}
