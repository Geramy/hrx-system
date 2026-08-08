// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_CMD_BASE_H_
#define LOOMC_TARGET_CMD_BASE_H_

#include "loomc/program_plan.h"

/// @file
/// Command-program compiler capability package.

#ifdef __cplusplus
extern "C" {
#endif

/// Command-program plan preparation options.
///
/// Attach this descriptor to `loomc_program_plan_options_t::next`. The exact
/// dependency artifact format becomes part of the immutable plan and is used
/// for every independently compiled dependency unit.
typedef struct loomc_cmd_program_plan_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PLAN_OPTIONS`.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Additional provider-owned options in the same unordered chain.
  const void* next;

  /// Exact loadable artifact format emitted for dependency units.
  loomc_string_view_t dependency_artifact_format;
} loomc_cmd_program_plan_options_t;

/// Creates a program environment containing command-program compilation.
///
/// Install the returned environment through
/// `loomc_compiler_program_options_t` when creating a compiler that will
/// prepare command-program roots. The environment is independent of hardware
/// target environments and may prepare roots whose dependency units target any
/// architecture linked into the compiler.
LOOMC_API_EXPORT loomc_status_t loomc_program_environment_create_command(
    loomc_allocator_t allocator,
    loomc_program_environment_t** out_program_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_CMD_BASE_H_
