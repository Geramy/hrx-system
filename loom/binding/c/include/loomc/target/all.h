// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_ALL_H_
#define LOOMC_TARGET_ALL_H_

#include "loomc/target.h"

/// @file
/// Heterogeneous target environment linked into the embedding binary.

#ifdef __cplusplus
extern "C" {
#endif

/// Creates the heterogeneous target environment for this compiler build.
///
/// The environment contains the portable command representation plus every
/// target family and artifact emitter linked into the embedding binary. Target
/// profiles still select concrete targets independently for each function;
/// creating this environment does not select a device or artifact format.
///
/// Embedders that intentionally package one target family may instead use its
/// leaf target-environment constructor. Leaf products still include the
/// portable command representation but do not link unrelated target families.
///
/// @param allocator Host allocator used for environment storage.
/// @param out_target_environment Receives one retained environment on success.
/// @return OK when all linked provider contributions compose successfully.
///
/// @ownership
/// The caller owns the returned environment and releases it with
/// `loomc_target_environment_release`.
LOOMC_API_EXPORT loomc_status_t loomc_target_environment_create_all(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_ALL_H_
