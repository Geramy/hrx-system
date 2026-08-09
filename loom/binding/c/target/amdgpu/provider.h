// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU artifact-emitter contribution for C API target products.

#ifndef LOOM_BINDING_C_TARGET_AMDGPU_PROVIDER_H_
#define LOOM_BINDING_C_TARGET_AMDGPU_PROVIDER_H_

#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Provider contribution containing the public AMDGPU HSACO emitter.
extern const loom_target_provider_t loomc_amdgpu_artifact_emitter_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_BINDING_C_TARGET_AMDGPU_PROVIDER_H_
