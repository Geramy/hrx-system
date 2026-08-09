// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/all.h"

#include "loom/target/arch/cmd/provider.h"
#include "target.h"

#ifndef LOOMC_TARGET_ALL_HAVE_AMDGPU
#define LOOMC_TARGET_ALL_HAVE_AMDGPU 0
#endif  // LOOMC_TARGET_ALL_HAVE_AMDGPU
#ifndef LOOMC_TARGET_ALL_HAVE_IREE_VM
#define LOOMC_TARGET_ALL_HAVE_IREE_VM 0
#endif  // LOOMC_TARGET_ALL_HAVE_IREE_VM
#ifndef LOOMC_TARGET_ALL_HAVE_LLVMIR
#define LOOMC_TARGET_ALL_HAVE_LLVMIR 0
#endif  // LOOMC_TARGET_ALL_HAVE_LLVMIR
#ifndef LOOMC_TARGET_ALL_HAVE_SPIRV
#define LOOMC_TARGET_ALL_HAVE_SPIRV 0
#endif  // LOOMC_TARGET_ALL_HAVE_SPIRV
#ifndef LOOMC_TARGET_ALL_HAVE_WASM
#define LOOMC_TARGET_ALL_HAVE_WASM 0
#endif  // LOOMC_TARGET_ALL_HAVE_WASM
#ifndef LOOMC_TARGET_ALL_HAVE_X86
#define LOOMC_TARGET_ALL_HAVE_X86 0
#endif  // LOOMC_TARGET_ALL_HAVE_X86

#if LOOMC_TARGET_ALL_HAVE_AMDGPU
#include "loom/binding/c/target/amdgpu/provider.h"
#include "loom/target/arch/amdgpu/provider.h"
#endif  // LOOMC_TARGET_ALL_HAVE_AMDGPU
#if LOOMC_TARGET_ALL_HAVE_IREE_VM
#include "loom/target/arch/ireevm/provider.h"
#endif  // LOOMC_TARGET_ALL_HAVE_IREE_VM
#if LOOMC_TARGET_ALL_HAVE_LLVMIR
#include "loom/target/arch/llvmir/provider.h"
#include "loom/target/emit/llvmir/artifact_emitter.h"
#endif  // LOOMC_TARGET_ALL_HAVE_LLVMIR
#if LOOMC_TARGET_ALL_HAVE_SPIRV
#include "loom/binding/c/target/spirv/provider.h"
#include "loom/target/arch/spirv/provider.h"
#endif  // LOOMC_TARGET_ALL_HAVE_SPIRV
#if LOOMC_TARGET_ALL_HAVE_WASM
#include "loom/target/arch/wasm/provider.h"
#endif  // LOOMC_TARGET_ALL_HAVE_WASM
#if LOOMC_TARGET_ALL_HAVE_X86
#include "loom/target/arch/x86/provider.h"
#endif  // LOOMC_TARGET_ALL_HAVE_X86

static const loom_target_provider_t* const kLoomcAllTargetProviders[] = {
    &loom_cmd_target_provider,
#if LOOMC_TARGET_ALL_HAVE_IREE_VM
    &loom_ireevm_target_provider,
#endif  // LOOMC_TARGET_ALL_HAVE_IREE_VM
#if LOOMC_TARGET_ALL_HAVE_LLVMIR
    &loom_llvmir_target_provider, &loom_llvmir_artifact_emitter_provider,
#endif  // LOOMC_TARGET_ALL_HAVE_LLVMIR
#if LOOMC_TARGET_ALL_HAVE_SPIRV
    &loom_spirv_target_provider,  &loomc_spirv_artifact_emitter_provider,
#endif  // LOOMC_TARGET_ALL_HAVE_SPIRV
#if LOOMC_TARGET_ALL_HAVE_WASM
    &loom_wasm_target_provider,
#endif  // LOOMC_TARGET_ALL_HAVE_WASM
#if LOOMC_TARGET_ALL_HAVE_X86
    &loom_x86_target_provider,
#endif  // LOOMC_TARGET_ALL_HAVE_X86
#if LOOMC_TARGET_ALL_HAVE_AMDGPU
    &loom_amdgpu_target_provider, &loomc_amdgpu_artifact_emitter_provider,
#endif  // LOOMC_TARGET_ALL_HAVE_AMDGPU
};

static const loom_target_provider_set_t kLoomcAllTargetProviderSet = {
    .providers = kLoomcAllTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomcAllTargetProviders),
};

loomc_status_t loomc_target_environment_create_all(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment) {
  return loomc_target_environment_create_from_provider_set(
      &kLoomcAllTargetProviderSet, allocator, out_target_environment);
}
