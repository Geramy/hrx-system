// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_ARTIFACT_STORAGE_H_
#define LOOMC_ARTIFACT_STORAGE_H_

#include "loomc/artifact.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared artifact storage owned by immutable public objects.
typedef struct loomc_artifact_storage_t loomc_artifact_storage_t;

// Creates empty artifact storage.
LOOMC_API_PRIVATE loomc_status_t loomc_artifact_storage_create(
    loomc_allocator_t allocator, loomc_artifact_storage_t** out_storage);

// Retains |storage| for another immutable owner.
LOOMC_API_PRIVATE void loomc_artifact_storage_retain(
    loomc_artifact_storage_t* storage);

// Releases |storage| from one owner.
LOOMC_API_PRIVATE void loomc_artifact_storage_release(
    loomc_artifact_storage_t* storage);

// Adds a copied artifact to |storage| while its owner is being built.
LOOMC_API_PRIVATE loomc_status_t loomc_artifact_storage_add(
    loomc_artifact_storage_t* storage, const loomc_artifact_t* artifact);

// Adds an artifact whose allocator-owned contents transfer on success.
LOOMC_API_PRIVATE loomc_status_t loomc_artifact_storage_add_take_contents(
    loomc_artifact_storage_t* storage, loomc_artifact_kind_t kind,
    loomc_string_view_t format, loomc_string_view_t identifier,
    loomc_byte_span_t contents);

// Returns the number of artifacts in |storage|.
LOOMC_API_PRIVATE loomc_host_size_t
loomc_artifact_storage_count(const loomc_artifact_storage_t* storage);

// Returns a borrowed artifact by index, or NULL when out of range.
LOOMC_API_PRIVATE const loomc_artifact_t* loomc_artifact_storage_at(
    const loomc_artifact_storage_t* storage, loomc_host_size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_ARTIFACT_STORAGE_H_
