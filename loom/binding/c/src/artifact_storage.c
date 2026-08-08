// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "artifact_storage.h"

#include <string.h>

#include "iree/base/internal/atomics.h"

typedef struct loomc_owned_artifact_t {
  // Public artifact view returned to callers.
  loomc_artifact_t value;
  // Storage for value.format.
  loomc_string_view_t format_storage;
  // Storage for value.identifier.
  loomc_string_view_t identifier_storage;
  // Storage for value.contents.
  loomc_byte_span_t contents_storage;
} loomc_owned_artifact_t;

typedef enum loomc_artifact_storage_mode_e {
  // Copy artifact contents into storage-owned memory.
  LOOMC_ARTIFACT_STORAGE_MODE_COPY = 0,

  // Take allocator-owned artifact contents on success.
  LOOMC_ARTIFACT_STORAGE_MODE_TAKE = 1,
} loomc_artifact_storage_mode_t;

struct loomc_artifact_storage_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used to release this storage.
  loomc_allocator_t allocator;
  // Growable artifact array used while the storage is being built.
  loomc_owned_artifact_t* artifacts;
  // Number of live artifacts.
  loomc_host_size_t artifact_count;
  // Allocated artifact capacity.
  loomc_host_size_t artifact_capacity;
};

static void loomc_owned_artifact_deinitialize(
    loomc_allocator_t allocator, loomc_owned_artifact_t* artifact) {
  loomc_allocator_free(allocator, (void*)artifact->format_storage.data);
  loomc_allocator_free(allocator, (void*)artifact->identifier_storage.data);
  loomc_allocator_free(allocator, (void*)artifact->contents_storage.data);
  *artifact = (loomc_owned_artifact_t){0};
}

static void loomc_artifact_storage_destroy(loomc_artifact_storage_t* storage) {
  loomc_allocator_t allocator = storage->allocator;
  for (loomc_host_size_t i = 0; i < storage->artifact_count; ++i) {
    loomc_owned_artifact_deinitialize(allocator, &storage->artifacts[i]);
  }
  loomc_allocator_free(allocator, storage->artifacts);
  loomc_allocator_free(allocator, storage);
}

loomc_status_t loomc_artifact_storage_create(
    loomc_allocator_t allocator, loomc_artifact_storage_t** out_storage) {
  if (out_storage == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_storage must not be NULL");
  }
  *out_storage = NULL;
  loomc_artifact_storage_t* storage = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  memset(storage, 0, sizeof(*storage));
  iree_atomic_ref_count_init(&storage->ref_count);
  storage->allocator = allocator;
  *out_storage = storage;
  return loomc_ok_status();
}

void loomc_artifact_storage_retain(loomc_artifact_storage_t* storage) {
  if (storage == NULL) return;
  iree_atomic_ref_count_inc(&storage->ref_count);
}

void loomc_artifact_storage_release(loomc_artifact_storage_t* storage) {
  if (storage == NULL) return;
  if (iree_atomic_ref_count_dec(&storage->ref_count) == 1) {
    loomc_artifact_storage_destroy(storage);
  }
}

static loomc_status_t loomc_artifact_storage_reserve(
    loomc_artifact_storage_t* storage, loomc_host_size_t required_count) {
  if (required_count <= storage->artifact_capacity) {
    return loomc_ok_status();
  }
  loomc_host_size_t new_capacity = 4;
  if (storage->artifact_capacity != 0) {
    new_capacity = storage->artifact_capacity > LOOMC_HOST_SIZE_MAX / 2
                       ? required_count
                       : storage->artifact_capacity * 2;
  }
  while (new_capacity < required_count) {
    if (new_capacity > LOOMC_HOST_SIZE_MAX / 2) {
      new_capacity = required_count;
      break;
    }
    new_capacity *= 2;
  }
  if (new_capacity > LOOMC_HOST_SIZE_MAX / sizeof(*storage->artifacts)) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "artifact table is too large");
  }
  loomc_owned_artifact_t* new_artifacts = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      storage->allocator, new_capacity * sizeof(*storage->artifacts),
      (void**)&new_artifacts));
  if (storage->artifact_count != 0) {
    memcpy(new_artifacts, storage->artifacts,
           storage->artifact_count * sizeof(*storage->artifacts));
  }
  loomc_allocator_free(storage->allocator, storage->artifacts);
  storage->artifacts = new_artifacts;
  storage->artifact_capacity = new_capacity;
  return loomc_ok_status();
}

static loomc_status_t loomc_artifact_storage_add_with_mode(
    loomc_artifact_storage_t* storage, const loomc_artifact_t* artifact,
    loomc_artifact_storage_mode_t mode) {
  if (storage == NULL || artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "storage and artifact must not be NULL");
  }
  if (artifact->contents.data == NULL && artifact->contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact contents have length but no data");
  }
  if (storage->artifact_count == LOOMC_HOST_SIZE_MAX) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "artifact table is too large");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_artifact_storage_reserve(storage, storage->artifact_count + 1));
  loomc_owned_artifact_t* target = &storage->artifacts[storage->artifact_count];
  *target = (loomc_owned_artifact_t){0};
  loomc_status_t status = loomc_string_view_clone(
      artifact->format, storage->allocator, &target->format_storage);
  if (loomc_status_is_ok(status)) {
    status = loomc_string_view_clone(artifact->identifier, storage->allocator,
                                     &target->identifier_storage);
  }
  if (loomc_status_is_ok(status) && mode == LOOMC_ARTIFACT_STORAGE_MODE_COPY &&
      artifact->contents.data_length != 0) {
    uint8_t* contents = NULL;
    status = loomc_allocator_malloc_uninitialized(
        storage->allocator, artifact->contents.data_length, (void**)&contents);
    if (loomc_status_is_ok(status)) {
      memcpy(contents, artifact->contents.data, artifact->contents.data_length);
      target->contents_storage =
          loomc_make_byte_span(contents, artifact->contents.data_length);
    }
  }
  if (loomc_status_is_ok(status) && mode == LOOMC_ARTIFACT_STORAGE_MODE_TAKE) {
    target->contents_storage = artifact->contents;
  }
  if (loomc_status_is_ok(status)) {
    target->value = *artifact;
    target->value.format = target->format_storage;
    target->value.identifier = target->identifier_storage;
    target->value.contents = target->contents_storage;
    ++storage->artifact_count;
  } else {
    loomc_owned_artifact_deinitialize(storage->allocator, target);
  }
  return status;
}

loomc_status_t loomc_artifact_storage_add(loomc_artifact_storage_t* storage,
                                          const loomc_artifact_t* artifact) {
  return loomc_artifact_storage_add_with_mode(storage, artifact,
                                              LOOMC_ARTIFACT_STORAGE_MODE_COPY);
}

loomc_status_t loomc_artifact_storage_add_take_contents(
    loomc_artifact_storage_t* storage, loomc_artifact_kind_t kind,
    loomc_string_view_t format, loomc_string_view_t identifier,
    loomc_byte_span_t contents) {
  const loomc_artifact_t artifact = {
      .kind = kind,
      .format = format,
      .identifier = identifier,
      .contents = contents,
  };
  return loomc_artifact_storage_add_with_mode(storage, &artifact,
                                              LOOMC_ARTIFACT_STORAGE_MODE_TAKE);
}

loomc_host_size_t loomc_artifact_storage_count(
    const loomc_artifact_storage_t* storage) {
  return storage ? storage->artifact_count : 0;
}

const loomc_artifact_t* loomc_artifact_storage_at(
    const loomc_artifact_storage_t* storage, loomc_host_size_t index) {
  if (storage == NULL || index >= storage->artifact_count) return NULL;
  return &storage->artifacts[index].value;
}
