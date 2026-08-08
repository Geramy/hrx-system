// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "program.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

typedef struct loomc_program_export_storage_t {
  // Owned public export name.
  loomc_string_view_t name;
} loomc_program_export_storage_t;

struct loomc_program_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release this program.
  loomc_allocator_t allocator;

  // Owned export metadata in program-local token order.
  loomc_program_export_storage_t* exports;

  // Number of entries in |exports|.
  loomc_host_size_t export_count;

  // Shared direct target artifacts.
  loomc_artifact_storage_t* artifact_storage;

  // Retained ancillary programs in target-defined slot order.
  loomc_program_dependency_t* dependencies;

  // Number of entries in |dependencies|.
  loomc_host_size_t dependency_count;

  // Optional target-owned immutable representation.
  void* target_storage;

  // Callback used to release |target_storage|.
  loomc_program_target_storage_release_fn_t target_storage_release;
};

static void loomc_program_destroy(loomc_program_t* program) {
  loomc_allocator_t allocator = program->allocator;
  if (program->target_storage != NULL) {
    program->target_storage_release(program->target_storage, allocator);
  }
  for (loomc_host_size_t i = 0; i < program->dependency_count; ++i) {
    loomc_program_release(program->dependencies[i].program);
  }
  for (loomc_host_size_t i = 0; i < program->export_count; ++i) {
    loomc_allocator_free(allocator, (void*)program->exports[i].name.data);
  }
  loomc_artifact_storage_release(program->artifact_storage);
  loomc_allocator_free(allocator, program->dependencies);
  loomc_allocator_free(allocator, program->exports);
  loomc_allocator_free(allocator, program);
}

loomc_status_t loomc_program_create(const loomc_program_create_params_t* params,
                                    loomc_allocator_t allocator,
                                    loomc_program_t** out_program) {
  if (out_program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_program must not be NULL");
  }
  *out_program = NULL;
  if (params == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "params must not be NULL");
  }
  IREE_ASSERT(params->export_count == 0 || params->export_names != NULL);
  IREE_ASSERT(params->dependency_count == 0 || params->dependencies != NULL);
  IREE_ASSERT((params->target_storage == NULL) ==
              (params->target_storage_release == NULL));
  if (params->export_count > UINT32_MAX ||
      params->export_count >
          LOOMC_HOST_SIZE_MAX / sizeof(loomc_program_export_storage_t) ||
      params->dependency_count >
          LOOMC_HOST_SIZE_MAX / sizeof(loomc_program_dependency_t)) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "program metadata tables are too large");
  }

  loomc_program_t* program = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*program), (void**)&program));
  memset(program, 0, sizeof(*program));
  iree_atomic_ref_count_init(&program->ref_count);
  program->allocator = allocator;

  loomc_status_t status = loomc_ok_status();
  if (params->export_count != 0) {
    status = loomc_allocator_malloc(
        allocator, params->export_count * sizeof(*program->exports),
        (void**)&program->exports);
  }
  for (loomc_host_size_t i = 0;
       i < params->export_count && loomc_status_is_ok(status); ++i) {
    status = loomc_string_view_clone(params->export_names[i], allocator,
                                     &program->exports[i].name);
    if (loomc_status_is_ok(status)) ++program->export_count;
  }
  if (loomc_status_is_ok(status) && params->dependency_count != 0) {
    status = loomc_allocator_malloc(
        allocator, params->dependency_count * sizeof(*program->dependencies),
        (void**)&program->dependencies);
  }
  if (loomc_status_is_ok(status)) {
    for (loomc_host_size_t i = 0; i < params->dependency_count; ++i) {
      program->dependencies[i] = params->dependencies[i];
      loomc_program_retain(program->dependencies[i].program);
      ++program->dependency_count;
    }
    program->artifact_storage = params->artifact_storage;
    loomc_artifact_storage_retain(program->artifact_storage);
    program->target_storage = params->target_storage;
    program->target_storage_release = params->target_storage_release;
    *out_program = program;
  } else {
    loomc_program_destroy(program);
  }
  return status;
}

void loomc_program_retain(loomc_program_t* program) {
  if (program == NULL) return;
  iree_atomic_ref_count_inc(&program->ref_count);
}

void loomc_program_release(loomc_program_t* program) {
  if (program == NULL) return;
  if (iree_atomic_ref_count_dec(&program->ref_count) == 1) {
    loomc_program_destroy(program);
  }
}

loomc_host_size_t loomc_program_export_count(const loomc_program_t* program) {
  return program ? program->export_count : 0;
}

loomc_status_t loomc_program_lookup_export(const loomc_program_t* program,
                                           loomc_string_view_t name,
                                           loomc_program_export_t* out_export) {
  if (program == NULL || out_export == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program and out_export must not be NULL");
  }
  *out_export = loomc_program_export_invalid();
  if (name.data == NULL || name.size == 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "export name must be a non-empty string view");
  }
  for (loomc_host_size_t i = 0; i < program->export_count; ++i) {
    if (loomc_string_view_equal(program->exports[i].name, name)) {
      *out_export = loomc_program_export_from_index((uint32_t)i);
      return loomc_ok_status();
    }
  }
  return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                           "program export was not found");
}

static loomc_status_t loomc_program_validate_export_info(
    const loomc_program_export_info_t* info) {
  if (info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  if (info->type != LOOMC_STRUCTURE_TYPE_NONE &&
      info->type != LOOMC_STRUCTURE_TYPE_PROGRAM_EXPORT_INFO) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program export info has an unknown structure type");
  }
  if (info->structure_size != 0 && info->structure_size < sizeof(*info)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program export info structure_size is too small");
  }
  if (info->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "program export info extensions are not supported");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_program_export_info(
    const loomc_program_t* program, loomc_program_export_t export_token,
    loomc_program_export_info_t* out_info) {
  LOOMC_RETURN_IF_ERROR(loomc_program_validate_export_info(out_info));
  if (program == NULL || export_token.value > UINT32_MAX ||
      export_token.value >= program->export_count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program export token is out of range");
  }
  *out_info = (loomc_program_export_info_t){
      .type = LOOMC_STRUCTURE_TYPE_PROGRAM_EXPORT_INFO,
      .structure_size = sizeof(*out_info),
      .name = program->exports[(uint32_t)export_token.value].name,
  };
  return loomc_ok_status();
}

loomc_host_size_t loomc_program_artifact_count(const loomc_program_t* program) {
  return program ? loomc_artifact_storage_count(program->artifact_storage) : 0;
}

const loomc_artifact_t* loomc_program_artifact_at(
    const loomc_program_t* program, loomc_host_size_t index) {
  return program ? loomc_artifact_storage_at(program->artifact_storage, index)
                 : NULL;
}

loomc_host_size_t loomc_program_dependency_count(
    const loomc_program_t* program) {
  return program ? program->dependency_count : 0;
}

static loomc_status_t loomc_program_validate_dependency_info(
    const loomc_program_dependency_info_t* info) {
  if (info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  if (info->type != LOOMC_STRUCTURE_TYPE_NONE &&
      info->type != LOOMC_STRUCTURE_TYPE_PROGRAM_DEPENDENCY_INFO) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program dependency info has an unknown structure type");
  }
  if (info->structure_size != 0 && info->structure_size < sizeof(*info)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program dependency info structure_size is too small");
  }
  if (info->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "program dependency info extensions are not supported");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_program_dependency_info(
    const loomc_program_t* program, loomc_host_size_t index,
    loomc_program_dependency_info_t* out_info) {
  LOOMC_RETURN_IF_ERROR(loomc_program_validate_dependency_info(out_info));
  if (program == NULL || index >= program->dependency_count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program dependency index is out of range");
  }
  *out_info = (loomc_program_dependency_info_t){
      .type = LOOMC_STRUCTURE_TYPE_PROGRAM_DEPENDENCY_INFO,
      .structure_size = sizeof(*out_info),
      .slot = program->dependencies[index].slot,
      .program = program->dependencies[index].program,
  };
  return loomc_ok_status();
}
