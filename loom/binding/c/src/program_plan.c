// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "program_plan.h"

#include <string.h>

#include "compile.h"
#include "config.h"
#include "diagnostic.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "module.h"
#include "pass_program.h"
#include "program_environment.h"
#include "result.h"

typedef struct loomc_program_plan_root_storage_t {
  // Owned public root name.
  loomc_string_view_t name;

  // Plan-local unit containing the root implementation.
  uint32_t program_unit_index;

  // Owned root-local dependency table.
  loomc_program_plan_dependency_t* dependencies;

  // Number of entries in |dependencies|.
  loomc_host_size_t dependency_count;
} loomc_program_plan_root_storage_t;

typedef struct loomc_program_plan_unit_storage_t {
  // Owned stable diagnostic identifier.
  loomc_string_view_t identifier;

  // Owned exact compiled-program cache key.
  loomc_byte_span_t cache_key;
} loomc_program_plan_unit_storage_t;

struct loomc_program_plan_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release this plan.
  loomc_allocator_t allocator;

  // Selected roots in plan-local token order.
  loomc_program_plan_root_storage_t* roots;

  // Number of entries in |roots|.
  loomc_host_size_t root_count;

  // Independently producible units in plan-local token order.
  loomc_program_plan_unit_storage_t* units;

  // Number of entries in |units|.
  loomc_host_size_t unit_count;

  // Static target-owned operation table.
  const loomc_program_plan_operations_t* operations;

  // Target-owned immutable plan representation.
  void* target_storage;
};

static void loomc_program_plan_destroy(loomc_program_plan_t* program_plan) {
  loomc_allocator_t allocator = program_plan->allocator;
  if (program_plan->target_storage != NULL) {
    program_plan->operations->destroy(program_plan->target_storage, allocator);
  }
  for (loomc_host_size_t i = 0; i < program_plan->unit_count; ++i) {
    loomc_allocator_free(allocator,
                         (void*)program_plan->units[i].cache_key.data);
    loomc_allocator_free(allocator,
                         (void*)program_plan->units[i].identifier.data);
  }
  for (loomc_host_size_t i = 0; i < program_plan->root_count; ++i) {
    loomc_allocator_free(allocator, program_plan->roots[i].dependencies);
    loomc_allocator_free(allocator, (void*)program_plan->roots[i].name.data);
  }
  loomc_allocator_free(allocator, program_plan->units);
  loomc_allocator_free(allocator, program_plan->roots);
  loomc_allocator_free(allocator, program_plan);
}

static loomc_status_t loomc_program_plan_clone_byte_span(
    loomc_byte_span_t source, loomc_allocator_t allocator,
    loomc_byte_span_t* out_span) {
  *out_span = loomc_byte_span_empty();
  if (source.data_length == 0) return loomc_ok_status();
  uint8_t* data = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, source.data_length, (void**)&data));
  memcpy(data, source.data, source.data_length);
  *out_span = loomc_make_byte_span(data, source.data_length);
  return loomc_ok_status();
}

loomc_status_t loomc_program_plan_create(
    const loomc_program_plan_create_params_t* params,
    loomc_allocator_t allocator, loomc_program_plan_t** out_program_plan) {
  if (out_program_plan == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_program_plan must not be NULL");
  }
  *out_program_plan = NULL;
  if (params == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "params must not be NULL");
  }
  IREE_ASSERT(params->root_count == 0 || params->roots != NULL);
  IREE_ASSERT(params->unit_count == 0 || params->units != NULL);
  IREE_ASSERT(params->operations != NULL);
  IREE_ASSERT(params->operations->compile_unit != NULL);
  IREE_ASSERT(params->operations->load_unit_program != NULL);
  IREE_ASSERT(params->operations->assemble != NULL);
  IREE_ASSERT(params->operations->destroy != NULL);
  if (params->root_count > UINT32_MAX || params->unit_count > UINT32_MAX ||
      params->root_count >
          LOOMC_HOST_SIZE_MAX / sizeof(loomc_program_plan_root_storage_t) ||
      params->unit_count >
          LOOMC_HOST_SIZE_MAX / sizeof(loomc_program_plan_unit_storage_t)) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "program plan metadata tables are too large");
  }

  loomc_program_plan_t* program_plan = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(allocator, sizeof(*program_plan),
                                               (void**)&program_plan));
  memset(program_plan, 0, sizeof(*program_plan));
  iree_atomic_ref_count_init(&program_plan->ref_count);
  program_plan->allocator = allocator;
  program_plan->operations = params->operations;

  loomc_status_t status = loomc_ok_status();
  if (params->root_count != 0) {
    status = loomc_allocator_malloc(
        allocator, params->root_count * sizeof(*program_plan->roots),
        (void**)&program_plan->roots);
  }
  for (loomc_host_size_t i = 0;
       i < params->root_count && loomc_status_is_ok(status); ++i) {
    const loomc_program_plan_root_create_params_t* source = &params->roots[i];
    IREE_ASSERT_LT(source->program_unit_index, params->unit_count);
    IREE_ASSERT(source->dependency_count == 0 || source->dependencies != NULL);
    IREE_ASSERT(source->dependency_count <=
                LOOMC_HOST_SIZE_MAX / sizeof(*source->dependencies));
    loomc_program_plan_root_storage_t* target = &program_plan->roots[i];
    status = loomc_string_view_clone(source->name, allocator, &target->name);
    if (!loomc_status_is_ok(status)) break;
    target->program_unit_index = source->program_unit_index;
    ++program_plan->root_count;
    if (source->dependency_count != 0) {
      status = loomc_allocator_malloc(
          allocator, source->dependency_count * sizeof(*target->dependencies),
          (void**)&target->dependencies);
    }
    if (!loomc_status_is_ok(status)) break;
    for (loomc_host_size_t j = 0; j < source->dependency_count; ++j) {
      IREE_ASSERT_LT(source->dependencies[j].unit_index, params->unit_count);
    }
    if (source->dependency_count != 0) {
      memcpy(target->dependencies, source->dependencies,
             source->dependency_count * sizeof(*target->dependencies));
    }
    target->dependency_count = source->dependency_count;
  }

  if (loomc_status_is_ok(status) && params->unit_count != 0) {
    status = loomc_allocator_malloc(
        allocator, params->unit_count * sizeof(*program_plan->units),
        (void**)&program_plan->units);
  }
  for (loomc_host_size_t i = 0;
       i < params->unit_count && loomc_status_is_ok(status); ++i) {
    const loomc_program_plan_unit_create_params_t* source = &params->units[i];
    IREE_ASSERT(source->cache_key.data != NULL ||
                source->cache_key.data_length == 0);
    loomc_program_plan_unit_storage_t* target = &program_plan->units[i];
    status = loomc_string_view_clone(source->identifier, allocator,
                                     &target->identifier);
    if (!loomc_status_is_ok(status)) break;
    ++program_plan->unit_count;
    status = loomc_program_plan_clone_byte_span(source->cache_key, allocator,
                                                &target->cache_key);
  }

  if (loomc_status_is_ok(status)) {
    program_plan->target_storage = params->target_storage;
    *out_program_plan = program_plan;
  } else {
    loomc_program_plan_destroy(program_plan);
  }
  return status;
}

static loomc_status_t loomc_program_plan_validate_options(
    const loomc_program_plan_options_t* options) {
  if (options == NULL) return loomc_ok_status();
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan options structure_size is too small");
  }
  return loomc_config_validate_options(&options->config);
}

static loomc_status_t loomc_program_plan_finish_preparation_failure(
    loomc_result_t* result, loomc_status_t status,
    loomc_result_t** out_result) {
  if (!loomc_status_is_result_diagnostic(status)) {
    loomc_result_release(result);
    return status;
  }
  loomc_status_t add_status = loomc_result_fail_status_diagnostic_consume(
      result, /*source=*/NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      loomc_make_cstring_view("PROGRAM_PLAN/PREPARE"), status);
  if (loomc_status_is_ok(add_status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return add_status;
}

loomc_status_t loomc_prepare_programs(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* preparation_pass_program,
    const loomc_pass_program_t* unit_pass_program, loomc_module_t* module,
    const loomc_program_plan_options_t* options, loomc_allocator_t allocator,
    loomc_program_plan_t** out_program_plan, loomc_result_t** out_result) {
  if (out_program_plan == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "out_program_plan and out_result must not be NULL");
  }
  *out_program_plan = NULL;
  *out_result = NULL;
  if (compiler == NULL || workspace == NULL ||
      preparation_pass_program == NULL || unit_pass_program == NULL ||
      module == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compiler, workspace, pass programs, and module are required");
  }
  LOOMC_RETURN_IF_ERROR(loomc_program_plan_validate_options(options));
  loomc_context_t* module_context = loomc_module_context(module);
  if (loomc_pass_program_context(preparation_pass_program) != module_context ||
      loomc_pass_program_context(unit_pass_program) != module_context) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan pass programs and source module contexts must match");
  }
  loomc_program_environment_t* program_environment =
      loomc_compiler_program_environment(compiler);
  if (program_environment == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "compiler has no program-root providers");
  }

  const loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = options ? options->target_specialization : NULL,
      .config = options ? options->config : (loomc_config_options_t){0},
  };
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_compile_module(compiler, workspace, preparation_pass_program,
                           module, &compile_options, allocator, &result));
  if (!loomc_result_succeeded(result)) {
    *out_result = result;
    return loomc_ok_status();
  }

  const loomc_program_provider_t* provider = NULL;
  loomc_status_t status = loomc_program_environment_select_provider(
      program_environment, module, &provider);
  if (loomc_status_is_ok(status)) {
    status = provider->prepare(compiler, workspace, unit_pass_program, module,
                               options, result, allocator, out_program_plan);
  }
  if (loomc_status_is_ok(status)) {
    IREE_ASSERT(*out_program_plan != NULL);
    *out_result = result;
    return loomc_ok_status();
  }
  return loomc_program_plan_finish_preparation_failure(result, status,
                                                       out_result);
}

void loomc_program_plan_retain(loomc_program_plan_t* program_plan) {
  if (program_plan == NULL) return;
  iree_atomic_ref_count_inc(&program_plan->ref_count);
}

void loomc_program_plan_release(loomc_program_plan_t* program_plan) {
  if (program_plan == NULL) return;
  if (iree_atomic_ref_count_dec(&program_plan->ref_count) == 1) {
    loomc_program_plan_destroy(program_plan);
  }
}

loomc_host_size_t loomc_program_plan_root_count(
    const loomc_program_plan_t* program_plan) {
  return program_plan ? program_plan->root_count : 0;
}

loomc_status_t loomc_program_plan_lookup_root(
    const loomc_program_plan_t* program_plan, loomc_string_view_t name,
    loomc_program_plan_root_t* out_root) {
  if (program_plan == NULL || out_root == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program_plan and out_root must not be NULL");
  }
  *out_root = loomc_program_plan_root_invalid();
  if (name.data == NULL || name.size == 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "root name must be a non-empty string view");
  }
  for (loomc_host_size_t i = 0; i < program_plan->root_count; ++i) {
    if (loomc_string_view_equal(program_plan->roots[i].name, name)) {
      *out_root = loomc_program_plan_root_from_index((uint32_t)i);
      return loomc_ok_status();
    }
  }
  return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                           "program plan root was not found");
}

static loomc_status_t loomc_program_plan_validate_output_descriptor(
    loomc_structure_type_t actual_type, loomc_structure_type_t expected_type,
    loomc_host_size_t actual_size, loomc_host_size_t expected_size,
    const void* next) {
  if (actual_type != LOOMC_STRUCTURE_TYPE_NONE &&
      actual_type != expected_type) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan output descriptor has an unknown structure type");
  }
  if (actual_size != 0 && actual_size < expected_size) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan output descriptor structure_size is too small");
  }
  if (next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "program plan output descriptor extensions are not supported");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_program_plan_root_info(
    const loomc_program_plan_t* program_plan, loomc_program_plan_root_t root,
    loomc_program_plan_root_info_t* out_info) {
  if (out_info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_program_plan_validate_output_descriptor(
      out_info->type, LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_ROOT_INFO,
      out_info->structure_size, sizeof(*out_info), out_info->next));
  if (program_plan == NULL || root.value > UINT32_MAX ||
      root.value >= program_plan->root_count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program plan root token is out of range");
  }
  const loomc_program_plan_root_storage_t* storage =
      &program_plan->roots[(uint32_t)root.value];
  *out_info = (loomc_program_plan_root_info_t){
      .type = LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_ROOT_INFO,
      .structure_size = sizeof(*out_info),
      .name = storage->name,
      .program_unit =
          loomc_program_plan_unit_from_index(storage->program_unit_index),
      .dependency_count = storage->dependency_count,
  };
  return loomc_ok_status();
}

loomc_status_t loomc_program_plan_root_dependency_info(
    const loomc_program_plan_t* program_plan, loomc_program_plan_root_t root,
    loomc_host_size_t index, loomc_program_plan_dependency_info_t* out_info) {
  if (out_info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_program_plan_validate_output_descriptor(
      out_info->type, LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_DEPENDENCY_INFO,
      out_info->structure_size, sizeof(*out_info), out_info->next));
  if (program_plan == NULL || root.value > UINT32_MAX ||
      root.value >= program_plan->root_count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program plan root token is out of range");
  }
  const loomc_program_plan_root_storage_t* root_storage =
      &program_plan->roots[(uint32_t)root.value];
  if (index >= root_storage->dependency_count) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan root dependency index is out of range");
  }
  const loomc_program_plan_dependency_t* dependency =
      &root_storage->dependencies[index];
  *out_info = (loomc_program_plan_dependency_info_t){
      .type = LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_DEPENDENCY_INFO,
      .structure_size = sizeof(*out_info),
      .slot = dependency->slot,
      .unit = loomc_program_plan_unit_from_index(dependency->unit_index),
  };
  return loomc_ok_status();
}

loomc_host_size_t loomc_program_plan_unit_count(
    const loomc_program_plan_t* program_plan) {
  return program_plan ? program_plan->unit_count : 0;
}

loomc_status_t loomc_program_plan_unit_info(
    const loomc_program_plan_t* program_plan, loomc_program_plan_unit_t unit,
    loomc_program_plan_unit_info_t* out_info) {
  if (out_info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_program_plan_validate_output_descriptor(
      out_info->type, LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_INFO,
      out_info->structure_size, sizeof(*out_info), out_info->next));
  if (program_plan == NULL || unit.value > UINT32_MAX ||
      unit.value >= program_plan->unit_count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program plan unit token is out of range");
  }
  const loomc_program_plan_unit_storage_t* storage =
      &program_plan->units[(uint32_t)unit.value];
  *out_info = (loomc_program_plan_unit_info_t){
      .type = LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_INFO,
      .structure_size = sizeof(*out_info),
      .identifier = storage->identifier,
      .cache_key = storage->cache_key,
  };
  return loomc_ok_status();
}

static loomc_status_t loomc_program_plan_validate_compile_options(
    const loomc_program_plan_unit_compile_options_t* options) {
  if (options == NULL) return loomc_ok_status();
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_COMPILE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan unit compile options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan unit compile options structure_size is too small");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_program_plan_compile_unit(
    const loomc_program_plan_t* program_plan, loomc_workspace_t* workspace,
    loomc_program_plan_unit_t unit,
    const loomc_program_plan_unit_compile_options_t* options,
    loomc_allocator_t allocator, loomc_program_t** out_program,
    loomc_result_t** out_result) {
  if (out_program == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_program and out_result must not be NULL");
  }
  *out_program = NULL;
  *out_result = NULL;
  if (program_plan == NULL || workspace == NULL || unit.value > UINT32_MAX ||
      unit.value >= program_plan->unit_count) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program_plan, workspace, and a valid unit token are required");
  }
  LOOMC_RETURN_IF_ERROR(loomc_program_plan_validate_compile_options(options));
  return program_plan->operations->compile_unit(
      program_plan->target_storage, workspace, (uint32_t)unit.value, options,
      allocator, out_program, out_result);
}

loomc_status_t loomc_program_plan_load_unit_program(
    const loomc_program_plan_t* program_plan, loomc_program_plan_unit_t unit,
    const loomc_artifact_t* artifacts, loomc_host_size_t artifact_count,
    loomc_allocator_t allocator, loomc_program_t** out_program,
    loomc_result_t** out_result) {
  if (out_program == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_program and out_result must not be NULL");
  }
  *out_program = NULL;
  *out_result = NULL;
  if (program_plan == NULL || unit.value > UINT32_MAX ||
      unit.value >= program_plan->unit_count ||
      (artifact_count != 0 && artifacts == NULL)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program_plan, a valid unit token, and artifact table are required");
  }
  return program_plan->operations->load_unit_program(
      program_plan->target_storage, (uint32_t)unit.value, artifacts,
      artifact_count, allocator, out_program, out_result);
}

static loomc_status_t loomc_program_plan_validate_assembly_options(
    const loomc_program_plan_assembly_options_t* options) {
  if (options == NULL) return loomc_ok_status();
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_ASSEMBLY_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan assembly options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan assembly options structure_size is too small");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_program_plan_validate_unit_table(
    const loomc_program_plan_t* program_plan,
    const loomc_program_plan_unit_table_t* unit_table) {
  if (unit_table == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "unit_table must not be NULL");
  }
  if (unit_table->type != LOOMC_STRUCTURE_TYPE_NONE &&
      unit_table->type != LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_TABLE) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan unit table has an unknown structure type");
  }
  if (unit_table->structure_size != 0 &&
      unit_table->structure_size < sizeof(*unit_table)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan unit table structure_size is too small");
  }
  if (unit_table->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "program plan unit table extensions are not supported");
  }
  if (unit_table->program_count != program_plan->unit_count ||
      (unit_table->program_count != 0 && unit_table->programs == NULL)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan unit table must match the dense plan unit count");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_program_plan_assemble(
    const loomc_program_plan_t* program_plan, loomc_workspace_t* workspace,
    const loomc_program_plan_root_t* roots, loomc_host_size_t root_count,
    const loomc_program_plan_unit_table_t* unit_table,
    const loomc_program_plan_assembly_options_t* options,
    loomc_allocator_t allocator, loomc_program_t** out_program,
    loomc_result_t** out_result) {
  if (out_program == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_program and out_result must not be NULL");
  }
  *out_program = NULL;
  *out_result = NULL;
  if (program_plan == NULL || workspace == NULL || roots == NULL ||
      root_count == 0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program_plan, workspace, and at least one root are required");
  }
  for (loomc_host_size_t i = 0; i < root_count; ++i) {
    if (roots[i].value > UINT32_MAX ||
        roots[i].value >= program_plan->root_count) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "program plan root token is out of range");
    }
    for (loomc_host_size_t j = 0; j < i; ++j) {
      if (roots[i].value == roots[j].value) {
        return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                                 "program plan assembly roots must be unique");
      }
    }
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_program_plan_validate_unit_table(program_plan, unit_table));
  LOOMC_RETURN_IF_ERROR(loomc_program_plan_validate_assembly_options(options));
  return program_plan->operations->assemble(
      program_plan->target_storage, workspace, roots, root_count, unit_table,
      options, allocator, out_program, out_result);
}
