// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/launch_config_module.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "context.h"
#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"
#include "loom/analysis/exact_function.h"
#include "loom/format/bytecode/reader.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loomc/iree.h"

enum {
  // Small host artifacts should remain small while retaining enough room for
  // decoded metadata and ordinary launch arithmetic.
  LOOMC_LAUNCH_CONFIG_MODULE_MINIMUM_BLOCK_SIZE = 4 * 1024,

  // One block owns the direct value table and one remains available for
  // touched-value or transient fact storage during steady evaluation.
  LOOMC_LAUNCH_CONFIG_CONTEXT_PREALLOCATED_BLOCK_COUNT = 2,
};

typedef struct loomc_launch_config_function_storage_t {
  // Prepared exact evaluator binding borrowed from the immutable module.
  loom_exact_function_t exact_function;
} loomc_launch_config_function_storage_t;

struct loomc_launch_config_module_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for handle and metadata storage.
  loomc_allocator_t allocator;

  // Immutable language context retained by the loaded module.
  loomc_context_t* context;

  // Arena block pool owning decoded module IR.
  iree_arena_block_pool_t block_pool;

  // Verified evaluation-ready host module.
  loom_module_t* module;

  // Dense exported launch-function table.
  loomc_launch_config_function_storage_t* functions;

  // Number of entries in functions.
  iree_host_size_t function_count;

  // Function ordinal for each module symbol, or UINT32_MAX when the symbol is
  // not an exported launch function.
  uint32_t* symbol_function_ordinals;
};

struct loomc_launch_config_context_t {
  // Atomic reference count for explicit shared ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release context storage.
  loomc_allocator_t allocator;

  // Immutable launch module retained by the context.
  loomc_launch_config_module_t* module;

  // Preallocated block pool for reusable evaluator state.
  iree_arena_block_pool_t block_pool;

  // Mutable exact evaluator state.
  loom_exact_function_context_t exact_context;
};

static bool loomc_launch_config_string_view_is_well_formed(
    loomc_string_view_t value) {
  return value.data != NULL || value.size == 0;
}

static loomc_status_t loomc_launch_config_module_validate_options(
    const loomc_launch_config_module_options_t* options) {
  if (options == NULL) return loomc_ok_status();
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_MODULE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config module options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config module options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config module option extensions are not supported");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_context_validate_options(
    const loomc_launch_config_context_options_t* options) {
  if (options == NULL) return loomc_ok_status();
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_CONTEXT_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config context options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config context options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config context option extensions are not supported");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_module_block_size(
    loomc_host_size_t artifact_length, iree_host_size_t* out_block_size) {
  *out_block_size = 0;
  const iree_host_size_t maximum_scaled_length = IREE_HOST_SIZE_MAX / 4;
  if (artifact_length > maximum_scaled_length) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "launch config artifact is too large to materialize");
  }
  iree_host_size_t requested_size = artifact_length * 4;
  requested_size =
      iree_max(requested_size,
               (iree_host_size_t)LOOMC_LAUNCH_CONFIG_MODULE_MINIMUM_BLOCK_SIZE);
  const iree_host_size_t block_size =
      iree_host_size_next_power_of_two(requested_size);
  if (!iree_arena_block_pool_is_valid_total_size(block_size)) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "launch config artifact requires an unsupported arena block size");
  }
  *out_block_size = block_size;
  return loomc_ok_status();
}

static void loomc_launch_config_module_destroy(
    loomc_launch_config_module_t* module) {
  loomc_allocator_t allocator = module->allocator;
  loomc_allocator_free(allocator, module->symbol_function_ordinals);
  loomc_allocator_free(allocator, module->functions);
  loom_module_free(module->module);
  iree_arena_block_pool_deinitialize(&module->block_pool);
  loomc_context_release(module->context);
  loomc_allocator_free(allocator, module);
}

static loomc_status_t loomc_launch_config_module_bind_functions(
    loomc_launch_config_module_t* module) {
  const loom_module_t* internal_module = module->module;
  iree_host_size_t function_count = 0;
  for (iree_host_size_t i = 0; i < internal_module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &internal_module->symbols.entries[i];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    const loom_func_like_t function =
        loom_func_like_cast(internal_module, symbol->defining_op);
    if (loom_func_like_visibility(function) != LOOM_FUNC_VISIBILITY_PUBLIC) {
      continue;
    }
    if (!loom_func_def_isa(function.op)) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "launch config artifact exports a function that is not a func.def");
    }
    ++function_count;
  }
  if (function_count == 0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config artifact contains no public functions");
  }
  if (function_count > UINT32_MAX) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "launch config artifact contains too many public functions");
  }

  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(
      module->allocator, function_count * sizeof(*module->functions),
      (void**)&module->functions));
  memset(module->functions, 0, function_count * sizeof(*module->functions));
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(module->allocator,
                             internal_module->symbols.count *
                                 sizeof(*module->symbol_function_ordinals),
                             (void**)&module->symbol_function_ordinals));
  memset(module->symbol_function_ordinals, 0xFF,
         internal_module->symbols.count *
             sizeof(*module->symbol_function_ordinals));

  iree_host_size_t function_ordinal = 0;
  for (iree_host_size_t symbol_ordinal = 0;
       symbol_ordinal < internal_module->symbols.count; ++symbol_ordinal) {
    const loom_symbol_t* symbol =
        &internal_module->symbols.entries[symbol_ordinal];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    const loom_func_like_t function =
        loom_func_like_cast(internal_module, symbol->defining_op);
    if (loom_func_like_visibility(function) != LOOM_FUNC_VISIBILITY_PUBLIC) {
      continue;
    }

    loomc_launch_config_function_storage_t* function_storage =
        &module->functions[function_ordinal];
    LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(loom_exact_function_bind(
        internal_module, function.op, &function_storage->exact_function)));
    if (function_storage->exact_function.result_count % 3 != 0) {
      return loomc_status_from_iree(iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "launch config function '@%.*s' must return complete xyz tuples",
          (int)function_storage->exact_function.name.size,
          function_storage->exact_function.name.data));
    }
    module->symbol_function_ordinals[symbol_ordinal] =
        (uint32_t)function_ordinal;
    ++function_ordinal;
  }
  module->function_count = function_count;
  return loomc_ok_status();
}

loomc_status_t loomc_launch_config_module_load(
    const loomc_artifact_t* artifact,
    const loomc_launch_config_module_options_t* options,
    loomc_allocator_t allocator, loomc_launch_config_module_t** out_module) {
  if (artifact == NULL || out_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact and out_module must not be NULL");
  }
  *out_module = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_validate_options(options));
  if (artifact->kind != LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "artifact kind is not LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG");
  }
  if (!loomc_launch_config_string_view_is_well_formed(artifact->format) ||
      !loomc_launch_config_string_view_is_well_formed(artifact->identifier)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact string view is malformed");
  }
  if (!loomc_string_view_equal(
          artifact->format,
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE))) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "launch config artifact format '%.*s' is not supported",
        (int)artifact->format.size, artifact->format.data));
  }
  if (artifact->contents.data == NULL && artifact->contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact contents have length but no data");
  }
  iree_host_size_t module_block_size = 0;
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_block_size(
      artifact->contents.data_length, &module_block_size));

  loomc_launch_config_module_t* module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  iree_atomic_ref_count_init(&module->ref_count);
  module->allocator = allocator;
  loomc_status_t status = loomc_context_create(
      /*options=*/NULL, allocator, &module->context);
  if (!loomc_status_is_ok(status)) {
    loomc_allocator_free(allocator, module);
    return status;
  }
  iree_arena_block_pool_initialize(module_block_size,
                                   iree_allocator_from_loomc(allocator),
                                   &module->block_pool);

  loom_bytecode_read_options_t read_options = {
      .verify_module = true,
  };
  loom_bytecode_read_result_t read_result = {0};
  iree_string_view_t identifier =
      iree_string_view_from_loomc(artifact->identifier);
  if (iree_string_view_is_empty(identifier)) {
    identifier = IREE_SV("launch_config.loombc");
  }
  status = loomc_status_from_iree(loom_bytecode_read_module(
      iree_make_const_byte_span(artifact->contents.data,
                                artifact->contents.data_length),
      identifier, loomc_context_loom_context(module->context),
      &module->block_pool, &read_options, &read_result, &module->module,
      iree_allocator_from_loomc(allocator)));
  if (loomc_status_is_ok(status) &&
      (read_result.error_count != 0 || module->module == NULL)) {
    status = loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config artifact is not valid verified Loom bytecode");
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_launch_config_module_bind_functions(module);
  }
  if (loomc_status_is_ok(status)) {
    *out_module = module;
  } else {
    loomc_launch_config_module_destroy(module);
  }
  return status;
}

void loomc_launch_config_module_retain(loomc_launch_config_module_t* module) {
  if (module == NULL) return;
  iree_atomic_ref_count_inc(&module->ref_count);
}

void loomc_launch_config_module_release(loomc_launch_config_module_t* module) {
  if (module == NULL) return;
  if (iree_atomic_ref_count_dec(&module->ref_count) == 1) {
    loomc_launch_config_module_destroy(module);
  }
}

loomc_host_size_t loomc_launch_config_module_function_count(
    const loomc_launch_config_module_t* module) {
  return module ? module->function_count : 0;
}

static loomc_status_t loomc_launch_config_module_resolve_function(
    const loomc_launch_config_module_t* module,
    loomc_launch_config_function_t function,
    const loomc_launch_config_function_storage_t** out_function) {
  *out_function = NULL;
  if (module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config module must not be NULL");
  }
  if (function.value > UINT32_MAX || function.value >= module->function_count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config function token is out of range");
  }
  *out_function = &module->functions[(uint32_t)function.value];
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_validate_function_info(
    const loomc_launch_config_function_info_t* info) {
  if (info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  if (info->type != LOOMC_STRUCTURE_TYPE_NONE &&
      info->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_FUNCTION_INFO) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config function info has an unknown structure type");
  }
  if (info->structure_size != 0 && info->structure_size < sizeof(*info)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config function info structure_size is too small");
  }
  if (info->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config function info extensions are not supported");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_launch_config_module_function_info(
    const loomc_launch_config_module_t* module,
    loomc_launch_config_function_t function,
    loomc_launch_config_function_info_t* out_info) {
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_validate_function_info(out_info));
  const loomc_launch_config_function_storage_t* function_storage = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_resolve_function(
      module, function, &function_storage));
  const loom_exact_function_t* exact_function =
      &function_storage->exact_function;
  *out_info = (loomc_launch_config_function_info_t){
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_FUNCTION_INFO,
      .structure_size = sizeof(*out_info),
      .name = loomc_string_view_from_iree(exact_function->name),
      .workload_argument_count = exact_function->argument_count,
      .result_count = exact_function->result_count / 3,
      .output_byte_length =
          (exact_function->result_count / 3) * sizeof(loomc_dimension3_t),
      .output_alignment = iree_alignof(uint32_t),
  };
  return loomc_ok_status();
}

loomc_status_t loomc_launch_config_module_lookup_function_by_name(
    const loomc_launch_config_module_t* module, loomc_string_view_t name,
    loomc_launch_config_function_t* out_function) {
  if (module == NULL || out_function == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config module and out_function must not be NULL");
  }
  *out_function = loomc_launch_config_function_invalid();
  if (!loomc_launch_config_string_view_is_well_formed(name) ||
      loomc_string_view_is_empty(name)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "function name must be a non-empty string view");
  }

  const loom_string_id_t name_id = loom_module_lookup_string(
      module->module, iree_string_view_from_loomc(name));
  if (name_id == LOOM_STRING_ID_INVALID) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "launch config function was not found");
  }
  const loom_symbol_id_t symbol_id =
      loom_module_find_symbol(module->module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "launch config function was not found");
  }
  const uint32_t function_ordinal = module->symbol_function_ordinals[symbol_id];
  if (function_ordinal == UINT32_MAX) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "launch config function was not found");
  }
  *out_function = loomc_launch_config_function_from_index(function_ordinal);
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_context_block_size(
    const loomc_launch_config_module_t* module,
    iree_host_size_t* out_block_size) {
  *out_block_size = 0;
  const iree_host_size_t value_capacity =
      loom_value_table_capacity(&module->module->values);
  const iree_host_size_t maximum_capacity =
      (IREE_HOST_SIZE_MAX - 1024) / sizeof(loom_value_facts_t);
  if (value_capacity > maximum_capacity) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "launch config module requires too much evaluation storage");
  }
  const iree_host_size_t requested_size =
      iree_max(value_capacity * sizeof(loom_value_facts_t) + 1024,
               (iree_host_size_t)LOOMC_LAUNCH_CONFIG_MODULE_MINIMUM_BLOCK_SIZE);
  const iree_host_size_t block_size =
      iree_host_size_next_power_of_two(requested_size);
  if (!iree_arena_block_pool_is_valid_total_size(block_size)) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "launch config module requires an unsupported arena block size");
  }
  *out_block_size = block_size;
  return loomc_ok_status();
}

static void loomc_launch_config_context_destroy(
    loomc_launch_config_context_t* context) {
  loomc_allocator_t allocator = context->allocator;
  loom_exact_function_context_deinitialize(&context->exact_context);
  iree_arena_block_pool_deinitialize(&context->block_pool);
  loomc_launch_config_module_release(context->module);
  loomc_allocator_free(allocator, context);
}

loomc_status_t loomc_launch_config_context_create(
    loomc_launch_config_module_t* module,
    const loomc_launch_config_context_options_t* options,
    loomc_allocator_t allocator, loomc_launch_config_context_t** out_context) {
  if (module == NULL || out_context == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config module and out_context must not be NULL");
  }
  *out_context = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_context_validate_options(options));
  iree_host_size_t context_block_size = 0;
  LOOMC_RETURN_IF_ERROR(
      loomc_launch_config_context_block_size(module, &context_block_size));

  loomc_launch_config_context_t* context = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*context), (void**)&context));
  memset(context, 0, sizeof(*context));
  iree_atomic_ref_count_init(&context->ref_count);
  context->allocator = allocator;
  context->module = module;
  loomc_launch_config_module_retain(module);
  iree_arena_block_pool_initialize(context_block_size,
                                   iree_allocator_from_loomc(allocator),
                                   &context->block_pool);
  loomc_status_t status =
      loomc_status_from_iree(iree_arena_block_pool_preallocate(
          &context->block_pool,
          LOOMC_LAUNCH_CONFIG_CONTEXT_PREALLOCATED_BLOCK_COUNT));
  if (loomc_status_is_ok(status)) {
    loom_exact_function_context_initialize(module->module, &context->block_pool,
                                           &context->exact_context);
    status = loomc_status_from_iree(loom_exact_function_context_prepare(
        &context->exact_context, &module->functions[0].exact_function));
  }
  if (loomc_status_is_ok(status)) {
    *out_context = context;
  } else {
    if (context->exact_context.module != NULL) {
      loom_exact_function_context_deinitialize(&context->exact_context);
    }
    iree_arena_block_pool_deinitialize(&context->block_pool);
    loomc_launch_config_module_release(context->module);
    loomc_allocator_free(allocator, context);
  }
  return status;
}

void loomc_launch_config_context_retain(
    loomc_launch_config_context_t* context) {
  if (context == NULL) return;
  iree_atomic_ref_count_inc(&context->ref_count);
}

void loomc_launch_config_context_release(
    loomc_launch_config_context_t* context) {
  if (context == NULL) return;
  if (iree_atomic_ref_count_dec(&context->ref_count) == 1) {
    loomc_launch_config_context_destroy(context);
  }
}

static loomc_status_t loomc_launch_config_validate_arguments(
    const loomc_launch_config_arguments_t* arguments) {
  if (arguments == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config arguments must not be NULL");
  }
  if (arguments->type != LOOMC_STRUCTURE_TYPE_NONE &&
      arguments->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_ARGUMENTS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config arguments have an unknown structure type");
  }
  if (arguments->structure_size != 0 &&
      arguments->structure_size < sizeof(*arguments)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config arguments structure_size is too small");
  }
  if (arguments->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config argument extensions are not supported");
  }
  if (arguments->workload_argument_count != 0 &&
      arguments->workload_arguments == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "workload_argument_count is non-zero but workload_arguments is NULL");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_validate_outputs(
    const loomc_launch_config_outputs_t* outputs) {
  if (outputs == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config outputs must not be NULL");
  }
  if (outputs->type != LOOMC_STRUCTURE_TYPE_NONE &&
      outputs->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_OUTPUTS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config outputs have an unknown structure type");
  }
  if (outputs->structure_size != 0 &&
      outputs->structure_size < sizeof(*outputs)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config outputs structure_size is too small");
  }
  if (outputs->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config output extensions are not supported");
  }
  if (outputs->storage_length != 0 && outputs->storage == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config output storage has length but no data");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_launch_config_context_evaluate(
    loomc_launch_config_context_t* context,
    loomc_launch_config_function_t function,
    const loomc_launch_config_arguments_t* arguments,
    loomc_launch_config_outputs_t* outputs) {
  if (context == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config context must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_validate_arguments(arguments));
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_validate_outputs(outputs));
  const loomc_launch_config_function_storage_t* function_storage = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_resolve_function(
      context->module, function, &function_storage));

  const iree_host_size_t output_count =
      function_storage->exact_function.result_count;
  const iree_host_size_t required_length = output_count * sizeof(uint32_t);
  if (outputs->storage_length < required_length) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "launch config output storage requires at least %" PRIhsz " bytes",
        required_length));
  }
  if (required_length != 0 &&
      !iree_host_ptr_has_alignment(outputs->storage, iree_alignof(uint32_t))) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config output storage must be aligned to %" PRIhsz " bytes",
        (iree_host_size_t)iree_alignof(uint32_t)));
  }

  return loomc_status_from_iree(loom_exact_function_evaluate_u32(
      &context->exact_context, &function_storage->exact_function,
      arguments->workload_arguments, arguments->workload_argument_count,
      (uint32_t*)outputs->storage, output_count));
}

static loomc_status_t loomc_launch_config_validate_result_config(
    const loomc_launch_config_t* config) {
  if (config == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_config must not be NULL");
  }
  if (config->type != LOOMC_STRUCTURE_TYPE_NONE &&
      config->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config has an unknown structure type");
  }
  if (config->structure_size != 0 && config->structure_size < sizeof(*config)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config structure_size is too small");
  }
  if (config->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config result extensions are not supported");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_launch_config_context_evaluate_one(
    loomc_launch_config_context_t* context,
    loomc_launch_config_function_t function,
    const loomc_launch_config_arguments_t* arguments,
    loomc_launch_config_t* out_config) {
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_validate_result_config(out_config));
  if (context == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config context must not be NULL");
  }
  const loomc_launch_config_function_storage_t* function_storage = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_resolve_function(
      context->module, function, &function_storage));
  if (function_storage->exact_function.result_count != 3) {
    return loomc_make_status(
        LOOMC_STATUS_FAILED_PRECONDITION,
        "single launch evaluation requires exactly one xyz result tuple");
  }

  loomc_dimension3_t workgroup_count = {0};
  loomc_launch_config_outputs_t outputs = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_OUTPUTS,
      .structure_size = sizeof(outputs),
      .storage = (uint8_t*)&workgroup_count,
      .storage_length = sizeof(workgroup_count),
  };
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_context_evaluate(
      context, function, arguments, &outputs));
  *out_config = (loomc_launch_config_t){
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(*out_config),
      .fields = LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT,
      .workgroup_count = workgroup_count,
  };
  return loomc_ok_status();
}
