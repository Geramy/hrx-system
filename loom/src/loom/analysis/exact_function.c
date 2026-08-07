// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/exact_function.h"

#include <inttypes.h>
#include <string.h>

#include "loom/analysis/symbol_value_constraints.h"
#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"

static bool loom_exact_function_type_is_supported(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  int64_t minimum = 0;
  int64_t maximum = 0;
  return loom_scalar_type_integer_domain(loom_type_element_type(type), &minimum,
                                         &maximum);
}

static iree_string_view_t loom_exact_function_name(const loom_module_t* module,
                                                   loom_func_like_t function) {
  const loom_symbol_ref_t symbol_ref = loom_func_like_callee(function);
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref));
  IREE_ASSERT_EQ(symbol_ref.module_id, 0u);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  IREE_ASSERT_LT(symbol->name_id, module->strings.count);
  return module->strings.entries[symbol->name_id];
}

static iree_status_t loom_exact_function_validate_signature(
    const loom_module_t* module, const loom_value_id_t* value_ids,
    uint16_t value_count, iree_string_view_t function_name,
    iree_string_view_t role) {
  for (uint16_t i = 0; i < value_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, value_ids[i]);
    if (!loom_exact_function_type_is_supported(type)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "exact host function '@%.*s' %.*s %u is not an integer-like scalar",
          (int)function_name.size, function_name.data, (int)role.size,
          role.data, (unsigned)i);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_exact_function_validate_body(
    const loom_module_t* module, loom_func_like_t function,
    iree_string_view_t function_name, loom_value_slice_t* out_results) {
  *out_results = (loom_value_slice_t){0};
  loom_region_t* body_region = loom_func_like_body(function);
  if (body_region == NULL || body_region->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "exact host function '@%.*s' must have one straight-line body block",
        (int)function_name.size, function_name.data);
  }
  const loom_block_t* body = loom_region_const_entry_block(body_region);
  if (body->last_op == NULL || !loom_func_return_isa(body->last_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exact host function '@%.*s' has no func.return",
                            (int)function_name.size, function_name.data);
  }

  const loom_op_t* op = NULL;
  loom_block_for_each_op(body, op) {
    if (op == body->last_op) continue;
    const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
    const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
    if (op->region_count != 0 || !iree_any_bit_set(traits, LOOM_TRAIT_PURE) ||
        vtable == NULL || vtable->infer_facts == NULL) {
      const iree_string_view_t op_name = loom_op_name(module, op);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "exact host function '@%.*s' operation `%.*s` is not a pure "
          "fact-evaluable leaf",
          (int)function_name.size, function_name.data, (int)op_name.size,
          op_name.data);
    }
  }

  *out_results = loom_func_return_operands(body->last_op);
  return iree_ok_status();
}

iree_status_t loom_exact_function_bind(const loom_module_t* module,
                                       loom_op_t* function_op,
                                       loom_exact_function_t* out_function) {
  if (module == NULL || function_op == NULL || out_function == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "module, function_op, and out_function must not be NULL");
  }
  *out_function = (loom_exact_function_t){0};
  if (!loom_func_def_isa(function_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exact host artifact entry must be a func.def");
  }

  const loom_func_like_t function = loom_func_like_cast(module, function_op);
  IREE_ASSERT(loom_func_like_isa(function));
  const iree_string_view_t function_name =
      loom_exact_function_name(module, function);
  if (loom_func_like_purity(function) != LOOM_FUNC_PURITY_PURE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exact host function '@%.*s' must be pure",
                            (int)function_name.size, function_name.data);
  }
  const uint8_t calling_convention = loom_func_like_cc(function);
  if (calling_convention != 0 && calling_convention != LOOM_FUNC_CC_HOST) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "exact host function '@%.*s' must use the host calling convention",
        (int)function_name.size, function_name.data);
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  IREE_RETURN_IF_ERROR(loom_exact_function_validate_signature(
      module, argument_ids, argument_count, function_name,
      IREE_SV("argument")));

  loom_value_slice_t results = {0};
  IREE_RETURN_IF_ERROR(loom_exact_function_validate_body(
      module, function, function_name, &results));
  IREE_RETURN_IF_ERROR(loom_exact_function_validate_signature(
      module, results.values, results.count, function_name, IREE_SV("result")));

  *out_function = (loom_exact_function_t){
      .module = module,
      .function = function,
      .name = function_name,
      .argument_ids = argument_ids,
      .argument_count = argument_count,
      .result_ids = results.values,
      .result_count = results.count,
  };
  return iree_ok_status();
}

void loom_exact_function_context_initialize(
    const loom_module_t* module, iree_arena_block_pool_t* block_pool,
    loom_exact_function_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  out_context->module = module;
  loom_pass_value_fact_owner_initialize(block_pool, &out_context->fact_owner);
}

void loom_exact_function_context_deinitialize(
    loom_exact_function_context_t* context) {
  loom_pass_value_fact_owner_deinitialize(&context->fact_owner);
  memset(context, 0, sizeof(*context));
}

static iree_status_t loom_exact_function_check_argument(
    const loom_exact_function_t* function, uint16_t argument_ordinal,
    int64_t value) {
  const loom_value_id_t value_id = function->argument_ids[argument_ordinal];
  const loom_type_t type = loom_module_value_type(function->module, value_id);
  int64_t minimum = 0;
  int64_t maximum = 0;
  const bool has_integer_domain = loom_scalar_type_integer_domain(
      loom_type_element_type(type), &minimum, &maximum);
  IREE_ASSERT(has_integer_domain);
  (void)has_integer_domain;
  if (value < minimum || value > maximum) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "exact host function '@%.*s' argument %u value %" PRId64
        " is outside [%" PRId64 ", %" PRId64 "]",
        (int)function->name.size, function->name.data,
        (unsigned)argument_ordinal, value, minimum, maximum);
  }

  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function->function, &predicate_count);
  return loom_symbol_value_constraints_check_exact(
      function->name, type, value_id, loom_attr_i64(value),
      loom_attr_predicate_list((loom_predicate_t*)predicates, predicate_count));
}

static iree_status_t loom_exact_function_compute(
    loom_exact_function_context_t* context,
    const loom_exact_function_t* function, const int64_t* arguments,
    iree_host_size_t argument_count, iree_host_size_t output_count,
    loom_value_fact_table_t** out_fact_table) {
  *out_fact_table = NULL;
  if (context == NULL || function == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "context and function must not be NULL");
  }
  if (context->module != function->module) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "exact host function belongs to a different evaluation module");
  }
  if (argument_count != function->argument_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exact host function '@%.*s' expects %u arguments "
                            "but received %" PRIhsz,
                            (int)function->name.size, function->name.data,
                            (unsigned)function->argument_count, argument_count);
  }
  if (argument_count != 0 && arguments == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty arguments must not be NULL");
  }
  if (output_count != function->result_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "exact host function '@%.*s' produces %u results but received %" PRIhsz
        " output slots",
        (int)function->name.size, function->name.data,
        (unsigned)function->result_count, output_count);
  }

  for (uint16_t i = 0; i < function->argument_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_exact_function_check_argument(function, i, arguments[i]));
  }

  loom_value_fact_table_t* fact_table = NULL;
  iree_status_t status = loom_pass_value_fact_owner_prepare(
      &context->fact_owner, context->module,
      loom_pass_value_fact_scope_function(function->function), &fact_table);
  for (uint16_t i = 0;
       i < function->argument_count && iree_status_is_ok(status); ++i) {
    status =
        loom_value_fact_table_define(fact_table, function->argument_ids[i],
                                     loom_value_facts_exact_i64(arguments[i]));
  }
  if (iree_status_is_ok(status)) {
    status = loom_value_fact_table_compute(fact_table, context->module,
                                           function->function);
  }
  if (iree_status_is_ok(status)) {
    *out_fact_table = fact_table;
  } else {
    loom_pass_value_fact_owner_invalidate(&context->fact_owner);
  }
  return status;
}

iree_status_t loom_exact_function_evaluate_i64(
    loom_exact_function_context_t* context,
    const loom_exact_function_t* function, const int64_t* arguments,
    iree_host_size_t argument_count, int64_t* outputs,
    iree_host_size_t output_count) {
  if (output_count != 0 && outputs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty outputs must not be NULL");
  }
  loom_value_fact_table_t* fact_table = NULL;
  IREE_RETURN_IF_ERROR(loom_exact_function_compute(
      context, function, arguments, argument_count, output_count, &fact_table));
  for (uint16_t i = 0; i < function->result_count; ++i) {
    if (!loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(fact_table, function->result_ids[i]),
            &outputs[i])) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "exact host function '@%.*s' result %u is not exact",
          (int)function->name.size, function->name.data, (unsigned)i);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_exact_function_evaluate_u32(
    loom_exact_function_context_t* context,
    const loom_exact_function_t* function, const int64_t* arguments,
    iree_host_size_t argument_count, uint32_t* outputs,
    iree_host_size_t output_count) {
  if (output_count != 0 && outputs == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "non-empty outputs must not be NULL");
  }
  loom_value_fact_table_t* fact_table = NULL;
  IREE_RETURN_IF_ERROR(loom_exact_function_compute(
      context, function, arguments, argument_count, output_count, &fact_table));
  for (uint16_t i = 0; i < function->result_count; ++i) {
    int64_t value = 0;
    if (!loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(fact_table, function->result_ids[i]),
            &value)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "exact host function '@%.*s' result %u is not exact",
          (int)function->name.size, function->name.data, (unsigned)i);
    }
    if (value < 0 || value > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "exact host function '@%.*s' result %u value %" PRId64
          " does not fit in u32",
          (int)function->name.size, function->name.data, (unsigned)i, value);
    }
    outputs[i] = (uint32_t)value;
  }
  return iree_ok_status();
}
