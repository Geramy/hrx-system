// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepared exact evaluation of pure host functions.
//
// This is the evaluation backing for compact Loombc host artifacts. Binding
// validates the closed straight-line subset once. Mutable fact storage belongs
// to an exclusive reusable context, so hot evaluation performs no parsing,
// symbol lookup, compiler passes, or per-call allocation.

#ifndef LOOM_ANALYSIS_EXACT_FUNCTION_H_
#define LOOM_ANALYSIS_EXACT_FUNCTION_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/pass/value_facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable module-local binding for one evaluation-ready host function.
//
// All pointers borrow from |module|. The module must remain immutable and alive
// while the binding is used.
typedef struct loom_exact_function_t {
  // Module owning the bound function and all borrowed metadata.
  const loom_module_t* module;

  // Bound pure function.
  loom_func_like_t function;

  // Public function name borrowed from the module string table.
  iree_string_view_t name;

  // Positional function argument value IDs.
  const loom_value_id_t* argument_ids;

  // Number of entries in |argument_ids|.
  uint16_t argument_count;

  // Positional return operand value IDs.
  const loom_value_id_t* result_ids;

  // Number of entries in |result_ids|.
  uint16_t result_count;
} loom_exact_function_t;

// Mutable reusable state for exact host-function evaluation.
//
// A context may evaluate any bound function from |module|. It is reusable but
// not concurrently enterable. Independent contexts share the immutable module
// without locking.
typedef struct loom_exact_function_context_t {
  // Immutable module accepted by this context.
  const loom_module_t* module;

  // Reusable value-fact storage for one active invocation.
  loom_pass_value_fact_owner_t fact_owner;
} loom_exact_function_context_t;

// Binds and validates one evaluation-ready pure function.
//
// The module must already have passed ordinary Loom verification. This checks
// the additional artifact contract: one straight-line body, integer-like
// scalar arguments and results, and pure leaf operations with fact transfer
// functions. Unsupported artifact contents return a non-OK status.
iree_status_t loom_exact_function_bind(const loom_module_t* module,
                                       loom_op_t* function_op,
                                       loom_exact_function_t* out_function);

// Initializes a dormant reusable context for |module|.
void loom_exact_function_context_initialize(
    const loom_module_t* module, iree_arena_block_pool_t* block_pool,
    loom_exact_function_context_t* out_context);

// Deinitializes all mutable context storage.
void loom_exact_function_context_deinitialize(
    loom_exact_function_context_t* context);

// Prepares reusable fact storage for |function| without evaluating it.
//
// This moves storage growth to context preparation so subsequent evaluations
// of functions from the same module can remain allocation-free.
iree_status_t loom_exact_function_context_prepare(
    loom_exact_function_context_t* context,
    const loom_exact_function_t* function);

// Evaluates |function| and writes exact signed results positionally.
//
// Invocation values are checked against their scalar domains and authored
// function predicates. |arguments| and |outputs| are caller-owned. Evaluation
// reuses context storage and performs no symbol lookup or compiler pass.
iree_status_t loom_exact_function_evaluate_i64(
    loom_exact_function_context_t* context,
    const loom_exact_function_t* function, const int64_t* arguments,
    iree_host_size_t argument_count, int64_t* outputs,
    iree_host_size_t output_count);

// Evaluates |function| directly into unsigned 32-bit result storage.
//
// This is the workgroup-count table path. Every exact result must fit in u32;
// no intermediate result block or packing copy is produced.
iree_status_t loom_exact_function_evaluate_u32(
    loom_exact_function_context_t* context,
    const loom_exact_function_t* function, const int64_t* arguments,
    iree_host_size_t argument_count, uint32_t* outputs,
    iree_host_size_t output_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_EXACT_FUNCTION_H_
