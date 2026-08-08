// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PROGRAM_PLAN_H_
#define LOOMC_PROGRAM_PLAN_H_

#include "loomc/program.h"
#include "loomc/result.h"
#include "loomc/workspace.h"

/// @file
/// Root-selected compiled-program plans.
///
/// A program plan is an immutable exact production graph formed after linking,
/// specialization, and partitioning. It names the selected roots, the
/// independently producible program units needed by those roots, and the dense
/// dependency slots connecting them. No program class is privileged: a unit
/// may produce a command program, device executable, VM module, native host
/// program, or another target-defined output.
///
/// Plan units can be compiled or loaded independently and in parallel. Program
/// assembly consumes an already-populated dense unit table and never compiles a
/// missing unit implicitly.

#ifdef __cplusplus
extern "C" {
#endif

/// Immutable exact compiled-program production plan.
///
/// @thread_safety
/// Plans are immutable after creation. Retained plan handles may be shared and
/// queried concurrently. Unit compilation is concurrent when each invocation
/// receives a distinct workspace.
typedef struct loomc_program_plan_t loomc_program_plan_t;

/// Plan-local selected-root token.
typedef struct loomc_program_plan_root_t {
  /// Plan-local root table value.
  uint64_t value;
} loomc_program_plan_root_t;

/// Plan-local independently producible unit token.
typedef struct loomc_program_plan_unit_t {
  /// Plan-local unit table value.
  uint64_t value;
} loomc_program_plan_unit_t;

/// Invalid program-plan root token value.
#define LOOMC_PROGRAM_PLAN_ROOT_INVALID_VALUE UINT64_MAX

/// Invalid program-plan unit token value.
#define LOOMC_PROGRAM_PLAN_UNIT_INVALID_VALUE UINT64_MAX

/// Returns an invalid program-plan root token.
static inline loomc_program_plan_root_t loomc_program_plan_root_invalid(void) {
  loomc_program_plan_root_t root = {
      LOOMC_PROGRAM_PLAN_ROOT_INVALID_VALUE,
  };
  return root;
}

/// Returns a program-plan root token for a dense plan index.
static inline loomc_program_plan_root_t loomc_program_plan_root_from_index(
    uint32_t index) {
  loomc_program_plan_root_t root = {index};
  return root;
}

/// Returns true when `root` contains a valid token value.
static inline bool loomc_program_plan_root_is_valid(
    loomc_program_plan_root_t root) {
  return root.value != LOOMC_PROGRAM_PLAN_ROOT_INVALID_VALUE;
}

/// Returns an invalid program-plan unit token.
static inline loomc_program_plan_unit_t loomc_program_plan_unit_invalid(void) {
  loomc_program_plan_unit_t unit = {
      LOOMC_PROGRAM_PLAN_UNIT_INVALID_VALUE,
  };
  return unit;
}

/// Returns a program-plan unit token for a dense plan index.
static inline loomc_program_plan_unit_t loomc_program_plan_unit_from_index(
    uint32_t index) {
  loomc_program_plan_unit_t unit = {index};
  return unit;
}

/// Returns true when `unit` contains a valid token value.
static inline bool loomc_program_plan_unit_is_valid(
    loomc_program_plan_unit_t unit) {
  return unit.value != LOOMC_PROGRAM_PLAN_UNIT_INVALID_VALUE;
}

/// Immutable metadata for one selected root.
typedef struct loomc_program_plan_root_info_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_ROOT_INFO`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Public root name borrowed from the plan.
  loomc_string_view_t name;

  /// Unit containing the root implementation.
  loomc_program_plan_unit_t program_unit;

  /// Number of root-local dependency slots.
  loomc_host_size_t dependency_count;
} loomc_program_plan_root_info_t;

/// Immutable metadata for one root-local dependency slot.
typedef struct loomc_program_plan_dependency_info_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_DEPENDENCY_INFO` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Dense target-defined slot populated by this dependency.
  uint32_t slot;

  /// Independently producible unit that populates `slot`.
  loomc_program_plan_unit_t unit;
} loomc_program_plan_dependency_info_t;

/// Immutable metadata for one independently producible unit.
typedef struct loomc_program_plan_unit_info_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_INFO`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Stable diagnostic identifier borrowed from the plan.
  loomc_string_view_t identifier;

  /// Opaque exact compiled-program cache key borrowed from the plan.
  ///
  /// The key identifies compiler inputs. It does not provide storage,
  /// transport, checksumming, or cache-integrity policy. An empty key means
  /// the target cannot provide a complete identity and the unit must not be
  /// loaded or stored through an identity-based cache.
  loomc_byte_span_t cache_key;
} loomc_program_plan_unit_info_t;

/// Unit compilation options.
typedef struct loomc_program_plan_unit_compile_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_COMPILE_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Unordered target-owned unit compilation extensions.
  const void* next;
} loomc_program_plan_unit_compile_options_t;

/// Dense plan-unit table supplied during assembly.
typedef struct loomc_program_plan_unit_table_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_TABLE`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  const void* next;

  /// Borrowed program array indexed by plan-unit token value.
  ///
  /// Entries outside the requested roots' dependency closure may be `NULL`.
  /// Every unit required by a requested root must be present.
  loomc_program_t* const* programs;

  /// Number of entries in `programs`.
  loomc_host_size_t program_count;
} loomc_program_plan_unit_table_t;

/// Program assembly options.
typedef struct loomc_program_plan_assembly_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_ASSEMBLY_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Unordered target-owned assembly extensions.
  const void* next;
} loomc_program_plan_assembly_options_t;

/// Retains `program_plan` for another owner.
LOOMC_API_EXPORT void loomc_program_plan_retain(
    loomc_program_plan_t* program_plan);

/// Releases `program_plan` from one owner. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_program_plan_release(
    loomc_program_plan_t* program_plan);

/// Returns the number of selected roots in `program_plan`.
LOOMC_API_EXPORT loomc_host_size_t
loomc_program_plan_root_count(const loomc_program_plan_t* program_plan);

/// Looks up a selected root by its public name.
LOOMC_API_EXPORT loomc_status_t loomc_program_plan_lookup_root(
    const loomc_program_plan_t* program_plan, loomc_string_view_t name,
    loomc_program_plan_root_t* out_root);

/// Returns metadata for one selected root.
LOOMC_API_EXPORT loomc_status_t loomc_program_plan_root_info(
    const loomc_program_plan_t* program_plan, loomc_program_plan_root_t root,
    loomc_program_plan_root_info_t* out_info);

/// Returns metadata for one root-local dependency slot.
LOOMC_API_EXPORT loomc_status_t loomc_program_plan_root_dependency_info(
    const loomc_program_plan_t* program_plan, loomc_program_plan_root_t root,
    loomc_host_size_t index, loomc_program_plan_dependency_info_t* out_info);

/// Returns the number of independently producible units in `program_plan`.
LOOMC_API_EXPORT loomc_host_size_t
loomc_program_plan_unit_count(const loomc_program_plan_t* program_plan);

/// Returns metadata for one independently producible unit.
LOOMC_API_EXPORT loomc_status_t loomc_program_plan_unit_info(
    const loomc_program_plan_t* program_plan, loomc_program_plan_unit_t unit,
    loomc_program_plan_unit_info_t* out_info);

/// Compiles one plan unit without compiling or finalizing another unit.
///
/// @ownership
/// The caller owns `out_program` and `out_result` on an OK return and releases
/// them with `loomc_program_release` and `loomc_result_release`.
///
/// @thread_safety
/// Calls for different or identical units may run concurrently when each call
/// receives a distinct workspace.
LOOMC_API_EXPORT loomc_status_t loomc_program_plan_compile_unit(
    const loomc_program_plan_t* program_plan, loomc_workspace_t* workspace,
    loomc_program_plan_unit_t unit,
    const loomc_program_plan_unit_compile_options_t* options,
    loomc_allocator_t allocator, loomc_program_t** out_program,
    loomc_result_t** out_result);

/// Loads one unit program from application-owned cached artifacts.
///
/// Input artifact views are borrowed for the call. The returned program owns
/// its loaded representation and artifacts.
LOOMC_API_EXPORT loomc_status_t loomc_program_plan_load_unit_program(
    const loomc_program_plan_t* program_plan, loomc_program_plan_unit_t unit,
    const loomc_artifact_t* artifacts, loomc_host_size_t artifact_count,
    loomc_allocator_t allocator, loomc_program_t** out_program,
    loomc_result_t** out_result);

/// Assembles selected roots from independently produced unit programs.
///
/// Assembly performs no unit compilation. The returned program exports every
/// requested root and owns or retains everything it requires after the plan,
/// unit table, workspaces, and operation results are released.
LOOMC_API_EXPORT loomc_status_t loomc_program_plan_assemble(
    const loomc_program_plan_t* program_plan, loomc_workspace_t* workspace,
    const loomc_program_plan_root_t* roots, loomc_host_size_t root_count,
    const loomc_program_plan_unit_table_t* unit_table,
    const loomc_program_plan_assembly_options_t* options,
    loomc_allocator_t allocator, loomc_program_t** out_program,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PROGRAM_PLAN_H_
