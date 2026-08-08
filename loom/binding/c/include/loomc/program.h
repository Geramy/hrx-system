// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PROGRAM_H_
#define LOOMC_PROGRAM_H_

#include "loomc/artifact.h"

/// @file
/// Immutable compiled target programs.
///
/// A program is one compiled node in a root-selected compilation graph. It may
/// represent a command program, device executable, VM module, native host
/// program, or another target-defined output. The generic API deliberately does
/// not classify programs by target kind or privilege one class of output.
///
/// Programs own their direct target artifacts and may retain ancillary programs
/// in target-defined dependency slots. Artifacts belonging to dependencies are
/// not flattened into the root program. One program may export several entries,
/// such as prefill, decode, and MTP, which share the same compiled storage and
/// dependency set.

#ifdef __cplusplus
extern "C" {
#endif

/// Immutable compiled target program.
///
/// @thread_safety
/// Programs are immutable after creation. Retained program handles may be
/// shared and queried concurrently across threads.
typedef struct loomc_program_t loomc_program_t;

/// Program-local exported-entry token.
///
/// Tokens are meaningful only with the program that returned them. Export names
/// are the stable identity across independent compilation and loading.
typedef struct loomc_program_export_t {
  /// Program-local export table value.
  uint64_t value;
} loomc_program_export_t;

/// Invalid program export token value.
#define LOOMC_PROGRAM_EXPORT_INVALID_VALUE UINT64_MAX

/// Returns an invalid program export token.
static inline loomc_program_export_t loomc_program_export_invalid(void) {
  loomc_program_export_t export_token = {LOOMC_PROGRAM_EXPORT_INVALID_VALUE};
  return export_token;
}

/// Returns a program export token for a dense program index.
static inline loomc_program_export_t loomc_program_export_from_index(
    uint32_t index) {
  loomc_program_export_t export_token = {index};
  return export_token;
}

/// Returns true when `export_token` contains a valid token value.
static inline bool loomc_program_export_is_valid(
    loomc_program_export_t export_token) {
  return export_token.value != LOOMC_PROGRAM_EXPORT_INVALID_VALUE;
}

/// Immutable metadata for one program export.
typedef struct loomc_program_export_info_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_PROGRAM_EXPORT_INFO` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Public export name borrowed from the program.
  loomc_string_view_t name;
} loomc_program_export_info_t;

/// Immutable metadata for one retained ancillary program.
typedef struct loomc_program_dependency_info_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_PROGRAM_DEPENDENCY_INFO`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  void* next;

  /// Dense target-defined slot populated by this dependency.
  uint32_t slot;

  /// Borrowed ancillary program retained by the owning program.
  const loomc_program_t* program;
} loomc_program_dependency_info_t;

/// Retains `program` for another owner.
///
/// @param program Program to retain.
///
/// @thread_safety
/// Retain and release operations are safe from multiple threads.
LOOMC_API_EXPORT void loomc_program_retain(loomc_program_t* program);

/// Releases `program` from one owner.
///
/// @param program Program to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain and release operations are safe from multiple threads. The program
/// and its retained dependencies are destroyed when the final reference is
/// released.
LOOMC_API_EXPORT void loomc_program_release(loomc_program_t* program);

/// Returns the number of exported entries in `program`.
LOOMC_API_EXPORT loomc_host_size_t
loomc_program_export_count(const loomc_program_t* program);

/// Looks up a program export by its public name.
///
/// @param program Program to query.
/// @param name Non-empty public export name.
/// @param out_export Receives a program-local token on success.
/// @return OK when the export was found, `LOOMC_STATUS_NOT_FOUND` when no
/// export has the requested name, or an argument status for malformed input.
LOOMC_API_EXPORT loomc_status_t loomc_program_lookup_export(
    const loomc_program_t* program, loomc_string_view_t name,
    loomc_program_export_t* out_export);

/// Returns metadata for one program export.
///
/// @param program Program to query.
/// @param export_token Program-local export token.
/// @param out_info Caller-initialized metadata storage.
/// @return OK when the export token and output descriptor are valid.
///
/// @lifetime
/// Strings in `out_info` remain valid until `program` is released.
LOOMC_API_EXPORT loomc_status_t loomc_program_export_info(
    const loomc_program_t* program, loomc_program_export_t export_token,
    loomc_program_export_info_t* out_info);

/// Returns the number of direct target artifacts owned by `program`.
///
/// Artifacts owned by ancillary programs are intentionally excluded.
LOOMC_API_EXPORT loomc_host_size_t
loomc_program_artifact_count(const loomc_program_t* program);

/// Returns a direct target artifact by index.
///
/// @return Borrowed artifact view, or `NULL` when `index` is out of range.
///
/// @lifetime
/// The returned view remains valid until `program` is released.
LOOMC_API_EXPORT const loomc_artifact_t* loomc_program_artifact_at(
    const loomc_program_t* program, loomc_host_size_t index);

/// Returns the number of ancillary programs retained by `program`.
LOOMC_API_EXPORT loomc_host_size_t
loomc_program_dependency_count(const loomc_program_t* program);

/// Returns metadata for one retained ancillary program.
///
/// @param program Program to query.
/// @param index Dense dependency index.
/// @param out_info Caller-initialized metadata storage.
/// @return OK when the index and output descriptor are valid.
///
/// @lifetime
/// The program in `out_info` is borrowed from `program` and remains valid until
/// `program` is released.
LOOMC_API_EXPORT loomc_status_t loomc_program_dependency_info(
    const loomc_program_t* program, loomc_host_size_t index,
    loomc_program_dependency_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PROGRAM_H_
