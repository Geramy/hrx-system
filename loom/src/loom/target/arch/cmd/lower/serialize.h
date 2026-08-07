// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Serialization of closed cmd.core low functions.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_SERIALIZE_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_SERIALIZE_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Serializes one verified cmd.core command_program low function.
//
// Pure scalar, buffer-reference, and immutable argument-list SSA is evaluated
// once while serializing. The resulting artifact contains canonical flat
// buffer-reference, logical-argument, and command tables and retains no module,
// operation, value, symbol, or string storage. The caller owns the returned
// bytes and must free them with |host_allocator|.
iree_status_t loom_cmd_program_serialize_low(loom_module_t* module,
                                             const loom_op_t* function_op,
                                             iree_byte_span_t* out_data,
                                             iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_SERIALIZE_H_
