// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Serialization of aggregate command-program launch configuration.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_LAUNCH_ARTIFACT_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_LAUNCH_ARTIFACT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/cmd/lower/launch_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Serializes the aggregate host module in |graph| as evaluation-ready Loombc.
//
// The returned bytes contain only the public host function and its scalar
// dependency closure. Source locations and command-program compiler state are
// omitted. The caller owns |out_data| and frees it with |host_allocator|.
iree_status_t loom_cmd_launch_graph_serialize(
    const loom_cmd_launch_graph_t* graph, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator, iree_byte_span_t* out_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_LAUNCH_ARTIFACT_H_
