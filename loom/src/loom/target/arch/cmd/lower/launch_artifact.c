// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/launch_artifact.h"

#include "iree/io/vec_stream.h"
#include "loom/format/bytecode/writer.h"

iree_status_t loom_cmd_launch_program_serialize(
    const loom_module_t* module, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator, iree_byte_span_t* out_data) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_data);
  *out_data = iree_byte_span_empty();

  iree_io_stream_t* stream = NULL;
  iree_status_t status = iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
          IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
      4096, host_allocator, &stream);
  if (iree_status_is_ok(status)) {
    const loom_bytecode_write_options_t options = {
        .producer = IREE_SV("loom-command-program"),
        .location_mode = LOOM_BYTECODE_LOCATION_MODE_NO_LOCATIONS,
    };
    status = loom_bytecode_write_module(module, stream, &options, block_pool);
  }

  iree_host_size_t data_length = 0;
  uint8_t* data = NULL;
  if (iree_status_is_ok(status)) {
    const iree_io_stream_pos_t stream_length = iree_io_stream_length(stream);
    if (stream_length < 0 || stream_length > IREE_HOST_SIZE_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "launch configuration artifact is not host-addressable");
    } else {
      data_length = (iree_host_size_t)stream_length;
    }
  }
  if (iree_status_is_ok(status) && data_length != 0) {
    status = iree_allocator_malloc(host_allocator, data_length, (void**)&data);
  }
  if (iree_status_is_ok(status) && data_length != 0) {
    status = iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0);
  }
  if (iree_status_is_ok(status) && data_length != 0) {
    status = iree_io_stream_read(stream, data_length, data, NULL);
  }

  iree_io_stream_release(stream);
  if (iree_status_is_ok(status)) {
    *out_data = iree_make_byte_span(data, data_length);
  } else {
    iree_allocator_free(host_allocator, data);
  }
  return status;
}
