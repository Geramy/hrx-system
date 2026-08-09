// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/program.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "loom/target/arch/cmd/format.h"

iree_status_t loom_cmd_program_format_calculate_layout(
    uint32_t buffer_ref_count, uint32_t argument_count, uint32_t command_count,
    uint32_t parameter_root_count, uint32_t parameter_count,
    uint32_t parameter_key_length,
    loom_cmd_program_format_layout_t* out_layout) {
  uint64_t offset = LOOM_CMD_PROGRAM_HEADER_SIZE;
  const uint64_t buffer_ref_offset = offset;
  uint64_t table_length = 0;
  if (!iree_checked_mul_u64(buffer_ref_count, LOOM_CMD_PROGRAM_BUFFER_REF_SIZE,
                            &table_length) ||
      !iree_checked_add_u64(offset, table_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command buffer-reference table is too large");
  }
  const uint64_t argument_offset = offset;
  if (!iree_checked_mul_u64(argument_count, LOOM_CMD_PROGRAM_ARGUMENT_SIZE,
                            &table_length) ||
      !iree_checked_add_u64(offset, table_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command argument table is too large");
  }
  const uint64_t command_offset = offset;
  if (!iree_checked_mul_u64(command_count, LOOM_CMD_PROGRAM_COMMAND_SIZE,
                            &table_length) ||
      !iree_checked_add_u64(offset, table_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command table is too large");
  }
  const uint64_t parameter_root_offset = offset;
  if (!iree_checked_mul_u64(parameter_root_count,
                            LOOM_CMD_PROGRAM_PARAMETER_ROOT_SIZE,
                            &table_length) ||
      !iree_checked_add_u64(offset, table_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command parameter-root table is too large");
  }
  const uint64_t parameter_offset = offset;
  if (!iree_checked_mul_u64(parameter_count, LOOM_CMD_PROGRAM_PARAMETER_SIZE,
                            &table_length) ||
      !iree_checked_add_u64(offset, table_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command parameter table is too large");
  }
  const uint64_t parameter_key_offset = offset;
  if (!iree_checked_add_u64(offset, parameter_key_length, &offset) ||
      offset > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command parameter keys are too large");
  }
  *out_layout = (loom_cmd_program_format_layout_t){
      .buffer_ref_offset = (uint32_t)buffer_ref_offset,
      .argument_offset = (uint32_t)argument_offset,
      .command_offset = (uint32_t)command_offset,
      .parameter_root_offset = (uint32_t)parameter_root_offset,
      .parameter_offset = (uint32_t)parameter_offset,
      .parameter_key_offset = (uint32_t)parameter_key_offset,
      .total_length = (uint32_t)offset,
  };
  return iree_ok_status();
}

loom_cmd_program_buffer_ref_t loom_cmd_program_buffer_ref_at(
    const loom_cmd_program_t* program, uint32_t index) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_LT(index, program->buffer_refs.count);
  const uint8_t* record =
      program->buffer_refs.data + index * LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
  return (loom_cmd_program_buffer_ref_t){
      .role = (loom_cmd_program_buffer_role_t)iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET),
      .root_index = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET),
      .byte_offset = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET),
      .byte_length = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET),
  };
}

loom_cmd_program_argument_t loom_cmd_program_argument_at(
    const loom_cmd_program_t* program, uint32_t index) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_LT(index, program->arguments.count);
  const uint8_t* record =
      program->arguments.data + index * LOOM_CMD_PROGRAM_ARGUMENT_SIZE;
  return (loom_cmd_program_argument_t){
      .kind = (loom_cmd_program_argument_kind_t)iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_ARGUMENT_KIND_OFFSET),
      .payload = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_ARGUMENT_PAYLOAD_OFFSET),
  };
}

loom_cmd_program_command_t loom_cmd_program_command_at(
    const loom_cmd_program_t* program, uint32_t index) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_LT(index, program->commands.count);
  const uint8_t* record =
      program->commands.data + index * LOOM_CMD_PROGRAM_COMMAND_SIZE;
  const loom_cmd_program_command_kind_t kind =
      (loom_cmd_program_command_kind_t)iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_COMMAND_KIND_OFFSET);
  const uint32_t operand_0 = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_0_OFFSET);
  const uint32_t operand_1 = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_1_OFFSET);
  const uint32_t operand_2 = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_2_OFFSET);
  const uint32_t operand_3 = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_3_OFFSET);
  const uint32_t operand_4 = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_4_OFFSET);
  loom_cmd_program_command_t command = {
      .kind = kind,
      .argument_offset = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_OFFSET_OFFSET),
      .argument_count = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_COMMAND_ARGUMENT_COUNT_OFFSET),
  };
  switch (kind) {
    case LOOM_CMD_PROGRAM_COMMAND_KIND_FILL:
      command.payload.fill.target_buffer_ref = operand_0;
      command.payload.fill.pattern = operand_1;
      command.payload.fill.pattern_length = operand_2;
      break;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_COPY:
      command.payload.copy.source_buffer_ref = operand_0;
      command.payload.copy.target_buffer_ref = operand_1;
      break;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT:
      command.payload.dispatch_direct.executable_index = operand_0;
      command.payload.dispatch_direct.entry_index = operand_1;
      command.payload.dispatch_direct.workgroup_count_x = operand_2;
      command.payload.dispatch_direct.workgroup_count_y = operand_3;
      command.payload.dispatch_direct.workgroup_count_z = operand_4;
      break;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC:
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC:
      command.payload.dispatch_indirect.executable_index = operand_0;
      command.payload.dispatch_indirect.entry_index = operand_1;
      command.payload.dispatch_indirect.workgroup_count_buffer_ref = operand_2;
      break;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION:
      break;
  }
  return command;
}

loom_cmd_program_parameter_root_t loom_cmd_program_parameter_root_at(
    const loom_cmd_program_t* program, uint32_t index) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_LT(index, program->parameter_roots.count);
  const uint8_t* record = program->parameter_roots.data +
                          index * LOOM_CMD_PROGRAM_PARAMETER_ROOT_SIZE;
  return (loom_cmd_program_parameter_root_t){
      .fixed_buffer_index = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_FIXED_BUFFER_INDEX_OFFSET),
      .required_byte_length = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_REQUIRED_BYTE_LENGTH_OFFSET),
      .minimum_alignment = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_MINIMUM_ALIGNMENT_OFFSET),
  };
}

loom_cmd_program_parameter_t loom_cmd_program_parameter_at(
    const loom_cmd_program_t* program, uint32_t index) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_LT(index, program->parameters.count);
  const uint8_t* record =
      program->parameters.data + index * LOOM_CMD_PROGRAM_PARAMETER_SIZE;
  const uint32_t key_offset = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_PARAMETER_KEY_OFFSET_OFFSET);
  const uint32_t key_length = iree_unaligned_load_le_u32(
      record + LOOM_CMD_PROGRAM_PARAMETER_KEY_LENGTH_OFFSET);
  IREE_ASSERT_LE(key_offset, program->parameter_keys.data_length);
  IREE_ASSERT_LE(key_length, program->parameter_keys.data_length - key_offset);
  return (loom_cmd_program_parameter_t){
      .key = iree_make_string_view(
          (const char*)program->parameter_keys.data + key_offset, key_length),
      .fixed_buffer_index = iree_unaligned_load_le_u32(
          record + LOOM_CMD_PROGRAM_PARAMETER_FIXED_BUFFER_INDEX_OFFSET),
      .byte_offset = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_PARAMETER_BYTE_OFFSET_OFFSET),
      .byte_length = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_PARAMETER_BYTE_LENGTH_OFFSET),
      .minimum_alignment = iree_unaligned_load_le_u64(
          record + LOOM_CMD_PROGRAM_PARAMETER_MINIMUM_ALIGNMENT_OFFSET),
  };
}

iree_status_t loom_cmd_program_relocate_dependencies(
    const loom_cmd_program_t* program,
    const loom_cmd_program_dependency_relocation_t* relocation,
    iree_byte_span_t* out_data, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_ARGUMENT(relocation);
  IREE_ASSERT_ARGUMENT(out_data);
  IREE_ASSERT(relocation->executable_indices ||
              program->requirements.executable_count == 0);
  IREE_ASSERT(relocation->entry_indices ||
              program->requirements.entry_count == 0);
  *out_data = iree_make_byte_span(NULL, 0);

  uint8_t* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_clone(host_allocator, program->storage, (void**)&data));
  iree_unaligned_store_le_u32(
      data + LOOM_CMD_PROGRAM_HEADER_EXECUTABLE_COUNT_OFFSET,
      relocation->executable_count);
  iree_unaligned_store_le_u32(data + LOOM_CMD_PROGRAM_HEADER_ENTRY_COUNT_OFFSET,
                              relocation->entry_count);

  const iree_host_size_t command_table_offset =
      (iree_host_size_t)(program->commands.data - program->storage.data);
  for (uint32_t i = 0; i < program->commands.count; ++i) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(program, i);
    if (command.kind != LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT &&
        command.kind !=
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC &&
        command.kind !=
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC) {
      continue;
    }
    const bool is_direct =
        command.kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT;
    const uint32_t executable_index =
        is_direct ? command.payload.dispatch_direct.executable_index
                  : command.payload.dispatch_indirect.executable_index;
    const uint32_t entry_index =
        is_direct ? command.payload.dispatch_direct.entry_index
                  : command.payload.dispatch_indirect.entry_index;
    const uint32_t relocated_executable_index =
        relocation->executable_indices[executable_index];
    const uint32_t relocated_entry_index =
        relocation->entry_indices[entry_index];
    IREE_ASSERT_LT(relocated_executable_index, relocation->executable_count);
    IREE_ASSERT_LT(relocated_entry_index, relocation->entry_count);
    uint8_t* record =
        data + command_table_offset + i * LOOM_CMD_PROGRAM_COMMAND_SIZE;
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_0_OFFSET,
        relocated_executable_index);
    iree_unaligned_store_le_u32(
        record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_1_OFFSET,
        relocated_entry_index);
  }

  *out_data = iree_make_byte_span(data, program->storage.data_length);
  return iree_ok_status();
}

static bool loom_cmd_program_slice_is_valid(uint32_t offset, uint32_t count,
                                            uint32_t total_count) {
  if (count == 0) return offset == 0;
  return offset < total_count && count <= total_count - offset;
}

static iree_status_t loom_cmd_program_validate_buffer_refs(
    const loom_cmd_program_t* program) {
  for (uint32_t i = 0; i < program->buffer_refs.count; ++i) {
    const loom_cmd_program_buffer_ref_t buffer_ref =
        loom_cmd_program_buffer_ref_at(program, i);
    if (buffer_ref.role == LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED) {
      if (buffer_ref.root_index >= program->requirements.fixed_buffer_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "command buffer reference %" PRIu32
                                " selects fixed root %" PRIu32
                                " outside the fixed-buffer table",
                                i, buffer_ref.root_index);
      }
    } else if (buffer_ref.role == LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE) {
      if (buffer_ref.root_index >=
          program->requirements.rebindable_binding_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "command buffer reference %" PRIu32
                                " selects binding root %" PRIu32
                                " outside the rebindable table",
                                i, buffer_ref.root_index);
      }
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command buffer reference %" PRIu32
                              " has unknown role %u",
                              i, (unsigned)buffer_ref.role);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_validate_arguments(
    const loom_cmd_program_t* program) {
  for (uint32_t i = 0; i < program->arguments.count; ++i) {
    const loom_cmd_program_argument_t argument =
        loom_cmd_program_argument_at(program, i);
    switch (argument.kind) {
      case LOOM_CMD_PROGRAM_ARGUMENT_KIND_U32:
        if (argument.payload > UINT32_MAX) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "command argument %" PRIu32
                                  " exceeds its unsigned 32-bit range",
                                  i);
        }
        break;
      case LOOM_CMD_PROGRAM_ARGUMENT_KIND_U64:
        break;
      case LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER_REF:
        if (argument.payload >= program->buffer_refs.count) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "command argument %" PRIu32
                                  " selects buffer reference %" PRIu64
                                  " outside the table",
                                  i, argument.payload);
        }
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "command argument %" PRIu32
                                " has unknown kind %u",
                                i, (unsigned)argument.kind);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_validate_dispatch(
    const loom_cmd_program_t* program, uint32_t command_index,
    const loom_cmd_program_command_t* command) {
  if (!loom_cmd_program_slice_is_valid(command->argument_offset,
                                       command->argument_count,
                                       program->arguments.count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command dispatch %" PRIu32
                            " has argument slice [%" PRIu32 ", %" PRIu32
                            ") outside the argument table",
                            command_index, command->argument_offset,
                            command->argument_offset + command->argument_count);
  }
  const bool is_direct =
      command->kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT;
  const uint32_t executable_index =
      is_direct ? command->payload.dispatch_direct.executable_index
                : command->payload.dispatch_indirect.executable_index;
  const uint32_t entry_index =
      is_direct ? command->payload.dispatch_direct.entry_index
                : command->payload.dispatch_indirect.entry_index;
  if (executable_index >= program->requirements.executable_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command dispatch %" PRIu32
                            " selects executable %" PRIu32
                            " outside the requirement table",
                            command_index, executable_index);
  }
  if (entry_index >= program->requirements.entry_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command dispatch %" PRIu32
                            " selects entry %" PRIu32
                            " outside the requirement table",
                            command_index, entry_index);
  }
  if (!is_direct &&
      command->payload.dispatch_indirect.workgroup_count_buffer_ref >=
          program->buffer_refs.count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "command dispatch %" PRIu32
        " selects workgroup-count buffer reference %" PRIu32
        " outside the table",
        command_index,
        command->payload.dispatch_indirect.workgroup_count_buffer_ref);
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_validate_commands(
    const loom_cmd_program_t* program) {
  for (uint32_t i = 0; i < program->commands.count; ++i) {
    const uint8_t* record =
        program->commands.data + i * LOOM_CMD_PROGRAM_COMMAND_SIZE;
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(program, i);
    const uint32_t operand_3 = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_3_OFFSET);
    const uint32_t operand_4 = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_4_OFFSET);
    switch (command.kind) {
      case LOOM_CMD_PROGRAM_COMMAND_KIND_FILL:
        if (!loom_cmd_program_slice_is_valid(command.argument_offset,
                                             command.argument_count, 0) ||
            command.payload.fill.target_buffer_ref >=
                program->buffer_refs.count ||
            (command.payload.fill.pattern_length != 1 &&
             command.payload.fill.pattern_length != 2 &&
             command.payload.fill.pattern_length != 4) ||
            operand_3 != 0 || operand_4 != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "command fill %" PRIu32 " has a noncanonical payload", i);
        }
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_COPY:
        if (!loom_cmd_program_slice_is_valid(command.argument_offset,
                                             command.argument_count, 0) ||
            command.payload.copy.source_buffer_ref >=
                program->buffer_refs.count ||
            command.payload.copy.target_buffer_ref >=
                program->buffer_refs.count ||
            iree_unaligned_load_le_u32(
                record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_2_OFFSET) != 0 ||
            operand_3 != 0 || operand_4 != 0) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "command copy %" PRIu32 " has a noncanonical payload", i);
        }
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT: {
        IREE_RETURN_IF_ERROR(
            loom_cmd_program_validate_dispatch(program, i, &command));
        break;
      }
      case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC:
      case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC:
        if (operand_3 != 0 || operand_4 != 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "indirect command dispatch %" PRIu32
                                  " has a noncanonical payload",
                                  i);
        }
        IREE_RETURN_IF_ERROR(
            loom_cmd_program_validate_dispatch(program, i, &command));
        break;
      case LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION:
        if (!loom_cmd_program_slice_is_valid(command.argument_offset,
                                             command.argument_count, 0)) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "command barrier %" PRIu32 " has a noncanonical payload", i);
        }
        for (uint32_t operand_index = 0; operand_index < 5; ++operand_index) {
          if (iree_unaligned_load_le_u32(
                  record + LOOM_CMD_PROGRAM_COMMAND_OPERAND_0_OFFSET +
                  operand_index * sizeof(uint32_t)) != 0) {
            return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "command barrier %" PRIu32 " has a noncanonical payload", i);
          }
        }
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "command %" PRIu32 " has unknown kind %u", i,
                                (unsigned)command.kind);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_validate_parameter_roots(
    const loom_cmd_program_t* program) {
  uint32_t previous_fixed_buffer_index = 0;
  for (uint32_t i = 0; i < program->parameter_roots.count; ++i) {
    const uint8_t* record = program->parameter_roots.data +
                            i * LOOM_CMD_PROGRAM_PARAMETER_ROOT_SIZE;
    const loom_cmd_program_parameter_root_t root =
        loom_cmd_program_parameter_root_at(program, i);
    if (iree_unaligned_load_le_u32(
            record + LOOM_CMD_PROGRAM_PARAMETER_ROOT_RESERVED_OFFSET) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command parameter root %" PRIu32 " has reserved fields set", i);
    }
    if (root.fixed_buffer_index >= program->requirements.fixed_buffer_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "command parameter root %" PRIu32
                              " selects fixed buffer %" PRIu32
                              " outside the fixed-buffer table",
                              i, root.fixed_buffer_index);
    }
    if (i != 0 && root.fixed_buffer_index <= previous_fixed_buffer_index) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command parameter roots are not in canonical ascending order");
    }
    if (!iree_is_power_of_two_uint64(root.minimum_alignment)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command parameter root %" PRIu32
                              " has invalid minimum alignment %" PRIu64,
                              i, root.minimum_alignment);
    }
    previous_fixed_buffer_index = root.fixed_buffer_index;
  }
  return iree_ok_status();
}

static bool loom_cmd_program_find_parameter_root(
    const loom_cmd_program_t* program, uint32_t fixed_buffer_index,
    loom_cmd_program_parameter_root_t* out_root) {
  uint32_t begin = 0;
  uint32_t end = program->parameter_roots.count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    const loom_cmd_program_parameter_root_t root =
        loom_cmd_program_parameter_root_at(program, middle);
    if (root.fixed_buffer_index < fixed_buffer_index) {
      begin = middle + 1;
    } else if (root.fixed_buffer_index > fixed_buffer_index) {
      end = middle;
    } else {
      *out_root = root;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_cmd_program_validate_parameters(
    const loom_cmd_program_t* program) {
  uint32_t expected_key_offset = 0;
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    const uint8_t* record =
        program->parameters.data + i * LOOM_CMD_PROGRAM_PARAMETER_SIZE;
    const uint32_t key_offset = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_PARAMETER_KEY_OFFSET_OFFSET);
    const uint32_t key_length = iree_unaligned_load_le_u32(
        record + LOOM_CMD_PROGRAM_PARAMETER_KEY_LENGTH_OFFSET);
    if (iree_unaligned_load_le_u32(
            record + LOOM_CMD_PROGRAM_PARAMETER_RESERVED_OFFSET) != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command parameter %" PRIu32 " has reserved fields set", i);
    }
    if (key_length == 0 || key_offset != expected_key_offset ||
        key_offset > program->parameter_keys.data_length ||
        key_length > program->parameter_keys.data_length - key_offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command parameter %" PRIu32
                              " has a noncanonical key slice [%" PRIu32
                              ", %" PRIu64 ")",
                              i, key_offset, (uint64_t)key_offset + key_length);
    }
    expected_key_offset += key_length;

    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    loom_cmd_program_parameter_root_t root = {0};
    if (!loom_cmd_program_find_parameter_root(
            program, parameter.fixed_buffer_index, &root)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "command parameter %" PRIu32
                              " selects fixed buffer %" PRIu32
                              " without a parameter-root requirement",
                              i, parameter.fixed_buffer_index);
    }
    if (!iree_is_power_of_two_uint64(parameter.minimum_alignment) ||
        parameter.byte_offset % parameter.minimum_alignment != 0 ||
        root.minimum_alignment < parameter.minimum_alignment) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command parameter %" PRIu32 " has inconsistent alignment", i);
    }
    uint64_t byte_end = 0;
    if (!iree_checked_add_u64(parameter.byte_offset, parameter.byte_length,
                              &byte_end) ||
        byte_end > root.required_byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "command parameter %" PRIu32
                              " exceeds fixed buffer %" PRIu32 " requirement",
                              i, parameter.fixed_buffer_index);
    }
  }
  if (expected_key_offset != program->parameter_keys.data_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command parameter key storage has unreferenced trailing bytes");
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_validate_transient(
    const loom_cmd_program_t* program) {
  const loom_cmd_program_transient_requirement_t transient =
      program->requirements.transient;
  if (transient.binding_index == UINT32_MAX) {
    if (transient.required_byte_length != 0 ||
        transient.minimum_alignment != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command program without a transient binding declares storage");
    }
    return iree_ok_status();
  }
  if (transient.binding_index >=
      program->requirements.rebindable_binding_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command transient selects binding %" PRIu32
                            " outside the rebindable table",
                            transient.binding_index);
  }
  if (transient.required_byte_length == 0 ||
      !iree_is_power_of_two_uint64(transient.minimum_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command transient requires a positive length and power-of-two "
        "alignment");
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_validate_launch_counts(
    const loom_cmd_program_t* program) {
  const loom_cmd_program_launch_count_requirement_t requirement =
      program->requirements.launch_counts;
  if (requirement.binding_index == UINT32_MAX) {
    if (requirement.required_byte_length != 0 ||
        requirement.minimum_alignment != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command program without host launch counts declares storage");
    }
  } else if (requirement.binding_index >=
                 program->requirements.rebindable_binding_count ||
             requirement.required_byte_length == 0 ||
             requirement.minimum_alignment !=
                 LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_ALIGNMENT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command host launch-count requirement is not representable");
  }

  bool found_static_dispatch = false;
  for (uint32_t i = 0; i < program->commands.count; ++i) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(program, i);
    if (command.kind !=
        LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC) {
      continue;
    }
    found_static_dispatch = true;
    if (requirement.binding_index == UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "static indirect dispatch has no host launch-count requirement");
    }
    const loom_cmd_program_buffer_ref_t buffer_ref =
        loom_cmd_program_buffer_ref_at(
            program,
            command.payload.dispatch_indirect.workgroup_count_buffer_ref);
    uint64_t byte_end = 0;
    if (buffer_ref.role != LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE ||
        buffer_ref.root_index != requirement.binding_index ||
        buffer_ref.byte_length !=
            LOOM_CMD_PROGRAM_LAUNCH_COUNT_TUPLE_BYTE_LENGTH ||
        !iree_checked_add_u64(buffer_ref.byte_offset, buffer_ref.byte_length,
                              &byte_end) ||
        byte_end > requirement.required_byte_length ||
        buffer_ref.byte_offset % requirement.minimum_alignment != 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "static indirect dispatch references inconsistent host launch "
          "counts");
    }
  }
  if (!found_static_dispatch && requirement.binding_index != UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "host launch-count storage has no static indirect dispatch");
  }
  return iree_ok_status();
}

iree_status_t loom_cmd_program_parse(iree_const_byte_span_t data,
                                     loom_cmd_program_t* out_program) {
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = (loom_cmd_program_t){0};
  if (data.data_length < LOOM_CMD_PROGRAM_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program is shorter than its header");
  }
  if (memcmp(data.data, LOOM_CMD_PROGRAM_FORMAT_MAGIC,
             LOOM_CMD_PROGRAM_FORMAT_MAGIC_LENGTH) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program has invalid magic bytes");
  }
  const uint16_t version = iree_unaligned_load_le_u16(
      data.data + LOOM_CMD_PROGRAM_HEADER_VERSION_OFFSET);
  if (version != LOOM_CMD_PROGRAM_FORMAT_VERSION) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "command program format version %" PRIu16 " is unsupported", version);
  }
  const uint16_t header_size = iree_unaligned_load_le_u16(
      data.data + LOOM_CMD_PROGRAM_HEADER_SIZE_OFFSET);
  if (header_size != LOOM_CMD_PROGRAM_HEADER_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program header size %" PRIu16
                            " is not canonical",
                            header_size);
  }
  const uint32_t total_length = iree_unaligned_load_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_TOTAL_LENGTH_OFFSET);
  if (total_length != data.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program declares %" PRIu32
                            " bytes but received %" PRIhsz,
                            total_length, data.data_length);
  }
  if (iree_unaligned_load_le_u32(
          data.data + LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_RESERVED_OFFSET) !=
          0 ||
      iree_unaligned_load_le_u32(
          data.data + LOOM_CMD_PROGRAM_HEADER_TRANSIENT_RESERVED_OFFSET) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program header reserved fields are set");
  }

  loom_cmd_program_t program = {
      .storage = data,
      .requirements =
          {
              .fixed_buffer_count = iree_unaligned_load_le_u32(
                  data.data +
                  LOOM_CMD_PROGRAM_HEADER_FIXED_BUFFER_COUNT_OFFSET),
              .rebindable_binding_count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_BINDING_COUNT_OFFSET),
              .executable_count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_EXECUTABLE_COUNT_OFFSET),
              .entry_count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_ENTRY_COUNT_OFFSET),
              .transient =
                  {
                      .binding_index = iree_unaligned_load_le_u32(
                          data.data +
                          LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BINDING_INDEX_OFFSET),
                      .required_byte_length = iree_unaligned_load_le_u64(
                          data.data +
                          LOOM_CMD_PROGRAM_HEADER_TRANSIENT_BYTE_LENGTH_OFFSET),
                      .minimum_alignment = iree_unaligned_load_le_u64(
                          data.data +
                          LOOM_CMD_PROGRAM_HEADER_TRANSIENT_MINIMUM_ALIGNMENT_OFFSET),
                  },
              .launch_counts =
                  {
                      .binding_index = iree_unaligned_load_le_u32(
                          data.data +
                          LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BINDING_INDEX_OFFSET),
                      .required_byte_length = iree_unaligned_load_le_u64(
                          data.data +
                          LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_BYTE_LENGTH_OFFSET),
                      .minimum_alignment = iree_unaligned_load_le_u64(
                          data.data +
                          LOOM_CMD_PROGRAM_HEADER_LAUNCH_COUNT_MINIMUM_ALIGNMENT_OFFSET),
                  },
          },
      .buffer_refs =
          {
              .count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_COUNT_OFFSET),
          },
      .arguments =
          {
              .count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_COUNT_OFFSET),
          },
      .commands =
          {
              .count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_COMMAND_COUNT_OFFSET),
          },
      .parameter_roots =
          {
              .count = iree_unaligned_load_le_u32(
                  data.data +
                  LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_COUNT_OFFSET),
          },
      .parameters =
          {
              .count = iree_unaligned_load_le_u32(
                  data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_COUNT_OFFSET),
          },
  };
  const uint32_t parameter_key_length = iree_unaligned_load_le_u32(
      data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_KEY_LENGTH_OFFSET);
  loom_cmd_program_format_layout_t layout = {0};
  IREE_RETURN_IF_ERROR(loom_cmd_program_format_calculate_layout(
      program.buffer_refs.count, program.arguments.count,
      program.commands.count, program.parameter_roots.count,
      program.parameters.count, parameter_key_length, &layout));
  if (layout.total_length != total_length ||
      layout.buffer_ref_offset !=
          iree_unaligned_load_le_u32(
              data.data + LOOM_CMD_PROGRAM_HEADER_BUFFER_REF_TABLE_OFFSET) ||
      layout.argument_offset !=
          iree_unaligned_load_le_u32(
              data.data + LOOM_CMD_PROGRAM_HEADER_ARGUMENT_TABLE_OFFSET) ||
      layout.command_offset !=
          iree_unaligned_load_le_u32(
              data.data + LOOM_CMD_PROGRAM_HEADER_COMMAND_TABLE_OFFSET) ||
      layout.parameter_root_offset !=
          iree_unaligned_load_le_u32(
              data.data +
              LOOM_CMD_PROGRAM_HEADER_PARAMETER_ROOT_TABLE_OFFSET) ||
      layout.parameter_offset !=
          iree_unaligned_load_le_u32(
              data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_TABLE_OFFSET) ||
      layout.parameter_key_offset !=
          iree_unaligned_load_le_u32(
              data.data + LOOM_CMD_PROGRAM_HEADER_PARAMETER_KEY_TABLE_OFFSET)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program table layout is not canonical");
  }
  program.buffer_refs.data = data.data + layout.buffer_ref_offset;
  program.arguments.data = data.data + layout.argument_offset;
  program.commands.data = data.data + layout.command_offset;
  program.parameter_roots.data = data.data + layout.parameter_root_offset;
  program.parameters.data = data.data + layout.parameter_offset;
  program.parameter_keys = iree_make_const_byte_span(
      data.data + layout.parameter_key_offset, parameter_key_length);

  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_transient(&program));
  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_buffer_refs(&program));
  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_arguments(&program));
  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_commands(&program));
  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_launch_counts(&program));
  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_parameter_roots(&program));
  IREE_RETURN_IF_ERROR(loom_cmd_program_validate_parameters(&program));
  *out_program = program;
  return iree_ok_status();
}
