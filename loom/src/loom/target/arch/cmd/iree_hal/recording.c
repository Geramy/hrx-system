// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/iree_hal/recording.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/cmd/abi_layout.h"
#include "loom/target/arch/cmd/descriptors/descriptors.h"
#include "loom/target/arch/cmd/format.h"
#include "loom/target/registers.h"

typedef enum loom_cmd_iree_hal_argument_kind_e {
  LOOM_CMD_IREE_HAL_ARGUMENT_NONE = 0,
  LOOM_CMD_IREE_HAL_ARGUMENT_B8 = 1,
  LOOM_CMD_IREE_HAL_ARGUMENT_B16 = 2,
  LOOM_CMD_IREE_HAL_ARGUMENT_B32 = 3,
  LOOM_CMD_IREE_HAL_ARGUMENT_B64 = 4,
  LOOM_CMD_IREE_HAL_ARGUMENT_FIXED_BUFFER = 5,
  LOOM_CMD_IREE_HAL_ARGUMENT_BINDING = 6,
} loom_cmd_iree_hal_argument_kind_t;

typedef union loom_cmd_iree_hal_value_t {
  uint32_t u32;
  uint64_t u64;
  uint32_t table_index;
  iree_hal_buffer_ref_t buffer_ref;
} loom_cmd_iree_hal_value_t;

typedef struct loom_cmd_iree_hal_workspace_t {
  const loom_module_t* module;
  const loom_low_descriptor_set_t* descriptor_set;
  const loom_cmd_iree_hal_inputs_t* inputs;
  iree_hal_command_buffer_t* command_buffer;
  loom_cmd_iree_hal_value_t* values;
  uint8_t* constants;
  iree_hal_buffer_ref_t* bindings;
} loom_cmd_iree_hal_workspace_t;

static bool loom_cmd_iree_hal_packet_is(
    const loom_cmd_iree_hal_workspace_t* workspace,
    const loom_low_descriptor_packet_t* packet, uint32_t descriptor_ordinal) {
  IREE_ASSERT_LT(descriptor_ordinal,
                 workspace->descriptor_set->descriptor_count);
  return packet->descriptor_ordinal == descriptor_ordinal;
}

static iree_string_view_t loom_cmd_iree_hal_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static int64_t loom_cmd_iree_hal_constant_value(const loom_op_t* op) {
  const loom_named_attr_slice_t attrs = loom_low_const_attrs(op);
  IREE_ASSERT_EQ(attrs.count, 1u);
  IREE_ASSERT_EQ(attrs.entries[0].value.kind, LOOM_ATTR_I64);
  return loom_attr_as_i64(attrs.entries[0].value);
}

static iree_status_t loom_cmd_iree_hal_validate_entry(
    const loom_cmd_iree_hal_inputs_t* inputs, iree_host_size_t entry_index) {
  const loom_cmd_iree_hal_entry_t* entry = &inputs->entries[entry_index];
  if (entry->executable_index >= inputs->executable_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command entry %" PRIhsz
                            " references executable %" PRIu32
                            " outside the executable table",
                            entry_index, entry->executable_index);
  }
  if (inputs->executables[entry->executable_index] == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command entry %" PRIhsz " references a null executable", entry_index);
  }
  if (!iree_hal_executable_function_is_valid(entry->function)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command entry %" PRIhsz " has an invalid function token", entry_index);
  }
  if (entry->info.parameter_count != 0 && entry->parameters == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command entry %" PRIhsz
                            " is missing its parameter table",
                            entry_index);
  }
  for (uint16_t i = 0; i < entry->info.parameter_count; ++i) {
    const iree_hal_executable_function_parameter_t* parameter =
        &entry->parameters[i];
    switch (parameter->type) {
      case IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT:
      case IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BUFFER_PTR:
        if (parameter->offset > entry->info.constant_byte_length ||
            parameter->size >
                entry->info.constant_byte_length - parameter->offset) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "command entry %" PRIhsz " parameter %" PRIu16
                                  " exceeds its constant table",
                                  entry_index, i);
        }
        break;
      case IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING:
        if (parameter->offset >= entry->info.binding_count) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "command entry %" PRIhsz " parameter %" PRIu16
                                  " exceeds its binding table",
                                  entry_index, i);
        }
        break;
      default:
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "command entry %" PRIhsz " parameter %" PRIu16
                                " has unknown type %u",
                                entry_index, i, (unsigned)parameter->type);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_iree_hal_validate_inputs(
    const loom_cmd_iree_hal_inputs_t* inputs,
    iree_host_size_t* out_max_constant_byte_length,
    iree_host_size_t* out_max_binding_count) {
  *out_max_constant_byte_length = 0;
  *out_max_binding_count = 0;
  if (inputs->binding_count > (iree_host_size_t)UINT32_C(0x1000000)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command program binding count %" PRIhsz
                            " exceeds the 24-bit HAL binding-slot namespace",
                            inputs->binding_count);
  }
  if (inputs->fixed_buffer_count != 0 && inputs->fixed_buffers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command fixed-buffer table is missing");
  }
  if (inputs->executable_count != 0 && inputs->executables == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command executable table is missing");
  }
  if (inputs->entry_count != 0 && inputs->entries == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command entry table is missing");
  }
  for (iree_host_size_t i = 0; i < inputs->entry_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_validate_entry(inputs, i));
    const iree_hal_executable_function_info_t* info = &inputs->entries[i].info;
    *out_max_constant_byte_length =
        iree_max(*out_max_constant_byte_length,
                 (iree_host_size_t)info->constant_byte_length);
    *out_max_binding_count =
        iree_max(*out_max_binding_count, (iree_host_size_t)info->binding_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_iree_hal_validate_input_counts(
    uint32_t fixed_buffer_count, uint32_t binding_count,
    uint32_t executable_count, uint32_t entry_count,
    const loom_cmd_iree_hal_inputs_t* inputs) {
  if (inputs->fixed_buffer_count != fixed_buffer_count ||
      inputs->binding_count != binding_count ||
      inputs->executable_count != executable_count ||
      inputs->entry_count != entry_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command program requires fixed/binding/executable/entry counts "
        "(%" PRIu32 ", %" PRIu32 ", %" PRIu32 ", %" PRIu32
        ") but received (%" PRIhsz ", %" PRIhsz ", %" PRIhsz ", %" PRIhsz ")",
        fixed_buffer_count, binding_count, executable_count, entry_count,
        inputs->fixed_buffer_count, inputs->binding_count,
        inputs->executable_count, inputs->entry_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_iree_hal_import_resource(
    loom_cmd_iree_hal_workspace_t* workspace, const loom_op_t* op) {
  const loom_value_id_t result_id = loom_low_resource_result(op);
  const loom_type_t result_type =
      loom_module_value_type(workspace->module, result_id);
  const uint16_t register_class_id =
      loom_low_register_type_class_id(result_type);
  const uint64_t resource_index = (uint64_t)loom_low_resource_index(op);
  loom_cmd_iree_hal_value_t* result = &workspace->values[result_id];
  switch (register_class_id) {
    case CMD_CORE_REG_CLASS_ID_BUFFER: {
      if (resource_index >= workspace->inputs->fixed_buffer_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "fixed-buffer resource index %" PRIu64
                                " is out of range",
                                resource_index);
      }
      const iree_hal_buffer_ref_t buffer_ref =
          workspace->inputs->fixed_buffers[resource_index];
      if (buffer_ref.buffer == NULL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "fixed-buffer resource index %" PRIu64
                                " is not a direct buffer",
                                resource_index);
      }
      result->buffer_ref = buffer_ref;
      return iree_ok_status();
    }
    case CMD_CORE_REG_CLASS_ID_BINDING:
      if (resource_index >= workspace->inputs->binding_count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "binding resource index %" PRIu64
                                " is out of range",
                                resource_index);
      }
      result->table_index = (uint32_t)resource_index;
      return iree_ok_status();
    case CMD_CORE_REG_CLASS_ID_EXECUTABLE:
      if (resource_index >= workspace->inputs->executable_count ||
          workspace->inputs->executables[resource_index] == NULL) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "executable resource index %" PRIu64
                                " is out of range",
                                resource_index);
      }
      result->table_index = (uint32_t)resource_index;
      return iree_ok_status();
    case CMD_CORE_REG_CLASS_ID_ENTRY:
      if (resource_index >= workspace->inputs->entry_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "entry resource index %" PRIu64 " is out of range", resource_index);
      }
      result->table_index = (uint32_t)resource_index;
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command_input resource has unsupported cmd register class %" PRIu16,
          register_class_id);
  }
}

static iree_status_t loom_cmd_iree_hal_make_direct_buffer_ref(
    loom_cmd_iree_hal_workspace_t* workspace, const loom_op_t* op) {
  const loom_value_slice_t operands = loom_low_op_operands(op);
  const loom_value_slice_t results = loom_low_op_results(op);
  const iree_hal_buffer_ref_t root =
      workspace->values[operands.values[0]].buffer_ref;
  const iree_device_size_t offset = workspace->values[operands.values[1]].u64;
  const iree_device_size_t length = workspace->values[operands.values[2]].u64;
  iree_device_size_t adjusted_offset = 0;
  iree_device_size_t adjusted_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_calculate_range(root.offset, root.length, offset, length,
                                      &adjusted_offset, &adjusted_length));
  workspace->values[results.values[0]].buffer_ref =
      iree_hal_make_buffer_ref(root.buffer, adjusted_offset, adjusted_length);
  return iree_ok_status();
}

static void loom_cmd_iree_hal_make_binding_buffer_ref(
    loom_cmd_iree_hal_workspace_t* workspace, const loom_op_t* op) {
  const loom_value_slice_t operands = loom_low_op_operands(op);
  const loom_value_slice_t results = loom_low_op_results(op);
  const uint32_t binding = workspace->values[operands.values[0]].table_index;
  const iree_device_size_t offset = workspace->values[operands.values[1]].u64;
  const iree_device_size_t length = workspace->values[operands.values[2]].u64;
  workspace->values[results.values[0]].buffer_ref =
      iree_hal_make_indirect_buffer_ref(binding, offset, length);
}

static loom_cmd_iree_hal_argument_kind_t loom_cmd_iree_hal_argument_kind(
    const loom_cmd_iree_hal_workspace_t* workspace, loom_value_id_t value_id) {
  const loom_type_t type = loom_module_value_type(workspace->module, value_id);
  switch (loom_low_register_type_class_id(type)) {
    case CMD_CORE_REG_CLASS_ID_B8:
      return LOOM_CMD_IREE_HAL_ARGUMENT_B8;
    case CMD_CORE_REG_CLASS_ID_B16:
      return LOOM_CMD_IREE_HAL_ARGUMENT_B16;
    case CMD_CORE_REG_CLASS_ID_B32:
      return LOOM_CMD_IREE_HAL_ARGUMENT_B32;
    case CMD_CORE_REG_CLASS_ID_B64:
      return LOOM_CMD_IREE_HAL_ARGUMENT_B64;
    case CMD_CORE_REG_CLASS_ID_BUFFER:
      return LOOM_CMD_IREE_HAL_ARGUMENT_FIXED_BUFFER;
    case CMD_CORE_REG_CLASS_ID_BINDING:
      return LOOM_CMD_IREE_HAL_ARGUMENT_BINDING;
    default:
      return LOOM_CMD_IREE_HAL_ARGUMENT_NONE;
  }
}

static iree_host_size_t loom_cmd_iree_hal_scalar_byte_length(
    loom_cmd_iree_hal_argument_kind_t kind) {
  switch (kind) {
    case LOOM_CMD_IREE_HAL_ARGUMENT_B8:
      return 1;
    case LOOM_CMD_IREE_HAL_ARGUMENT_B16:
      return 2;
    case LOOM_CMD_IREE_HAL_ARGUMENT_B32:
      return 4;
    case LOOM_CMD_IREE_HAL_ARGUMENT_B64:
      return 8;
    default:
      return 0;
  }
}

static void loom_cmd_iree_hal_store_scalar_argument(
    uint64_t value, iree_host_size_t byte_length, uint8_t* target) {
  switch (byte_length) {
    case 1:
      target[0] = (uint8_t)value;
      break;
    case 2:
      iree_unaligned_store_le_u16(target, (uint16_t)value);
      break;
    case 4:
      iree_unaligned_store_le_u32(target, (uint32_t)value);
      break;
    case 8:
      iree_unaligned_store_le_u64(target, value);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("portable scalar argument byte length");
  }
}

static iree_status_t loom_cmd_iree_hal_resolve_argument_buffer(
    const loom_cmd_iree_hal_workspace_t* workspace,
    loom_cmd_iree_hal_argument_kind_t kind, loom_value_id_t root_id,
    loom_value_id_t byte_offset_id, loom_value_id_t byte_length_id,
    iree_hal_buffer_ref_t* out_buffer_ref) {
  const iree_device_size_t byte_offset = workspace->values[byte_offset_id].u64;
  const iree_device_size_t byte_length = workspace->values[byte_length_id].u64;
  if (kind == LOOM_CMD_IREE_HAL_ARGUMENT_BINDING) {
    *out_buffer_ref = iree_hal_make_indirect_buffer_ref(
        workspace->values[root_id].table_index, byte_offset, byte_length);
    return iree_ok_status();
  }

  IREE_ASSERT_EQ(kind, LOOM_CMD_IREE_HAL_ARGUMENT_FIXED_BUFFER);
  const iree_hal_buffer_ref_t root = workspace->values[root_id].buffer_ref;
  iree_device_size_t adjusted_offset = 0;
  iree_device_size_t adjusted_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      root.offset, root.length, byte_offset, byte_length, &adjusted_offset,
      &adjusted_length));
  *out_buffer_ref =
      iree_hal_make_buffer_ref(root.buffer, adjusted_offset, adjusted_length);
  return iree_ok_status();
}

static iree_status_t loom_cmd_iree_hal_pack_arguments(
    loom_cmd_iree_hal_workspace_t* workspace,
    const loom_cmd_iree_hal_entry_t* entry, loom_value_slice_t operands,
    uint16_t argument_start, iree_const_byte_span_t* out_constants,
    iree_hal_buffer_ref_list_t* out_bindings) {
  IREE_ASSERT_LE(argument_start, operands.count);
  if (entry->info.constant_byte_length != 0) {
    memset(workspace->constants, 0, entry->info.constant_byte_length);
  }
  if (entry->info.binding_count != 0) {
    memset(workspace->bindings, 0,
           entry->info.binding_count * sizeof(*workspace->bindings));
  }
  uint16_t logical_index = 0;
  uint16_t operand_index = argument_start;
  while (operand_index < operands.count) {
    if (logical_index >= entry->info.parameter_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command dispatch has more logical arguments than its entry");
    }
    const loom_value_id_t argument = operands.values[operand_index];
    const loom_cmd_iree_hal_argument_kind_t argument_kind =
        loom_cmd_iree_hal_argument_kind(workspace, argument);
    const iree_hal_executable_function_parameter_t* parameter =
        &entry->parameters[logical_index];
    const iree_host_size_t scalar_byte_length =
        loom_cmd_iree_hal_scalar_byte_length(argument_kind);
    if (scalar_byte_length != 0) {
      if (parameter->type !=
              IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT ||
          parameter->size != scalar_byte_length) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "command argument %" PRIu16
                                " does not match its %" PRIu16
                                "-byte constant parameter",
                                logical_index, parameter->size);
      }
      loom_cmd_iree_hal_store_scalar_argument(
          workspace->values[argument].u64, scalar_byte_length,
          workspace->constants + parameter->offset);
      ++operand_index;
      ++logical_index;
      continue;
    }

    if ((argument_kind != LOOM_CMD_IREE_HAL_ARGUMENT_FIXED_BUFFER &&
         argument_kind != LOOM_CMD_IREE_HAL_ARGUMENT_BINDING) ||
        (iree_host_size_t)operand_index + 3 > operands.count ||
        loom_low_register_type_class_id(loom_module_value_type(
            workspace->module, operands.values[operand_index + 1])) !=
            CMD_CORE_REG_CLASS_ID_U64 ||
        loom_low_register_type_class_id(loom_module_value_type(
            workspace->module, operands.values[operand_index + 2])) !=
            CMD_CORE_REG_CLASS_ID_U64) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command buffer argument %" PRIu16
          " is not a root, byte-offset, and byte-length tuple",
          logical_index);
    }
    if (parameter->type ==
        IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BUFFER_PTR) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "command entry raw buffer-pointer parameters cannot preserve "
          "issue-time buffer rebinding");
    }
    if (parameter->type !=
        IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command argument %" PRIu16
                              " is not a scalar constant",
                              logical_index);
    }
    IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_resolve_argument_buffer(
        workspace, argument_kind, argument, operands.values[operand_index + 1],
        operands.values[operand_index + 2],
        &workspace->bindings[parameter->offset]));
    operand_index += 3;
    ++logical_index;
  }
  if (logical_index != entry->info.parameter_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command dispatch has %" PRIu16
                            " logical arguments but its entry expects %" PRIu16,
                            logical_index, entry->info.parameter_count);
  }

  *out_constants = iree_make_const_byte_span(workspace->constants,
                                             entry->info.constant_byte_length);
  *out_bindings = (iree_hal_buffer_ref_list_t){
      .count = entry->info.binding_count,
      .values = workspace->bindings,
  };
  return iree_ok_status();
}

static iree_status_t loom_cmd_iree_hal_dispatch(
    loom_cmd_iree_hal_workspace_t* workspace, const loom_op_t* op,
    iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags,
    uint16_t argument_start) {
  const loom_value_slice_t operands = loom_low_op_operands(op);
  const uint32_t executable_index =
      workspace->values[operands.values[0]].table_index;
  const uint32_t entry_index =
      workspace->values[operands.values[1]].table_index;
  const loom_cmd_iree_hal_entry_t* entry =
      &workspace->inputs->entries[entry_index];
  if (entry->executable_index != executable_index) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command dispatch pairs executable %" PRIu32 " with entry %" PRIu32
        " owned by executable %" PRIu32,
        executable_index, entry_index, entry->executable_index);
  }

  iree_const_byte_span_t constants = iree_const_byte_span_empty();
  iree_hal_buffer_ref_list_t bindings = iree_hal_buffer_ref_list_empty();
  IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_pack_arguments(
      workspace, entry, operands, argument_start, &constants, &bindings));
  return iree_hal_command_buffer_dispatch(
      workspace->command_buffer,
      workspace->inputs->executables[executable_index], entry->function, config,
      constants, bindings, flags);
}

static iree_status_t loom_cmd_iree_hal_record_execution_barrier(
    iree_hal_command_buffer_t* command_buffer) {
  const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE |
                      IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ |
                      IREE_HAL_ACCESS_SCOPE_CONSTANT_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
                      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ |
                      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE |
                      IREE_HAL_ACCESS_SCOPE_MEMORY_READ |
                      IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE,
  };
  return iree_hal_command_buffer_execution_barrier(
      command_buffer,
      IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 1, &memory_barrier, 0, NULL);
}

static iree_status_t loom_cmd_iree_hal_record_packet(
    loom_cmd_iree_hal_workspace_t* workspace,
    const loom_low_descriptor_packet_t* packet) {
  const loom_op_t* op = packet->op;
  if (loom_cmd_iree_hal_packet_is(workspace, packet,
                                  CMD_CORE_DESCRIPTOR_REF_CONSTANT_U32)) {
    workspace->values[loom_low_const_result(op)].u32 =
        (uint32_t)loom_cmd_iree_hal_constant_value(op);
    return iree_ok_status();
  }
  if (loom_cmd_iree_hal_packet_is(workspace, packet,
                                  CMD_CORE_DESCRIPTOR_REF_CONSTANT_U64)) {
    workspace->values[loom_low_const_result(op)].u64 =
        (uint64_t)loom_cmd_iree_hal_constant_value(op);
    return iree_ok_status();
  }
  if (loom_cmd_iree_hal_packet_is(workspace, packet,
                                  CMD_CORE_DESCRIPTOR_REF_CONSTANT_B8) ||
      loom_cmd_iree_hal_packet_is(workspace, packet,
                                  CMD_CORE_DESCRIPTOR_REF_CONSTANT_B16) ||
      loom_cmd_iree_hal_packet_is(workspace, packet,
                                  CMD_CORE_DESCRIPTOR_REF_CONSTANT_B32) ||
      loom_cmd_iree_hal_packet_is(workspace, packet,
                                  CMD_CORE_DESCRIPTOR_REF_CONSTANT_B64)) {
    workspace->values[loom_low_const_result(op)].u64 =
        (uint64_t)loom_cmd_iree_hal_constant_value(op);
    return iree_ok_status();
  }
  if (loom_cmd_iree_hal_packet_is(workspace, packet,
                                  CMD_CORE_DESCRIPTOR_REF_BUFFER_REF_DIRECT)) {
    return loom_cmd_iree_hal_make_direct_buffer_ref(workspace, op);
  }
  if (loom_cmd_iree_hal_packet_is(workspace, packet,
                                  CMD_CORE_DESCRIPTOR_REF_BUFFER_REF_BINDING)) {
    loom_cmd_iree_hal_make_binding_buffer_ref(workspace, op);
    return iree_ok_status();
  }
  const bool is_fill = loom_cmd_iree_hal_packet_is(
      workspace, packet, CMD_CORE_DESCRIPTOR_REF_FILL);
  const bool is_fill_barrier = loom_cmd_iree_hal_packet_is(
      workspace, packet, CMD_CORE_DESCRIPTOR_REF_FILL_BARRIER);
  if (is_fill || is_fill_barrier) {
    if (is_fill_barrier) {
      IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_record_execution_barrier(
          workspace->command_buffer));
    }
    const loom_value_slice_t operands = loom_low_op_operands(op);
    const uint32_t pattern = workspace->values[operands.values[1]].u32;
    return iree_hal_command_buffer_fill_buffer(
        workspace->command_buffer,
        workspace->values[operands.values[0]].buffer_ref, &pattern,
        workspace->values[operands.values[2]].u32, IREE_HAL_FILL_FLAG_NONE);
  }
  const bool is_copy = loom_cmd_iree_hal_packet_is(
      workspace, packet, CMD_CORE_DESCRIPTOR_REF_COPY);
  const bool is_copy_barrier = loom_cmd_iree_hal_packet_is(
      workspace, packet, CMD_CORE_DESCRIPTOR_REF_COPY_BARRIER);
  if (is_copy || is_copy_barrier) {
    if (is_copy_barrier) {
      IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_record_execution_barrier(
          workspace->command_buffer));
    }
    const loom_value_slice_t operands = loom_low_op_operands(op);
    return iree_hal_command_buffer_copy_buffer(
        workspace->command_buffer,
        workspace->values[operands.values[0]].buffer_ref,
        workspace->values[operands.values[1]].buffer_ref,
        IREE_HAL_COPY_FLAG_NONE);
  }
  const bool is_direct = loom_cmd_iree_hal_packet_is(
      workspace, packet, CMD_CORE_DESCRIPTOR_REF_DISPATCH_DIRECT);
  const bool is_direct_barrier = loom_cmd_iree_hal_packet_is(
      workspace, packet, CMD_CORE_DESCRIPTOR_REF_DISPATCH_DIRECT_BARRIER);
  if (is_direct || is_direct_barrier) {
    if (is_direct_barrier) {
      IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_record_execution_barrier(
          workspace->command_buffer));
    }
    const loom_value_slice_t operands = loom_low_op_operands(op);
    const iree_hal_dispatch_config_t config =
        iree_hal_make_static_dispatch_config(
            workspace->values[operands.values[2]].u32,
            workspace->values[operands.values[3]].u32,
            workspace->values[operands.values[4]].u32);
    return loom_cmd_iree_hal_dispatch(workspace, op, config,
                                      IREE_HAL_DISPATCH_FLAG_NONE,
                                      /*argument_start=*/5);
  }
  const bool is_indirect_static = loom_cmd_iree_hal_packet_is(
      workspace, packet, CMD_CORE_DESCRIPTOR_REF_DISPATCH_INDIRECT_STATIC);
  const bool is_indirect_static_barrier = loom_cmd_iree_hal_packet_is(
      workspace, packet,
      CMD_CORE_DESCRIPTOR_REF_DISPATCH_INDIRECT_STATIC_BARRIER);
  const bool is_indirect_dynamic = loom_cmd_iree_hal_packet_is(
      workspace, packet, CMD_CORE_DESCRIPTOR_REF_DISPATCH_INDIRECT_DYNAMIC);
  const bool is_indirect_dynamic_barrier = loom_cmd_iree_hal_packet_is(
      workspace, packet,
      CMD_CORE_DESCRIPTOR_REF_DISPATCH_INDIRECT_DYNAMIC_BARRIER);
  if (is_indirect_static || is_indirect_static_barrier || is_indirect_dynamic ||
      is_indirect_dynamic_barrier) {
    if (is_indirect_static_barrier || is_indirect_dynamic_barrier) {
      IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_record_execution_barrier(
          workspace->command_buffer));
    }
    const loom_value_slice_t operands = loom_low_op_operands(op);
    iree_hal_dispatch_config_t config = {0};
    config.workgroup_count_ref =
        workspace->values[operands.values[2]].buffer_ref;
    const iree_hal_dispatch_flags_t flags =
        is_indirect_static || is_indirect_static_barrier
            ? IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS
            : IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS;
    return loom_cmd_iree_hal_dispatch(workspace, op, config, flags,
                                      /*argument_start=*/3);
  }
  if (loom_cmd_iree_hal_packet_is(workspace, packet,
                                  CMD_CORE_DESCRIPTOR_REF_BARRIER_EXECUTION)) {
    return loom_cmd_iree_hal_record_execution_barrier(
        workspace->command_buffer);
  }
  const iree_string_view_t descriptor_key =
      loom_low_descriptor_packet_diagnostic_key(workspace->descriptor_set,
                                                packet);
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "cmd low descriptor `%.*s` cannot be recorded as an IREE HAL command",
      (int)descriptor_key.size, descriptor_key.data);
}

typedef struct loom_cmd_iree_hal_program_workspace_t {
  // Parsed artifact being recorded.
  const loom_cmd_program_t* program;
  // Package resources supplied by the application.
  const loom_cmd_iree_hal_inputs_t* inputs;
  // Begun command buffer receiving portable commands.
  iree_hal_command_buffer_t* command_buffer;
  // Optional recorder-relative HAL operation identity map.
  loom_cmd_iree_hal_operation_map_t* operation_map;
  // Barrier wave containing the next payload command.
  uint32_t barrier_wave_ordinal;
  // Resolved ranges used by fills, copies, and indirect launch counts.
  iree_hal_buffer_ref_t* buffer_refs;
  // Scratch storage for one dispatch constant block.
  uint8_t* constants;
  // Scratch storage for one dispatch binding table.
  iree_hal_buffer_ref_t* bindings;
} loom_cmd_iree_hal_program_workspace_t;

static void loom_cmd_iree_hal_append_operation_map_entry(
    loom_cmd_iree_hal_program_workspace_t* workspace, uint32_t command_ordinal,
    loom_cmd_iree_hal_operation_phase_t phase) {
  if (workspace->operation_map == NULL) return;
  IREE_ASSERT_LT(workspace->operation_map->count,
                 workspace->operation_map->capacity);
  workspace->operation_map->entries[workspace->operation_map->count++] =
      (loom_cmd_iree_hal_operation_map_entry_t){
          .command_ordinal = command_ordinal,
          .barrier_wave_ordinal = workspace->barrier_wave_ordinal,
          .phase = phase,
      };
}

static iree_status_t loom_cmd_iree_hal_record_program_barrier(
    loom_cmd_iree_hal_program_workspace_t* workspace,
    uint32_t command_ordinal) {
  IREE_RETURN_IF_ERROR(
      loom_cmd_iree_hal_record_execution_barrier(workspace->command_buffer));
  ++workspace->barrier_wave_ordinal;
  loom_cmd_iree_hal_append_operation_map_entry(
      workspace, command_ordinal, LOOM_CMD_IREE_HAL_OPERATION_PHASE_BARRIER);
  return iree_ok_status();
}

static iree_status_t loom_cmd_iree_hal_validate_program_inputs(
    const loom_cmd_program_t* program, const loom_cmd_iree_hal_inputs_t* inputs,
    iree_host_size_t* out_max_constant_byte_length,
    iree_host_size_t* out_max_binding_count) {
  IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_validate_inputs(
      inputs, out_max_constant_byte_length, out_max_binding_count));
  IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_validate_input_counts(
      program->requirements.fixed_buffer_count,
      program->requirements.rebindable_binding_count,
      program->requirements.executable_count, program->requirements.entry_count,
      inputs));
  for (uint32_t schema_index = 0; schema_index < program->entry_schemas.count;
       ++schema_index) {
    const loom_cmd_program_entry_schema_t schema =
        loom_cmd_program_entry_schema_at(program, schema_index);
    const loom_cmd_iree_hal_entry_t* entry =
        &inputs->entries[schema.entry_index];
    if (schema.argument_count != entry->info.parameter_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "command entry %" PRIu32 " schema has %" PRIu32
          " logical arguments but its implementation expects %" PRIu16,
          schema.entry_index, schema.argument_count,
          entry->info.parameter_count);
    }
    for (uint32_t argument_index = 0; argument_index < schema.argument_count;
         ++argument_index) {
      const loom_cmd_program_argument_kind_t kind =
          loom_cmd_program_entry_schema_kind_at(program, &schema,
                                                argument_index);
      const iree_hal_executable_function_parameter_t* parameter =
          &entry->parameters[argument_index];
      uint16_t scalar_byte_length = 0;
      switch (kind) {
        case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B8:
          scalar_byte_length = 1;
          break;
        case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B16:
          scalar_byte_length = 2;
          break;
        case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B32:
          scalar_byte_length = 4;
          break;
        case LOOM_CMD_PROGRAM_ARGUMENT_KIND_B64:
          scalar_byte_length = 8;
          break;
        case LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER:
          if (parameter->type ==
              IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING) {
            continue;
          }
          if (parameter->type ==
              IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BUFFER_PTR) {
            return iree_make_status(
                IREE_STATUS_UNIMPLEMENTED,
                "command entry raw buffer-pointer parameters cannot preserve "
                "issue-time buffer rebinding");
          }
          break;
      }
      if (scalar_byte_length == 0 ||
          parameter->type !=
              IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT ||
          parameter->size != scalar_byte_length) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "command entry %" PRIu32 " argument %" PRIu32
            " is incompatible with its portable logical kind",
            schema.entry_index, argument_index);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_iree_hal_resolve_program_buffer_ref(
    const loom_cmd_iree_hal_program_workspace_t* workspace,
    loom_cmd_program_buffer_ref_t source,
    iree_hal_buffer_ref_t* out_buffer_ref) {
  if (source.role == LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE) {
    *out_buffer_ref = iree_hal_make_indirect_buffer_ref(
        source.root_index, source.byte_offset, source.byte_length);
    return iree_ok_status();
  }

  const iree_hal_buffer_ref_t root =
      workspace->inputs->fixed_buffers[source.root_index];
  if (root.buffer == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fixed-buffer root %" PRIu32
                            " is not a direct buffer",
                            source.root_index);
  }
  iree_device_size_t adjusted_offset = 0;
  iree_device_size_t adjusted_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      root.offset, root.length, source.byte_offset, source.byte_length,
      &adjusted_offset, &adjusted_length));
  *out_buffer_ref =
      iree_hal_make_buffer_ref(root.buffer, adjusted_offset, adjusted_length);
  return iree_ok_status();
}

static iree_status_t loom_cmd_iree_hal_resolve_program_buffer_refs(
    loom_cmd_iree_hal_program_workspace_t* workspace) {
  for (uint32_t i = 0; i < workspace->program->buffer_refs.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_resolve_program_buffer_ref(
        workspace, loom_cmd_program_buffer_ref_at(workspace->program, i),
        &workspace->buffer_refs[i]));
  }
  return iree_ok_status();
}

static loom_cmd_program_buffer_ref_t
loom_cmd_iree_hal_decode_program_buffer_argument(const uint8_t* data) {
  return (loom_cmd_program_buffer_ref_t){
      .role = (loom_cmd_program_buffer_role_t)iree_unaligned_load_le_u32(
          data + LOOM_CMD_PROGRAM_BUFFER_REF_ROLE_OFFSET),
      .root_index = iree_unaligned_load_le_u32(
          data + LOOM_CMD_PROGRAM_BUFFER_REF_ROOT_INDEX_OFFSET),
      .byte_offset = iree_unaligned_load_le_u64(
          data + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_OFFSET_OFFSET),
      .byte_length = iree_unaligned_load_le_u64(
          data + LOOM_CMD_PROGRAM_BUFFER_REF_BYTE_LENGTH_OFFSET),
  };
}

static iree_status_t loom_cmd_iree_hal_pack_program_arguments(
    loom_cmd_iree_hal_program_workspace_t* workspace,
    const loom_cmd_program_command_t* command,
    const loom_cmd_iree_hal_entry_t* entry,
    iree_const_byte_span_t* out_constants,
    iree_hal_buffer_ref_list_t* out_bindings) {
  if (entry->info.constant_byte_length != 0) {
    memset(workspace->constants, 0, entry->info.constant_byte_length);
  }
  if (entry->info.binding_count != 0) {
    memset(workspace->bindings, 0,
           entry->info.binding_count * sizeof(*workspace->bindings));
  }
  const loom_cmd_program_entry_schema_t schema =
      loom_cmd_program_entry_schema_at(workspace->program,
                                       command->argument_schema_index);
  iree_const_byte_span_t argument_data =
      loom_cmd_program_command_argument_data(workspace->program, command);
  const uint8_t* cursor = argument_data.data;
  for (uint32_t i = 0; i < schema.argument_count; ++i) {
    const loom_cmd_program_argument_kind_t kind =
        loom_cmd_program_entry_schema_kind_at(workspace->program, &schema, i);
    const iree_hal_executable_function_parameter_t* parameter =
        &entry->parameters[i];
    if (kind == LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER) {
      const loom_cmd_program_buffer_ref_t source =
          loom_cmd_iree_hal_decode_program_buffer_argument(cursor);
      IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_resolve_program_buffer_ref(
          workspace, source, &workspace->bindings[parameter->offset]));
      cursor += LOOM_CMD_PROGRAM_BUFFER_REF_SIZE;
    } else {
      memcpy(workspace->constants + parameter->offset, cursor, parameter->size);
      cursor += parameter->size;
    }
  }
  IREE_ASSERT_EQ(cursor, argument_data.data + argument_data.data_length);

  *out_constants = iree_make_const_byte_span(workspace->constants,
                                             entry->info.constant_byte_length);
  *out_bindings = (iree_hal_buffer_ref_list_t){
      .count = entry->info.binding_count,
      .values = workspace->bindings,
  };
  return iree_ok_status();
}

static iree_status_t loom_cmd_iree_hal_record_program_dispatch(
    loom_cmd_iree_hal_program_workspace_t* workspace,
    const loom_cmd_program_command_t* command) {
  const loom_cmd_program_command_kind_t base_kind =
      loom_cmd_program_command_kind_base(command->kind);
  const bool is_direct =
      base_kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT;
  const uint32_t executable_index =
      is_direct ? command->payload.dispatch_direct.executable_index
                : command->payload.dispatch_indirect.executable_index;
  const uint32_t entry_index =
      is_direct ? command->payload.dispatch_direct.entry_index
                : command->payload.dispatch_indirect.entry_index;
  const loom_cmd_iree_hal_entry_t* entry =
      &workspace->inputs->entries[entry_index];
  if (entry->executable_index != executable_index) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command dispatch pairs executable %" PRIu32 " with entry %" PRIu32
        " owned by executable %" PRIu32,
        executable_index, entry_index, entry->executable_index);
  }

  iree_const_byte_span_t constants = iree_const_byte_span_empty();
  iree_hal_buffer_ref_list_t bindings = iree_hal_buffer_ref_list_empty();
  IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_pack_program_arguments(
      workspace, command, entry, &constants, &bindings));
  iree_hal_dispatch_config_t config = {0};
  iree_hal_dispatch_flags_t flags = IREE_HAL_DISPATCH_FLAG_NONE;
  if (is_direct) {
    config = iree_hal_make_static_dispatch_config(
        command->payload.dispatch_direct.workgroup_count_x,
        command->payload.dispatch_direct.workgroup_count_y,
        command->payload.dispatch_direct.workgroup_count_z);
  } else {
    config.workgroup_count_ref =
        workspace->buffer_refs[command->payload.dispatch_indirect
                                   .workgroup_count_buffer_ref];
    const bool is_static =
        base_kind == LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC;
    flags = is_static ? IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS
                      : IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS;
  }
  return iree_hal_command_buffer_dispatch(
      workspace->command_buffer,
      workspace->inputs->executables[executable_index], entry->function, config,
      constants, bindings, flags);
}

static iree_status_t loom_cmd_iree_hal_record_program_command(
    loom_cmd_iree_hal_program_workspace_t* workspace, uint32_t command_ordinal,
    const loom_cmd_program_command_t* command) {
  if (loom_cmd_program_command_kind_has_barrier(command->kind)) {
    IREE_RETURN_IF_ERROR(
        loom_cmd_iree_hal_record_program_barrier(workspace, command_ordinal));
  }
  iree_status_t status = iree_ok_status();
  switch ((uint32_t)loom_cmd_program_command_kind_base(command->kind)) {
    case LOOM_CMD_PROGRAM_COMMAND_KIND_FILL: {
      const uint32_t pattern = command->payload.fill.pattern;
      status = iree_hal_command_buffer_fill_buffer(
          workspace->command_buffer,
          workspace->buffer_refs[command->payload.fill.target_buffer_ref],
          &pattern, command->payload.fill.pattern_length,
          IREE_HAL_FILL_FLAG_NONE);
      break;
    }
    case LOOM_CMD_PROGRAM_COMMAND_KIND_COPY:
      status = iree_hal_command_buffer_copy_buffer(
          workspace->command_buffer,
          workspace->buffer_refs[command->payload.copy.source_buffer_ref],
          workspace->buffer_refs[command->payload.copy.target_buffer_ref],
          IREE_HAL_COPY_FLAG_NONE);
      break;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT:
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC:
    case LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_DYNAMIC:
      status = loom_cmd_iree_hal_record_program_dispatch(workspace, command);
      break;
    case LOOM_CMD_PROGRAM_COMMAND_KIND_BARRIER_EXECUTION:
      return loom_cmd_iree_hal_record_program_barrier(workspace,
                                                      command_ordinal);
    default:
      IREE_ASSERT_UNREACHABLE("parsed command kinds are exhaustive");
      return iree_ok_status();
  }
  if (iree_status_is_ok(status)) {
    loom_cmd_iree_hal_append_operation_map_entry(
        workspace, command_ordinal, LOOM_CMD_IREE_HAL_OPERATION_PHASE_PAYLOAD);
  }
  return status;
}

static uint32_t loom_cmd_iree_hal_program_first_barrier_wave_ordinal(
    const loom_cmd_program_t* program) {
  if (program->commands.count == 0) return 0;
  const loom_cmd_program_command_t first_command =
      loom_cmd_program_command_at(program, 0);
  return loom_cmd_program_command_kind_begins_barrier_wave(first_command.kind)
             ? 1
             : 0;
}

iree_status_t loom_cmd_iree_hal_record_function(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_cmd_iree_hal_inputs_t* inputs,
    iree_hal_command_buffer_t* command_buffer,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(function_op);
  IREE_ASSERT_ARGUMENT(inputs);
  IREE_ASSERT_ARGUMENT(command_buffer);
  if (!loom_low_func_def_isa(function_op) ||
      loom_low_func_def_abi(function_op) != LOOM_TARGET_ABI_COMMAND_PROGRAM) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected a command_program low.func.def");
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_cmd_core_descriptor_set();
  const iree_string_view_t descriptor_set_key = loom_cmd_iree_hal_module_string(
      module, loom_low_func_def_descriptor_set(function_op));
  if (!iree_string_view_equal(
          descriptor_set_key,
          loom_low_descriptor_set_string(descriptor_set,
                                         descriptor_set->key_string_offset))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program must use the cmd.core contract");
  }

  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL || body->block_count != 1 ||
      loom_region_const_entry_block(body)->arg_count != 0 ||
      function_op->result_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "IREE HAL command recording requires a zero-signature, "
        "single-block cmd low function");
  }

  loom_cmd_abi_layout_t abi_layout = {0};
  IREE_RETURN_IF_ERROR(
      loom_cmd_abi_layout_from_low(module, function_op, &abi_layout));

  iree_host_size_t max_constant_byte_length = 0;
  iree_host_size_t max_binding_count = 0;
  IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_validate_inputs(
      inputs, &max_constant_byte_length, &max_binding_count));
  IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_validate_input_counts(
      abi_layout.fixed_buffer_count, abi_layout.rebindable_binding_count,
      abi_layout.executable_count, abi_layout.entry_count, inputs));

  iree_host_size_t total_size = 0;
  iree_host_size_t values_offset = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t bindings_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &total_size,
      IREE_STRUCT_FIELD_ALIGNED(module->values.count, loom_cmd_iree_hal_value_t,
                                iree_alignof(loom_cmd_iree_hal_value_t),
                                &values_offset),
      IREE_STRUCT_FIELD_ALIGNED(max_binding_count, iree_hal_buffer_ref_t,
                                iree_alignof(iree_hal_buffer_ref_t),
                                &bindings_offset),
      IREE_STRUCT_FIELD(max_constant_byte_length, uint8_t, &constants_offset)));

  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&storage));
  memset(storage, 0, total_size);
  loom_cmd_iree_hal_workspace_t workspace = {
      .module = module,
      .descriptor_set = descriptor_set,
      .inputs = inputs,
      .command_buffer = command_buffer,
      .values = (loom_cmd_iree_hal_value_t*)(storage + values_offset),
      .constants = storage + constants_offset,
      .bindings = (iree_hal_buffer_ref_t*)(storage + bindings_offset),
  };

  iree_status_t status = iree_ok_status();
  const loom_block_t* block = loom_region_const_entry_block(body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (!iree_status_is_ok(status)) break;
    loom_low_descriptor_packet_t packet = {0};
    loom_low_descriptor_packet_initialize(descriptor_set, op, &packet);
    if (packet.kind != LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
      status = loom_cmd_iree_hal_record_packet(&workspace, &packet);
    } else if (loom_low_resource_isa(op)) {
      if (loom_low_resource_import_kind(op) !=
          LOOM_LOW_RESOURCE_IMPORT_KIND_COMMAND_INPUT) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "cmd low function contains a non-command resource import");
      } else {
        status = loom_cmd_iree_hal_import_resource(&workspace, op);
      }
    } else if (loom_low_return_isa(op)) {
      if (op->operand_count != 0) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "IREE HAL command programs cannot return low values");
      }
    } else {
      const iree_string_view_t op_name = loom_op_name(module, op);
      status = iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "low operation `%.*s` cannot be recorded as an IREE HAL command",
          (int)op_name.size, op_name.data);
    }
  }

  iree_allocator_free(host_allocator, storage);
  return status;
}

iree_status_t loom_cmd_iree_hal_materialize_function(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_cmd_iree_hal_inputs_t* inputs, iree_hal_device_t* device,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(function_op);
  IREE_ASSERT_ARGUMENT(inputs);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device, mode, IREE_HAL_COMMAND_CATEGORY_ANY, queue_affinity,
      inputs->binding_count, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_iree_hal_record_function(module, function_op, inputs,
                                               command_buffer, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = command_buffer;
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  return status;
}

iree_status_t loom_cmd_iree_hal_record_program_range(
    const loom_cmd_program_t* program, loom_cmd_program_command_range_t range,
    uint32_t barrier_wave_ordinal, const loom_cmd_iree_hal_inputs_t* inputs,
    iree_hal_command_buffer_t* command_buffer,
    loom_cmd_iree_hal_operation_map_t* operation_map,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_ARGUMENT(inputs);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_LE(range.first_command, program->commands.count);
  IREE_ASSERT_LE(range.command_count,
                 program->commands.count - range.first_command);
  if (operation_map != NULL) operation_map->count = 0;

  iree_host_size_t max_constant_byte_length = 0;
  iree_host_size_t max_binding_count = 0;
  IREE_RETURN_IF_ERROR(loom_cmd_iree_hal_validate_program_inputs(
      program, inputs, &max_constant_byte_length, &max_binding_count));

  iree_host_size_t total_size = 0;
  iree_host_size_t buffer_refs_offset = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t bindings_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          program->buffer_refs.count, iree_hal_buffer_ref_t,
          iree_alignof(iree_hal_buffer_ref_t), &buffer_refs_offset),
      IREE_STRUCT_FIELD_ALIGNED(max_binding_count, iree_hal_buffer_ref_t,
                                iree_alignof(iree_hal_buffer_ref_t),
                                &bindings_offset),
      IREE_STRUCT_FIELD(max_constant_byte_length, uint8_t, &constants_offset)));

  uint8_t* storage = NULL;
  iree_status_t status = iree_ok_status();
  if (total_size != 0) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&storage);
  }
  if (iree_status_is_ok(status) && total_size != 0) {
    memset(storage, 0, total_size);
  }
  loom_cmd_iree_hal_program_workspace_t workspace = {
      .program = program,
      .inputs = inputs,
      .command_buffer = command_buffer,
      .operation_map = operation_map,
      .barrier_wave_ordinal = barrier_wave_ordinal,
      .buffer_refs =
          storage ? (iree_hal_buffer_ref_t*)(storage + buffer_refs_offset)
                  : NULL,
      .constants = storage ? storage + constants_offset : NULL,
      .bindings =
          storage ? (iree_hal_buffer_ref_t*)(storage + bindings_offset) : NULL,
  };
  if (iree_status_is_ok(status)) {
    status = loom_cmd_iree_hal_resolve_program_buffer_refs(&workspace);
  }
  if (range.command_count != 0) {
    const loom_cmd_program_command_t first_command =
        loom_cmd_program_command_at(program, range.first_command);
    if (loom_cmd_program_command_kind_begins_barrier_wave(first_command.kind)) {
      IREE_ASSERT_GT(workspace.barrier_wave_ordinal, 0u);
      --workspace.barrier_wave_ordinal;
    }
  }
  const uint32_t end_command = range.first_command + range.command_count;
  for (uint32_t i = range.first_command;
       i < end_command && iree_status_is_ok(status); ++i) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(program, i);
    status = loom_cmd_iree_hal_record_program_command(&workspace, i, &command);
  }

  iree_allocator_free(host_allocator, storage);
  return status;
}

iree_status_t loom_cmd_iree_hal_record_program(
    const loom_cmd_program_t* program, const loom_cmd_iree_hal_inputs_t* inputs,
    iree_hal_command_buffer_t* command_buffer,
    loom_cmd_iree_hal_operation_map_t* operation_map,
    iree_allocator_t host_allocator) {
  const loom_cmd_program_command_range_t range =
      loom_cmd_program_command_range_all(program);
  const uint32_t first_wave_ordinal =
      loom_cmd_iree_hal_program_first_barrier_wave_ordinal(program);
  return loom_cmd_iree_hal_record_program_range(
      program, range, first_wave_ordinal, inputs, command_buffer, operation_map,
      host_allocator);
}

iree_status_t loom_cmd_iree_hal_materialize_program_range(
    const loom_cmd_program_t* program, loom_cmd_program_command_range_t range,
    uint32_t barrier_wave_ordinal, const loom_cmd_iree_hal_inputs_t* inputs,
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_queue_affinity_t queue_affinity,
    loom_cmd_iree_hal_operation_map_t* operation_map,
    iree_hal_command_buffer_t** out_command_buffer,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(program);
  IREE_ASSERT_ARGUMENT(inputs);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  iree_hal_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device, mode, IREE_HAL_COMMAND_CATEGORY_ANY, queue_affinity,
      program->requirements.rebindable_binding_count, &command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_begin(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_iree_hal_record_program_range(
        program, range, barrier_wave_ordinal, inputs, command_buffer,
        operation_map, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(command_buffer);
  }
  if (iree_status_is_ok(status)) {
    *out_command_buffer = command_buffer;
  } else {
    iree_hal_command_buffer_release(command_buffer);
  }
  return status;
}

iree_status_t loom_cmd_iree_hal_materialize_program(
    const loom_cmd_program_t* program, const loom_cmd_iree_hal_inputs_t* inputs,
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_queue_affinity_t queue_affinity,
    loom_cmd_iree_hal_operation_map_t* operation_map,
    iree_hal_command_buffer_t** out_command_buffer,
    iree_allocator_t host_allocator) {
  const loom_cmd_program_command_range_t range =
      loom_cmd_program_command_range_all(program);
  const uint32_t first_wave_ordinal =
      loom_cmd_iree_hal_program_first_barrier_wave_ordinal(program);
  return loom_cmd_iree_hal_materialize_program_range(
      program, range, first_wave_ordinal, inputs, device, mode, queue_affinity,
      operation_map, out_command_buffer, host_allocator);
}
