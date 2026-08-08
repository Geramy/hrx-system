// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/lower.h"

#include <array>
#include <cstdint>
#include <vector>

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/exact_function.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/cmd/descriptors/low_registry.h"
#include "loom/target/arch/cmd/iree_hal/recording.h"
#include "loom/target/arch/cmd/lower/launch_artifact.h"
#include "loom/target/arch/cmd/lower/schedule.h"
#include "loom/target/arch/cmd/lower/serialize.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/testing/module_ptr.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ::loom::testing::DiagnosticEmissionCapture;
using ModulePtr = ::loom::testing::ModulePtr;

enum class CapturedCommandKind {
  // Full execution barrier between portable schedule waves.
  kBarrier,
  // Direct kernel dispatch within one portable schedule wave.
  kDispatch,
};

struct CapturedCommand {
  // Kind selecting the populated command payload.
  CapturedCommandKind kind;
  // Executable passed to a captured dispatch.
  iree_hal_executable_t* executable = nullptr;
  // Executable-local function token passed to a captured dispatch.
  iree_hal_executable_function_t function = {};
  // Workgroup counts passed to a captured dispatch.
  iree_hal_dispatch_config_t dispatch_config = {};
  // Dispatch flags selecting direct or indirect parameter stability.
  iree_hal_dispatch_flags_t dispatch_flags = IREE_HAL_DISPATCH_FLAG_NONE;
  // Buffer references passed to a captured dispatch.
  std::vector<iree_hal_buffer_ref_t> bindings;
};

struct CaptureCommandBuffer {
  // Base HAL command buffer implemented by the capture vtable.
  iree_hal_command_buffer_t base;
  // Number of observed begin operations.
  uint32_t begin_count = 0;
  // Number of observed end operations.
  uint32_t end_count = 0;
  // Recorded barriers and dispatches in invocation order.
  std::vector<CapturedCommand> commands;
};

static CaptureCommandBuffer* CastCommandBuffer(
    iree_hal_command_buffer_t* base_command_buffer) {
  return reinterpret_cast<CaptureCommandBuffer*>(base_command_buffer);
}

static void DestroyCommandBuffer(iree_hal_command_buffer_t*) {}

static iree_status_t BeginCommandBuffer(
    iree_hal_command_buffer_t* base_command_buffer) {
  ++CastCommandBuffer(base_command_buffer)->begin_count;
  return iree_ok_status();
}

static iree_status_t EndCommandBuffer(
    iree_hal_command_buffer_t* base_command_buffer) {
  ++CastCommandBuffer(base_command_buffer)->end_count;
  return iree_ok_status();
}

static iree_status_t CaptureExecutionBarrier(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_execution_stage_t,
    iree_hal_execution_stage_t, iree_hal_execution_barrier_flags_t,
    iree_host_size_t, const iree_hal_memory_barrier_t*, iree_host_size_t,
    const iree_hal_buffer_barrier_t*) {
  CastCommandBuffer(base_command_buffer)
      ->commands.push_back({/*.kind=*/CapturedCommandKind::kBarrier});
  return iree_ok_status();
}

static iree_status_t CaptureDispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  IREE_ASSERT(iree_const_byte_span_is_empty(constants));
  CapturedCommand command = {
      /*.kind=*/CapturedCommandKind::kDispatch,
      /*.executable=*/executable,
      /*.function=*/function,
      /*.dispatch_config=*/config,
      /*.dispatch_flags=*/flags,
  };
  command.bindings.assign(bindings.values, bindings.values + bindings.count);
  CastCommandBuffer(base_command_buffer)
      ->commands.push_back(std::move(command));
  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t kCaptureCommandBufferVtable = {
    /*.destroy=*/DestroyCommandBuffer,
    /*.begin=*/BeginCommandBuffer,
    /*.end=*/EndCommandBuffer,
    /*.begin_debug_group=*/nullptr,
    /*.end_debug_group=*/nullptr,
    /*.execution_barrier=*/CaptureExecutionBarrier,
    /*.signal_event=*/nullptr,
    /*.reset_event=*/nullptr,
    /*.wait_events=*/nullptr,
    /*.advise_buffer=*/nullptr,
    /*.fill_buffer=*/nullptr,
    /*.update_buffer=*/nullptr,
    /*.copy_buffer=*/nullptr,
    /*.collective=*/nullptr,
    /*.dispatch=*/CaptureDispatch,
};

static void InitializeCommandBuffer(iree_host_size_t binding_count,
                                    CaptureCommandBuffer* command_buffer) {
  iree_hal_command_buffer_initialize(
      /*device_allocator=*/nullptr,
      IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED |
          IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
      IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY, binding_count,
      /*validation_state=*/nullptr, &kCaptureCommandBufferVtable,
      &command_buffer->base);
}

static void ExpectCapturedProgramsEqual(const CaptureCommandBuffer& expected,
                                        const CaptureCommandBuffer& actual) {
  EXPECT_EQ(actual.begin_count, expected.begin_count);
  EXPECT_EQ(actual.end_count, expected.end_count);
  ASSERT_EQ(actual.commands.size(), expected.commands.size());
  for (iree_host_size_t i = 0; i < expected.commands.size(); ++i) {
    const CapturedCommand& expected_command = expected.commands[i];
    const CapturedCommand& actual_command = actual.commands[i];
    ASSERT_EQ(actual_command.kind, expected_command.kind);
    if (actual_command.kind != CapturedCommandKind::kDispatch) continue;
    EXPECT_EQ(actual_command.executable, expected_command.executable);
    EXPECT_EQ(actual_command.function.value, expected_command.function.value);
    EXPECT_EQ(actual_command.dispatch_flags, expected_command.dispatch_flags);
    for (iree_host_size_t axis = 0;
         axis < LOOM_CMD_LAUNCH_COUNT_DIMENSION_COUNT; ++axis) {
      EXPECT_EQ(actual_command.dispatch_config.workgroup_size[axis],
                expected_command.dispatch_config.workgroup_size[axis]);
      EXPECT_EQ(actual_command.dispatch_config.workgroup_count[axis],
                expected_command.dispatch_config.workgroup_count[axis]);
    }
    EXPECT_EQ(actual_command.dispatch_config.dynamic_workgroup_local_memory,
              expected_command.dispatch_config.dynamic_workgroup_local_memory);
    EXPECT_EQ(actual_command.dispatch_config.workgroup_count_ref.buffer,
              expected_command.dispatch_config.workgroup_count_ref.buffer);
    EXPECT_EQ(actual_command.dispatch_config.workgroup_count_ref.buffer_slot,
              expected_command.dispatch_config.workgroup_count_ref.buffer_slot);
    EXPECT_EQ(actual_command.dispatch_config.workgroup_count_ref.offset,
              expected_command.dispatch_config.workgroup_count_ref.offset);
    EXPECT_EQ(actual_command.dispatch_config.workgroup_count_ref.length,
              expected_command.dispatch_config.workgroup_count_ref.length);
    ASSERT_EQ(actual_command.bindings.size(), expected_command.bindings.size());
    for (iree_host_size_t binding = 0;
         binding < expected_command.bindings.size(); ++binding) {
      EXPECT_EQ(actual_command.bindings[binding].buffer,
                expected_command.bindings[binding].buffer);
      EXPECT_EQ(actual_command.bindings[binding].buffer_slot,
                expected_command.bindings[binding].buffer_slot);
      EXPECT_EQ(actual_command.bindings[binding].offset,
                expected_command.bindings[binding].offset);
      EXPECT_EQ(actual_command.bindings[binding].length,
                expected_command.bindings[binding].length);
    }
  }
}

class CmdLowerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_cmd_low_descriptor_registry_initialize(&registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseAndVerifySource(const char* source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("cmd_lower_test.loom"), &context_,
                                  &block_pool_, &parse_options, &module));
    ModulePtr module_ptr(module);

    loom_verify_options_t verify_options = {};
    verify_options.max_errors = 20;
    loom_verify_result_t result = {};
    IREE_CHECK_OK(loom_verify_module(module, &verify_options, &result));
    IREE_ASSERT_EQ(result.error_count, 0u);
    return module_ptr;
  }

  void VerifyLowModule(loom_module_t* module) {
    DiagnosticEmissionCapture capture;
    loom_low_verify_options_t verify_options = {};
    verify_options.descriptor_registry = &registry_.registry;
    verify_options.emitter = capture.emitter();
    verify_options.max_errors = 20;
    loom_low_verify_scratch_t scratch =
        loom_low_verify_scratch_for_module(module);
    loom_low_verify_result_t result = {};
    IREE_ASSERT_OK(
        loom_low_verify_module(module, &verify_options, &scratch, &result));
    IREE_ASSERT_EQ(result.error_count, 0u);
    IREE_ASSERT(capture.emissions.empty());
  }

  loom_op_t* FindSymbol(loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return module->symbols.entries[symbol_id].defining_op;
  }

  loom_symbol_ref_t FindSymbolRef(loom_module_t* module,
                                  iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return loom_symbol_ref_t{
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
  }

  ModulePtr ReadAndVerifyModule(const std::vector<uint8_t>& bytes) {
    loom_bytecode_read_options_t options = {};
    options.verify_module = true;
    loom_bytecode_read_result_t result = {};
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_bytecode_read_module(
        iree_make_const_byte_span(bytes.data(), bytes.size()),
        IREE_SV("command_launch_config.loombc"), &context_, &block_pool_,
        &options, &result, &module, iree_allocator_system()));
    EXPECT_EQ(result.error_count, 0u);
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  // Shared arena block pool backing each parsed test module.
  iree_arena_block_pool_t block_pool_;
  // Source dialect context used by the text parser and verifier.
  loom_context_t context_;
  // Portable command descriptor registry used by low verification.
  loom_target_low_descriptor_registry_t registry_ = {};
};

TEST_F(CmdLowerTest, LowersAndRecordsPrepareThenConcurrentQkv) {
  ModulePtr module = ParseAndVerifySource(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def @prepare() {
  %sixteen = index.constant 16 : index
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%sixteen, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%parameters: view<128xi32, #dense>, %input: buffer, %scratch: buffer) {
  kernel.return
}

kernel.def @query(%token_count: index) {
  %one = index.constant 1 : index
  %row_groups = index.add %token_count, %one : index
  kernel.launch.config workgroups(%row_groups, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%parameters: buffer, %scratch: buffer, %output: buffer) {
  kernel.return
}

kernel.def @key(%token_count: index) {
  %one = index.constant 1 : index
  %row_groups = index.add %token_count, %one : index
  kernel.launch.config workgroups(%row_groups, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%parameters: buffer, %scratch: buffer, %output: buffer) {
  kernel.return
}

kernel.def @value(%token_count: index) {
  %one = index.constant 1 : index
  %row_groups = index.add %token_count, %one : index
  kernel.launch.config workgroups(%row_groups, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%parameters: buffer, %scratch: buffer, %output: buffer) {
  kernel.return
}

command.program.def public @attention(%token_count: index) launch(%parameters: buffer, %input: buffer, %scratch: buffer, %query_output: buffer, %key_output: buffer, %value_output: buffer) where [range(%token_count, 1, 512)] {
  %parameter_offset = index.constant 256 : offset
  %parameter_view = buffer.view %parameters[%parameter_offset] : buffer -> view<128xi32, #dense>
  kernel.launch @prepare[](%parameter_view, %input, %scratch) : [](view<128xi32, #dense>, buffer, buffer)
  command.concurrent {
    kernel.launch @query[%token_count](%parameters, %scratch, %query_output) : [index](buffer, buffer, buffer)
    kernel.launch @key[%token_count](%parameters, %scratch, %key_output) : [index](buffer, buffer, buffer)
    kernel.launch @value[%token_count](%parameters, %scratch, %value_output) : [index](buffer, buffer, buffer)
  }
  command.return
}
)");

  loom_op_t* source_program = FindSymbol(module.get(), IREE_SV("attention"));
  loom_func_like_t source_program_like =
      loom_func_like_cast(module.get(), source_program);
  iree_arena_allocator_t schedule_arena;
  iree_arena_initialize(&block_pool_, &schedule_arena);
  loom_cmd_schedule_plan_t schedule = {};
  IREE_ASSERT_OK(loom_cmd_schedule_plan_build(
      module.get(), loom_func_like_body(source_program_like), &schedule_arena,
      &schedule));
  ASSERT_EQ(schedule.command_count, 4u);
  loom_cmd_launch_graph_t launch_graph = {};
  IREE_ASSERT_OK(loom_cmd_launch_graph_materialize(
      module.get(), source_program, &schedule, &block_pool_,
      iree_allocator_system(), &launch_graph));
  ASSERT_EQ(launch_graph.launch_count, 4u);
  ASSERT_EQ(launch_graph.wave_count, 2u);
  ASSERT_EQ(launch_graph.waves[0].command_offset, 0u);
  ASSERT_EQ(launch_graph.waves[0].command_count, 1u);
  ASSERT_EQ(launch_graph.waves[1].command_offset, 1u);
  ASSERT_EQ(launch_graph.waves[1].command_count, 3u);
  ASSERT_EQ(launch_graph.host_tuple_count, 1u);
  ASSERT_EQ(launch_graph.launches[0].kind, LOOM_CMD_LAUNCH_COUNT_KIND_DIRECT);
  for (iree_host_size_t i = 1; i < launch_graph.launch_count; ++i) {
    ASSERT_EQ(launch_graph.launches[i].kind, LOOM_CMD_LAUNCH_COUNT_KIND_HOST);
    ASSERT_EQ(launch_graph.launches[i].payload.host_tuple_ordinal, 0u);
  }
  iree_byte_span_t launch_config_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_launch_program_serialize(
      launch_graph.module, &block_pool_, iree_allocator_system(),
      &launch_config_data));
  const loom_value_id_t parameter_view =
      loom_kernel_launch_arguments(schedule.commands[0]).values[0];
  iree_arena_deinitialize(&schedule_arena);

  static constexpr uint64_t kBufferLength = 4096;
  static constexpr uint64_t kLaunchCountOffset = 64;
  const std::array<loom_cmd_lower_binding_t, 6> binding_plan = {{
      {LOOM_CMD_LOWER_BUFFER_ROLE_FIXED, 0, 0, kBufferLength},
      {LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE, 0, 0, kBufferLength},
      {LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE, 1, 0, kBufferLength},
      {LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE, 2, 0, kBufferLength},
      {LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE, 3, 0, kBufferLength},
      {LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE, 4, 0, kBufferLength},
  }};
  const loom_cmd_lower_buffer_range_t buffer_range = {
      /*.source_value=*/parameter_view,
      /*.role=*/LOOM_CMD_LOWER_BUFFER_ROLE_FIXED,
      /*.resource_index=*/0,
      /*.byte_offset=*/256,
      /*.byte_length=*/512,
  };
  const std::array<uint16_t, 3> prepare_argument_ordinals = {0, 1, 2};
  const std::array<uint16_t, 2> projection_argument_ordinals = {0, 2};
  const std::array<loom_cmd_lower_launch_t, 4> launch_plan = {{
      {0, 0, prepare_argument_ordinals.size(),
       prepare_argument_ordinals.data()},
      {1, 1, projection_argument_ordinals.size(),
       projection_argument_ordinals.data()},
      {1, 2, projection_argument_ordinals.size(),
       projection_argument_ordinals.data()},
      {1, 3, projection_argument_ordinals.size(),
       projection_argument_ordinals.data()},
  }};
  const loom_cmd_lower_plan_t plan = {
      /*.command_target=*/FindSymbolRef(module.get(),
                                        IREE_SV("command_target")),
      /*.bindings=*/binding_plan.data(),
      /*.binding_count=*/binding_plan.size(),
      /*.buffer_ranges=*/&buffer_range,
      /*.buffer_range_count=*/1,
      /*.fixed_buffer_count=*/1,
      /*.rebindable_binding_count=*/6,
      /*.executable_count=*/2,
      /*.entry_count=*/4,
      /*.launch_graph=*/&launch_graph,
      /*.launch_count_binding=*/{5, kLaunchCountOffset},
      /*.launches=*/launch_plan.data(),
  };
  loom_op_t* low_function = nullptr;
  IREE_ASSERT_OK(loom_cmd_lower_program_to_low(module.get(), source_program,
                                               &plan, &low_function));
  loom_cmd_launch_graph_deinitialize(&launch_graph);

  ASSERT_NE(low_function, nullptr);
  EXPECT_TRUE(loom_low_func_def_isa(low_function));
  EXPECT_EQ(loom_low_func_def_abi(low_function),
            LOOM_TARGET_ABI_COMMAND_PROGRAM);
  EXPECT_EQ(FindSymbol(module.get(), IREE_SV("attention")), low_function);
  VerifyLowModule(module.get());

  std::array<iree_hal_executable_function_parameter_t, 3> prepare_parameters =
      {};
  for (uint32_t i = 0; i < prepare_parameters.size(); ++i) {
    prepare_parameters[i].type =
        IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING;
    prepare_parameters[i].offset = i;
  }
  std::array<iree_hal_executable_function_parameter_t, 2>
      projection_parameters = {};
  for (uint32_t i = 0; i < projection_parameters.size(); ++i) {
    projection_parameters[i].type =
        IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING;
    projection_parameters[i].offset = i;
  }
  std::array<iree_hal_resource_t, 2> executable_storage = {};
  const std::array<iree_hal_executable_t*, 2> executables = {
      reinterpret_cast<iree_hal_executable_t*>(&executable_storage[0]),
      reinterpret_cast<iree_hal_executable_t*>(&executable_storage[1]),
  };
  std::array<loom_cmd_iree_hal_entry_t, 4> entries = {};
  for (uint32_t i = 0; i < entries.size(); ++i) {
    entries[i].executable_index = i == 0 ? 0 : 1;
    entries[i].function = iree_hal_executable_function_from_value(100 + i);
    if (i == 0) {
      entries[i].info.binding_count = prepare_parameters.size();
      entries[i].info.parameter_count = prepare_parameters.size();
      entries[i].parameters = prepare_parameters.data();
    } else {
      entries[i].info.binding_count = projection_parameters.size();
      entries[i].info.parameter_count = projection_parameters.size();
      entries[i].parameters = projection_parameters.data();
    }
  }
  iree_hal_resource_t fixed_buffer_storage = {};
  const iree_hal_buffer_ref_t fixed_buffer = iree_hal_make_buffer_ref(
      reinterpret_cast<iree_hal_buffer_t*>(&fixed_buffer_storage), 64,
      kBufferLength);
  const loom_cmd_iree_hal_inputs_t inputs = {
      /*.binding_count=*/6,
      /*.fixed_buffer_count=*/1,
      /*.fixed_buffers=*/&fixed_buffer,
      /*.executable_count=*/executables.size(),
      /*.executables=*/executables.data(),
      /*.entry_count=*/entries.size(),
      /*.entries=*/entries.data(),
  };

  CaptureCommandBuffer low_command_buffer = {};
  InitializeCommandBuffer(inputs.binding_count, &low_command_buffer);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(&low_command_buffer.base));
  IREE_ASSERT_OK(loom_cmd_iree_hal_record_function(
      module.get(), low_function, &inputs, &low_command_buffer.base,
      iree_allocator_system()));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(&low_command_buffer.base));

  EXPECT_EQ(low_command_buffer.begin_count, 1u);
  EXPECT_EQ(low_command_buffer.end_count, 1u);
  ASSERT_EQ(low_command_buffer.commands.size(), 5u);
  EXPECT_EQ(low_command_buffer.commands[1].kind, CapturedCommandKind::kBarrier);
  const std::array<iree_host_size_t, 4> command_indices = {0, 2, 3, 4};
  for (iree_host_size_t i = 0; i < command_indices.size(); ++i) {
    const CapturedCommand& dispatch =
        low_command_buffer.commands[command_indices[i]];
    ASSERT_EQ(dispatch.kind, CapturedCommandKind::kDispatch);
    EXPECT_EQ(dispatch.executable, executables[i == 0 ? 0 : 1]);
    EXPECT_EQ(dispatch.function.value, 100u + i);
    if (i == 0) {
      EXPECT_EQ(dispatch.dispatch_flags, IREE_HAL_DISPATCH_FLAG_NONE);
      EXPECT_EQ(dispatch.dispatch_config.workgroup_count[0], 16u);
      EXPECT_EQ(dispatch.dispatch_config.workgroup_count[1], 1u);
      EXPECT_EQ(dispatch.dispatch_config.workgroup_count[2], 1u);
    } else {
      EXPECT_EQ(dispatch.dispatch_flags,
                IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS);
      EXPECT_EQ(dispatch.dispatch_config.workgroup_count_ref.buffer, nullptr);
      EXPECT_EQ(dispatch.dispatch_config.workgroup_count_ref.buffer_slot, 5u);
      EXPECT_EQ(dispatch.dispatch_config.workgroup_count_ref.offset,
                kLaunchCountOffset);
      EXPECT_EQ(dispatch.dispatch_config.workgroup_count_ref.length,
                LOOM_CMD_LAUNCH_COUNT_TUPLE_BYTE_LENGTH);
    }
    ASSERT_EQ(dispatch.bindings.size(), i == 0 ? 3u : 2u);
    EXPECT_EQ(dispatch.bindings[0].buffer, fixed_buffer.buffer);
    EXPECT_EQ(dispatch.bindings[0].offset,
              fixed_buffer.offset + (i == 0 ? 256u : 0u));
    EXPECT_EQ(dispatch.bindings[0].length, i == 0 ? 512u : kBufferLength);
    EXPECT_EQ(dispatch.bindings[1].buffer, nullptr);
    EXPECT_EQ(dispatch.bindings[1].buffer_slot, i == 0 ? 0u : 1u + i);
    EXPECT_EQ(dispatch.bindings[1].offset, 0u);
    EXPECT_EQ(dispatch.bindings[1].length, kBufferLength);
    if (i == 0) {
      EXPECT_EQ(dispatch.bindings[2].buffer, nullptr);
      EXPECT_EQ(dispatch.bindings[2].buffer_slot, 1u);
      EXPECT_EQ(dispatch.bindings[2].offset, 0u);
      EXPECT_EQ(dispatch.bindings[2].length, kBufferLength);
    }
  }

  iree_byte_span_t program_data = iree_byte_span_empty();
  IREE_ASSERT_OK(loom_cmd_program_serialize_low(
      module.get(), low_function, /*parameter_requirements=*/nullptr,
      /*transient_requirement=*/nullptr, &program_data,
      iree_allocator_system()));
  loom_cmd_program_t program = {};
  IREE_ASSERT_OK(loom_cmd_program_parse(
      iree_make_const_byte_span(program_data.data, program_data.data_length),
      &program));
  EXPECT_EQ(program.requirements.fixed_buffer_count, 1u);
  EXPECT_EQ(program.requirements.rebindable_binding_count, 6u);
  EXPECT_EQ(program.requirements.executable_count, 2u);
  EXPECT_EQ(program.requirements.entry_count, 4u);
  ASSERT_EQ(program.buffer_refs.count, 8u);
  EXPECT_EQ(program.arguments.count, 9u);
  ASSERT_EQ(program.commands.count, 5u);
  const loom_cmd_program_buffer_ref_t launch_count_ref =
      loom_cmd_program_buffer_ref_at(&program, 7);
  EXPECT_EQ(launch_count_ref.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_REBINDABLE);
  EXPECT_EQ(launch_count_ref.root_index, 5u);
  EXPECT_EQ(launch_count_ref.byte_offset, kLaunchCountOffset);
  EXPECT_EQ(launch_count_ref.byte_length,
            LOOM_CMD_LAUNCH_COUNT_TUPLE_BYTE_LENGTH);
  EXPECT_EQ(loom_cmd_program_command_at(&program, 0).kind,
            LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_DIRECT);
  const loom_cmd_program_command_t prepare_command =
      loom_cmd_program_command_at(&program, 0);
  const loom_cmd_program_argument_t parameter_argument =
      loom_cmd_program_argument_at(&program, prepare_command.argument_offset);
  ASSERT_EQ(parameter_argument.kind, LOOM_CMD_PROGRAM_ARGUMENT_KIND_BUFFER_REF);
  EXPECT_EQ(parameter_argument.payload, 6u);
  const loom_cmd_program_buffer_ref_t parameter_ref =
      loom_cmd_program_buffer_ref_at(&program, parameter_argument.payload);
  EXPECT_EQ(parameter_ref.role, LOOM_CMD_PROGRAM_BUFFER_ROLE_FIXED);
  EXPECT_EQ(parameter_ref.root_index, 0u);
  EXPECT_EQ(parameter_ref.byte_offset, 256u);
  EXPECT_EQ(parameter_ref.byte_length, 512u);
  for (uint32_t i = 2; i < program.commands.count; ++i) {
    const loom_cmd_program_command_t command =
        loom_cmd_program_command_at(&program, i);
    EXPECT_EQ(command.kind,
              LOOM_CMD_PROGRAM_COMMAND_KIND_DISPATCH_INDIRECT_STATIC);
    EXPECT_EQ(command.payload.dispatch_indirect.workgroup_count_buffer_ref, 7u);
  }

  module.reset();
  const std::vector<uint8_t> launch_config_bytes(
      launch_config_data.data,
      launch_config_data.data + launch_config_data.data_length);
  iree_allocator_free(iree_allocator_system(), launch_config_data.data);
  ModulePtr launch_config_module = ReadAndVerifyModule(launch_config_bytes);
  ASSERT_NE(launch_config_module.get(), nullptr);
  loom_exact_function_t launch_config_function = {};
  IREE_ASSERT_OK(loom_exact_function_bind(
      launch_config_module.get(),
      FindSymbol(launch_config_module.get(), IREE_SV("attention")),
      &launch_config_function));
  loom_exact_function_context_t launch_config_context = {};
  loom_exact_function_context_initialize(launch_config_module.get(),
                                         &block_pool_, &launch_config_context);
  std::array<uint32_t, LOOM_CMD_LAUNCH_COUNT_DIMENSION_COUNT>
      launch_count_table = {};
  const int64_t first_arguments[] = {1};
  IREE_ASSERT_OK(loom_exact_function_evaluate_u32(
      &launch_config_context, &launch_config_function, first_arguments,
      IREE_ARRAYSIZE(first_arguments), launch_count_table.data(),
      launch_count_table.size()));
  EXPECT_EQ(launch_count_table, (std::array<uint32_t, 3>{2u, 1u, 1u}));

  CaptureCommandBuffer artifact_command_buffer = {};
  InitializeCommandBuffer(inputs.binding_count, &artifact_command_buffer);
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(&artifact_command_buffer.base));
  IREE_ASSERT_OK(loom_cmd_iree_hal_record_program(&program, &inputs,
                                                  &artifact_command_buffer.base,
                                                  iree_allocator_system()));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(&artifact_command_buffer.base));
  ExpectCapturedProgramsEqual(low_command_buffer, artifact_command_buffer);

  const int64_t second_arguments[] = {127};
  IREE_ASSERT_OK(loom_exact_function_evaluate_u32(
      &launch_config_context, &launch_config_function, second_arguments,
      IREE_ARRAYSIZE(second_arguments), launch_count_table.data(),
      launch_count_table.size()));
  EXPECT_EQ(launch_count_table, (std::array<uint32_t, 3>{128u, 1u, 1u}));
  EXPECT_EQ(artifact_command_buffer.begin_count, 1u);
  EXPECT_EQ(artifact_command_buffer.end_count, 1u);

  loom_exact_function_context_deinitialize(&launch_config_context);
  iree_allocator_free(iree_allocator_system(), program_data.data);
}

TEST_F(CmdLowerTest, RejectsUnsupportedKernelArgumentWithoutMutation) {
  ModulePtr module = ParseAndVerifySource(R"(
target.generic<reference> @command_target {abi = command_program, contract_set_key = "cmd.core"}

kernel.def @unsupported() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%value: index) {
  kernel.return
}

command.program.def @residual(%value: index) launch() {
  kernel.launch @unsupported[](%value) : [](index)
  command.return
}
)");

  loom_op_t* source_program = FindSymbol(module.get(), IREE_SV("residual"));
  const loom_func_like_t source_program_like =
      loom_func_like_cast(module.get(), source_program);
  iree_arena_allocator_t schedule_arena;
  iree_arena_initialize(&block_pool_, &schedule_arena);
  loom_cmd_schedule_plan_t schedule = {};
  IREE_ASSERT_OK(loom_cmd_schedule_plan_build(
      module.get(), loom_func_like_body(source_program_like), &schedule_arena,
      &schedule));
  loom_cmd_launch_graph_t launch_graph = {};
  IREE_ASSERT_OK(loom_cmd_launch_graph_materialize(
      module.get(), source_program, &schedule, &block_pool_,
      iree_allocator_system(), &launch_graph));
  iree_arena_deinitialize(&schedule_arena);

  const uint16_t source_argument_ordinal = 0;
  const loom_cmd_lower_launch_t launch = {
      /*.executable_index=*/0,
      /*.entry_index=*/0,
      /*.argument_count=*/1,
      /*.source_argument_ordinals=*/&source_argument_ordinal,
  };
  const loom_cmd_lower_plan_t plan = {
      /*.command_target=*/FindSymbolRef(module.get(),
                                        IREE_SV("command_target")),
      /*.bindings=*/nullptr,
      /*.binding_count=*/0,
      /*.buffer_ranges=*/nullptr,
      /*.buffer_range_count=*/0,
      /*.fixed_buffer_count=*/0,
      /*.rebindable_binding_count=*/0,
      /*.executable_count=*/1,
      /*.entry_count=*/1,
      /*.launch_graph=*/&launch_graph,
      /*.launch_count_binding=*/{},
      /*.launches=*/&launch,
  };
  loom_op_t* low_function = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      loom_cmd_lower_program_to_low(module.get(), source_program, &plan,
                                    &low_function));
  loom_cmd_launch_graph_deinitialize(&launch_graph);

  EXPECT_EQ(low_function, nullptr);
  EXPECT_EQ(FindSymbol(module.get(), IREE_SV("residual")), source_program);
  EXPECT_FALSE(iree_any_bit_set(source_program->flags, LOOM_OP_FLAG_DEAD));
}

}  // namespace
}  // namespace loom
