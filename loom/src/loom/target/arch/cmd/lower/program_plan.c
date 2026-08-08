// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_plan.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/link/linker.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_registry.h"
#include "loom/target/arch/cmd/lower/lower.h"
#include "loom/target/arch/cmd/lower/schedule.h"
#include "loom/util/fact_table.h"

static iree_string_view_t loom_cmd_program_plan_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref));
  IREE_ASSERT_EQ(symbol_ref.module_id, 0u);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  IREE_ASSERT_LT(symbol->name_id, module->strings.count);
  return module->strings.entries[symbol->name_id];
}

static loom_op_t* loom_cmd_program_plan_find_symbol(
    loom_module_t* module, iree_string_view_t symbol_name) {
  const loom_string_id_t name_id =
      loom_module_lookup_string(module, symbol_name);
  IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
  const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
  IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
  loom_op_t* defining_op = module->symbols.entries[symbol_id].defining_op;
  IREE_ASSERT(defining_op != NULL);
  return defining_op;
}

static iree_status_t loom_cmd_program_plan_allocate_dependencies(
    iree_host_size_t dependency_count, loom_cmd_program_plan_t* plan) {
  if (dependency_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      plan->host_allocator, dependency_count * sizeof(*plan->dependency_units),
      (void**)&plan->dependency_units));
  memset(plan->dependency_units, 0,
         dependency_count * sizeof(*plan->dependency_units));
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_plan_build_lower_plan(
    loom_cmd_program_plan_t* plan, loom_module_t* preparation_module,
    loom_op_t* root_program_op, const loom_cmd_schedule_plan_t* schedule,
    const loom_value_fact_table_t* source_facts,
    const loom_cmd_launch_graph_t* launch_graph,
    iree_arena_allocator_t* scratch_arena, iree_arena_block_pool_t* block_pool,
    loom_cmd_lower_plan_t* out_lower_plan) {
  if (schedule->command_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command root has too many dependency units");
  }

  IREE_RETURN_IF_ERROR(loom_cmd_program_plan_allocate_dependencies(
      schedule->command_count, plan));

  loom_cmd_lower_launch_t* launches = NULL;
  const loom_op_t** dependency_launches = NULL;
  if (schedule->command_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, schedule->command_count,
                                  sizeof(*launches), (void**)&launches));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, schedule->command_count, sizeof(*dependency_launches),
        (void**)&dependency_launches));
  }
  for (iree_host_size_t i = 0; i < schedule->command_count; ++i) {
    const loom_op_t* source_launch = schedule->commands[i];
    uint32_t dependency_index = 0;
    while (dependency_index < plan->dependency_count &&
           !loom_cmd_kernel_unit_launches_equivalent(
               preparation_module, source_launch,
               dependency_launches[dependency_index], source_facts)) {
      ++dependency_index;
    }
    if (dependency_index == plan->dependency_count) {
      loom_cmd_kernel_unit_t* new_unit =
          &plan->dependency_units[dependency_index];
      IREE_RETURN_IF_ERROR(loom_cmd_kernel_unit_materialize(
          preparation_module, source_launch, source_facts, block_pool,
          plan->host_allocator, new_unit));
      dependency_launches[dependency_index] = source_launch;
      ++plan->dependency_count;
    }
    const loom_cmd_kernel_unit_t* unit =
        &plan->dependency_units[dependency_index];
    launches[i] = (loom_cmd_lower_launch_t){
        .executable_index = dependency_index,
        .entry_index = dependency_index,
        .argument_count = unit->argument_count,
        .source_argument_ordinals = unit->source_argument_ordinals,
    };
  }

  const loom_func_like_t root_program =
      loom_func_like_cast(preparation_module, root_program_op);
  uint16_t argument_count = 0;
  loom_func_like_arg_ids(root_program, &argument_count);
  const int64_t specialization_count_i64 =
      loom_func_like_specialization_count(root_program);
  IREE_ASSERT_GE(specialization_count_i64, 0);
  IREE_ASSERT_LE(specialization_count_i64, argument_count);
  const uint16_t binding_count =
      argument_count - (uint16_t)specialization_count_i64;

  loom_cmd_lower_binding_t* bindings = NULL;
  if (binding_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, binding_count, sizeof(*bindings), (void**)&bindings));
  }
  for (uint16_t i = 0; i < binding_count; ++i) {
    bindings[i] = (loom_cmd_lower_binding_t){
        .role = LOOM_CMD_LOWER_BUFFER_ROLE_REBINDABLE,
        .resource_index = i,
        .byte_offset = 0,
        .byte_length = UINT64_MAX,
    };
  }

  const bool has_host_launch_counts = launch_graph->host_tuple_count > 0;
  *out_lower_plan = (loom_cmd_lower_plan_t){
      .command_target = loom_command_program_def_target(root_program_op),
      .bindings = bindings,
      .binding_count = binding_count,
      .fixed_buffer_count = 0,
      .rebindable_binding_count =
          (uint32_t)binding_count + (has_host_launch_counts ? 1u : 0u),
      .executable_count = (uint32_t)plan->dependency_count,
      .entry_count = (uint32_t)plan->dependency_count,
      .launch_graph = launch_graph,
      .launch_count_binding =
          {
              .resource_index = binding_count,
              .byte_offset = 0,
          },
      .launches = launches,
  };
  return iree_ok_status();
}

iree_status_t loom_cmd_program_plan_prepare(const loom_module_t* source_module,
                                            const loom_op_t* source_program_op,
                                            iree_arena_block_pool_t* block_pool,
                                            iree_allocator_t host_allocator,
                                            loom_cmd_program_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(source_module);
  IREE_ASSERT_ARGUMENT(source_program_op);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_plan);
  IREE_ASSERT(loom_command_program_def_isa(source_program_op));
  memset(out_plan, 0, sizeof(*out_plan));

  loom_cmd_program_plan_t plan = {
      .host_allocator = host_allocator,
  };
  const loom_symbol_ref_t source_program_ref =
      loom_command_program_def_callee(source_program_op);
  const iree_string_view_t root_name =
      loom_cmd_program_plan_symbol_name(source_module, source_program_ref);
  const iree_string_view_t root_names[] = {root_name};
  const loom_module_t* source_modules[] = {source_module};
  loom_module_t* preparation_module = NULL;
  iree_status_t status = loom_link_materialized_modules(
      source_modules, IREE_ARRAYSIZE(source_modules),
      &(loom_link_options_t){
          .module_name = IREE_SV("command_program_root"),
          .root_symbols =
              {
                  .count = IREE_ARRAYSIZE(root_names),
                  .values = root_names,
              },
      },
      block_pool, host_allocator, &preparation_module);

  loom_op_t* root_program_op = NULL;
  loom_func_like_t root_program = {0};
  iree_arena_allocator_t scratch_arena;
  bool scratch_arena_initialized = false;
  loom_cmd_schedule_plan_t schedule = {0};
  loom_value_fact_table_t source_facts = {0};
  loom_cmd_launch_graph_t launch_graph = {0};
  if (iree_status_is_ok(status)) {
    root_program_op =
        loom_cmd_program_plan_find_symbol(preparation_module, root_name);
    root_program = loom_func_like_cast(preparation_module, root_program_op);
    IREE_ASSERT(loom_func_like_isa(root_program));
    if (!loom_symbol_ref_is_valid(
            loom_command_program_def_target(root_program_op))) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "command root must have a selected target before preparation");
    }
  }
  if (iree_status_is_ok(status)) {
    iree_arena_initialize(block_pool, &scratch_arena);
    scratch_arena_initialized = true;
    status = loom_cmd_schedule_plan_build(preparation_module,
                                          loom_func_like_body(root_program),
                                          &scratch_arena, &schedule);
  }
  if (iree_status_is_ok(status)) {
    status = loom_value_fact_table_initialize(&source_facts, &scratch_arena,
                                              preparation_module->values.count);
  }
  if (iree_status_is_ok(status)) {
    loom_type_registry_configure_fact_context(&source_facts.context);
    status = loom_value_fact_table_compute(&source_facts, preparation_module,
                                           root_program);
  }
  if (iree_status_is_ok(status)) {
    status = loom_cmd_launch_graph_materialize(
        preparation_module, root_program_op, &schedule, block_pool,
        host_allocator, &launch_graph);
  }

  loom_cmd_lower_plan_t lower_plan = {0};
  if (iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_build_lower_plan(
        &plan, preparation_module, root_program_op, &schedule, &source_facts,
        &launch_graph, &scratch_arena, block_pool, &lower_plan);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* preparation_root_function = NULL;
    status =
        loom_cmd_lower_program_to_low(preparation_module, root_program_op,
                                      &lower_plan, &preparation_root_function);
  }
  if (iree_status_is_ok(status)) {
    const loom_module_t* root_source_modules[] = {preparation_module};
    status = loom_link_materialized_modules(
        root_source_modules, IREE_ARRAYSIZE(root_source_modules),
        &(loom_link_options_t){
            .module_name = IREE_SV("command_program_root"),
            .root_symbols =
                {
                    .count = IREE_ARRAYSIZE(root_names),
                    .values = root_names,
                },
        },
        block_pool, host_allocator, &plan.root_module);
  }
  if (iree_status_is_ok(status)) {
    plan.root_function_op =
        loom_cmd_program_plan_find_symbol(plan.root_module, root_name);
    plan.launch_module = launch_graph.module;
    plan.launch_function_op = launch_graph.host_function_op;
    plan.launch_tuple_count = launch_graph.host_tuple_count;
    launch_graph.module = NULL;
  }

  if (scratch_arena_initialized) {
    iree_arena_deinitialize(&scratch_arena);
  }
  loom_cmd_launch_graph_deinitialize(&launch_graph);
  if (preparation_module) loom_module_free(preparation_module);
  if (!iree_status_is_ok(status)) {
    loom_cmd_program_plan_deinitialize(&plan);
    return status;
  }

  *out_plan = plan;
  return iree_ok_status();
}

void loom_cmd_program_plan_deinitialize(loom_cmd_program_plan_t* plan) {
  if (!plan) return;
  for (iree_host_size_t i = 0; i < plan->dependency_count; ++i) {
    loom_cmd_kernel_unit_deinitialize(&plan->dependency_units[i]);
  }
  iree_allocator_free(plan->host_allocator, plan->dependency_units);
  if (plan->launch_module) loom_module_free(plan->launch_module);
  if (plan->root_module) loom_module_free(plan->root_module);
  memset(plan, 0, sizeof(*plan));
}
