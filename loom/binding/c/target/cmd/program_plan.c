// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/target/cmd/program_plan.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/binding/c/src/diagnostic.h"
#include "loom/binding/c/src/module.h"
#include "loom/binding/c/src/pass_program.h"
#include "loom/binding/c/src/program.h"
#include "loom/binding/c/src/program_plan.h"
#include "loom/binding/c/src/result.h"
#include "loom/binding/c/src/workspace.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/cmd/lower/launch_artifact.h"
#include "loom/target/arch/cmd/lower/program_plan.h"
#include "loom/target/arch/cmd/lower/serialize.h"
#include "loom/target/arch/cmd/program.h"
#include "loomc/iree.h"
#include "loomc/launch_config_module.h"

typedef struct loomc_cmd_program_plan_storage_t {
  // Compiler retained for independent dependency-unit compilation.
  loomc_compiler_t* compiler;

  // Per-unit pass program retained for independent compilation.
  loomc_pass_program_t* unit_pass_program;

  // Compiler-owned root metadata and dependency maps.
  loom_cmd_program_plan_t plan;

  // Public owner of |plan.roots[*].function_op| storage.
  loomc_module_t* root_module;

  // Public owner of |plan.roots[*].launch_function_op| storage.
  loomc_module_t* launch_module;

  // Public immutable dependency modules in plan-unit order after the root.
  loomc_module_t** dependency_modules;
} loomc_cmd_program_plan_storage_t;

static iree_string_view_t loomc_cmd_program_plan_symbol_name(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_symbol_ref_t symbol_ref =
      loom_func_like_callee(loom_func_like_cast(module, (loom_op_t*)op));
  IREE_ASSERT(loom_symbol_ref_is_valid(symbol_ref));
  IREE_ASSERT_EQ(symbol_ref.module_id, 0u);
  IREE_ASSERT_LT(symbol_ref.symbol_id, module->symbols.count);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  IREE_ASSERT_LT(symbol->name_id, module->strings.count);
  return module->strings.entries[symbol->name_id];
}

static loomc_string_view_t loomc_cmd_program_plan_root_name(
    const loomc_cmd_program_plan_storage_t* storage, uint32_t root_index) {
  IREE_ASSERT_LT(root_index, storage->plan.root_count);
  return loomc_string_view_from_iree(loomc_cmd_program_plan_symbol_name(
      loomc_module_const_loom_module(storage->root_module),
      storage->plan.roots[root_index].function_op));
}

static loomc_string_view_t loomc_cmd_program_plan_dependency_name(
    const loomc_cmd_program_plan_storage_t* storage,
    uint32_t dependency_index) {
  IREE_ASSERT_LT(dependency_index, storage->plan.dependency_count);
  return loomc_string_view_from_iree(loomc_cmd_program_plan_symbol_name(
      loomc_module_const_loom_module(
          storage->dependency_modules[dependency_index]),
      storage->plan.dependency_units[dependency_index].kernel_op));
}

static const loomc_artifact_t* loomc_cmd_program_find_artifact(
    const loomc_program_t* program, loomc_artifact_kind_t kind,
    loomc_string_view_t format, loomc_string_view_t identifier) {
  const loomc_artifact_t* found = NULL;
  const loomc_host_size_t artifact_count =
      loomc_program_artifact_count(program);
  for (loomc_host_size_t i = 0; i < artifact_count; ++i) {
    const loomc_artifact_t* artifact = loomc_program_artifact_at(program, i);
    if (artifact->kind != kind ||
        !loomc_string_view_equal(artifact->format, format) ||
        !loomc_string_view_equal(artifact->identifier, identifier)) {
      continue;
    }
    if (found != NULL) return NULL;
    found = artifact;
  }
  return found;
}

static loomc_status_t loomc_cmd_program_create(
    const loomc_string_view_t* export_names, loomc_host_size_t export_count,
    const loomc_program_dependency_t* dependencies,
    loomc_host_size_t dependency_count, loomc_result_t* result,
    loomc_allocator_t allocator, loomc_program_t** out_program) {
  const loomc_program_create_params_t params = {
      .export_names = export_names,
      .export_count = export_count,
      .artifact_storage = loomc_result_artifact_storage(result),
      .dependencies = dependencies,
      .dependency_count = dependency_count,
  };
  return loomc_program_create(&params, allocator, out_program);
}

static loomc_status_t loomc_cmd_program_finish_failure(
    loomc_result_t* result, loomc_status_t status,
    loomc_string_view_t diagnostic_code, loomc_result_t** out_result) {
  if (!loomc_status_is_result_diagnostic(status)) {
    loomc_result_release(result);
    return status;
  }
  loomc_status_t add_status = loomc_result_fail_status_diagnostic_consume(
      result, /*source=*/NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR, diagnostic_code,
      status);
  if (loomc_status_is_ok(add_status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return add_status;
}

static loomc_status_t loomc_cmd_program_compile_root_unit(
    const loomc_cmd_program_plan_storage_t* storage,
    loomc_workspace_t* workspace, loomc_allocator_t allocator,
    loomc_program_t** out_program, loomc_result_t** out_result) {
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  loomc_string_view_t* export_names = NULL;
  loomc_status_t status = loomc_status_from_iree(
      iree_arena_allocate_array(&scratch_arena, storage->plan.root_count,
                                sizeof(*export_names), (void**)&export_names));

  iree_byte_span_t artifact_data = iree_byte_span_empty();
  for (uint32_t i = 0;
       i < storage->plan.root_count && loomc_status_is_ok(status); ++i) {
    export_names[i] = loomc_cmd_program_plan_root_name(storage, i);
    status = loomc_status_from_iree(loom_cmd_program_serialize_low(
        loomc_module_loom_module(storage->root_module),
        storage->plan.roots[i].function_op, &artifact_data,
        iree_allocator_from_loomc(allocator)));
    if (loomc_status_is_ok(status)) {
      status = loomc_result_add_artifact_take_contents(
          result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM),
          export_names[i],
          loomc_byte_span_from_iree(iree_make_const_byte_span(
              artifact_data.data, artifact_data.data_length)));
      if (loomc_status_is_ok(status)) {
        artifact_data = iree_byte_span_empty();
      }
    }
  }

  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(loom_cmd_launch_program_serialize(
        loomc_module_const_loom_module(storage->launch_module),
        loomc_workspace_block_pool(workspace),
        iree_allocator_from_loomc(allocator), &artifact_data));
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_result_add_artifact_take_contents(
        result, LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
        loomc_make_cstring_view("command-program-launch.loombc"),
        loomc_byte_span_from_iree(iree_make_const_byte_span(
            artifact_data.data, artifact_data.data_length)));
    if (loomc_status_is_ok(status)) {
      artifact_data = iree_byte_span_empty();
    }
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_cmd_program_create(
        export_names, storage->plan.root_count, /*dependencies=*/NULL,
        /*dependency_count=*/0, result, allocator, out_program);
  }

  iree_allocator_free(iree_allocator_from_loomc(allocator), artifact_data.data);
  iree_arena_deinitialize(&scratch_arena);
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    return loomc_ok_status();
  }
  return loomc_cmd_program_finish_failure(
      result, status, loomc_make_cstring_view("PROGRAM_PLAN/ROOT_COMPILE"),
      out_result);
}

static loomc_status_t loomc_cmd_program_compile_dependency_unit(
    const loomc_cmd_program_plan_storage_t* storage,
    loomc_workspace_t* workspace, uint32_t dependency_index,
    loomc_allocator_t allocator, loomc_program_t** out_program,
    loomc_result_t** out_result) {
  loomc_module_t* module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_clone(storage->dependency_modules[dependency_index],
                         workspace, allocator, &module));

  const loomc_string_view_t identifier =
      loomc_cmd_program_plan_dependency_name(storage, dependency_index);
  const loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .module_name = identifier,
      .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE,
  };
  loomc_result_t* result = NULL;
  loomc_status_t status = loomc_compile_module(
      storage->compiler, workspace, storage->unit_pass_program, module,
      &compile_options, allocator, &result);
  loomc_module_release(module);
  if (!loomc_status_is_ok(status)) return status;
  if (!loomc_result_succeeded(result)) {
    *out_result = result;
    return loomc_ok_status();
  }

  status = loomc_cmd_program_create(
      &identifier, /*export_count=*/1, /*dependencies=*/NULL,
      /*dependency_count=*/0, result, allocator, out_program);
  if (loomc_status_is_ok(status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return status;
}

static loomc_status_t loomc_cmd_program_compile_unit(
    const void* storage_ptr, loomc_workspace_t* workspace, uint32_t unit_index,
    const loomc_program_plan_unit_compile_options_t* options,
    loomc_allocator_t allocator, loomc_program_t** out_program,
    loomc_result_t** out_result) {
  const loomc_cmd_program_plan_storage_t* storage =
      (const loomc_cmd_program_plan_storage_t*)storage_ptr;
  IREE_ASSERT_LT(unit_index, storage->plan.dependency_count + 1);
  if (options != NULL && options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "command program unit compile extensions are not supported");
  }
  if (unit_index == 0) {
    return loomc_cmd_program_compile_root_unit(storage, workspace, allocator,
                                               out_program, out_result);
  }
  return loomc_cmd_program_compile_dependency_unit(
      storage, workspace, unit_index - 1, allocator, out_program, out_result);
}

static loomc_status_t loomc_cmd_program_validate_root_artifacts(
    const loomc_cmd_program_plan_storage_t* storage,
    const loomc_artifact_t* artifacts, loomc_host_size_t artifact_count,
    loomc_allocator_t allocator) {
  if (artifact_count != storage->plan.root_count + 1) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command root unit artifact table does not match the plan roots");
  }
  for (uint32_t i = 0; i < storage->plan.root_count; ++i) {
    const loomc_string_view_t root_name =
        loomc_cmd_program_plan_root_name(storage, i);
    const loomc_artifact_t* artifact = NULL;
    for (loomc_host_size_t j = 0; j < artifact_count; ++j) {
      if (artifacts[j].kind == LOOMC_ARTIFACT_KIND_EXECUTABLE &&
          loomc_string_view_equal(
              artifacts[j].format,
              loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM)) &&
          loomc_string_view_equal(artifacts[j].identifier, root_name)) {
        if (artifact != NULL) {
          return loomc_make_status(
              LOOMC_STATUS_INVALID_ARGUMENT,
              "command root unit contains duplicate root artifacts");
        }
        artifact = &artifacts[j];
      }
    }
    if (artifact == NULL) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "command root unit is missing a root artifact");
    }
    loom_cmd_program_t program = {0};
    loomc_status_t status = loomc_status_from_iree(loom_cmd_program_parse(
        iree_const_byte_span_from_loomc(artifact->contents), &program));
    if (!loomc_status_is_ok(status)) return status;
    if (program.requirements.executable_count !=
            storage->plan.roots[i].dependency_count ||
        program.requirements.entry_count !=
            storage->plan.roots[i].dependency_count) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "command root dependency requirements do not match the plan");
    }
  }

  const loomc_artifact_t* launch_artifact = NULL;
  for (loomc_host_size_t i = 0; i < artifact_count; ++i) {
    if (artifacts[i].kind == LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG &&
        loomc_string_view_equal(
            artifacts[i].format,
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE)) &&
        loomc_string_view_equal(
            artifacts[i].identifier,
            loomc_make_cstring_view("command-program-launch.loombc"))) {
      if (launch_artifact != NULL) {
        return loomc_make_status(
            LOOMC_STATUS_INVALID_ARGUMENT,
            "command root unit contains duplicate launch artifacts");
      }
      launch_artifact = &artifacts[i];
    }
  }
  if (launch_artifact == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command root unit is missing its launch configuration artifact");
  }
  loomc_launch_config_module_t* launch_module = NULL;
  loomc_status_t status = loomc_launch_config_module_load(
      launch_artifact, /*options=*/NULL, allocator, &launch_module);
  if (loomc_status_is_ok(status) &&
      loomc_launch_config_module_function_count(launch_module) !=
          storage->plan.root_count) {
    status = loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command launch artifact export table does not match the plan roots");
  }
  for (uint32_t i = 0;
       i < storage->plan.root_count && loomc_status_is_ok(status); ++i) {
    loomc_launch_config_function_t function =
        loomc_launch_config_function_invalid();
    status = loomc_launch_config_module_lookup_function_by_name(
        launch_module, loomc_cmd_program_plan_root_name(storage, i), &function);
  }
  loomc_launch_config_module_release(launch_module);
  return status;
}

static loomc_status_t loomc_cmd_program_validate_dependency_artifact(
    const loomc_cmd_program_plan_storage_t* storage, uint32_t dependency_index,
    const loomc_artifact_t* artifact, loomc_allocator_t allocator) {
  loomc_workspace_t* workspace = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_workspace_create(/*options=*/NULL, allocator, &workspace));
  loomc_source_t* source = NULL;
  loomc_status_t status = loomc_artifact_create_source(
      artifact, LOOMC_SOURCE_FORMAT_BYTECODE, allocator, &source);
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  if (loomc_status_is_ok(status)) {
    status = loomc_module_deserialize_from_source(
        loomc_module_context(storage->root_module), workspace, source,
        /*options=*/NULL, allocator, &module, &result);
  }
  if (loomc_status_is_ok(status) && !loomc_result_succeeded(result)) {
    status = loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command dependency unit contains invalid Loom bytecode");
  }
  if (loomc_status_is_ok(status)) {
    const loom_module_t* internal_module =
        loomc_module_const_loom_module(module);
    const iree_string_view_t expected_name = iree_string_view_from_loomc(
        loomc_cmd_program_plan_dependency_name(storage, dependency_index));
    const loom_string_id_t name_id =
        loom_module_lookup_string(internal_module, expected_name);
    const loom_symbol_id_t symbol_id =
        name_id == LOOM_STRING_ID_INVALID
            ? LOOM_SYMBOL_ID_INVALID
            : loom_module_find_symbol(internal_module, name_id);
    if (symbol_id == LOOM_SYMBOL_ID_INVALID ||
        internal_module->symbols.entries[symbol_id].defining_op == NULL) {
      status = loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "command dependency unit does not define its planned export");
    }
  }
  loomc_result_release(result);
  loomc_module_release(module);
  loomc_source_release(source);
  loomc_workspace_release(workspace);
  return status;
}

static loomc_status_t loomc_cmd_program_load_unit_program(
    const void* storage_ptr, uint32_t unit_index,
    const loomc_artifact_t* artifacts, loomc_host_size_t artifact_count,
    loomc_allocator_t allocator, loomc_program_t** out_program,
    loomc_result_t** out_result) {
  const loomc_cmd_program_plan_storage_t* storage =
      (const loomc_cmd_program_plan_storage_t*)storage_ptr;
  IREE_ASSERT_LT(unit_index, storage->plan.dependency_count + 1);

  loomc_status_t status = loomc_ok_status();
  if (unit_index == 0) {
    status = loomc_cmd_program_validate_root_artifacts(
        storage, artifacts, artifact_count, allocator);
  } else if (artifact_count != 1 ||
             artifacts[0].kind != LOOMC_ARTIFACT_KIND_MODULE ||
             !loomc_string_view_equal(
                 artifacts[0].format,
                 loomc_make_cstring_view(
                     LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE))) {
    status = loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command dependency unit requires one Loom bytecode module artifact");
  } else {
    status = loomc_cmd_program_validate_dependency_artifact(
        storage, unit_index - 1, &artifacts[0], allocator);
  }
  if (!loomc_status_is_ok(status)) return status;

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));
  for (loomc_host_size_t i = 0;
       i < artifact_count && loomc_status_is_ok(status); ++i) {
    status = loomc_result_add_artifact(result, &artifacts[i]);
  }

  loomc_string_view_t* export_names = NULL;
  const loomc_host_size_t export_count =
      unit_index == 0 ? storage->plan.root_count : 1;
  if (loomc_status_is_ok(status)) {
    status = loomc_allocator_malloc(
        allocator, export_count * sizeof(*export_names), (void**)&export_names);
  }
  if (loomc_status_is_ok(status)) {
    if (unit_index == 0) {
      for (uint32_t i = 0; i < storage->plan.root_count; ++i) {
        export_names[i] = loomc_cmd_program_plan_root_name(storage, i);
      }
    } else {
      export_names[0] =
          loomc_cmd_program_plan_dependency_name(storage, unit_index - 1);
    }
    status = loomc_cmd_program_create(
        export_names, export_count, /*dependencies=*/NULL,
        /*dependency_count=*/0, result, allocator, out_program);
  }
  loomc_allocator_free(allocator, export_names);

  if (loomc_status_is_ok(status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return status;
}

static loomc_status_t loomc_cmd_program_require_export(
    const loomc_program_t* program, loomc_string_view_t name) {
  if (program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "required program unit is missing");
  }
  loomc_program_export_t export_token = loomc_program_export_invalid();
  loomc_status_t status =
      loomc_program_lookup_export(program, name, &export_token);
  if (!loomc_status_is_ok(status)) {
    return loomc_status_from_iree(iree_status_annotate(
        iree_status_from_loomc(status),
        IREE_SV("program unit does not match its plan slot")));
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_cmd_program_assemble(
    const void* storage_ptr, loomc_workspace_t* workspace,
    const loomc_program_plan_root_t* roots, loomc_host_size_t root_count,
    const loomc_program_plan_unit_table_t* unit_table,
    const loomc_program_plan_assembly_options_t* options,
    loomc_allocator_t allocator, loomc_program_t** out_program,
    loomc_result_t** out_result) {
  const loomc_cmd_program_plan_storage_t* storage =
      (const loomc_cmd_program_plan_storage_t*)storage_ptr;
  if (options != NULL && options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "command program assembly extensions are not supported");
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  uint32_t* global_to_slot = NULL;
  loomc_string_view_t* export_names = NULL;
  loomc_program_dependency_t* dependencies = NULL;
  loomc_status_t status = loomc_ok_status();
  if (storage->plan.dependency_count != 0) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, storage->plan.dependency_count, sizeof(*global_to_slot),
        (void**)&global_to_slot));
  }
  if (loomc_status_is_ok(status)) {
    for (uint32_t i = 0; i < storage->plan.dependency_count; ++i) {
      global_to_slot[i] = UINT32_MAX;
    }
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, root_count, sizeof(*export_names),
        (void**)&export_names));
  }
  if (loomc_status_is_ok(status) && storage->plan.dependency_count != 0) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, storage->plan.dependency_count, sizeof(*dependencies),
        (void**)&dependencies));
  }

  uint32_t dependency_count = 0;
  for (loomc_host_size_t i = 0; i < root_count && loomc_status_is_ok(status);
       ++i) {
    const uint32_t root_index = (uint32_t)roots[i].value;
    const loom_cmd_program_root_t* root = &storage->plan.roots[root_index];
    export_names[i] = loomc_cmd_program_plan_root_name(storage, root_index);
    status = loomc_cmd_program_require_export(unit_table->programs[0],
                                              export_names[i]);
    for (uint32_t j = 0;
         j < root->dependency_count && loomc_status_is_ok(status); ++j) {
      const uint32_t global_index = root->dependency_unit_indices[j];
      if (global_to_slot[global_index] != UINT32_MAX) continue;
      const uint32_t unit_index = global_index + 1;
      status = loomc_cmd_program_require_export(
          unit_table->programs[unit_index],
          loomc_cmd_program_plan_dependency_name(storage, global_index));
      if (loomc_status_is_ok(status)) {
        global_to_slot[global_index] = dependency_count;
        dependencies[dependency_count] = (loomc_program_dependency_t){
            .slot = dependency_count,
            .program = unit_table->programs[unit_index],
        };
        ++dependency_count;
      }
    }
  }

  loomc_result_t* result = NULL;
  if (loomc_status_is_ok(status)) {
    status =
        loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result);
  }
  for (loomc_host_size_t i = 0; i < root_count && loomc_status_is_ok(status);
       ++i) {
    const uint32_t root_index = (uint32_t)roots[i].value;
    const loom_cmd_program_root_t* root = &storage->plan.roots[root_index];
    const loomc_artifact_t* source_artifact = loomc_cmd_program_find_artifact(
        unit_table->programs[0], LOOMC_ARTIFACT_KIND_EXECUTABLE,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM),
        export_names[i]);
    if (source_artifact == NULL) {
      status = loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "command root program is missing its unique root artifact");
      break;
    }

    loom_cmd_program_t source_program = {0};
    status = loomc_status_from_iree(loom_cmd_program_parse(
        iree_const_byte_span_from_loomc(source_artifact->contents),
        &source_program));
    uint32_t* local_to_assembled = NULL;
    if (loomc_status_is_ok(status) && root->dependency_count != 0) {
      status = loomc_status_from_iree(iree_arena_allocate_array(
          &scratch_arena, root->dependency_count, sizeof(*local_to_assembled),
          (void**)&local_to_assembled));
    }
    if (loomc_status_is_ok(status)) {
      for (uint32_t j = 0; j < root->dependency_count; ++j) {
        local_to_assembled[j] =
            global_to_slot[root->dependency_unit_indices[j]];
        IREE_ASSERT_NE(local_to_assembled[j], UINT32_MAX);
      }
      const loom_cmd_program_dependency_relocation_t relocation = {
          .executable_indices = local_to_assembled,
          .executable_count = dependency_count,
          .entry_indices = local_to_assembled,
          .entry_count = dependency_count,
      };
      iree_byte_span_t relocated_data = iree_byte_span_empty();
      status = loomc_status_from_iree(loom_cmd_program_relocate_dependencies(
          &source_program, &relocation, &relocated_data,
          iree_allocator_from_loomc(allocator)));
      if (loomc_status_is_ok(status)) {
        status = loomc_result_add_artifact_take_contents(
            result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMMAND_PROGRAM),
            export_names[i],
            loomc_byte_span_from_iree(iree_make_const_byte_span(
                relocated_data.data, relocated_data.data_length)));
        if (loomc_status_is_ok(status)) {
          relocated_data = iree_byte_span_empty();
        }
      }
      iree_allocator_free(iree_allocator_from_loomc(allocator),
                          relocated_data.data);
    }
  }

  if (loomc_status_is_ok(status)) {
    const loomc_artifact_t* launch_artifact = loomc_cmd_program_find_artifact(
        unit_table->programs[0], LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
        loomc_make_cstring_view("command-program-launch.loombc"));
    if (launch_artifact == NULL) {
      status = loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "command root program is missing its unique launch artifact");
    } else {
      status = loomc_result_add_artifact(result, launch_artifact);
    }
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_cmd_program_create(export_names, root_count, dependencies,
                                      dependency_count, result, allocator,
                                      out_program);
  }

  iree_arena_deinitialize(&scratch_arena);
  if (loomc_status_is_ok(status)) {
    *out_result = result;
  } else {
    loomc_result_release(result);
  }
  return status;
}

static void loomc_cmd_program_plan_destroy(void* storage_ptr,
                                           loomc_allocator_t allocator) {
  if (storage_ptr == NULL) return;
  loomc_cmd_program_plan_storage_t* storage =
      (loomc_cmd_program_plan_storage_t*)storage_ptr;
  if (storage->dependency_modules != NULL) {
    for (uint32_t i = 0; i < storage->plan.dependency_count; ++i) {
      loomc_module_release(storage->dependency_modules[i]);
    }
  }
  loomc_allocator_free(allocator, storage->dependency_modules);
  loomc_module_release(storage->launch_module);
  loomc_module_release(storage->root_module);
  loom_cmd_program_plan_deinitialize(&storage->plan);
  loomc_pass_program_release(storage->unit_pass_program);
  loomc_compiler_release(storage->compiler);
  loomc_allocator_free(allocator, storage);
}

static const loomc_program_plan_operations_t loomc_cmd_program_plan_operations =
    {
        .compile_unit = loomc_cmd_program_compile_unit,
        .load_unit_program = loomc_cmd_program_load_unit_program,
        .assemble = loomc_cmd_program_assemble,
        .destroy = loomc_cmd_program_plan_destroy,
};

static loomc_status_t loomc_cmd_program_plan_take_module(
    loomc_context_t* context, loomc_workspace_t* workspace,
    loom_module_t** internal_module, loomc_allocator_t allocator,
    loomc_module_t** out_module) {
  *out_module = NULL;
  loomc_module_t* module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_create_empty(context, workspace, allocator, &module));
  loomc_status_t status =
      loomc_module_set_loom_module(module, *internal_module);
  if (loomc_status_is_ok(status)) {
    *internal_module = NULL;
    *out_module = module;
  } else {
    loomc_module_release(module);
  }
  return status;
}

static loomc_status_t loomc_cmd_program_plan_create_storage(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* unit_pass_program,
    loomc_module_t* source_module, loom_cmd_program_plan_t* internal_plan,
    loomc_allocator_t allocator,
    loomc_cmd_program_plan_storage_t** out_storage) {
  *out_storage = NULL;
  loomc_cmd_program_plan_storage_t* storage = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  memset(storage, 0, sizeof(*storage));
  storage->plan = *internal_plan;
  memset(internal_plan, 0, sizeof(*internal_plan));

  loomc_context_t* context = loomc_module_context(source_module);
  loomc_status_t status = loomc_cmd_program_plan_take_module(
      context, workspace, &storage->plan.root_module, allocator,
      &storage->root_module);
  if (loomc_status_is_ok(status)) {
    status = loomc_cmd_program_plan_take_module(
        context, workspace, &storage->plan.launch_module, allocator,
        &storage->launch_module);
  }
  if (loomc_status_is_ok(status) && storage->plan.dependency_count != 0) {
    status = loomc_allocator_malloc(
        allocator,
        storage->plan.dependency_count * sizeof(*storage->dependency_modules),
        (void**)&storage->dependency_modules);
    if (loomc_status_is_ok(status)) {
      memset(storage->dependency_modules, 0,
             storage->plan.dependency_count *
                 sizeof(*storage->dependency_modules));
    }
  }
  for (uint32_t i = 0;
       i < storage->plan.dependency_count && loomc_status_is_ok(status); ++i) {
    status = loomc_cmd_program_plan_take_module(
        context, workspace, &storage->plan.dependency_units[i].module,
        allocator, &storage->dependency_modules[i]);
  }
  if (loomc_status_is_ok(status)) {
    storage->compiler = compiler;
    loomc_compiler_retain(compiler);
    storage->unit_pass_program = (loomc_pass_program_t*)unit_pass_program;
    loomc_pass_program_retain(storage->unit_pass_program);
    *out_storage = storage;
  } else {
    loomc_cmd_program_plan_destroy(storage, allocator);
  }
  return status;
}

static loomc_status_t loomc_cmd_program_plan_create_public_plan(
    loomc_cmd_program_plan_storage_t* storage, loomc_workspace_t* workspace,
    loomc_allocator_t allocator, loomc_program_plan_t** out_program_plan) {
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  loomc_program_plan_root_create_params_t* roots = NULL;
  loomc_program_plan_unit_create_params_t* units = NULL;
  loomc_program_plan_dependency_t* dependencies = NULL;
  loomc_status_t status = loomc_status_from_iree(
      iree_arena_allocate_array(&scratch_arena, storage->plan.root_count,
                                sizeof(*roots), (void**)&roots));
  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, storage->plan.dependency_count + 1, sizeof(*units),
        (void**)&units));
  }
  iree_host_size_t root_dependency_count = 0;
  for (uint32_t i = 0; i < storage->plan.root_count; ++i) {
    root_dependency_count += storage->plan.roots[i].dependency_count;
  }
  if (loomc_status_is_ok(status) && root_dependency_count != 0) {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, root_dependency_count, sizeof(*dependencies),
        (void**)&dependencies));
  }

  iree_host_size_t dependency_offset = 0;
  for (uint32_t i = 0;
       i < storage->plan.root_count && loomc_status_is_ok(status); ++i) {
    const loom_cmd_program_root_t* root = &storage->plan.roots[i];
    roots[i] = (loomc_program_plan_root_create_params_t){
        .name = loomc_cmd_program_plan_root_name(storage, i),
        .program_unit_index = 0,
        .dependencies = root->dependency_count != 0
                            ? dependencies + dependency_offset
                            : NULL,
        .dependency_count = root->dependency_count,
    };
    for (uint32_t j = 0; j < root->dependency_count; ++j) {
      dependencies[dependency_offset + j] = (loomc_program_plan_dependency_t){
          .slot = j,
          .unit_index = root->dependency_unit_indices[j] + 1,
      };
    }
    dependency_offset += root->dependency_count;
  }
  if (loomc_status_is_ok(status)) {
    units[0] = (loomc_program_plan_unit_create_params_t){
        .identifier = loomc_make_cstring_view("command-program-roots"),
        .cache_key = loomc_byte_span_empty(),
    };
    for (uint32_t i = 0; i < storage->plan.dependency_count; ++i) {
      units[i + 1] = (loomc_program_plan_unit_create_params_t){
          .identifier = loomc_cmd_program_plan_dependency_name(storage, i),
          .cache_key = loomc_byte_span_empty(),
      };
    }
    const loomc_program_plan_create_params_t params = {
        .roots = roots,
        .root_count = storage->plan.root_count,
        .units = units,
        .unit_count = storage->plan.dependency_count + 1,
        .operations = &loomc_cmd_program_plan_operations,
        .target_storage = storage,
    };
    status = loomc_program_plan_create(&params, allocator, out_program_plan);
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

loomc_status_t loomc_cmd_program_plan_prepare_module(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* unit_pass_program, loomc_module_t* module,
    loomc_allocator_t allocator, loomc_program_plan_t** out_program_plan,
    loomc_result_t** out_result) {
  if (out_program_plan == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "out_program_plan and out_result must not be NULL");
  }
  *out_program_plan = NULL;
  *out_result = NULL;
  if (compiler == NULL || workspace == NULL || unit_pass_program == NULL ||
      module == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "compiler, workspace, unit_pass_program, and module are required");
  }
  if (loomc_pass_program_context(unit_pass_program) !=
      loomc_module_context(module)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "unit pass program and source module contexts must match");
  }

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  const loom_module_t* internal_module = loomc_module_const_loom_module(module);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(loomc_workspace_block_pool(workspace), &scratch_arena);
  const loom_op_t** root_ops = NULL;
  iree_host_size_t root_count = 0;
  for (loom_symbol_id_t i = 0; i < internal_module->symbols.count; ++i) {
    const loom_op_t* op = internal_module->symbols.entries[i].defining_op;
    if (op != NULL && loom_command_program_def_isa(op) &&
        loom_func_like_visibility(
            loom_func_like_cast(internal_module, (loom_op_t*)op)) != 0) {
      ++root_count;
    }
  }
  loomc_status_t status = loomc_ok_status();
  if (root_count == 0) {
    status = loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                               "module has no public command program roots");
  } else {
    status = loomc_status_from_iree(iree_arena_allocate_array(
        &scratch_arena, root_count, sizeof(*root_ops), (void**)&root_ops));
  }
  if (loomc_status_is_ok(status)) {
    iree_host_size_t root_index = 0;
    for (loom_symbol_id_t i = 0; i < internal_module->symbols.count; ++i) {
      const loom_op_t* op = internal_module->symbols.entries[i].defining_op;
      if (op != NULL && loom_command_program_def_isa(op) &&
          loom_func_like_visibility(
              loom_func_like_cast(internal_module, (loom_op_t*)op)) != 0) {
        root_ops[root_index++] = op;
      }
    }
    IREE_ASSERT_EQ(root_index, root_count);
  }

  loom_cmd_program_plan_t internal_plan = {0};
  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(loom_cmd_program_plan_prepare(
        internal_module, root_ops, root_count,
        loomc_workspace_block_pool(workspace),
        iree_allocator_from_loomc(allocator), &internal_plan));
  }
  loomc_cmd_program_plan_storage_t* storage = NULL;
  if (loomc_status_is_ok(status)) {
    status = loomc_cmd_program_plan_create_storage(
        compiler, workspace, unit_pass_program, module, &internal_plan,
        allocator, &storage);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_cmd_program_plan_create_public_plan(
        storage, workspace, allocator, out_program_plan);
    if (loomc_status_is_ok(status)) storage = NULL;
  }

  loomc_cmd_program_plan_destroy(storage, allocator);
  loom_cmd_program_plan_deinitialize(&internal_plan);
  iree_arena_deinitialize(&scratch_arena);
  if (loomc_status_is_ok(status)) {
    *out_result = result;
    return loomc_ok_status();
  }
  return loomc_cmd_program_finish_failure(
      result, status, loomc_make_cstring_view("PROGRAM_PLAN/PREPARE"),
      out_result);
}
