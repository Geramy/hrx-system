// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "program_environment.h"

#include <string.h>

#include "iree/base/internal/atomics.h"

struct loomc_program_environment_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release this environment.
  loomc_allocator_t allocator;

  // Owned table of borrowed static provider descriptors.
  const loomc_program_provider_t** providers;

  // Number of entries in |providers|.
  loomc_host_size_t provider_count;
};

static void loomc_program_environment_destroy(
    loomc_program_environment_t* program_environment) {
  loomc_allocator_t allocator = program_environment->allocator;
  loomc_allocator_free(allocator, program_environment->providers);
  loomc_allocator_free(allocator, program_environment);
}

loomc_status_t loomc_program_environment_create(
    const loomc_program_provider_t* const* providers,
    loomc_host_size_t provider_count, loomc_allocator_t allocator,
    loomc_program_environment_t** out_program_environment) {
  if (out_program_environment == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_program_environment must not be NULL");
  }
  *out_program_environment = NULL;
  IREE_ASSERT(provider_count == 0 || providers != NULL);
  if (provider_count == 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program environment requires a provider");
  }
  if (provider_count >
      LOOMC_HOST_SIZE_MAX / sizeof(loomc_program_provider_t*)) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "program provider table is too large");
  }

  loomc_program_environment_t* program_environment = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(
      allocator, sizeof(*program_environment), (void**)&program_environment));
  memset(program_environment, 0, sizeof(*program_environment));
  iree_atomic_ref_count_init(&program_environment->ref_count);
  program_environment->allocator = allocator;

  loomc_status_t status = loomc_allocator_malloc(
      allocator, provider_count * sizeof(*program_environment->providers),
      (void**)&program_environment->providers);
  if (loomc_status_is_ok(status)) {
    memcpy(program_environment->providers, providers,
           provider_count * sizeof(*program_environment->providers));
    program_environment->provider_count = provider_count;
    *out_program_environment = program_environment;
  } else {
    loomc_program_environment_destroy(program_environment);
  }
  return status;
}

void loomc_program_environment_retain(
    loomc_program_environment_t* program_environment) {
  if (program_environment == NULL) return;
  iree_atomic_ref_count_inc(&program_environment->ref_count);
}

void loomc_program_environment_release(
    loomc_program_environment_t* program_environment) {
  if (program_environment == NULL) return;
  if (iree_atomic_ref_count_dec(&program_environment->ref_count) == 1) {
    loomc_program_environment_destroy(program_environment);
  }
}

loomc_status_t loomc_program_environment_select_provider(
    const loomc_program_environment_t* program_environment,
    const loomc_module_t* module,
    const loomc_program_provider_t** out_provider) {
  *out_provider = NULL;
  for (loomc_host_size_t i = 0; i < program_environment->provider_count; ++i) {
    const loomc_program_provider_t* provider =
        program_environment->providers[i];
    if (!provider->matches(module)) continue;
    if (*out_provider != NULL) {
      return loomc_make_status(
          LOOMC_STATUS_FAILED_PRECONDITION,
          "selected program roots are owned by multiple providers");
    }
    *out_provider = provider;
  }
  if (*out_provider == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_NOT_FOUND,
        "no program provider owns the selected module roots");
  }
  return loomc_ok_status();
}
