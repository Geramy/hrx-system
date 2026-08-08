// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_COMPILE_STORAGE_H_
#define LOOMC_COMPILE_STORAGE_H_

#include "loomc/compile.h"
#include "loomc/program_plan.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the program environment retained by |compiler|, or NULL.
LOOMC_API_PRIVATE loomc_program_environment_t*
loomc_compiler_program_environment(const loomc_compiler_t* compiler);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_COMPILE_STORAGE_H_
