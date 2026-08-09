// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"

#include <array>

#include "iree/testing/gtest.h"

namespace iree::hal::amdgpu {
namespace {

TEST(HostQueueProfileEventsTest, SnapshotsFailedDispatchProgress) {
  std::array<iree_amd_signal_t, 4> completion_signals = {};
  completion_signals[0].value = 0;
  completion_signals[0].start_ts = 100;
  completion_signals[0].end_ts = 200;
  completion_signals[1].value = 1;
  completion_signals[1].start_ts = 300;
  completion_signals[2].value = 1;

  std::array<iree_hal_amdgpu_profile_dispatch_event_t, 4> events = {};
  for (uint32_t i = 0; i < 3; ++i) {
    events[i].flags =
        IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
    events[i].command_index = i * 2;
  }

  iree_hal_amdgpu_host_queue_t queue = {};
  iree_slim_mutex_initialize(&queue.profiling.event_mutex);
  queue.profiling.completion_signals = completion_signals.data();
  queue.profiling.dispatch_events.values = events.data();
  queue.profiling.dispatch_events.capacity = events.size();
  queue.profiling.dispatch_events.mask = events.size() - 1;

  const iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation = {
      /*.first_event_position=*/0,
      /*.event_count=*/3,
      /*.reserved0=*/0,
  };
  iree_hal_amdgpu_host_queue_snapshot_failed_profile_dispatch_events(
      &queue, reservation);

  const auto progress_valid =
      IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_PROGRESS_VALID;
  const auto started =
      IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_EXECUTION_STARTED;
  const auto completed =
      IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_EXECUTION_COMPLETED;
  EXPECT_EQ(events[0].flags,
            IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER |
                progress_valid | started | completed);
  EXPECT_EQ(events[0].start_tick, 100u);
  EXPECT_EQ(events[0].end_tick, 200u);
  EXPECT_EQ(events[1].flags,
            IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER |
                progress_valid | started);
  EXPECT_EQ(events[1].start_tick, 300u);
  EXPECT_EQ(events[1].end_tick, 0u);
  EXPECT_EQ(events[2].flags,
            IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER |
                progress_valid);
  EXPECT_EQ(events[2].start_tick, 0u);
  EXPECT_EQ(events[2].end_tick, 0u);
  EXPECT_EQ(queue.profiling.dispatch_events.ready_position, 3u);

  iree_slim_mutex_deinitialize(&queue.profiling.event_mutex);
}

TEST(HostQueueProfileEventsTest, PreservesHarvestedCompletion) {
  iree_amd_signal_t completion_signal = {};
  completion_signal.value = 1;
  iree_hal_amdgpu_profile_dispatch_event_t event = {};
  event.start_tick = 400;
  event.end_tick = 500;

  iree_hal_amdgpu_profile_dispatch_event_snapshot_progress(&completion_signal,
                                                           &event);

  EXPECT_TRUE(iree_all_bits_set(
      event.flags,
      IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_PROGRESS_VALID |
          IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_EXECUTION_STARTED |
          IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_EXECUTION_COMPLETED));
  EXPECT_EQ(event.start_tick, 400u);
  EXPECT_EQ(event.end_tick, 500u);
}

}  // namespace
}  // namespace iree::hal::amdgpu
