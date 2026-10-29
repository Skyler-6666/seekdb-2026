#include "lib/vector/ob_cuvs_worker.h"
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#ifdef CUVS_WORKER_STANDALONE_TEST
#include <cassert>
#define TEST(suite, name) static void suite##_##name()
#define EXPECT_TRUE(value) assert(value)
#define EXPECT_EQ(left, right) assert((left) == (right))
#else
#include <gtest/gtest.h>
#endif

namespace {
using oceanbase::common::obvsag::CuvsWorker;

struct WorkerJob {
  pthread_t thread{};
  int *counter = nullptr;
  int result = 0;
};

void *record_job(void *argument)
{
  WorkerJob *job = static_cast<WorkerJob *>(argument);
  job->thread = pthread_self();
  job->result = ++*job->counter;
  return nullptr;
}

TEST(CuvsWorker, ReusesThreadAndCompletesBeforeReturn)
{
  CuvsWorker worker;
  int counter = 0;
  WorkerJob job;
  job.counter = &counter;
  EXPECT_TRUE(!worker.run(nullptr, &job));
  EXPECT_TRUE(worker.run(record_job, &job));
  const pthread_t first_thread = job.thread;
  EXPECT_TRUE(!pthread_equal(first_thread, pthread_self()));
  for (int i = 0; i < 4096; ++i) {
    EXPECT_TRUE(worker.run(record_job, &job));
    EXPECT_TRUE(pthread_equal(first_thread, job.thread));
    EXPECT_EQ(i + 2, job.result);
  }
  EXPECT_EQ(4097, counter);
}

TEST(CuvsWorker, SerializesConcurrentCallers)
{
  CuvsWorker worker;
  int counter = 0;
  std::vector<std::thread> callers;
  for (int i = 0; i < 8; ++i) {
    callers.emplace_back([&] {
      for (int j = 0; j < 1000; ++j) {
        WorkerJob job;
        job.counter = &counter;
        EXPECT_TRUE(worker.run(record_job, &job));
        EXPECT_TRUE(job.result > 0);
      }
    });
  }
  for (auto &caller : callers) { caller.join(); }
  EXPECT_EQ(8000, counter);
}

#ifdef CUVS_WORKER_STANDALONE_TEST
void benchmark_dispatch()
{
  constexpr int count = 4096;
  int counter = 0;
  WorkerJob job;
  job.counter = &counter;
  pthread_attr_t attr;
  assert(pthread_attr_init(&attr) == 0);
  assert(pthread_attr_setstacksize(&attr, 32UL * 1024 * 1024) == 0);
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < count; ++i) {
    pthread_t thread;
    assert(pthread_create(&thread, &attr, record_job, &job) == 0);
    assert(pthread_join(thread, nullptr) == 0);
  }
  const auto fresh_end = std::chrono::steady_clock::now();
  pthread_attr_destroy(&attr);
  CuvsWorker worker;
  const auto reuse_start = std::chrono::steady_clock::now();
  for (int i = 0; i < count; ++i) { assert(worker.run(record_job, &job)); }
  const auto reuse_end = std::chrono::steady_clock::now();
  std::printf("CPU dispatch only, %d calls: create/join=%.3f ms, worker=%.3f ms\n", count,
      std::chrono::duration<double, std::milli>(fresh_end - start).count(),
      std::chrono::duration<double, std::milli>(reuse_end - reuse_start).count());
  assert(counter == 2 * count);
}
#endif
} // namespace

#ifdef CUVS_WORKER_STANDALONE_TEST
int main()
{
  CuvsWorker_ReusesThreadAndCompletesBeforeReturn();
  CuvsWorker_SerializesConcurrentCallers();
  benchmark_dispatch();
  std::puts("PASS: thread reuse, synchronous lifetime, concurrent callers, shutdown");
}
#endif
