#pragma once

#include <condition_variable>
#include <mutex>
#include <pthread.h>

namespace oceanbase {
namespace common {
namespace obvsag {

// Synchronous dispatch keeps caller-owned jobs alive and reuses one HIP thread.
// cuVS needs more stack than an observer worker provides.
class CuvsWorker {
public:
  CuvsWorker()
  {
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) == 0) {
      if (pthread_attr_setstacksize(&attr, 32UL * 1024 * 1024) == 0) {
        started_ = pthread_create(&thread_, &attr, entry, this) == 0;
      }
      pthread_attr_destroy(&attr);
    }
  }

  ~CuvsWorker()
  {
    if (started_) {
      std::lock_guard<std::mutex> submit(submit_mu_);
      {
        std::lock_guard<std::mutex> lock(mu_);
        stopping_ = true;
      }
      ready_.notify_one();
      pthread_join(thread_, nullptr);
    }
  }

  bool run(void *(*function)(void *), void *argument)
  {
    if (!started_ || function == nullptr) { return false; }
    std::lock_guard<std::mutex> submit(submit_mu_);
    std::unique_lock<std::mutex> lock(mu_);
    function_ = function;
    argument_ = argument;
    ready_.notify_one();
    done_.wait(lock, [this] { return function_ == nullptr; });
    return true;
  }

  CuvsWorker(const CuvsWorker &) = delete;
  CuvsWorker &operator=(const CuvsWorker &) = delete;

private:
  static void *entry(void *argument)
  {
    CuvsWorker *worker = static_cast<CuvsWorker *>(argument);
    std::unique_lock<std::mutex> lock(worker->mu_);
    for (;;) {
      worker->ready_.wait(lock, [worker] { return worker->stopping_ || worker->function_ != nullptr; });
      if (worker->stopping_) { return nullptr; }
      auto function = worker->function_;
      void *job = worker->argument_;
      lock.unlock();
      function(job);
      lock.lock();
      worker->argument_ = nullptr;
      worker->function_ = nullptr;
      worker->done_.notify_one();
    }
  }

  pthread_t thread_{};
  bool started_ = false;
  bool stopping_ = false;
  std::mutex submit_mu_;
  std::mutex mu_;
  std::condition_variable ready_;
  std::condition_variable done_;
  void *(*function_)(void *) = nullptr;
  void *argument_ = nullptr;
};

} // namespace obvsag
} // namespace common
} // namespace oceanbase
