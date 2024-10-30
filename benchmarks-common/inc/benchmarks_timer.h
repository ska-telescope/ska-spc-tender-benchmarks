#include <chrono>

namespace benchmarks_common
{
    class timer
    {
      public:
        timer()
        {
            reset();
        }

        void reset()
        {
            last_time_ = std::chrono::high_resolution_clock::now();
        }

        template <typename time_unit = std::chrono::microseconds>
        double time()
        {
            const auto previous_time = last_time_;
            reset();
            const auto current_time = last_time_;
            return std::chrono::duration_cast<time_unit>(current_time - previous_time).count();
        }

      private:
        std::chrono::high_resolution_clock::time_point last_time_;
    };
} // namespace benchmarks_common