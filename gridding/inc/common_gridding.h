#pragma once

namespace gridding_benchmark
{
    struct benchmark_result
    {
        double in_transfer_time{-1.};
        double out_transfer_time{-1.};
        double in_bandwidth{-1.};
        double out_bandwidth{-1.};
        double compute_time{-1.};
    };
} // namespace gridding_benchmark