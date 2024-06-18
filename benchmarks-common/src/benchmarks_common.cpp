#include <array>
#include <cmath>
#include <iostream>
#include <string>

#include "benchmarks_common.h"

namespace benchmarks_common
{
    std::string bytes_to_memory_size(const size_t n)
    {
        const std::array<std::string, 9> strs = {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"};

        const auto log2n = (benchmarks_common::log2_size_t(n) - 1);
        const auto logrank = log2n / 10;
        const auto base = 1 << (logrank * 10);
        const auto rounded = std::round(static_cast<double>(10. * n) / static_cast<double>(base)) / 10.;
        std::stringstream sstr;
        sstr << rounded << " " << strs[std::min(logrank, size_t(8))];
        return sstr.str();
    }

    size_t log2_size_t(size_t index)
    {
        size_t targetlevel = 0;
        while (index >>= 1)
            ++targetlevel;
        return targetlevel;
    }
} // namespace benchmarks_common