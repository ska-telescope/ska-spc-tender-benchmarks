#pragma once
#include <stdexcept>
#include <sys/mman.h>

template <typename T>
class hpallocator {
public:
    using value_type = T;

    hpallocator() = default;

    template <typename U>
    hpallocator(const hpallocator<U>&) {}

    T* allocate(std::size_t n) {
        std::size_t size = n * sizeof(T);

        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (ptr == MAP_FAILED) {
            throw std::runtime_error("mmap allocation failed");
        }

        if (madvise(ptr, size, MADV_HUGEPAGE) != 0) {
            throw std::runtime_error("madvise failed to enable huge pages");
        }

        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t n) {
        std::size_t size = n * sizeof(T);

        if (munmap(ptr, size) != 0) {
            throw std::runtime_error("munmap deallocation failed");
        }
    }
};

template <typename T, typename U>
bool operator==(const hpallocator<T>&, const hpallocator<U>&) {
    return true;
}

template <typename T, typename U>
bool operator!=(const hpallocator<T>&, const hpallocator<U>&) {
    return false;
}
