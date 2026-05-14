/*
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

#include "svt_malloc.h"
#include <cstddef>
#include <memory>
#include <new>
#include <vector>

namespace svt_av1_test_tool {

template <typename T>
class aligned_allocator {
  public:
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    aligned_allocator() noexcept = default;

    template <typename U>
    explicit aligned_allocator(const aligned_allocator<U> &) noexcept {
    }

    template <typename U>
    struct rebind {
        using other = aligned_allocator<U>;
    };

    static pointer allocate(size_type n, const void * = nullptr) {
        if (T *ptr = reinterpret_cast<T *>(svt_aom_memalign(32, n * sizeof(T))))
            return ptr;
        throw std::bad_alloc();
    }

    static void deallocate(pointer ptr, size_type) noexcept {
        svt_aom_free(ptr);
    }
};

template <typename T, typename U>
bool operator==(const aligned_allocator<T> &, const aligned_allocator<U> &) {
    return true;
}

template <typename T, typename U>
bool operator!=(const aligned_allocator<T> &, const aligned_allocator<U> &) {
    return false;
}

#define DEFINE_ALIGNED_NEW_DELETE(Derrived)                        \
    void *operator new(std::size_t size) {                         \
        if (void *ptr = svt_aom_memalign(alignof(Derrived), size)) \
            return ptr;                                            \
        throw std::bad_alloc();                                    \
    }                                                              \
    void operator delete(void *ptr) noexcept {                     \
        svt_aom_free(ptr);                                         \
    }

/**
 * @brief A vector that provides a constructor that only takes size.
 * Useful for defining a class member vector with a size, but stores an integer
 * type. This is mainly to prioritize the size constructor over an
 * initializer_list.
 *
 * @tparam T type to store in the vector
 * @tparam Allocator allocator type to use for the vector
 */
template <typename T, typename Allocator = std::allocator<T>>
struct SizeOnlyVec : public std::vector<T, Allocator> {
    // Constructor so we don't have to specify the allocator every time we
    // declare a vector. Similar to what we have in ResizeTest.cc
    explicit SizeOnlyVec(size_t size, const T &init_value = T(),
                         const Allocator &allocator = Allocator())
        : std::vector<T, Allocator>(size, init_value, allocator) {
    }
};

}  // namespace svt_av1_test_tool

#endif  // ALIGNED_ALLOCATOR_H
