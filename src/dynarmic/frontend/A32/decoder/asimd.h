/* This file is part of the dynarmic project.
 * Copyright (c) 2020 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <optional>
#include <set>
#include <vector>

#include <mcl/bit/bit_count.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/frontend/decoder/decoder_detail.h"
#include "dynarmic/frontend/decoder/matcher.h"

namespace Dynarmic::A32 {

namespace detail {
static_assert(std::atomic<u64>::is_always_lock_free);
constexpr size_t asimd_decode_cache_size = 1 << 16;
constexpr u64 asimd_decode_cache_valid = u64{1} << 63;
inline std::array<std::atomic<u64>, asimd_decode_cache_size> asimd_decode_cache{};
}  // namespace detail

template<typename Visitor>
using ASIMDMatcher = Decoder::Matcher<Visitor, u32>;

template<typename V>
std::vector<ASIMDMatcher<V>> GetASIMDDecodeTable() {
    std::vector<ASIMDMatcher<V>> table = {

#define INST(fn, name, bitstring) DYNARMIC_DECODER_GET_MATCHER(ASIMDMatcher, fn, name, Decoder::detail::StringToArray<32>(bitstring)),
#include "./asimd.inc"
#undef INST

    };

    // Exceptions to the rule of thumb.
    const std::set<std::string> comes_first{
        "VBIC, VMOV, VMVN, VORR (immediate)",
        "VEXT",
        "VTBL",
        "VTBX",
        "VDUP (scalar)",
    };
    const std::set<std::string> comes_last{
        "VMLA (scalar)",
        "VMLAL (scalar)",
        "VQDMLAL/VQDMLSL (scalar)",
        "VMUL (scalar)",
        "VMULL (scalar)",
        "VQDMULL (scalar)",
        "VQDMULH (scalar)",
        "VQRDMULH (scalar)",
    };
    const auto sort_begin = std::stable_partition(table.begin(), table.end(), [&](const auto& matcher) {
        return comes_first.count(matcher.GetName()) > 0;
    });
    const auto sort_end = std::stable_partition(table.begin(), table.end(), [&](const auto& matcher) {
        return comes_last.count(matcher.GetName()) == 0;
    });

    // If a matcher has more bits in its mask it is more specific, so it should come first.
    std::stable_sort(sort_begin, sort_end, [](const auto& matcher1, const auto& matcher2) {
        return mcl::bit::count_ones(matcher1.GetMask()) > mcl::bit::count_ones(matcher2.GetMask());
    });

    return table;
}

template<typename V>
std::optional<std::reference_wrapper<const ASIMDMatcher<V>>> DecodeASIMD(u32 instruction) {
    static const auto table = GetASIMDDecodeTable<V>();

    u32 hash = instruction * 0x9E3779B1U;
    hash ^= hash >> 16;
    const size_t cache_index = hash & (detail::asimd_decode_cache_size - 1);
    const u64 cached = detail::asimd_decode_cache[cache_index].load(std::memory_order_relaxed);
    if ((cached & detail::asimd_decode_cache_valid) != 0 &&
        static_cast<u32>(cached >> 16) == instruction) {
        const u16 result = static_cast<u16>(cached);
        return result == 0
                   ? std::nullopt
                   : std::optional<std::reference_wrapper<const ASIMDMatcher<V>>>{table[result - 1]};
    }

    const auto matches_instruction = [instruction](const auto& matcher) { return matcher.Matches(instruction); };

    const auto iter = std::find_if(table.begin(), table.end(), matches_instruction);
    const u16 result = iter == table.end() ? 0 : static_cast<u16>(iter - table.begin() + 1);
    detail::asimd_decode_cache[cache_index].store(
        detail::asimd_decode_cache_valid | (static_cast<u64>(instruction) << 16) | result,
        std::memory_order_relaxed);
    return iter != table.end() ? std::optional<std::reference_wrapper<const ASIMDMatcher<V>>>(*iter) : std::nullopt;
}

}  // namespace Dynarmic::A32
