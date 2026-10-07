/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 *
 * Original version of table by Lioncash.
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <optional>
#include <vector>

#include <mcl/bit/bit_count.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/frontend/decoder/decoder_detail.h"
#include "dynarmic/frontend/decoder/matcher.h"

namespace Dynarmic::A32 {

template<typename Visitor>
using ArmMatcher = Decoder::Matcher<Visitor, u32>;

template<typename Visitor>
using ArmDecodeTable = std::array<std::vector<ArmMatcher<Visitor>>, 0x1000>;

namespace detail {
constexpr size_t arm_decode_cache_size = 1 << 16;
constexpr u64 arm_decode_cache_valid = u64{1} << 63;
inline std::array<std::atomic<u64>, arm_decode_cache_size> arm_decode_cache{};

inline size_t ToFastLookupIndexArm(u32 instruction) {
    return ((instruction >> 4) & 0x00F) | ((instruction >> 16) & 0xFF0);
}
}  // namespace detail

template<typename V>
ArmDecodeTable<V> GetArmDecodeTable() {
    std::vector<ArmMatcher<V>> list = {

#define INST(fn, name, bitstring) DYNARMIC_DECODER_GET_MATCHER(ArmMatcher, fn, name, Decoder::detail::StringToArray<32>(bitstring)),
#include "./arm.inc"
#undef INST

    };

    // If a matcher has more bits in its mask it is more specific, so it should come first.
    std::stable_sort(list.begin(), list.end(), [](const auto& matcher1, const auto& matcher2) {
        return mcl::bit::count_ones(matcher1.GetMask()) > mcl::bit::count_ones(matcher2.GetMask());
    });

    ArmDecodeTable<V> table{};
    for (size_t i = 0; i < table.size(); ++i) {
        for (auto matcher : list) {
            const auto expect = detail::ToFastLookupIndexArm(matcher.GetExpected());
            const auto mask = detail::ToFastLookupIndexArm(matcher.GetMask());
            if ((i & mask) == expect) {
                table[i].push_back(matcher);
            }
        }
    }
    return table;
}

template<typename V>
std::optional<std::reference_wrapper<const ArmMatcher<V>>> DecodeArm(u32 instruction) {
    static const auto table = GetArmDecodeTable<V>();

    u32 hash = instruction * 0x9E3779B1U;
    hash ^= hash >> 16;
    const size_t cache_index = hash & (detail::arm_decode_cache_size - 1);
    const u64 cached = detail::arm_decode_cache[cache_index].load(std::memory_order_relaxed);

    const auto& subtable = table[detail::ToFastLookupIndexArm(instruction)];
    if ((cached & detail::arm_decode_cache_valid) != 0 &&
        static_cast<u32>(cached >> 16) == instruction) {
        const u16 result = static_cast<u16>(cached);
        return result == 0
                   ? std::nullopt
                   : std::optional<std::reference_wrapper<const ArmMatcher<V>>>{subtable[result - 1]};
    }

    const auto matches_instruction = [instruction](const auto& matcher) { return matcher.Matches(instruction); };

    const auto iter = std::find_if(subtable.begin(), subtable.end(), matches_instruction);
    const u16 result = iter == subtable.end() ? 0 : static_cast<u16>(iter - subtable.begin() + 1);
    detail::arm_decode_cache[cache_index].store(
        detail::arm_decode_cache_valid | (static_cast<u64>(instruction) << 16) | result,
        std::memory_order_relaxed);
    return iter != subtable.end() ? std::optional<std::reference_wrapper<const ArmMatcher<V>>>(*iter) : std::nullopt;
}

}  // namespace Dynarmic::A32
