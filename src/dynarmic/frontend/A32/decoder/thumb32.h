/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <optional>
#include <vector>

#include <mcl/stdint.hpp>

#include "dynarmic/frontend/decoder/decoder_detail.h"
#include "dynarmic/frontend/decoder/matcher.h"

namespace Dynarmic::A32 {

namespace detail {
constexpr size_t thumb32_decode_cache_size = 1 << 16;
constexpr u64 thumb32_decode_cache_valid = u64{1} << 63;
inline std::array<std::atomic<u64>, thumb32_decode_cache_size> thumb32_decode_cache{};
}  // namespace detail

template<typename Visitor>
using Thumb32Matcher = Decoder::Matcher<Visitor, u32>;

template<typename V>
std::optional<std::reference_wrapper<const Thumb32Matcher<V>>> DecodeThumb32(u32 instruction) {
    static const std::vector<Thumb32Matcher<V>> table = {

#define INST(fn, name, bitstring) DYNARMIC_DECODER_GET_MATCHER(Thumb32Matcher, fn, name, Decoder::detail::StringToArray<32>(bitstring)),
#include "./thumb32.inc"
#undef INST

    };

    u32 hash = instruction * 0x9E3779B1U;
    hash ^= hash >> 16;
    const size_t cache_index = hash & (detail::thumb32_decode_cache_size - 1);
    const u64 cached = detail::thumb32_decode_cache[cache_index].load(std::memory_order_relaxed);
    if ((cached & detail::thumb32_decode_cache_valid) != 0 &&
        static_cast<u32>(cached >> 16) == instruction) {
        const u16 result = static_cast<u16>(cached);
        return result == 0
                   ? std::nullopt
                   : std::optional<std::reference_wrapper<const Thumb32Matcher<V>>>{table[result - 1]};
    }

    const auto matches_instruction = [instruction](const auto& matcher) { return matcher.Matches(instruction); };

    const auto iter = std::find_if(table.begin(), table.end(), matches_instruction);
    const u16 result = iter == table.end() ? 0 : static_cast<u16>(iter - table.begin() + 1);
    detail::thumb32_decode_cache[cache_index].store(
        detail::thumb32_decode_cache_valid | (static_cast<u64>(instruction) << 16) | result,
        std::memory_order_relaxed);
    return iter != table.end() ? std::optional<std::reference_wrapper<const Thumb32Matcher<V>>>(*iter) : std::nullopt;
}

}  // namespace Dynarmic::A32
