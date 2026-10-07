/* This file is part of the dynarmic project.
 * Copyright (c) 2032 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <optional>
#include <vector>

#include <mcl/stdint.hpp>

#include "dynarmic/frontend/decoder/decoder_detail.h"
#include "dynarmic/frontend/decoder/matcher.h"

namespace Dynarmic::A32 {

namespace detail {
static_assert(std::atomic<u64>::is_always_lock_free);
constexpr size_t vfp_decode_cache_size = 1 << 16;
constexpr u64 vfp_decode_cache_valid = u64{1} << 63;
inline std::array<std::atomic<u64>, vfp_decode_cache_size> vfp_decode_cache{};
}  // namespace detail

template<typename Visitor>
using VFPMatcher = Decoder::Matcher<Visitor, u32>;

template<typename V>
std::optional<std::reference_wrapper<const VFPMatcher<V>>> DecodeVFP(u32 instruction) {
    using Table = std::vector<VFPMatcher<V>>;

    static const struct Tables {
        Table unconditional;
        Table conditional;
    } tables = [] {
        Table list = {

#define INST(fn, name, bitstring) DYNARMIC_DECODER_GET_MATCHER(VFPMatcher, fn, name, Decoder::detail::StringToArray<32>(bitstring)),
#include "./vfp.inc"
#undef INST

        };

        const auto division = std::stable_partition(list.begin(), list.end(), [&](const auto& matcher) {
            return (matcher.GetMask() & 0xF0000000) == 0xF0000000;
        });

        return Tables{
            Table{list.begin(), division},
            Table{division, list.end()},
        };
    }();

    const bool is_unconditional = (instruction & 0xF0000000) == 0xF0000000;
    const Table& table = is_unconditional ? tables.unconditional : tables.conditional;

    u32 hash = instruction * 0x9E3779B1U;
    hash ^= hash >> 16;
    const size_t cache_index = hash & (detail::vfp_decode_cache_size - 1);
    const u64 cached = detail::vfp_decode_cache[cache_index].load(std::memory_order_relaxed);
    if ((cached & detail::vfp_decode_cache_valid) != 0 &&
        static_cast<u32>(cached >> 16) == instruction) {
        const u16 result = static_cast<u16>(cached);
        return result == 0
                   ? std::nullopt
                   : std::optional<std::reference_wrapper<const VFPMatcher<V>>>{table[result - 1]};
    }

    const auto matches_instruction = [instruction](const auto& matcher) { return matcher.Matches(instruction); };

    const auto iter = std::find_if(table.begin(), table.end(), matches_instruction);
    const u16 result = iter == table.end() ? 0 : static_cast<u16>(iter - table.begin() + 1);
    detail::vfp_decode_cache[cache_index].store(
        detail::vfp_decode_cache_valid | (static_cast<u64>(instruction) << 16) | result,
        std::memory_order_relaxed);
    return iter != table.end() ? std::optional<std::reference_wrapper<const VFPMatcher<V>>>(*iter) : std::nullopt;
}

}  // namespace Dynarmic::A32
