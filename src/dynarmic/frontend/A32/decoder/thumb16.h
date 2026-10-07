/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
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
constexpr u16 thumb16_decode_cache_miss = 0xFFFF;
inline std::array<std::atomic<u16>, 1 << 16> thumb16_decode_cache{};
}  // namespace detail

template<typename Visitor>
using Thumb16Matcher = Decoder::Matcher<Visitor, u16>;

template<typename V>
std::optional<std::reference_wrapper<const Thumb16Matcher<V>>> DecodeThumb16(u16 instruction) {
    static const std::vector<Thumb16Matcher<V>> table = {

#define INST(fn, name, bitstring) DYNARMIC_DECODER_GET_MATCHER(Thumb16Matcher, fn, name, Decoder::detail::StringToArray<16>(bitstring)),
#include "./thumb16.inc"
#undef INST

    };

    const u16 cached = detail::thumb16_decode_cache[instruction].load(std::memory_order_relaxed);
    if (cached != 0) {
        return cached == detail::thumb16_decode_cache_miss
                   ? std::nullopt
                   : std::optional<std::reference_wrapper<const Thumb16Matcher<V>>>{table[cached - 1]};
    }

    const auto matches_instruction = [instruction](const auto& matcher) { return matcher.Matches(instruction); };

    const auto iter = std::find_if(table.begin(), table.end(), matches_instruction);
    const u16 result = iter == table.end()
                           ? detail::thumb16_decode_cache_miss
                           : static_cast<u16>(iter - table.begin() + 1);
    detail::thumb16_decode_cache[instruction].store(result, std::memory_order_relaxed);
    return iter != table.end() ? std::optional<std::reference_wrapper<const Thumb16Matcher<V>>>(*iter) : std::nullopt;
}

}  // namespace Dynarmic::A32
