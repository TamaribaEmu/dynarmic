/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <vector>

#include <boost/icl/interval_set.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
class BlockRangeInformation {
public:
    void AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location);
    void ClearCache();
    // The blocks whose code overlaps `ranges` (it changes nothing, despite the name).
    tsl::robin_set<IR::LocationDescriptor> InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges) const;
    // The same, also dropping those entries from the pages of `ranges`: the blocks are leaving.
    // Without this, every block ever compiled stayed indexed, so modules loaded in turn at one
    // address (Pokémon's field and battle) made each range walk slower (~20 ms per unload).
    tsl::robin_set<IR::LocationDescriptor> ExtractRanges(const boost::icl::interval_set<ProgramCounterType>& ranges);

    /// Calls fn(descriptor) for each indexed block that starts in [start, end], once per entry
    /// (an entry is visited from its first page only), without building a set.
    template<typename Fn>
    void ForEachStartingIn(ProgramCounterType start, ProgramCounterType end, Fn&& fn) const {
        ProgramCounterType page = start >> PageShift;
        const ProgramCounterType end_page = end >> PageShift;
        for (;;) {
            if (const auto bucket = block_ranges.find(page); bucket != block_ranges.end()) {
                for (const auto& entry : bucket->second) {
                    if ((entry.start >> PageShift) == page && entry.start >= start &&
                        entry.start <= end) {
                        fn(entry.descriptor);
                    }
                }
            }
            if (page == end_page) {
                break;
            }
            ++page;
        }
    }

private:
    struct RangeEntry {
        ProgramCounterType start;
        ProgramCounterType end;
        IR::LocationDescriptor descriptor;
    };

    static constexpr std::size_t PageShift = 12;
    tsl::robin_map<ProgramCounterType, std::vector<RangeEntry>> block_ranges;
};

}  // namespace Dynarmic::Backend
