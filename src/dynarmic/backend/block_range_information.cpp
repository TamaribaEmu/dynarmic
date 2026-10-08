/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/block_range_information.h"

#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include <mcl/stdint.hpp>
#include <tsl/robin_set.h>

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location) {
    if (boost::icl::is_empty(range)) {
        return;
    }
    const ProgramCounterType start = boost::icl::first(range);
    const ProgramCounterType end = boost::icl::last(range);
    ProgramCounterType page = start >> PageShift;
    const ProgramCounterType end_page = end >> PageShift;
    for (;;) {
        block_ranges[page].push_back({start, end, location});
        if (page == end_page) {
            break;
        }
        ++page;
    }
}

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::ClearCache() {
    block_ranges.clear();
}

template<typename ProgramCounterType>
tsl::robin_set<IR::LocationDescriptor> BlockRangeInformation<ProgramCounterType>::InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges) const {
    tsl::robin_set<IR::LocationDescriptor> erase_locations;
    for (const auto& invalidate_interval : ranges) {
        if (boost::icl::is_empty(invalidate_interval)) {
            continue;
        }
        const ProgramCounterType invalidate_start = boost::icl::first(invalidate_interval);
        const ProgramCounterType invalidate_end = boost::icl::last(invalidate_interval);
        ProgramCounterType page = invalidate_start >> PageShift;
        const ProgramCounterType end_page = invalidate_end >> PageShift;
        for (;;) {
            if (const auto bucket = block_ranges.find(page); bucket != block_ranges.end()) {
                for (const auto& entry : bucket->second) {
                    if (entry.start <= invalidate_end && entry.end >= invalidate_start) {
                        erase_locations.insert(entry.descriptor);
                    }
                }
            }
            if (page == end_page) {
                break;
            }
            ++page;
        }
    }
    // TODO: EFFICIENCY: Remove ranges that are to be erased.
    return erase_locations;
}

template<typename ProgramCounterType>
tsl::robin_set<IR::LocationDescriptor> BlockRangeInformation<ProgramCounterType>::ExtractRanges(const boost::icl::interval_set<ProgramCounterType>& ranges) {
    tsl::robin_set<IR::LocationDescriptor> erase_locations;
    for (const auto& invalidate_interval : ranges) {
        if (boost::icl::is_empty(invalidate_interval)) {
            continue;
        }
        const ProgramCounterType invalidate_start = boost::icl::first(invalidate_interval);
        const ProgramCounterType invalidate_end = boost::icl::last(invalidate_interval);
        ProgramCounterType page = invalidate_start >> PageShift;
        const ProgramCounterType end_page = invalidate_end >> PageShift;
        for (;;) {
            if (const auto bucket = block_ranges.find(page); bucket != block_ranges.end()) {
                auto& entries = bucket.value();
                std::erase_if(entries, [&](const RangeEntry& entry) {
                    if (entry.start <= invalidate_end && entry.end >= invalidate_start) {
                        erase_locations.insert(entry.descriptor);
                        return true;
                    }
                    return false;
                });
                if (entries.empty()) {
                    block_ranges.erase(bucket);
                }
            }
            if (page == end_page) {
                break;
            }
            ++page;
        }
    }
    return erase_locations;
}

template class BlockRangeInformation<u32>;
template class BlockRangeInformation<u64>;

}  // namespace Dynarmic::Backend
