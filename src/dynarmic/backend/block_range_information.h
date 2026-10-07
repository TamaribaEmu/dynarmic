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
    tsl::robin_set<IR::LocationDescriptor> InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges);

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
