/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <optional>

#include "dynarmic/backend/arm64/address_space.h"
#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/interface/A32/config.h"

namespace Dynarmic::Backend::Arm64 {

struct EmittedBlockInfo;

class A32AddressSpace final : public AddressSpace {
public:
    explicit A32AddressSpace(const A32::UserConfig& conf);

    IR::Block GenerateIR(IR::LocationDescriptor) const override;

    void InvalidateCacheRanges(const boost::icl::interval_set<u32>& ranges);

    void SetFastPrecompile(bool enabled) {
        fast_precompile = enabled;
    }

    std::optional<u64> GuestCodeHash(u64 descriptor, u64 end_descriptor) const;

    /// GetCompiledBlockEntries for the blocks starting in [start, start + length) only, found
    /// through the per-page range index instead of by walking every compiled block.
    std::vector<CachedBlockEntry> GetCompiledBlockEntriesInRange(u32 start, std::size_t length) const;
    std::vector<u64> GetCompiledBlockDescriptorsInRange(u32 start, std::size_t length) const;

protected:
    friend class A32Core;

    void EmitPrelude();
    EmitConfig GetEmitConfig() override;
    void RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo& block_info) override;
    bool CanReactivateBlock(CodePtr entry_point) const override;
    void ClearReactivationMetadata() override;
    void RegisterReactivatedBlock(const CachedBlockEntry& entry) override;

    const A32::UserConfig conf;
    BlockRangeInformation<u32> block_ranges;
    tsl::robin_set<CodePtr> fast_precompiled_blocks;
    bool fast_precompile = false;
};

}  // namespace Dynarmic::Backend::Arm64
