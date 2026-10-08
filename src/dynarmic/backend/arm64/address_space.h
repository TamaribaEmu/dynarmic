/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <mcl/stdint.hpp>
#include <oaknut/code_block.hpp>
#include <oaknut/oaknut.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/fastmem.h"
#include "dynarmic/interface/halt_reason.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend::Arm64 {

struct CacheStats {
    u64 emitted_blocks{};
    u64 emitted_bytes{};
    u64 invalidated_blocks{};
    u64 cache_clears{};
    u64 live_blocks{};
    u64 used_bytes{};
    u64 capacity_bytes{};
    u64 precompile_translate_nanoseconds{};
    u64 precompile_emit_nanoseconds{};
    u64 precompile_body_nanoseconds{};
    u64 precompile_terminal_nanoseconds{};
    u64 precompile_deferred_nanoseconds{};
    u64 precompile_metadata_nanoseconds{};
    u64 precompile_link_nanoseconds{};
};

struct CachedBlockEntry {
    u64 descriptor{};
    u64 end_descriptor{};
    CodePtr entry_point{};
};

class AddressSpace {
public:
    explicit AddressSpace(size_t code_cache_size);
    virtual ~AddressSpace();

    virtual IR::Block GenerateIR(IR::LocationDescriptor) const = 0;

    CodePtr Get(IR::LocationDescriptor descriptor);

    // Returns "most likely" LocationDescriptor assocated with the emitted code at that location
    std::optional<IR::LocationDescriptor> ReverseGetLocation(CodePtr host_pc);

    // Returns "most likely" entry_point associated with the emitted code at that location
    CodePtr ReverseGetEntryPoint(CodePtr host_pc);

    CodePtr GetOrEmit(IR::LocationDescriptor descriptor);
    CodePtr GetOrEmitPrecompiled(IR::LocationDescriptor descriptor);
    std::size_t PrecompileBlocks(const std::vector<u64>& descriptors);

    void BeginCodeWriteBatch();
    void EndCodeWriteBatch();
    void BeginPrecompileBatch();
    void EndPrecompileBatch();

    std::vector<u64> GetCompiledBlockDescriptors() const;
    std::vector<CachedBlockEntry> GetCompiledBlockEntries() const;
    std::size_t ReactivateBlocks(const std::vector<CachedBlockEntry>& entries);
    CacheStats GetAndResetCacheStats();

    /// `keep_links_within_set`: the blocks are leaving together (a module unload or a code write
    /// range), so branches between them are left as they are: the blocks are unreachable from then
    /// on, and ReactivateBlocks relies on those branches being intact. The fastmem path passes
    /// false, because the block being replaced may be running and may branch to itself.
    void InvalidateBasicBlocks(const tsl::robin_set<IR::LocationDescriptor>& descriptors,
                               bool keep_links_within_set = false);

    void ClearCache();

    void DumpDisassembly() const;

protected:
    virtual EmitConfig GetEmitConfig() = 0;
    std::array<void*, static_cast<size_t>(LinkTarget::Count)> GetLinkTargets() const;
    virtual void RegisterNewBasicBlock(const IR::Block& block, const EmittedBlockInfo& block_info) = 0;
    virtual bool CanReactivateBlock(CodePtr entry_point) const = 0;
    virtual void ClearReactivationMetadata() = 0;
    /// A block taken out by a range invalidation runs again: index its range again.
    virtual void RegisterReactivatedBlock(const CachedBlockEntry&) {}

    void ProtectCodeMemory() {
#if defined(DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT) || defined(__APPLE__) || defined(__OpenBSD__)
        mem.protect();
#endif
    }

    void UnprotectCodeMemory() {
#if defined(DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT) || defined(__APPLE__) || defined(__OpenBSD__)
        mem.unprotect();
#endif
    }

    size_t GetRemainingSize();
    CodePtr Emit(IR::Block ir_block);
    void Link(const EmittedBlockInfo& block);
    /// Whether the block at entry_point is the one its location currently runs. Blocks that are
    /// not (invalidated or dormant) are unreachable, so relinking skips them: their code is never
    /// patched after they leave, and ReactivateBlocks re-links what leaves their set.
    bool IsLiveBlock(CodePtr entry_point) const;
    void LinkBlockLinks(const CodePtr entry_point, const CodePtr target_ptr, const std::vector<BlockRelocation>& block_relocations_list);
    void LinkBlockLinks(const CodePtr entry_point, const CodePtr target_ptr,
                        const std::vector<BlockRelocation>& block_relocations_list,
                        oaknut::CodeGenerator& patcher);
    void RelinkForDescriptor(IR::LocationDescriptor target_descriptor, CodePtr target_ptr);
    void RelinkForDescriptorExcluding(IR::LocationDescriptor target_descriptor,
                                      CodePtr target_ptr,
                                      const tsl::robin_set<CodePtr>& excluded_entries);

    FakeCall FastmemCallback(u64 host_pc);

    const size_t code_cache_size;
    oaknut::CodeBlock mem;
    oaknut::CodeGenerator code;

    // A IR::LocationDescriptor will have one current CodePtr.
    // However, there can be multiple other CodePtrs which are older, previously invalidated blocks.
    tsl::robin_map<IR::LocationDescriptor, CodePtr> block_entries;
    /// Entry points of blocks taken out of block_entries (invalidated or dormant) whose code is
    /// still in the cache: IsLiveBlock in one lookup.
    tsl::robin_set<CodePtr> retired_entries;
    std::map<CodePtr, IR::LocationDescriptor> reverse_block_entries;
    tsl::robin_map<CodePtr, IR::LocationDescriptor> block_end_locations;
    tsl::robin_map<CodePtr, EmittedBlockInfo> block_infos;
    tsl::robin_map<IR::LocationDescriptor, tsl::robin_set<CodePtr>> block_references;

    ExceptionHandler exception_handler;
    FastmemManager fastmem_manager;
    std::optional<EmitConfig> cached_emit_config;
    bool code_write_batch_active = false;
    bool precompile_batch_active = false;
    // What a write batch changed, so EndCodeWriteBatch flushes the instruction cache for just
    // that: code emitted from batch_code_start on, and each patched branch site. A cache
    // clear inside the batch makes it flush everything, as it always used to.
    std::size_t batch_code_start = 0;
    std::vector<std::uintptr_t> batch_patches;
    bool batch_full_flush = false;
    void FlushBatchWrites();
    std::vector<std::pair<IR::LocationDescriptor, CodePtr>> deferred_precompile_entries;
    u64 emitted_blocks_since_reset{};
    u64 emitted_bytes_since_reset{};
    u64 invalidated_blocks_since_reset{};
    u64 cache_clears_since_reset{};
    u64 precompile_translate_nanoseconds_since_reset{};
    u64 precompile_emit_nanoseconds_since_reset{};
    u64 precompile_body_nanoseconds_since_reset{};
    u64 precompile_terminal_nanoseconds_since_reset{};
    u64 precompile_deferred_nanoseconds_since_reset{};
    u64 precompile_metadata_nanoseconds_since_reset{};
    u64 precompile_link_nanoseconds_since_reset{};

    struct PreludeInfo {
        std::ptrdiff_t end_of_prelude;

        using RunCodeFuncType = HaltReason (*)(CodePtr entry_point, void* jit_state, volatile u32* halt_reason);
        RunCodeFuncType run_code;
        RunCodeFuncType step_code;
        void* return_to_dispatcher;
        void* return_from_run_code;

        void* read_memory_8;
        void* read_memory_16;
        void* read_memory_32;
        void* read_memory_64;
        void* read_memory_128;
        void* wrapped_read_memory_8;
        void* wrapped_read_memory_16;
        void* wrapped_read_memory_32;
        void* wrapped_read_memory_64;
        void* wrapped_read_memory_128;
        void* exclusive_read_memory_8;
        void* exclusive_read_memory_16;
        void* exclusive_read_memory_32;
        void* exclusive_read_memory_64;
        void* exclusive_read_memory_128;
        void* write_memory_8;
        void* write_memory_16;
        void* write_memory_32;
        void* write_memory_64;
        void* write_memory_128;
        void* wrapped_write_memory_8;
        void* wrapped_write_memory_16;
        void* wrapped_write_memory_32;
        void* wrapped_write_memory_64;
        void* wrapped_write_memory_128;
        void* exclusive_write_memory_8;
        void* exclusive_write_memory_16;
        void* exclusive_write_memory_32;
        void* exclusive_write_memory_64;
        void* exclusive_write_memory_128;

        void* call_svc;
        void* exception_raised;
        void* dc_raised;
        void* ic_raised;
        void* isb_raised;

        void* get_cntpct;
        void* add_ticks;
        void* get_ticks_remaining;
    } prelude_info;
};

}  // namespace Dynarmic::Backend::Arm64
