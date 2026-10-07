/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <cstdio>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>

#include <mcl/bit_cast.hpp>
#include <mcl/scope_exit.hpp>

#include "dynarmic/backend/arm64/a64_address_space.h"
#include "dynarmic/backend/arm64/a64_jitstate.h"
#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/devirtualize.h"
#include "dynarmic/backend/arm64/emit_arm64.h"
#include "dynarmic/backend/arm64/stack_layout.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/llvm_disassemble.h"
#include "dynarmic/interface/exclusive_monitor.h"

namespace Dynarmic::Backend::Arm64 {

AddressSpace::AddressSpace(size_t code_cache_size)
        : code_cache_size(code_cache_size)
        , mem(code_cache_size)
        , code(mem.ptr(), mem.ptr())
        , fastmem_manager(exception_handler) {
    ASSERT_MSG(code_cache_size <= 128 * 1024 * 1024, "code_cache_size > 128 MiB not currently supported");

    exception_handler.Register(mem, code_cache_size);
    exception_handler.SetFastmemCallback([this](u64 host_pc) {
        return FastmemCallback(host_pc);
    });
}

AddressSpace::~AddressSpace() = default;

CodePtr AddressSpace::Get(IR::LocationDescriptor descriptor) {
    if (const auto iter = block_entries.find(descriptor); iter != block_entries.end()) {
        return iter->second;
    }
    return nullptr;
}

std::optional<IR::LocationDescriptor> AddressSpace::ReverseGetLocation(CodePtr host_pc) {
    if (auto iter = reverse_block_entries.upper_bound(host_pc); iter != reverse_block_entries.begin()) {
        // upper_bound locates the first value greater than host_pc, so we need to decrement
        --iter;
        return iter->second;
    }
    return std::nullopt;
}

CodePtr AddressSpace::ReverseGetEntryPoint(CodePtr host_pc) {
    if (auto iter = reverse_block_entries.upper_bound(host_pc); iter != reverse_block_entries.begin()) {
        // upper_bound locates the first value greater than host_pc, so we need to decrement
        --iter;
        return iter->first;
    }
    return nullptr;
}

CodePtr AddressSpace::GetOrEmit(IR::LocationDescriptor descriptor) {
    if (CodePtr block_entry = Get(descriptor)) {
        return block_entry;
    }

    IR::Block ir_block = GenerateIR(descriptor);
    return Emit(std::move(ir_block));
}

CodePtr AddressSpace::GetOrEmitPrecompiled(IR::LocationDescriptor descriptor) {
    if (CodePtr block_entry = Get(descriptor)) {
        return block_entry;
    }

    const auto translate_started = std::chrono::steady_clock::now();
    IR::Block ir_block = GenerateIR(descriptor);
    const auto emit_started = std::chrono::steady_clock::now();
    const CodePtr entry_point = Emit(std::move(ir_block));
    const auto finished = std::chrono::steady_clock::now();
    precompile_translate_nanoseconds_since_reset +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(emit_started - translate_started)
                    .count();
    precompile_emit_nanoseconds_since_reset +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(finished - emit_started).count();
    return entry_point;
}

std::size_t AddressSpace::PrecompileBlocks(const std::vector<u64>& descriptors) {
    std::vector<IR::LocationDescriptor> pending;
    pending.reserve(descriptors.size());
    tsl::robin_set<IR::LocationDescriptor> seen;
    seen.reserve(descriptors.size());
    for (const u64 descriptor_value : descriptors) {
        const IR::LocationDescriptor descriptor{descriptor_value};
        if (!Get(descriptor) && seen.insert(descriptor).second) {
            pending.push_back(descriptor);
        }
    }
    // Small batches do not repay thread startup and queue synchronization costs.
    constexpr std::size_t PipelineThreshold = 64;
    if (pending.size() < PipelineThreshold) {
        for (const auto descriptor : pending) {
            GetOrEmitPrecompiled(descriptor);
        }
        return descriptors.size();
    }

    struct QueuedBlock {
        IR::Block block;
        u64 translate_nanoseconds;
    };
    constexpr std::size_t QueueCapacity = 32;
    std::mutex queue_mutex;
    std::condition_variable queue_not_empty;
    std::condition_variable queue_not_full;
    std::deque<QueuedBlock> queue;
    bool producer_done = false;
    bool producer_cancelled = false;
    std::exception_ptr producer_error;

    std::thread producer{[&] {
        try {
            for (const auto descriptor : pending) {
                const auto started = std::chrono::steady_clock::now();
                IR::Block block = GenerateIR(descriptor);
                const u64 elapsed = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started)
                        .count());

                std::unique_lock lock{queue_mutex};
                queue_not_full.wait(lock, [&] {
                    return queue.size() < QueueCapacity || producer_cancelled;
                });
                if (producer_cancelled) {
                    break;
                }
                queue.push_back({std::move(block), elapsed});
                lock.unlock();
                queue_not_empty.notify_one();
            }
        } catch (...) {
            std::lock_guard lock{queue_mutex};
            producer_error = std::current_exception();
        }

        {
            std::lock_guard lock{queue_mutex};
            producer_done = true;
        }
        queue_not_empty.notify_one();
    }};
    SCOPE_EXIT {
        {
            std::lock_guard lock{queue_mutex};
            producer_cancelled = true;
        }
        queue_not_full.notify_one();
        if (producer.joinable()) {
            producer.join();
        }
    };

    for (;;) {
        std::unique_lock lock{queue_mutex};
        queue_not_empty.wait(lock, [&] { return !queue.empty() || producer_done; });
        if (queue.empty()) {
            break;
        }
        QueuedBlock queued = std::move(queue.front());
        queue.pop_front();
        lock.unlock();
        queue_not_full.notify_one();

        precompile_translate_nanoseconds_since_reset += queued.translate_nanoseconds;
        const auto emit_started = std::chrono::steady_clock::now();
        Emit(std::move(queued.block));
        precompile_emit_nanoseconds_since_reset += static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - emit_started)
                .count());
    }

    if (producer_error) {
        std::rethrow_exception(producer_error);
    }
    return descriptors.size();
}

void AddressSpace::BeginCodeWriteBatch() {
    ASSERT(!code_write_batch_active);
    UnprotectCodeMemory();
    code_write_batch_active = true;
}

void AddressSpace::EndCodeWriteBatch() {
    ASSERT(code_write_batch_active);
    mem.invalidate(mem.ptr(), static_cast<std::size_t>(code.offset()));
    code_write_batch_active = false;
    ProtectCodeMemory();
}

void AddressSpace::BeginPrecompileBatch() {
    BeginCodeWriteBatch();
    ASSERT(!precompile_batch_active);
    precompile_batch_active = true;
    deferred_precompile_entries.clear();
}

void AddressSpace::EndPrecompileBatch() {
    ASSERT(precompile_batch_active);
    const auto link_started = std::chrono::steady_clock::now();

    struct IncomingRelink {
        IR::LocationDescriptor descriptor;
        CodePtr target;
        std::vector<CodePtr> sources;
    };
    std::vector<IncomingRelink> incoming_relinks;
    incoming_relinks.reserve(deferred_precompile_entries.size());
    // Snapshot only references that existed before this batch. Link() below adds references from
    // every newly emitted block, but those outgoing links already see the complete target map and
    // must not be scanned and patched a second time.
    for (const auto& [descriptor, entry_point] : deferred_precompile_entries) {
        const auto references = block_references.find(descriptor);
        if (references == block_references.end() || references->second.empty()) {
            continue;
        }
        IncomingRelink incoming{descriptor, entry_point, {}};
        incoming.sources.reserve(references->second.size());
        incoming.sources.insert(incoming.sources.end(), references->second.begin(),
                                references->second.end());
        incoming_relinks.push_back(std::move(incoming));
    }

    for (const auto& entry : deferred_precompile_entries) {
        Link(block_infos.at(entry.second));
    }
    for (const auto& incoming : incoming_relinks) {
        for (const CodePtr source : incoming.sources) {
            const auto block = block_infos.find(source);
            if (block == block_infos.end()) {
                continue;
            }
            const auto relocations = block->second.block_relocations.find(incoming.descriptor);
            if (relocations != block->second.block_relocations.end()) {
                LinkBlockLinks(source, incoming.target, relocations->second);
            }
        }
    }
    deferred_precompile_entries.clear();
    precompile_batch_active = false;
    EndCodeWriteBatch();
    precompile_link_nanoseconds_since_reset += static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - link_started)
            .count());
}

std::vector<u64> AddressSpace::GetCompiledBlockDescriptors() const {
    std::vector<u64> descriptors;
    descriptors.reserve(block_entries.size());
    for (const auto& [descriptor, entry] : block_entries) {
        descriptors.push_back(descriptor.Value());
    }
    return descriptors;
}

std::vector<CachedBlockEntry> AddressSpace::GetCompiledBlockEntries() const {
    std::vector<CachedBlockEntry> entries;
    entries.reserve(block_entries.size());
    for (const auto& [descriptor, entry_point] : block_entries) {
        const auto end = block_end_locations.find(entry_point);
        if (end != block_end_locations.end() && CanReactivateBlock(entry_point)) {
            entries.push_back({descriptor.Value(), end->second.Value(), entry_point});
        }
    }
    return entries;
}

std::size_t AddressSpace::ReactivateBlocks(const std::vector<CachedBlockEntry>& entries) {
    std::vector<CachedBlockEntry> valid_entries;
    valid_entries.reserve(entries.size());
    for (const auto& entry : entries) {
        const IR::LocationDescriptor descriptor{entry.descriptor};
        const auto reverse = reverse_block_entries.find(entry.entry_point);
        if (block_entries.contains(descriptor) ||
            reverse == reverse_block_entries.end() || reverse->second != descriptor ||
            !block_infos.contains(entry.entry_point)) {
            continue;
        }
        valid_entries.push_back(entry);
    }
    if (valid_entries.empty()) {
        return 0;
    }

    BeginCodeWriteBatch();
    for (const auto& entry : valid_entries) {
        block_entries.emplace(IR::LocationDescriptor{entry.descriptor}, entry.entry_point);
    }
    for (const auto& entry : valid_entries) {
        Link(block_infos.at(entry.entry_point));
    }
    for (const auto& entry : valid_entries) {
        RelinkForDescriptor(IR::LocationDescriptor{entry.descriptor}, entry.entry_point);
    }
    EndCodeWriteBatch();
    return valid_entries.size();
}

CacheStats AddressSpace::GetAndResetCacheStats() {
    const u64 used_bytes = static_cast<u64>(code.offset() - prelude_info.end_of_prelude);
    const u64 capacity_bytes =
        static_cast<u64>(code_cache_size - prelude_info.end_of_prelude);
    const CacheStats stats{
        .emitted_blocks = emitted_blocks_since_reset,
        .emitted_bytes = emitted_bytes_since_reset,
        .invalidated_blocks = invalidated_blocks_since_reset,
        .cache_clears = cache_clears_since_reset,
        .live_blocks = static_cast<u64>(block_entries.size()),
        .used_bytes = used_bytes,
        .capacity_bytes = capacity_bytes,
        .precompile_translate_nanoseconds = precompile_translate_nanoseconds_since_reset,
        .precompile_emit_nanoseconds = precompile_emit_nanoseconds_since_reset,
        .precompile_body_nanoseconds = precompile_body_nanoseconds_since_reset,
        .precompile_terminal_nanoseconds = precompile_terminal_nanoseconds_since_reset,
        .precompile_deferred_nanoseconds = precompile_deferred_nanoseconds_since_reset,
        .precompile_metadata_nanoseconds = precompile_metadata_nanoseconds_since_reset,
        .precompile_link_nanoseconds = precompile_link_nanoseconds_since_reset,
    };
    emitted_blocks_since_reset = 0;
    emitted_bytes_since_reset = 0;
    invalidated_blocks_since_reset = 0;
    cache_clears_since_reset = 0;
    precompile_translate_nanoseconds_since_reset = 0;
    precompile_emit_nanoseconds_since_reset = 0;
    precompile_body_nanoseconds_since_reset = 0;
    precompile_terminal_nanoseconds_since_reset = 0;
    precompile_deferred_nanoseconds_since_reset = 0;
    precompile_metadata_nanoseconds_since_reset = 0;
    precompile_link_nanoseconds_since_reset = 0;
    return stats;
}

void AddressSpace::InvalidateBasicBlocks(const tsl::robin_set<IR::LocationDescriptor>& descriptors) {
    if (!code_write_batch_active) {
        UnprotectCodeMemory();
    }

    for (const auto& descriptor : descriptors) {
        const auto iter = block_entries.find(descriptor);
        if (iter == block_entries.end()) {
            continue;
        }

        ++invalidated_blocks_since_reset;

        // Unlink before removal because InvalidateBasicBlocks can be called within a fastmem callback,
        // and the currently executing block may have references to itself which need to be unlinked.
        RelinkForDescriptor(descriptor, nullptr);

        block_entries.erase(iter);
    }

    if (!code_write_batch_active) {
        ProtectCodeMemory();
    }
}

void AddressSpace::ClearCache() {
    ++cache_clears_since_reset;
    ClearReactivationMetadata();
    block_entries.clear();
    reverse_block_entries.clear();
    block_end_locations.clear();
    block_infos.clear();
    block_references.clear();
    deferred_precompile_entries.clear();
    code.set_offset(prelude_info.end_of_prelude);
}

void AddressSpace::DumpDisassembly() const {
    for (u32* ptr = mem.ptr(); ptr < code.xptr<u32*>(); ptr++) {
        std::printf("%s", Common::DisassembleAArch64(*ptr, mcl::bit_cast<u64>(ptr)).c_str());
    }
}

size_t AddressSpace::GetRemainingSize() {
    return code_cache_size - static_cast<size_t>(code.offset());
}

CodePtr AddressSpace::Emit(IR::Block block) {
    if (GetRemainingSize() < 1024 * 1024) {
        ClearCache();
    }

    if (!code_write_batch_active) {
        UnprotectCodeMemory();
    }

    const IR::LocationDescriptor location = block.Location();
    const IR::LocationDescriptor end_location = block.EndLocation();
    if (!cached_emit_config) {
        cached_emit_config.emplace(GetEmitConfig());
    }
    EmitTiming timing;
    EmittedBlockInfo block_info = EmitArm64(code, std::move(block), *cached_emit_config,
                                            fastmem_manager,
                                            precompile_batch_active ? &timing : nullptr);
    const auto metadata_started = precompile_batch_active ? std::chrono::steady_clock::now()
                                                          : std::chrono::steady_clock::time_point{};

    ++emitted_blocks_since_reset;
    emitted_bytes_since_reset += block_info.size;

    ASSERT(block_entries.insert({location, block_info.entry_point}).second);
    ASSERT(reverse_block_entries.insert({block_info.entry_point, location}).second);
    ASSERT(block_end_locations.insert({block_info.entry_point, end_location}).second);
    const CodePtr entry_point = block_info.entry_point;
    auto [stored_iter, inserted] = block_infos.emplace(entry_point, std::move(block_info));
    ASSERT(inserted);
    const EmittedBlockInfo& stored_info = stored_iter->second;

    if (precompile_batch_active) {
        deferred_precompile_entries.emplace_back(location, entry_point);
    } else {
        Link(stored_info);
        RelinkForDescriptor(location, entry_point);
    }

    if (!code_write_batch_active) {
        mem.invalidate(reinterpret_cast<u32*>(entry_point), stored_info.size);
        ProtectCodeMemory();
    }

    RegisterNewBasicBlock(block, stored_info);

    if (precompile_batch_active) {
        precompile_body_nanoseconds_since_reset += timing.body_nanoseconds;
        precompile_terminal_nanoseconds_since_reset += timing.terminal_nanoseconds;
        precompile_deferred_nanoseconds_since_reset += timing.deferred_nanoseconds;
        precompile_metadata_nanoseconds_since_reset += static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - metadata_started)
                .count());
    }

    return entry_point;
}

std::array<void*, static_cast<size_t>(LinkTarget::Count)> AddressSpace::GetLinkTargets() const {
    return {
        prelude_info.return_to_dispatcher,
        prelude_info.return_from_run_code,
        prelude_info.read_memory_8,
        prelude_info.read_memory_16,
        prelude_info.read_memory_32,
        prelude_info.read_memory_64,
        prelude_info.read_memory_128,
        prelude_info.wrapped_read_memory_8,
        prelude_info.wrapped_read_memory_16,
        prelude_info.wrapped_read_memory_32,
        prelude_info.wrapped_read_memory_64,
        prelude_info.wrapped_read_memory_128,
        prelude_info.exclusive_read_memory_8,
        prelude_info.exclusive_read_memory_16,
        prelude_info.exclusive_read_memory_32,
        prelude_info.exclusive_read_memory_64,
        prelude_info.exclusive_read_memory_128,
        prelude_info.write_memory_8,
        prelude_info.write_memory_16,
        prelude_info.write_memory_32,
        prelude_info.write_memory_64,
        prelude_info.write_memory_128,
        prelude_info.wrapped_write_memory_8,
        prelude_info.wrapped_write_memory_16,
        prelude_info.wrapped_write_memory_32,
        prelude_info.wrapped_write_memory_64,
        prelude_info.wrapped_write_memory_128,
        prelude_info.exclusive_write_memory_8,
        prelude_info.exclusive_write_memory_16,
        prelude_info.exclusive_write_memory_32,
        prelude_info.exclusive_write_memory_64,
        prelude_info.exclusive_write_memory_128,
        prelude_info.call_svc,
        prelude_info.exception_raised,
        prelude_info.isb_raised,
        prelude_info.ic_raised,
        prelude_info.dc_raised,
        prelude_info.get_cntpct,
        prelude_info.add_ticks,
        prelude_info.get_ticks_remaining,
    };
}

void AddressSpace::Link(const EmittedBlockInfo& block_info) {
    using namespace oaknut;
    using namespace oaknut::util;

    CodeGenerator c{mem.ptr(), mem.ptr()};

    for (const auto& [target_descriptor, list] : block_info.block_relocations) {
        block_references[target_descriptor].insert(block_info.entry_point);
        LinkBlockLinks(block_info.entry_point, Get(target_descriptor), list, c);
    }
}

void AddressSpace::LinkBlockLinks(const CodePtr entry_point, const CodePtr target_ptr, const std::vector<BlockRelocation>& block_relocations_list) {
    oaknut::CodeGenerator patcher{mem.ptr(), mem.ptr()};
    LinkBlockLinks(entry_point, target_ptr, block_relocations_list, patcher);
}

void AddressSpace::LinkBlockLinks(const CodePtr entry_point, const CodePtr target_ptr,
                                  const std::vector<BlockRelocation>& block_relocations_list,
                                  oaknut::CodeGenerator& patcher) {
    using namespace oaknut;
    using namespace oaknut::util;

    for (auto [ptr_offset, type] : block_relocations_list) {
        patcher.set_xptr(reinterpret_cast<u32*>(entry_point + ptr_offset));

        switch (type) {
        case BlockRelocationType::Branch:
            if (target_ptr) {
                patcher.B((void*)target_ptr);
            } else {
                patcher.NOP();
            }
            break;
        case BlockRelocationType::MoveToScratch1:
            if (target_ptr) {
                patcher.ADRL(Xscratch1, (void*)target_ptr);
            } else {
                patcher.ADRL(Xscratch1, prelude_info.return_to_dispatcher);
            }
            break;
        default:
            ASSERT_FALSE("Invalid BlockRelocationType");
        }
    }
}

void AddressSpace::RelinkForDescriptor(IR::LocationDescriptor target_descriptor, CodePtr target_ptr) {
    static const tsl::robin_set<CodePtr> no_exclusions;
    RelinkForDescriptorExcluding(target_descriptor, target_ptr, no_exclusions);
}

void AddressSpace::RelinkForDescriptorExcluding(
        IR::LocationDescriptor target_descriptor, CodePtr target_ptr,
        const tsl::robin_set<CodePtr>& excluded_entries) {
    for (auto code_ptr : block_references[target_descriptor]) {
        if (excluded_entries.contains(code_ptr)) {
            continue;
        }
        if (auto block_iter = block_infos.find(code_ptr); block_iter != block_infos.end()) {
            const EmittedBlockInfo& block_info = block_iter->second;

            if (auto relocation_iter = block_info.block_relocations.find(target_descriptor); relocation_iter != block_info.block_relocations.end()) {
                LinkBlockLinks(block_info.entry_point, target_ptr, relocation_iter->second);
            }

            if (!code_write_batch_active) {
                mem.invalidate(reinterpret_cast<u32*>(block_info.entry_point), block_info.size);
            }
        }
    }
}

FakeCall AddressSpace::FastmemCallback(u64 host_pc) {
    {
        const auto host_ptr = mcl::bit_cast<CodePtr>(host_pc);

        const auto entry_point = ReverseGetEntryPoint(host_ptr);
        if (!entry_point) {
            goto fail;
        }

        const auto block_info = block_infos.find(entry_point);
        if (block_info == block_infos.end()) {
            goto fail;
        }

        const auto patch_entry = block_info->second.fastmem_patch_info.find(host_ptr - entry_point);
        if (patch_entry == block_info->second.fastmem_patch_info.end()) {
            goto fail;
        }

        const auto fc = patch_entry->second.fc;

        if (patch_entry->second.recompile) {
            const auto marker = patch_entry->second.marker;
            fastmem_manager.MarkDoNotFastmem(marker);
            InvalidateBasicBlocks({std::get<0>(marker)});
        }

        return fc;
    }

fail:
    fmt::print("dynarmic: Segfault happened within JITted code at host_pc = {:016x}\n", host_pc);
    fmt::print("Segfault wasn't at a fastmem patch location!\n");
    ASSERT_FALSE("segfault");
}

}  // namespace Dynarmic::Backend::Arm64
