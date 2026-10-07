/* This file is part of the dynarmic project.
 * Copyright (c) 2021 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <memory>
#include <mutex>

#include <boost/icl/interval_set.hpp>
#include <mcl/assert.hpp>
#include <mcl/scope_exit.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/backend/arm64/a32_address_space.h"
#include "dynarmic/backend/arm64/a32_core.h"
#include "dynarmic/backend/arm64/a32_jitstate.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/interface/A32/a32.h"

namespace Dynarmic::A32 {

using namespace Backend::Arm64;

struct Jit::Impl final {
    Impl(Jit* jit_interface, A32::UserConfig conf)
            : jit_interface(jit_interface)
            , conf(conf)
            , current_address_space(conf)
            , core(conf) {}

    HaltReason Run() {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&halt_reason)));

        jit_interface->is_executing = true;
        SCOPE_EXIT {
            jit_interface->is_executing = false;
        };

        HaltReason hr = core.Run(current_address_space, current_state, &halt_reason);

        PerformRequestedCacheInvalidation(hr);

        return hr;
    }

    HaltReason Step() {
        ASSERT(!jit_interface->is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&halt_reason)));

        jit_interface->is_executing = true;
        SCOPE_EXIT {
            jit_interface->is_executing = false;
        };

        HaltReason hr = core.Step(current_address_space, current_state, &halt_reason);

        PerformRequestedCacheInvalidation(hr);

        return hr;
    }

    void ClearCache() {
        std::unique_lock lock{invalidation_mutex};
        invalidate_entire_cache = true;
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void InvalidateCacheRange(std::uint32_t start_address, std::size_t length) {
        std::unique_lock lock{invalidation_mutex};
        invalid_cache_ranges.add(boost::icl::discrete_interval<u32>::closed(start_address, static_cast<u32>(start_address + length - 1)));
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void Reset() {
        current_state = {};
    }

    void HaltExecution(HaltReason hr) {
        Atomic::Or(&halt_reason, static_cast<u32>(hr));
        Atomic::Barrier();
    }

    void ClearHalt(HaltReason hr) {
        Atomic::And(&halt_reason, ~static_cast<u32>(hr));
        Atomic::Barrier();
    }

    std::array<std::uint32_t, 16>& Regs() {
        return current_state.regs;
    }

    const std::array<std::uint32_t, 16>& Regs() const {
        return current_state.regs;
    }

    std::array<std::uint32_t, 64>& ExtRegs() {
        return current_state.ext_regs;
    }

    const std::array<std::uint32_t, 64>& ExtRegs() const {
        return current_state.ext_regs;
    }

    std::uint32_t Cpsr() const {
        return current_state.Cpsr();
    }

    void SetCpsr(std::uint32_t value) {
        current_state.SetCpsr(value);
    }

    std::uint32_t Fpscr() const {
        return current_state.Fpscr();
    }

    void SetFpscr(std::uint32_t value) {
        current_state.SetFpscr(value);
    }

    void ClearExclusiveState() {
        current_state.exclusive_state = false;
    }

    void DumpDisassembly() const {
        ASSERT_FALSE("Unimplemented");
    }

    std::size_t Precompile(const std::vector<std::uint64_t>& descriptors) {
        ASSERT(!jit_interface->is_executing);
        // None of these blocks can execute before Precompile returns. Defer the expensive ARM
        // instruction-cache barriers until the complete batch has been emitted and linked.
        current_address_space.BeginPrecompileBatch();
        current_address_space.SetFastPrecompile(true);
        SCOPE_EXIT {
            current_address_space.SetFastPrecompile(false);
            current_address_space.EndPrecompileBatch();
        };
        return current_address_space.PrecompileBlocks(descriptors);
    }

    std::vector<std::uint64_t> GetCompiledBlockDescriptors() const {
        return current_address_space.GetCompiledBlockDescriptors();
    }

    std::vector<JitBlockCacheEntry> GetCompiledBlockEntries() const {
        const auto entries = current_address_space.GetCompiledBlockEntries();
        std::vector<JitBlockCacheEntry> result;
        result.reserve(entries.size());
        for (const auto& entry : entries) {
            const auto guest_code_hash =
                current_address_space.GuestCodeHash(entry.descriptor, entry.end_descriptor);
            if (guest_code_hash) {
                result.push_back({entry.descriptor, entry.end_descriptor, *guest_code_hash,
                                  reinterpret_cast<std::uintptr_t>(entry.entry_point)});
            }
        }
        return result;
    }

    std::size_t ReactivateBlocks(const std::vector<JitBlockCacheEntry>& entries) {
        ASSERT(!jit_interface->is_executing);
        std::vector<Backend::Arm64::CachedBlockEntry> backend_entries;
        backend_entries.reserve(entries.size());
        for (const auto& entry : entries) {
            if (current_address_space.GuestCodeHash(entry.descriptor, entry.end_descriptor) ==
                entry.guest_code_hash) {
                backend_entries.push_back({entry.descriptor, entry.end_descriptor,
                                           reinterpret_cast<CodePtr>(entry.entry_point)});
            }
        }
        return current_address_space.ReactivateBlocks(backend_entries);
    }

    JitCacheStats GetAndResetCacheStats() {
        const auto stats = current_address_space.GetAndResetCacheStats();
        return {
            .emitted_blocks = stats.emitted_blocks,
            .emitted_bytes = stats.emitted_bytes,
            .invalidated_blocks = stats.invalidated_blocks,
            .cache_clears = stats.cache_clears,
            .live_blocks = stats.live_blocks,
            .used_bytes = stats.used_bytes,
            .capacity_bytes = stats.capacity_bytes,
            .precompile_translate_nanoseconds = stats.precompile_translate_nanoseconds,
            .precompile_emit_nanoseconds = stats.precompile_emit_nanoseconds,
            .precompile_body_nanoseconds = stats.precompile_body_nanoseconds,
            .precompile_terminal_nanoseconds = stats.precompile_terminal_nanoseconds,
            .precompile_deferred_nanoseconds = stats.precompile_deferred_nanoseconds,
            .precompile_metadata_nanoseconds = stats.precompile_metadata_nanoseconds,
            .precompile_link_nanoseconds = stats.precompile_link_nanoseconds,
        };
    }

private:
    void PerformRequestedCacheInvalidation(HaltReason hr) {
        if (Has(hr, HaltReason::CacheInvalidation)) {
            std::unique_lock lock{invalidation_mutex};

            ClearHalt(HaltReason::CacheInvalidation);

            if (invalidate_entire_cache) {
                current_address_space.ClearCache();

                invalidate_entire_cache = false;
                invalid_cache_ranges.clear();
                return;
            }

            if (!invalid_cache_ranges.empty()) {
                current_address_space.InvalidateCacheRanges(invalid_cache_ranges);

                invalid_cache_ranges.clear();
                return;
            }
        }
    }

    Jit* jit_interface;
    A32::UserConfig conf;
    A32JitState current_state{};
    A32AddressSpace current_address_space;
    A32Core core;

    volatile u32 halt_reason = 0;

    std::mutex invalidation_mutex;
    boost::icl::interval_set<u32> invalid_cache_ranges;
    bool invalidate_entire_cache = false;
};

Jit::Jit(UserConfig conf)
        : impl(std::make_unique<Impl>(this, conf)) {}

Jit::~Jit() = default;

HaltReason Jit::Run() {
    return impl->Run();
}

HaltReason Jit::Step() {
    return impl->Step();
}

void Jit::ClearCache() {
    impl->ClearCache();
}

void Jit::InvalidateCacheRange(std::uint32_t start_address, std::size_t length) {
    impl->InvalidateCacheRange(start_address, length);
}

void Jit::Reset() {
    impl->Reset();
}

void Jit::HaltExecution(HaltReason hr) {
    impl->HaltExecution(hr);
}

void Jit::ClearHalt(HaltReason hr) {
    impl->ClearHalt(hr);
}

std::array<std::uint32_t, 16>& Jit::Regs() {
    return impl->Regs();
}

const std::array<std::uint32_t, 16>& Jit::Regs() const {
    return impl->Regs();
}

std::array<std::uint32_t, 64>& Jit::ExtRegs() {
    return impl->ExtRegs();
}

const std::array<std::uint32_t, 64>& Jit::ExtRegs() const {
    return impl->ExtRegs();
}

std::uint32_t Jit::Cpsr() const {
    return impl->Cpsr();
}

void Jit::SetCpsr(std::uint32_t value) {
    impl->SetCpsr(value);
}

std::uint32_t Jit::Fpscr() const {
    return impl->Fpscr();
}

void Jit::SetFpscr(std::uint32_t value) {
    impl->SetFpscr(value);
}

void Jit::ClearExclusiveState() {
    impl->ClearExclusiveState();
}

void Jit::DumpDisassembly() const {
    impl->DumpDisassembly();
}

std::size_t Jit::Precompile(const std::vector<std::uint64_t>& descriptors) {
    return impl->Precompile(descriptors);
}

std::vector<std::uint64_t> Jit::GetCompiledBlockDescriptors() const {
    return impl->GetCompiledBlockDescriptors();
}

std::vector<JitBlockCacheEntry> Jit::GetCompiledBlockEntries() const {
    return impl->GetCompiledBlockEntries();
}

std::size_t Jit::ReactivateBlocks(const std::vector<JitBlockCacheEntry>& entries) {
    return impl->ReactivateBlocks(entries);
}

JitCacheStats Jit::GetAndResetCacheStats() {
    return impl->GetAndResetCacheStats();
}

}  // namespace Dynarmic::A32
