/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dynarmic/interface/A32/config.h"
#include "dynarmic/interface/halt_reason.h"

namespace Dynarmic {
namespace A32 {

struct JitCacheStats {
    std::uint64_t emitted_blocks{};
    std::uint64_t emitted_bytes{};
    std::uint64_t invalidated_blocks{};
    std::uint64_t cache_clears{};
    std::uint64_t live_blocks{};
    std::uint64_t used_bytes{};
    std::uint64_t capacity_bytes{};
    std::uint64_t precompile_translate_nanoseconds{};
    std::uint64_t precompile_emit_nanoseconds{};
    std::uint64_t precompile_body_nanoseconds{};
    std::uint64_t precompile_terminal_nanoseconds{};
    std::uint64_t precompile_deferred_nanoseconds{};
    std::uint64_t precompile_metadata_nanoseconds{};
    std::uint64_t precompile_link_nanoseconds{};
};

struct JitBlockCacheEntry {
    std::uint64_t descriptor{};
    std::uint64_t end_descriptor{};
    std::uint64_t guest_code_hash{};
    std::uintptr_t entry_point{};
};

class Jit final {
public:
    explicit Jit(UserConfig conf);
    ~Jit();

    /**
     * Runs the emulated CPU.
     * Cannot be recursively called.
     */
    HaltReason Run();

    /**
     * Steps the emulated CPU.
     * Cannot be recursively called.
     */
    HaltReason Step();

    /**
     * Clears the code cache of all compiled code.
     * Can be called at any time. Halts execution if called within a callback.
     */
    void ClearCache();

    /**
     * Invalidate the code cache at a range of addresses.
     * @param start_address The starting address of the range to invalidate.
     * @param length The length (in bytes) of the range to invalidate.
     */
    void InvalidateCacheRange(std::uint32_t start_address, std::size_t length);

    /// Like InvalidateCacheRange, but done before returning rather than at the next Run.
    /// Only while not executing. The ARM64 backend keeps the code, so the blocks can be
    /// reactivated (ReactivateBlocks); the other backends defer as InvalidateCacheRange does.
    void InvalidateCacheRangeNow(std::uint32_t start_address, std::size_t length);

    /**
     * Reset CPU state to state at startup. Does not clear code cache.
     * Cannot be called from a callback.
     */
    void Reset();

    /**
     * Stops execution in Jit::Run.
     */
    void HaltExecution(HaltReason hr = HaltReason::UserDefined1);

    /**
     * Clears a halt reason from flags.
     * Warning: Only use this if you're sure this won't introduce races.
     */
    void ClearHalt(HaltReason hr = HaltReason::UserDefined1);

    /// View and modify registers.
    std::array<std::uint32_t, 16>& Regs();
    const std::array<std::uint32_t, 16>& Regs() const;
    std::array<std::uint32_t, 64>& ExtRegs();
    const std::array<std::uint32_t, 64>& ExtRegs() const;

    /// View and modify CPSR.
    std::uint32_t Cpsr() const;
    void SetCpsr(std::uint32_t value);

    /// View and modify FPSCR.
    std::uint32_t Fpscr() const;
    void SetFpscr(std::uint32_t value);

    /// Clears exclusive state for this core.
    void ClearExclusiveState();

    /**
     * Returns true if Jit::Run was called but hasn't returned yet.
     * i.e.: We're in a callback.
     */
    bool IsExecuting() const {
        return is_executing;
    }

    /// Debugging: Dump a disassembly all compiled code to the console.
    void DumpDisassembly() const;

    /**
     * Disassemble the instructions following the current pc and return
     * the resulting instructions as a vector of their string representations.
     */
    std::vector<std::string> Disassemble() const;

    /**
     * Compiles previously observed guest block descriptors without executing them.
     * This is intended for validated frontend-managed persistent caches.
     */
    std::size_t Precompile(const std::vector<std::uint64_t>& descriptors);

    /// Returns descriptors for blocks currently present in the host code cache.
    std::vector<std::uint64_t> GetCompiledBlockDescriptors() const;
    /// Those starting in [start_address, start_address + length) where the backend can find them
    /// by range (arm64); other backends return all of them, so callers still filter.
    std::vector<std::uint64_t> GetCompiledBlockDescriptors(std::uint32_t start_address, std::size_t length) const;

    /// Captures active guest descriptors and their retained host-code entry points.
    std::vector<JitBlockCacheEntry> GetCompiledBlockEntries() const;
    /// Only the blocks starting in [start_address, start_address + length): cheaper, since
    /// each entry's instruction hash is computed only for those.
    std::vector<JitBlockCacheEntry> GetCompiledBlockEntries(std::uint32_t start_address, std::size_t length) const;

    /// Reactivates retained host code after its identical guest code has been mapped again.
    std::size_t ReactivateBlocks(const std::vector<JitBlockCacheEntry>& entries);

    /// Returns interval counters and current code-cache occupancy, then resets interval counters.
    JitCacheStats GetAndResetCacheStats();

private:
    bool is_executing = false;

    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace A32
}  // namespace Dynarmic
