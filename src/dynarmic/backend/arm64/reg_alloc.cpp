/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/reg_alloc.h"

#include <algorithm>
#include <array>
#include <bit>
#include <iterator>

#include <mcl/assert.hpp>
#include <mcl/bit/bit_field.hpp>
#include <mcl/bit_cast.hpp>
#include <mcl/mp/metavalue/lift_value.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/backend/arm64/abi.h"
#include "dynarmic/backend/arm64/emit_context.h"
#include "dynarmic/backend/arm64/fpsr_manager.h"
#include "dynarmic/backend/arm64/verbose_debugging_output.h"
#include "dynarmic/common/always_false.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

constexpr size_t spill_offset = offsetof(StackLayout, spill);
constexpr size_t spill_slot_size = sizeof(decltype(StackLayout::spill)::value_type);

static u8 EncodeValueLocation(HostLoc host_loc) {
    switch (host_loc.kind) {
    case HostLoc::Kind::Gpr:
        ASSERT(host_loc.index >= 0 && host_loc.index < 32);
        return static_cast<u8>(host_loc.index + 1);
    case HostLoc::Kind::Fpr:
        ASSERT(host_loc.index >= 0 && host_loc.index < 32);
        return static_cast<u8>(host_loc.index + 33);
    case HostLoc::Kind::Flags:
        ASSERT(host_loc.index == 0);
        return 65;
    case HostLoc::Kind::Spill:
        ASSERT(host_loc.index >= 0 && host_loc.index < static_cast<int>(SpillCount));
        return static_cast<u8>(host_loc.index + 66);
    }
    ASSERT_FALSE("Invalid host location");
}

static HostLoc DecodeValueLocation(u8 encoded) {
    ASSERT(encoded != 0 && encoded <= 129);
    if (encoded <= 32) {
        return HostLoc{HostLoc::Kind::Gpr, encoded - 1};
    }
    if (encoded <= 64) {
        return HostLoc{HostLoc::Kind::Fpr, encoded - 33};
    }
    if (encoded == 65) {
        return HostLoc{HostLoc::Kind::Flags, 0};
    }
    return HostLoc{HostLoc::Kind::Spill, encoded - 66};
}

static bool IsValuelessType(IR::Type type) {
    switch (type) {
    case IR::Type::Table:
        return true;
    default:
        return false;
    }
}

IR::Type Argument::GetType() const {
    return value.GetType();
}

bool Argument::IsImmediate() const {
    return value.IsImmediate();
}

bool Argument::GetImmediateU1() const {
    return value.GetU1();
}

u8 Argument::GetImmediateU8() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x100);
    return u8(imm);
}

u16 Argument::GetImmediateU16() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x10000);
    return u16(imm);
}

u32 Argument::GetImmediateU32() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x100000000);
    return u32(imm);
}

u64 Argument::GetImmediateU64() const {
    return value.GetImmediateAsU64();
}

IR::Cond Argument::GetImmediateCond() const {
    ASSERT(IsImmediate() && GetType() == IR::Type::Cond);
    return value.GetCond();
}

IR::AccType Argument::GetImmediateAccType() const {
    ASSERT(IsImmediate() && GetType() == IR::Type::AccType);
    return value.GetAccType();
}

HostLoc::Kind Argument::CurrentLocationKind() const {
    return reg_alloc.ValueLocation(value.GetInst())->kind;
}

bool HostLocInfo::Contains(const IR::Inst* value) const {
    return primary_value == value || secondary_value == value ||
           std::find(aliases.begin(), aliases.end(), value) != aliases.end();
}

void HostLocInfo::AddValue(const IR::Inst* value) {
    if (!primary_value) {
        primary_value = value;
    } else if (!secondary_value) {
        secondary_value = value;
    } else {
        aliases.push_back(value);
    }
}

void HostLocInfo::ClearValues() {
    primary_value = nullptr;
    secondary_value = nullptr;
    aliases.clear();
}

bool HostLocInfo::ValuesEmpty() const {
    return primary_value == nullptr;
}

void HostLocInfo::SetupScratchLocation() {
    ASSERT(IsCompletelyEmpty());
    realized = true;
}

void HostLocInfo::SetupLocation(const IR::Inst* value) {
    ASSERT(IsCompletelyEmpty());
    ClearValues();
    AddValue(value);
    realized = true;
    uses_this_inst = 0;
    accumulated_uses = 0;
    expected_uses = value->UseCount();
}

bool HostLocInfo::IsCompletelyEmpty() const {
    return ValuesEmpty() && !locked && !realized && !accumulated_uses && !expected_uses && !uses_this_inst && !pending_update;
}

bool HostLocInfo::MaybeAllocatable() const {
    return !locked && !realized;
}

bool HostLocInfo::IsOneRemainingUse() const {
    return accumulated_uses + 1 == expected_uses && uses_this_inst == 1;
}

void HostLocInfo::UpdateUses() {
    accumulated_uses += uses_this_inst;
    uses_this_inst = 0;

    if (accumulated_uses == expected_uses) {
        ClearValues();
        accumulated_uses = 0;
        expected_uses = 0;
    }
}

RegAlloc::ArgumentInfo RegAlloc::GetArgumentInfo(IR::Inst* inst) {
    ArgumentInfo ret = {Argument{*this}, Argument{*this}, Argument{*this}, Argument{*this}};
    for (size_t i = 0; i < inst->NumArgs(); i++) {
        const IR::Value arg = inst->GetArg(i);
        ret[i].value = arg;
        if (!arg.IsImmediate() && !IsValuelessType(arg.GetType())) {
            ASSERT_MSG(ValueLocation(arg.GetInst()), "argument must already been defined");
            auto& info = ValueInfo(arg.GetInst());
            info.uses_this_inst++;
            MarkPendingUpdate(info);
        }
    }
    return ret;
}

bool RegAlloc::WasValueDefined(IR::Inst* inst) const {
    const unsigned name = inst->GetName();
    if (name != 0) {
        const size_t bit_index = name - 1;
        const size_t word_index = bit_index / 64;
        if (word_index < defined_inst_bits.size()) {
            return (defined_inst_bits[word_index] & (u64{1} << (bit_index % 64))) != 0;
        }
    }
    return std::find(fallback_defined_insts.begin(), fallback_defined_insts.end(), inst) != fallback_defined_insts.end();
}

void RegAlloc::MarkValueDefined(const IR::Inst* inst) {
    const unsigned name = inst->GetName();
    if (name != 0) {
        const size_t bit_index = name - 1;
        const size_t word_index = bit_index / 64;
        if (word_index < defined_inst_bits.size()) {
            defined_inst_bits[word_index] |= u64{1} << (bit_index % 64);
            return;
        }
    }
    if (std::find(fallback_defined_insts.begin(), fallback_defined_insts.end(), inst) == fallback_defined_insts.end()) {
        fallback_defined_insts.push_back(inst);
    }
}

void RegAlloc::PrepareForCall(std::optional<Argument::copyable_reference> arg0, std::optional<Argument::copyable_reference> arg1, std::optional<Argument::copyable_reference> arg2, std::optional<Argument::copyable_reference> arg3) {
    fpsr_manager.Spill();
    SpillFlags();

    // TODO: Spill into callee-save registers

    for (int i = 0; i < 32; i++) {
        if (mcl::bit::get_bit(i, static_cast<u32>(ABI_CALLER_SAVE))) {
            SpillGpr(i);
        }
    }

    for (int i = 0; i < 32; i++) {
        if (mcl::bit::get_bit(i, static_cast<u32>(ABI_CALLER_SAVE >> 32))) {
            SpillFpr(i);
        }
    }

    const std::array<std::optional<Argument::copyable_reference>, 4> args{arg0, arg1, arg2, arg3};

    // AAPCS64 Next General-purpose Register Number
    int ngrn = 0;
    // AAPCS64 Next SIMD and Floating-point Register Number
    int nsrn = 0;

    for (int i = 0; i < 4; i++) {
        if (args[i]) {
            if (args[i]->get().GetType() == IR::Type::U128) {
                ASSERT(fprs[nsrn].IsCompletelyEmpty());
                LoadCopyInto(args[i]->get().value, oaknut::QReg{nsrn});
                nsrn++;
            } else {
                ASSERT(gprs[ngrn].IsCompletelyEmpty());
                LoadCopyInto(args[i]->get().value, oaknut::XReg{ngrn});
                ngrn++;
            }
        } else {
            // Gaps are assumed to be in general-purpose registers
            // TODO: should there be a separate list passed for FPRs instead?
            ngrn++;
        }
    }
}

void RegAlloc::DefineAsExisting(IR::Inst* inst, Argument& arg) {
    MarkValueDefined(inst);

    ASSERT(!ValueLocation(inst));

    if (arg.value.IsImmediate()) {
        inst->ReplaceUsesWith(arg.value);
        return;
    }

    const auto host_loc = ValueLocation(arg.value.GetInst());
    ASSERT(host_loc);
    auto& info = ValueInfo(*host_loc);
    info.AddValue(inst);
    info.expected_uses += inst->UseCount();
    SetValueLocation(inst, *host_loc);
    MarkPendingUpdate(info);
}

void RegAlloc::DefineAsRegister(IR::Inst* inst, oaknut::Reg reg) {
    MarkValueDefined(inst);

    ASSERT(!ValueLocation(inst));
    auto& info = reg.is_vector() ? fprs[reg.index()] : gprs[reg.index()];
    ASSERT(info.IsCompletelyEmpty());
    info.AddValue(inst);
    info.expected_uses += inst->UseCount();
    SetValueLocation(inst, HostLoc{reg.is_vector() ? HostLoc::Kind::Fpr : HostLoc::Kind::Gpr,
                                    reg.index()});
    MarkPendingUpdate(info);
}

void RegAlloc::UpdateAllUses() {
    for (size_t i = 0; i < pending_update_count; i++) {
        auto& info = *pending_updates[i];
        ASSERT(info.pending_update);
        if (info.accumulated_uses + info.uses_this_inst == info.expected_uses) {
            if (!info.ValuesEmpty()) {
                const auto location = ValueLocation(info.primary_value);
                ASSERT(location);
                if (location->kind == HostLoc::Kind::Spill) {
                    occupied_spills &= ~(u64{1} << location->index);
                }
                ClearValueLocation(info.primary_value);
                if (info.secondary_value) {
                    ClearValueLocation(info.secondary_value);
                }
                for (const IR::Inst* alias : info.aliases) {
                    ClearValueLocation(alias);
                }
            }
        }
        info.UpdateUses();
        info.pending_update = false;
    }
    pending_update_count = 0;
}

void RegAlloc::MarkPendingUpdate(HostLocInfo& info) {
    if (info.pending_update) {
        return;
    }

    ASSERT(pending_update_count < pending_updates.size());
    info.pending_update = true;
    pending_updates[pending_update_count++] = &info;
}

void RegAlloc::MoveLocation(HostLoc target_location, HostLoc source_location) {
    auto& target = ValueInfo(target_location);
    auto& source = ValueInfo(source_location);
    ASSERT(target.IsCompletelyEmpty());
    if (source.primary_value) {
        SetValueLocation(source.primary_value, target_location);
    }
    if (source.secondary_value) {
        SetValueLocation(source.secondary_value, target_location);
    }
    for (const IR::Inst* alias : source.aliases) {
        SetValueLocation(alias, target_location);
    }
    if (source.pending_update) {
        const auto end = pending_updates.begin() + static_cast<ptrdiff_t>(pending_update_count);
        const auto iter = std::find(pending_updates.begin(), end, &source);
        ASSERT(iter != end);
        *iter = &target;
    }
    if (source_location.kind == HostLoc::Kind::Spill) {
        occupied_spills &= ~(u64{1} << source_location.index);
    }
    if (target_location.kind == HostLoc::Kind::Spill) {
        occupied_spills |= u64{1} << target_location.index;
    }
    target = std::exchange(source, {});
}

void RegAlloc::AssertAllUnlocked() const {
    const auto is_unlocked = [](const auto& i) { return !i.locked && !i.realized; };
    ASSERT(std::all_of(gprs.begin(), gprs.end(), is_unlocked));
    ASSERT(std::all_of(fprs.begin(), fprs.end(), is_unlocked));
    ASSERT(is_unlocked(flags));
    ASSERT(std::all_of(spills.begin(), spills.end(), is_unlocked));
}

void RegAlloc::AssertNoMoreUses() const {
    const auto is_empty = [](const auto& i) { return i.IsCompletelyEmpty(); };
    ASSERT(std::all_of(gprs.begin(), gprs.end(), is_empty));
    ASSERT(std::all_of(fprs.begin(), fprs.end(), is_empty));
    ASSERT(is_empty(flags));
    ASSERT(std::all_of(spills.begin(), spills.end(), is_empty));
}

void RegAlloc::EmitVerboseDebuggingOutput() {
    code.MOV(X19, mcl::bit_cast<u64>(&PrintVerboseDebuggingOutputLine));  // Non-volatile register

    const auto do_location = [&](HostLocInfo& info, HostLocType type, size_t index) {
        using namespace oaknut::util;
        const auto emit_value = [&](const IR::Inst* value) {
            code.MOV(X0, SP);
            code.MOV(X1, static_cast<u64>(type));
            code.MOV(X2, index);
            code.MOV(X3, value->GetName());
            code.MOV(X4, static_cast<u64>(value->GetType()));
            code.BLR(X19);
        };
        if (info.primary_value) {
            emit_value(info.primary_value);
        }
        if (info.secondary_value) {
            emit_value(info.secondary_value);
        }
        for (const IR::Inst* value : info.aliases) {
            emit_value(value);
        }
    };

    for (size_t i = 0; i < gprs.size(); i++) {
        do_location(gprs[i], HostLocType::X, i);
    }
    for (size_t i = 0; i < fprs.size(); i++) {
        do_location(fprs[i], HostLocType::Q, i);
    }
    do_location(flags, HostLocType::Nzcv, 0);
    for (size_t i = 0; i < spills.size(); i++) {
        do_location(spills[i], HostLocType::Spill, i);
    }
}

template<HostLoc::Kind kind>
int RegAlloc::GenerateImmediate(const IR::Value& value) {
    ASSERT(value.GetType() != IR::Type::U1);
    if constexpr (kind == HostLoc::Kind::Gpr) {
        const int new_location_index = AllocateRegister(gprs, gpr_order);
        SpillGpr(new_location_index);
        gprs[new_location_index].SetupScratchLocation();

        code.MOV(oaknut::XReg{new_location_index}, value.GetImmediateAsU64());

        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Fpr) {
        const int new_location_index = AllocateRegister(fprs, fpr_order);
        SpillFpr(new_location_index);
        fprs[new_location_index].SetupScratchLocation();

        code.MOV(Xscratch0, value.GetImmediateAsU64());
        code.FMOV(oaknut::DReg{new_location_index}, Xscratch0);

        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Flags) {
        SpillFlags();
        flags.SetupScratchLocation();

        code.MOV(Xscratch0, value.GetImmediateAsU64());
        code.MSR(oaknut::SystemReg::NZCV, Xscratch0);

        return 0;
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<kind>>);
    }
}

template<HostLoc::Kind required_kind>
int RegAlloc::RealizeReadImpl(const IR::Value& value) {
    if (value.IsImmediate()) {
        return GenerateImmediate<required_kind>(value);
    }

    const auto current_location = ValueLocation(value.GetInst());
    ASSERT(current_location);

    if (current_location->kind == required_kind) {
        ValueInfo(*current_location).realized = true;
        return current_location->index;
    }

    ASSERT(!ValueInfo(*current_location).realized);
    ASSERT(ValueInfo(*current_location).locked);

    if constexpr (required_kind == HostLoc::Kind::Gpr) {
        const int new_location_index = AllocateRegister(gprs, gpr_order);
        SpillGpr(new_location_index);

        switch (current_location->kind) {
        case HostLoc::Kind::Gpr:
            ASSERT_FALSE("Logic error");
            break;
        case HostLoc::Kind::Fpr:
            code.FMOV(oaknut::XReg{new_location_index}, oaknut::DReg{current_location->index});
            // ASSERT size fits
            break;
        case HostLoc::Kind::Spill:
            code.LDR(oaknut::XReg{new_location_index}, SP, spill_offset + current_location->index * spill_slot_size);
            break;
        case HostLoc::Kind::Flags:
            code.MRS(oaknut::XReg{new_location_index}, oaknut::SystemReg::NZCV);
            break;
        }

        MoveLocation(HostLoc{HostLoc::Kind::Gpr, new_location_index}, *current_location);
        gprs[new_location_index].realized = true;
        return new_location_index;
    } else if constexpr (required_kind == HostLoc::Kind::Fpr) {
        const int new_location_index = AllocateRegister(fprs, fpr_order);
        SpillFpr(new_location_index);

        switch (current_location->kind) {
        case HostLoc::Kind::Gpr:
            code.FMOV(oaknut::DReg{new_location_index}, oaknut::XReg{current_location->index});
            break;
        case HostLoc::Kind::Fpr:
            ASSERT_FALSE("Logic error");
            break;
        case HostLoc::Kind::Spill:
            code.LDR(oaknut::QReg{new_location_index}, SP, spill_offset + current_location->index * spill_slot_size);
            break;
        case HostLoc::Kind::Flags:
            ASSERT_FALSE("Moving from flags into fprs is not currently supported");
            break;
        }

        MoveLocation(HostLoc{HostLoc::Kind::Fpr, new_location_index}, *current_location);
        fprs[new_location_index].realized = true;
        return new_location_index;
    } else if constexpr (required_kind == HostLoc::Kind::Flags) {
        ASSERT_FALSE("A simple read from flags is likely a logic error.");
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<required_kind>>);
    }
}

template<HostLoc::Kind kind>
int RegAlloc::RealizeWriteImpl(const IR::Inst* value) {
    MarkValueDefined(value);

    ASSERT(!ValueLocation(value));

    if constexpr (kind == HostLoc::Kind::Gpr) {
        const int new_location_index = AllocateRegister(gprs, gpr_order);
        SpillGpr(new_location_index);
        gprs[new_location_index].SetupLocation(value);
        SetValueLocation(value, HostLoc{HostLoc::Kind::Gpr, new_location_index});
        MarkPendingUpdate(gprs[new_location_index]);
        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Fpr) {
        const int new_location_index = AllocateRegister(fprs, fpr_order);
        SpillFpr(new_location_index);
        fprs[new_location_index].SetupLocation(value);
        SetValueLocation(value, HostLoc{HostLoc::Kind::Fpr, new_location_index});
        MarkPendingUpdate(fprs[new_location_index]);
        return new_location_index;
    } else if constexpr (kind == HostLoc::Kind::Flags) {
        SpillFlags();
        flags.SetupLocation(value);
        SetValueLocation(value, HostLoc{HostLoc::Kind::Flags, 0});
        MarkPendingUpdate(flags);
        return 0;
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<kind>>);
    }
}

template<HostLoc::Kind kind>
int RegAlloc::RealizeReadWriteImpl(const IR::Value& read_value, const IR::Inst* write_value) {
    MarkValueDefined(write_value);

    // TODO: Move elimination

    const int write_loc = RealizeWriteImpl<kind>(write_value);

    if constexpr (kind == HostLoc::Kind::Gpr) {
        LoadCopyInto(read_value, oaknut::XReg{write_loc});
        return write_loc;
    } else if constexpr (kind == HostLoc::Kind::Fpr) {
        LoadCopyInto(read_value, oaknut::QReg{write_loc});
        return write_loc;
    } else if constexpr (kind == HostLoc::Kind::Flags) {
        ASSERT_FALSE("Incorrect function for ReadWrite of flags");
    } else {
        static_assert(Common::always_false_v<mcl::mp::lift_value<kind>>);
    }
}

template int RegAlloc::RealizeReadImpl<HostLoc::Kind::Gpr>(const IR::Value& value);
template int RegAlloc::RealizeReadImpl<HostLoc::Kind::Fpr>(const IR::Value& value);
template int RegAlloc::RealizeReadImpl<HostLoc::Kind::Flags>(const IR::Value& value);
template int RegAlloc::RealizeWriteImpl<HostLoc::Kind::Gpr>(const IR::Inst* value);
template int RegAlloc::RealizeWriteImpl<HostLoc::Kind::Fpr>(const IR::Inst* value);
template int RegAlloc::RealizeWriteImpl<HostLoc::Kind::Flags>(const IR::Inst* value);
template int RegAlloc::RealizeReadWriteImpl<HostLoc::Kind::Gpr>(const IR::Value&, const IR::Inst*);
template int RegAlloc::RealizeReadWriteImpl<HostLoc::Kind::Fpr>(const IR::Value&, const IR::Inst*);
template int RegAlloc::RealizeReadWriteImpl<HostLoc::Kind::Flags>(const IR::Value&, const IR::Inst*);

int RegAlloc::AllocateRegister(const std::array<HostLocInfo, 32>& regs, std::span<const int> order) const {
    const auto empty = std::find_if(order.begin(), order.end(), [&](int i) { return regs[i].IsCompletelyEmpty(); });
    if (empty != order.end()) {
        return *empty;
    }

    const size_t candidate_count = static_cast<size_t>(
        std::count_if(order.begin(), order.end(), [&](int i) { return regs[i].MaybeAllocatable(); }));
    ASSERT(candidate_count != 0);
    size_t selected_candidate = alloc_candidate_index++ % candidate_count;

    // TODO: The candidate was chosen randomly before, and a LRU was
    // suggested as an improvement. However, using an incrementing index
    // seems to be close enough. Determine if an LRU is still needed.
    for (const int i : order) {
        if (!regs[i].MaybeAllocatable()) {
            continue;
        }
        if (selected_candidate-- == 0) {
            return i;
        }
    }

    ASSERT_FALSE("Register-allocation candidate disappeared");
}

void RegAlloc::SpillGpr(int index) {
    ASSERT(!gprs[index].locked && !gprs[index].realized);
    if (gprs[index].ValuesEmpty()) {
        return;
    }
    const int new_location_index = FindFreeSpill();
    code.STR(oaknut::XReg{index}, SP, spill_offset + new_location_index * spill_slot_size);
    MoveLocation(HostLoc{HostLoc::Kind::Spill, new_location_index},
                 HostLoc{HostLoc::Kind::Gpr, index});
}

void RegAlloc::SpillFpr(int index) {
    ASSERT(!fprs[index].locked && !fprs[index].realized);
    if (fprs[index].ValuesEmpty()) {
        return;
    }
    const int new_location_index = FindFreeSpill();
    code.STR(oaknut::QReg{index}, SP, spill_offset + new_location_index * spill_slot_size);
    MoveLocation(HostLoc{HostLoc::Kind::Spill, new_location_index},
                 HostLoc{HostLoc::Kind::Fpr, index});
}

void RegAlloc::ReadWriteFlags(Argument& read, IR::Inst* write) {
    if (write) {
        MarkValueDefined(write);
    }

    const auto current_location = ValueLocation(read.value.GetInst());
    ASSERT(current_location);

    if (current_location->kind == HostLoc::Kind::Flags) {
        if (!flags.IsOneRemainingUse()) {
            SpillFlags();
        }
    } else if (current_location->kind == HostLoc::Kind::Gpr) {
        if (!flags.ValuesEmpty()) {
            SpillFlags();
        }
        code.MSR(oaknut::SystemReg::NZCV, oaknut::XReg{current_location->index});
    } else if (current_location->kind == HostLoc::Kind::Spill) {
        if (!flags.ValuesEmpty()) {
            SpillFlags();
        }
        code.LDR(Wscratch0, SP, spill_offset + current_location->index * spill_slot_size);
        code.MSR(oaknut::SystemReg::NZCV, Xscratch0);
    } else {
        ASSERT_FALSE("Invalid current location for flags");
    }

    if (write) {
        flags.SetupLocation(write);
        SetValueLocation(write, HostLoc{HostLoc::Kind::Flags, 0});
        MarkPendingUpdate(flags);
        flags.realized = false;
    }
}

void RegAlloc::SpillFlags() {
    ASSERT(!flags.locked && !flags.realized);
    if (flags.ValuesEmpty()) {
        return;
    }
    const int new_location_index = AllocateRegister(gprs, gpr_order);
    SpillGpr(new_location_index);
    code.MRS(oaknut::XReg{new_location_index}, oaknut::SystemReg::NZCV);
    MoveLocation(HostLoc{HostLoc::Kind::Gpr, new_location_index},
                 HostLoc{HostLoc::Kind::Flags, 0});
}

int RegAlloc::FindFreeSpill() const {
    static_assert(SpillCount <= 64);
    const int index = static_cast<int>(std::countr_zero(~occupied_spills));
    ASSERT_MSG(index < static_cast<int>(SpillCount), "All spill locations are full");
    return index;
}

void RegAlloc::LoadCopyInto(const IR::Value& value, oaknut::XReg reg) {
    if (value.IsImmediate()) {
        code.MOV(reg, value.GetImmediateAsU64());
        return;
    }

    const auto current_location = ValueLocation(value.GetInst());
    ASSERT(current_location);
    switch (current_location->kind) {
    case HostLoc::Kind::Gpr:
        code.MOV(reg, oaknut::XReg{current_location->index});
        break;
    case HostLoc::Kind::Fpr:
        code.FMOV(reg, oaknut::DReg{current_location->index});
        // ASSERT size fits
        break;
    case HostLoc::Kind::Spill:
        code.LDR(reg, SP, spill_offset + current_location->index * spill_slot_size);
        break;
    case HostLoc::Kind::Flags:
        code.MRS(reg, oaknut::SystemReg::NZCV);
        break;
    }
}

void RegAlloc::LoadCopyInto(const IR::Value& value, oaknut::QReg reg) {
    if (value.IsImmediate()) {
        code.MOV(Xscratch0, value.GetImmediateAsU64());
        code.FMOV(reg.toD(), Xscratch0);
        return;
    }

    const auto current_location = ValueLocation(value.GetInst());
    ASSERT(current_location);
    switch (current_location->kind) {
    case HostLoc::Kind::Gpr:
        code.FMOV(reg.toD(), oaknut::XReg{current_location->index});
        break;
    case HostLoc::Kind::Fpr:
        code.MOV(reg.B16(), oaknut::QReg{current_location->index}.B16());
        break;
    case HostLoc::Kind::Spill:
        // TODO: Minimize move size to max value width
        code.LDR(reg, SP, spill_offset + current_location->index * spill_slot_size);
        break;
    case HostLoc::Kind::Flags:
        ASSERT_FALSE("Moving from flags into fprs is not currently supported");
        break;
    }
}

std::optional<HostLoc> RegAlloc::ValueLocation(const IR::Inst* value) const {
    const unsigned name = value->GetName();
    if (name == 0 || name >= value_locations.size()) {
        return ScanValueLocation(value);
    }

    const u8 encoded = value_locations[name];
    const std::optional<HostLoc> result = encoded == 0
                                              ? std::nullopt
                                              : std::optional<HostLoc>{DecodeValueLocation(encoded)};
#if !defined(NDEBUG)
    const auto scanned = ScanValueLocation(value);
    ASSERT(result.has_value() == scanned.has_value());
    if (result) {
        ASSERT(result->kind == scanned->kind && result->index == scanned->index);
    }
#endif
    return result;
}

std::optional<HostLoc> RegAlloc::ScanValueLocation(const IR::Inst* value) const {
    const auto contains_value = [value](const HostLocInfo& info) { return info.Contains(value); };

    if (const auto iter = std::find_if(gprs.begin(), gprs.end(), contains_value); iter != gprs.end()) {
        return HostLoc{HostLoc::Kind::Gpr, static_cast<int>(iter - gprs.begin())};
    }
    if (const auto iter = std::find_if(fprs.begin(), fprs.end(), contains_value); iter != fprs.end()) {
        return HostLoc{HostLoc::Kind::Fpr, static_cast<int>(iter - fprs.begin())};
    }
    if (contains_value(flags)) {
        return HostLoc{HostLoc::Kind::Flags, 0};
    }
    if (const auto iter = std::find_if(spills.begin(), spills.end(), contains_value); iter != spills.end()) {
        return HostLoc{HostLoc::Kind::Spill, static_cast<int>(iter - spills.begin())};
    }
    return std::nullopt;
}

void RegAlloc::SetValueLocation(const IR::Inst* value, HostLoc host_loc) {
    const unsigned name = value->GetName();
    if (name != 0 && name < value_locations.size()) {
        value_locations[name] = EncodeValueLocation(host_loc);
    }
}

void RegAlloc::ClearValueLocation(const IR::Inst* value) {
    const unsigned name = value->GetName();
    if (name != 0 && name < value_locations.size()) {
        ASSERT(value_locations[name] != 0);
        value_locations[name] = 0;
    }
}

HostLocInfo& RegAlloc::ValueInfo(HostLoc host_loc) {
    switch (host_loc.kind) {
    case HostLoc::Kind::Gpr:
        return gprs[static_cast<size_t>(host_loc.index)];
    case HostLoc::Kind::Fpr:
        return fprs[static_cast<size_t>(host_loc.index)];
    case HostLoc::Kind::Flags:
        return flags;
    case HostLoc::Kind::Spill:
        return spills[static_cast<size_t>(host_loc.index)];
    }
    ASSERT_FALSE("RegAlloc::ValueInfo: Invalid HostLoc::Kind");
}

HostLocInfo& RegAlloc::ValueInfo(const IR::Inst* value) {
    const auto host_loc = ValueLocation(value);
    ASSERT(host_loc);
    return ValueInfo(*host_loc);
}

}  // namespace Dynarmic::Backend::Arm64
