    // Castle Crashers (tools/xenonrecomp): instructions this XenonRecomp
    // commit does not handle.
    case PPC_INST_ADDC:
        println("\t{}.ca = {}.u32 + {}.u32 < {}.u32;", xer(), r(insn.operands[1]), r(insn.operands[2]), r(insn.operands[1]));
        println("\t{}.u64 = {}.u64 + {}.u64;", r(insn.operands[0]), r(insn.operands[1]), r(insn.operands[2]));
        if (strchr(insn.opcode->name, '.'))
            println("\t{}.compare<int32_t>({}.s32, 0, {});", cr(0), r(insn.operands[0]), xer());
        break;

    case PPC_INST_ADDME:
        println("\t{}.u64 = {}.u64 + {}.ca - 1;", temp(), r(insn.operands[1]), xer());
        println("\t{}.ca = ({}.u32 != 0) | {}.ca;", xer(), r(insn.operands[1]), xer());
        println("\t{}.u64 = {}.u64;", r(insn.operands[0]), temp());
        if (strchr(insn.opcode->name, '.'))
            println("\t{}.compare<int32_t>({}.s32, 0, {});", cr(0), r(insn.operands[0]), xer());
        break;

    case PPC_INST_SUBFZE:
        println("\t{}.u64 = ~{}.u64 + {}.ca;", temp(), r(insn.operands[1]), xer());
        println("\t{}.ca = ({}.u32 == 0) & {}.ca;", xer(), r(insn.operands[1]), xer());
        println("\t{}.u64 = {}.u64;", r(insn.operands[0]), temp());
        if (strchr(insn.opcode->name, '.'))
            println("\t{}.compare<int32_t>({}.s32, 0, {});", cr(0), r(insn.operands[0]), xer());
        break;

    case PPC_INST_EQV:
        println("\t{}.u64 = ~({}.u64 ^ {}.u64);", r(insn.operands[0]), r(insn.operands[1]), r(insn.operands[2]));
        if (strchr(insn.opcode->name, '.'))
            println("\t{}.compare<int32_t>({}.s32, 0, {});", cr(0), r(insn.operands[0]), xer());
        break;

    case PPC_INST_BDZF:
    case PPC_INST_BDZT:
    case PPC_INST_BDNZT:
    {
        constexpr std::string_view fields[] = { "lt", "gt", "eq", "so" };
        println("\t--{}.u64;", ctr());
        println("\tif ({}.u32 {} 0 && {}{}.{}) goto loc_{:X};", ctr(), id == PPC_INST_BDNZT ? "!=" : "==",
            id == PPC_INST_BDZF ? "!" : "", cr(insn.operands[0] / 4), fields[insn.operands[0] % 4], insn.operands[1]);
        break;
    }

    case PPC_INST_LHBRX:
        print("\t{}.u64 = __builtin_bswap16(PPC_LOAD_U16(", r(insn.operands[0]));
        if (insn.operands[1] != 0)
            print("{}.u32 + ", r(insn.operands[1]));
        println("{}.u32));", r(insn.operands[2]));
        break;

    // Update forms: the address is computed first, rA takes it last.
    case PPC_INST_LHZU:
    case PPC_INST_LHAU:
    case PPC_INST_LFSU:
    case PPC_INST_LFDU:
    case PPC_INST_STHU:
    case PPC_INST_STFSU:
    case PPC_INST_STFDU:
        println("\t{} = {} + {}.u32;", ea(), int32_t(insn.operands[1]), r(insn.operands[2]));
        goto update_form;

    case PPC_INST_LHZUX:
    case PPC_INST_LBZUX:
    case PPC_INST_LDUX:
    case PPC_INST_LFSUX:
    case PPC_INST_STHUX:
    case PPC_INST_STDUX:
        println("\t{} = {}.u32 + {}.u32;", ea(), r(insn.operands[1]), r(insn.operands[2]));
    update_form:
    {
        const auto ra = insn.operands[id == PPC_INST_LHZUX || id == PPC_INST_LBZUX || id == PPC_INST_LDUX ||
            id == PPC_INST_LFSUX || id == PPC_INST_STHUX || id == PPC_INST_STDUX ? 1 : 2];
        switch (id)
        {
        case PPC_INST_LHZU:
        case PPC_INST_LHZUX:
            println("\t{}.u64 = PPC_LOAD_U16({});", r(insn.operands[0]), ea());
            break;
        case PPC_INST_LHAU:
            println("\t{}.s64 = int16_t(PPC_LOAD_U16({}));", r(insn.operands[0]), ea());
            break;
        case PPC_INST_LBZUX:
            println("\t{}.u64 = PPC_LOAD_U8({});", r(insn.operands[0]), ea());
            break;
        case PPC_INST_LDUX:
            println("\t{}.u64 = PPC_LOAD_U64({});", r(insn.operands[0]), ea());
            break;
        case PPC_INST_LFSU:
        case PPC_INST_LFSUX:
            printSetFlushMode(false);
            println("\t{}.u32 = PPC_LOAD_U32({});", temp(), ea());
            println("\t{}.f64 = double({}.f32);", f(insn.operands[0]), temp());
            break;
        case PPC_INST_LFDU:
            printSetFlushMode(false);
            println("\t{}.u64 = PPC_LOAD_U64({});", f(insn.operands[0]), ea());
            break;
        case PPC_INST_STHU:
        case PPC_INST_STHUX:
            println("\tPPC_STORE_U16({}, {}.u16);", ea(), r(insn.operands[0]));
            break;
        case PPC_INST_STDUX:
            println("\tPPC_STORE_U64({}, {}.u64);", ea(), r(insn.operands[0]));
            break;
        case PPC_INST_STFSU:
            printSetFlushMode(false);
            println("\t{}.f32 = float({}.f64);", temp(), f(insn.operands[0]));
            println("\tPPC_STORE_U32({}, {}.u32);", ea(), temp());
            break;
        case PPC_INST_STFDU:
            printSetFlushMode(false);
            println("\tPPC_STORE_U64({}, {}.u64);", ea(), f(insn.operands[0]));
            break;
        }
        println("\t{}.u32 = {};", r(ra), ea());
        break;
    }

    case PPC_INST_VPKSWSS:
    case PPC_INST_VPKSWSS128:
        println("\tsimde_mm_store_si128((simde__m128i*){}.u8, simde_mm_packs_epi32(simde_mm_load_si128((simde__m128i*){}.s32), simde_mm_load_si128((simde__m128i*){}.s32)));", v(insn.operands[0]), v(insn.operands[2]), v(insn.operands[1]));
        break;

