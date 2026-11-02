"""An 8-bit stored-program computer as a synchronous machine of 2-input gates and D flip-flops.

Three units talk over one 8-bit bus (SAP-1 style, hardwired control):

  Memory unit   32 x 8 RAM of flip-flops, memory address register MAR (5 bits),
                5 -> 32 decoders, a 32:1 read select per bit, per-word write enable.
                A program port writes RAM while the CPU is held in reset.
  ALU           8-bit ripple adder/subtractor and 8 NANDs, result select, flags C and Z,
                operand register B.
  Control unit  program counter PC (5), instruction register IR (8), step counter T (0-4),
                opcode decoder, control lines, HALT.
  Registers     accumulator A, output register OUT (address 31 is memory-mapped to it).

Instruction = 3-bit opcode + 5-bit address:
  0 LDA a   A <- M[a]                       T0 MAR<-PC  T1 IR<-M, PC+1  T2 MAR<-IR.addr  T3 A<-M
  1 ADD a   A <- A + M[a]   flags C, Z      ... T3 B<-M  T4 A<-ALU, flags
  2 SUB a   A <- A - M[a]   C = no borrow   ... T3 B<-M  T4 A<-ALU, flags
  3 NAND a  A <- ~(A & M[a]) C = 0          ... T3 B<-M  T4 A<-ALU, flags
  4 STA a   M[a] <- A  (a = 31 also -> OUT) ... T3 M<-A
  5 JMP a   PC <- a; a jump to itself sets HALT        T2 PC<-IR.addr
  6 JZ a    if Z: PC <- a                              T2
  7 JC a    if C: PC <- a                              T2

Every rule here is mirrored in gatecpu.ref.CpuRef, and the two are compared tick by tick.
"""
from gatesnake.hdl import Circuit

NBIT, ABIT, NWORDS, OUT_ADDR = 8, 5, 32, 31
OPS = ["LDA", "ADD", "SUB", "NAND", "STA", "JMP", "JZ", "JC"]
LINES = ["PC_out", "PC_inc", "PC_in", "MAR_in", "RAM_out", "RAM_in", "IR_out", "IR_in",
         "A_out", "A_in", "B_in", "ALU_out", "F_in", "OUT_in"]


def build_cpu():
    c = Circuit()
    c.scope = "control"
    rst = c.input("rst")
    c.scope = "program port"
    prog = c.input("prog"); pwe = c.input("prog_we")
    paddr = [c.input(f"prog_addr{i}") for i in range(ABIT)]
    pdata = [c.input(f"prog_data{i}") for i in range(NBIT)]

    # ---- state ----
    pc = c.regs_bus("pc", ABIT); ir = c.regs_bus("ir", NBIT); t = c.regs_bus("t", 3); halt = c.reg("halt")
    mar = c.regs_bus("mar", ABIT)
    ram = [c.regs_bus(f"ram{w}", NBIT) for w in range(NWORDS)]
    a = c.regs_bus("a", NBIT); out = c.regs_bus("out", NBIT)
    b = c.regs_bus("b", NBIT); fc = c.reg("fc"); fz = c.reg("fz")

    # ---- control unit: decode step and opcode, make control lines ----
    c.scope = "control"
    cpu_rst = c.OR(rst, prog)
    keep = c.NOT(cpu_rst)
    run = c.AND(keep, c.NOT(halt))
    T = c.decoder(t)                       # one-hot T0..T7 (only 0-4 occur)
    opd = c.decoder(ir[5:8])               # one-hot opcode
    addr = ir[0:5]
    LDA, ADD, SUB, NAND_, STA, JMP, JZ, JC = opd
    alu_op = c.OR(c.OR(ADD, SUB), NAND_)
    mem_op = c.OR(c.OR(LDA, STA), alu_op)
    jump_op = c.OR(c.OR(JMP, JZ), JC)
    taken = c.OR(JMP, c.OR(c.AND(JZ, fz), c.AND(JC, fc)))
    L = {}
    g = lambda w: c.AND(w, run)
    T0, T1, T2, T3, T4 = (g(T[i]) for i in range(5))
    L["PC_out"] = T0
    L["PC_inc"] = T1
    L["IR_in"] = T1
    s2_mem = c.AND(T2, mem_op); s2_jmp = c.AND(T2, c.AND(jump_op, taken))
    L["MAR_in"] = c.OR(T0, s2_mem)
    L["IR_out"] = c.OR(s2_mem, s2_jmp)
    L["PC_in"] = s2_jmp
    s3_lda, s3_sta, s3_alu = c.AND(T3, LDA), c.AND(T3, STA), c.AND(T3, alu_op)
    s4_alu = c.AND(T4, alu_op)
    L["RAM_out"] = c.OR(T1, c.OR(s3_lda, s3_alu))
    L["RAM_in"] = s3_sta
    L["A_out"] = s3_sta
    L["A_in"] = c.OR(s3_lda, s4_alu)
    L["B_in"] = s3_alu
    L["ALU_out"] = s4_alu
    L["F_in"] = s4_alu
    halt_set = c.AND(c.AND(T2, JMP), c.equal(addr, mar))
    last = c.OR(c.OR(c.AND(T2, jump_op), c.AND(T3, c.OR(LDA, STA))), s4_alu)

    # ---- memory unit: read select (address = MAR) ----
    c.scope = "memory"
    rsel = c.decoder(mar)
    ram_rd = [c.or_tree([c.AND(rsel[w], ram[w][i]) for w in range(NWORDS)]) for i in range(NBIT)]

    # ---- ALU ----
    c.scope = "ALU"
    bx = [c.XOR(bi, SUB) for bi in b]
    carry, ssum = SUB, []
    for ai, bi in zip(a, bx):
        p = c.XOR(ai, bi)
        ssum.append(c.XOR(p, carry))
        carry = c.OR(c.AND(ai, bi), c.AND(p, carry))
    nandv = [c.NAND(ai, bi) for ai, bi in zip(a, b)]
    alu = c.mux_bus(NAND_, ssum, nandv)
    alu_c = c.AND(carry, c.NOT(NAND_))
    alu_z = c.NOT(c.or_tree(alu))

    # ---- bus: OR over (enable AND value) of every source ----
    c.scope = "bus"
    srcs = [(L["PC_out"], pc + [0] * (NBIT - ABIT)), (L["RAM_out"], ram_rd),
            (L["IR_out"], addr + [0] * (NBIT - ABIT)), (L["A_out"], a), (L["ALU_out"], alu)]
    bus = [c.or_tree([c.AND(en, v[i]) for en, v in srcs]) for i in range(NBIT)]

    # ---- register loads ----
    def load(regs, en, val, part):
        c.scope = part
        for q, v in zip(regs, c.mux_bus(en, regs, val)):
            c.set_d(q, c.AND(v, keep))

    load(mar, L["MAR_in"], bus[:ABIT], "memory")
    load(ir, L["IR_in"], bus, "control")
    load(a, L["A_in"], bus, "registers")
    load(b, L["B_in"], bus, "ALU")
    c.scope = "registers"
    L["OUT_in"] = c.AND(L["RAM_in"], c.and_tree(mar))          # writing address 31
    load(out, L["OUT_in"], bus, "registers")
    load([fc, fz], L["F_in"], [alu_c, alu_z], "ALU")
    c.scope = "control"
    pc_inc = c.increment(pc, L["PC_inc"])
    for q, v in zip(pc, c.mux_bus(L["PC_in"], pc_inc, bus[:ABIT])):
        c.set_d(q, c.AND(v, keep))
    t_inc = c.increment(t, c.NOT(c.OR(last, halt)))             # count up, hold while halted
    for q, v in zip(t, t_inc):
        c.set_d(q, c.AND(c.AND(v, c.NOT(last)), keep))
    c.set_d(halt, c.AND(c.OR(halt, halt_set), keep))

    # ---- memory unit: write (CPU at MAR from the bus, or the program port) ----
    c.scope = "program port"
    waddr = c.mux_bus(prog, mar, paddr)
    wdata = c.mux_bus(prog, bus, pdata)
    we = c.mux(prog, L["RAM_in"], pwe)
    c.scope = "memory"
    wsel = c.decoder(waddr)
    for w in range(NWORDS):
        en = c.AND(wsel[w], we)
        for q, v in zip(ram[w], c.mux_bus(en, ram[w], wdata)):
            c.set_d(q, v)

    # ---- outputs: the front panel ----
    c.scope = "bus"
    c.output("bus", bus)
    for k in LINES:
        c.output(k, [L[k]])
    c.output("halt_set", [halt_set])
    return c, {"NBIT": NBIT, "ABIT": ABIT, "NWORDS": NWORDS}


def reg_part(name):
    base = name.split("[")[0]
    if base.startswith("ram") or base == "mar": return "memory"
    if base in ("b", "fc", "fz"): return "ALU"
    if base in ("pc", "ir", "t", "halt"): return "control"
    return "registers"
