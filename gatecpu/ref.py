"""Plain-Python computer with exactly the micro-steps of gatecpu.design (checked tick by tick)."""
from .design import NBIT, ABIT, NWORDS, OUT_ADDR, OPS

M8, M5 = (1 << NBIT) - 1, (1 << ABIT) - 1


class CpuRef:
    def __init__(self):
        self.ram = [0] * NWORDS
        self.pc = self.ir = self.t = self.halt = self.mar = self.a = self.out = self.b = self.fc = self.fz = 0

    def lines(self, rst=0, prog=0):
        """Control lines for the current state (all zero while reset, programming or halted)."""
        L = dict.fromkeys(["PC_out", "PC_inc", "PC_in", "MAR_in", "RAM_out", "RAM_in", "IR_out", "IR_in",
                           "A_out", "A_in", "B_in", "ALU_out", "F_in", "OUT_in"], 0)
        halt_set = last = 0
        if rst or prog or self.halt:
            return L, halt_set, last
        op, addr, t = OPS[self.ir >> 5], self.ir & M5, self.t
        alu_op, mem_op, jump_op = op in ("ADD", "SUB", "NAND"), op in ("LDA", "STA", "ADD", "SUB", "NAND"), op in ("JMP", "JZ", "JC")
        taken = op == "JMP" or (op == "JZ" and self.fz) or (op == "JC" and self.fc)
        if t == 0:
            L["PC_out"] = L["MAR_in"] = 1
        elif t == 1:
            L["RAM_out"] = L["IR_in"] = L["PC_inc"] = 1
        elif t == 2:
            if mem_op:
                L["IR_out"] = L["MAR_in"] = 1
            if jump_op:
                last = 1
                if taken:
                    L["IR_out"] = L["PC_in"] = 1
                halt_set = int(op == "JMP" and addr == self.mar)
        elif t == 3:
            if op == "LDA":
                L["RAM_out"] = L["A_in"] = 1; last = 1
            elif op == "STA":
                L["A_out"] = L["RAM_in"] = 1; last = 1
                L["OUT_in"] = int(self.mar == OUT_ADDR)
            elif alu_op:
                L["RAM_out"] = L["B_in"] = 1
        elif t == 4 and alu_op:
            L["ALU_out"] = L["A_in"] = L["F_in"] = 1; last = 1
        return L, halt_set, last

    def alu(self):
        op = OPS[self.ir >> 5]
        if op == "NAND":
            r = ~(self.a & self.b) & M8
            return r, 0, int(r == 0)
        if op == "SUB":
            full = self.a + ((~self.b) & M8) + 1
        else:
            full = self.a + self.b
        r = full & M8
        return r, full >> NBIT, int(r == 0)

    def step(self, rst=0, prog=0, prog_we=0, prog_addr=0, prog_data=0):
        L, halt_set, last = self.lines(rst, prog)
        alu_r, alu_c, alu_z = self.alu()
        bus = 0
        if L["PC_out"]: bus |= self.pc
        if L["RAM_out"]: bus |= self.ram[self.mar]
        if L["IR_out"]: bus |= self.ir & M5
        if L["A_out"]: bus |= self.a
        if L["ALU_out"]: bus |= alu_r
        self.bus = bus
        # memory write
        if prog:
            if prog_we:
                self.ram[prog_addr & M5] = prog_data & M8
        elif L["RAM_in"]:
            self.ram[self.mar] = bus
        if rst or prog:
            self.pc = self.ir = self.t = self.halt = self.mar = self.a = self.out = self.b = self.fc = self.fz = 0
            return L
        if L["MAR_in"]: self.mar = bus & M5
        if L["IR_in"]: self.ir = bus
        if L["A_in"]: self.a = bus
        if L["B_in"]: self.b = bus
        if L["OUT_in"]: self.out = bus
        if L["F_in"]: self.fc, self.fz = alu_c, alu_z
        if L["PC_in"]: self.pc = bus & M5
        elif L["PC_inc"]: self.pc = (self.pc + 1) & M5
        self.t = 0 if last else (self.t if self.halt else self.t + 1)
        self.halt = int(self.halt or halt_set)
        return L

    def regs(self):
        r = {}
        def bus(name, v, n):
            for i in range(n):
                r[f"{name}[{i}]"] = (v >> i) & 1
        bus("pc", self.pc, ABIT); bus("ir", self.ir, NBIT); bus("t", self.t, 3); bus("mar", self.mar, ABIT)
        bus("a", self.a, NBIT); bus("out", self.out, NBIT); bus("b", self.b, NBIT)
        r["halt"] = self.halt; r["fc"] = self.fc; r["fz"] = self.fz
        for w in range(NWORDS):
            bus(f"ram{w}", self.ram[w], NBIT)
        return r


def port_inputs(addr=0, data=0, prog=0, we=0, rst=0):
    """Input dict for the gate machine (bit-level names)."""
    d = {"rst": rst, "prog": prog, "prog_we": we}
    for i in range(ABIT):
        d[f"prog_addr{i}"] = (addr >> i) & 1
    for i in range(NBIT):
        d[f"prog_data{i}"] = (data >> i) & 1
    return d


def load_ticks(program):
    """The ticks that load a 32-byte program through the program port, then one idle tick."""
    seq = [dict(prog=1, we=1, addr=i, data=program[i] if i < len(program) else 0) for i in range(NWORDS)]
    return seq + [dict()]


def run_isa(program, max_instr=10000):
    """Instruction-level run: returns (values written to OUT in order, halted, instructions executed)."""
    ram = list(program) + [0] * (NWORDS - len(program))
    pc = a = fc = fz = 0
    outs = []
    for n in range(max_instr):
        ins = ram[pc]; op, addr = OPS[ins >> 5], ins & M5
        here = pc; pc = (pc + 1) & M5
        if op == "LDA": a = ram[addr]
        elif op in ("ADD", "SUB"):
            full = a + ram[addr] if op == "ADD" else a + ((~ram[addr]) & M8) + 1
            a = full & M8; fc = full >> NBIT; fz = int(a == 0)
        elif op == "NAND":
            a = ~(a & ram[addr]) & M8; fc = 0; fz = int(a == 0)
        elif op == "STA":
            ram[addr] = a
            if addr == OUT_ADDR: outs.append(a)
        elif op == "JMP":
            if addr == here: return outs, True, n + 1
            pc = addr
        elif op == "JZ" and fz: pc = addr
        elif op == "JC" and fc: pc = addr
    return outs, False, max_instr
