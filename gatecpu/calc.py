"""A calculator program for the gate computer, used by the GPT's CPU head.

The program stays in RAM. A call writes x, y, op and prod = 0 through the program port
(4 clock ticks, CPU held in reset), then the CPU runs from address 0 until it halts or
CAP ticks pass, and the answer is read from the OUT register.

    op 1  x + y   (mod 256)
    op 2  x - y   (mod 256)
    op 3  x * y   by adding x, y times (mod 256); 54 + 32*y clock ticks
    else  0

If the CPU has not printed by tick CAP, the answer is 0 (a timeout).
"""
from .asm import assemble
from .design import OPS

SOURCE = """; calculator: OUT <- x op y
        LDA op
        SUB one
        JZ add       ; op 1
        SUB one
        JZ sub       ; op 2
        SUB one
        JZ mul       ; op 3
        LDA prod     ; anything else: 0
        JMP out
sub:    LDA x
        SUB y
        JMP out
mul:    LDA y        ; while y > 0: prod += x, y -= 1
        SUB one
        JC body
        LDA prod
        JMP out
body:   STA y
        LDA prod
        ADD x
        STA prod
        JMP mul
add:    LDA x
        ADD y
out:    OUT
        HLT
one:    .byte 1
x:      .byte 0
y:      .byte 0
op:     .byte 0
prod:   .byte 0
"""
PROGRAM = assemble(SOURCE)
ADDR = {"x": 27, "y": 28, "op": 29, "prod": 30}
CAP = 600
TICKS = {"LDA": 4, "STA": 4, "ADD": 5, "SUB": 5, "NAND": 5, "JMP": 3, "JZ": 3, "JC": 3}
OP_ADD, OP_SUB, OP_MUL = 1, 2, 3


def call_ticks(x, y, op):
    """The program-port ticks of one call: write x, y, op, prod (CPU held in reset)."""
    return [dict(prog=1, we=1, addr=ADDR[k], data=v & 255) for k, v in (("x", x), ("y", y), ("op", op), ("prod", 0))]


def run_calc(x, y, op, cap=CAP):
    """Instruction-level run with exact clock-tick counts. Returns (answer, ticks to print or None)."""
    ram = list(PROGRAM)
    ram[ADDR["x"]], ram[ADDR["y"]], ram[ADDR["op"]], ram[ADDR["prod"]] = x & 255, y & 255, op & 255, 0
    pc = a = fc = fz = 0
    t = 0
    while t < cap:
        ins = ram[pc]; name, addr = OPS[ins >> 5], ins & 31
        t += TICKS[name]
        if t > cap:
            break
        pc = (pc + 1) & 31
        if name == "LDA": a = ram[addr]
        elif name in ("ADD", "SUB"):
            full = a + ram[addr] if name == "ADD" else a + ((~ram[addr]) & 255) + 1
            a = full & 255; fc = full >> 8; fz = int(a == 0)
        elif name == "NAND":
            a = ~(a & ram[addr]) & 255; fc = 0; fz = int(a == 0)
        elif name == "STA":
            ram[addr] = a
            if addr == 31:
                return a, t
        elif name == "JMP": pc = addr
        elif name == "JZ" and fz: pc = addr
        elif name == "JC" and fc: pc = addr
    return 0, None


def expected(x, y, op):
    """What the program computes when it has time to finish."""
    return {OP_ADD: (x + y) & 255, OP_SUB: (x - y) & 255, OP_MUL: (x * y) & 255}.get(op, 0)
