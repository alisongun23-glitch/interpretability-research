"""Assembler for the gate computer, and the demo programs.

    label:  LDA x        ; comment
            OUT          ; = STA 31 (address 31 is the output register)
            HLT          ; = JMP to this instruction
    x:      .byte 5      ; data; numbers may be decimal, 0x.. or 0b..

Mnemonics: LDA ADD SUB NAND STA JMP JZ JC (each takes an address or a label), OUT, HLT.
The JavaScript assembler in cpu_ui/index.html follows the same rules.
"""
from .design import OPS, NWORDS, OUT_ADDR


class AsmError(ValueError):
    pass


def _num(tok, labels, line_no):
    t = tok.strip()
    if t in labels:
        return labels[t]
    try:
        return int(t, 0)
    except ValueError:
        raise AsmError(f"line {line_no}: unknown label or number '{t}'")


def assemble(src):
    """Returns 32 bytes. Raises AsmError with a line number on any problem."""
    items, labels = [], {}
    for n, raw in enumerate(src.splitlines(), 1):
        line = raw.split(";")[0].strip()
        while ":" in line:
            lab, line = line.split(":", 1)
            lab = lab.strip()
            if not lab.isidentifier():
                raise AsmError(f"line {n}: bad label '{lab}'")
            if lab in labels:
                raise AsmError(f"line {n}: label '{lab}' defined twice")
            labels[lab] = len(items)
            line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        items.append((n, parts[0].upper(), parts[1].strip() if len(parts) > 1 else None))
    if len(items) > NWORDS:
        raise AsmError(f"program needs {len(items)} bytes; memory has {NWORDS}")
    out = []
    for addr, (n, op, arg) in enumerate(items):
        if op == ".BYTE":
            if arg is None:
                raise AsmError(f"line {n}: .byte needs a value")
            out.append(_num(arg, labels, n) & 0xFF)
        elif op == "OUT":
            out.append((OPS.index("STA") << 5) | OUT_ADDR)
        elif op == "HLT":
            out.append((OPS.index("JMP") << 5) | addr)
        elif op in OPS:
            if arg is None:
                raise AsmError(f"line {n}: {op} needs an address")
            v = _num(arg, labels, n)
            if not 0 <= v < NWORDS:
                raise AsmError(f"line {n}: address {v} is outside 0-{NWORDS - 1}")
            out.append((OPS.index(op) << 5) | v)
        else:
            raise AsmError(f"line {n}: unknown instruction '{op}'")
    return out + [0] * (NWORDS - len(out))


def disassemble(byte):
    op, a = OPS[byte >> 5], byte & 31
    if op == "STA" and a == OUT_ADDR:
        return "OUT"
    return f"{op} {a}"


DEMOS = {
    "Fibonacci": ("""; prints Fibonacci numbers until the next one no longer fits in 8 bits
loop:   LDA x
        OUT
        ADD y        ; A = x + y
        JC done      ; carry: the sum passed 255
        STA t
        LDA y
        STA x
        LDA t
        STA y
        JMP loop
done:   HLT
x:      .byte 0
y:      .byte 1
t:      .byte 0
""", [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]),
    "Multiply": ("""; 7 x 6 by adding 7 six times
loop:   LDA prod
        ADD x
        STA prod
        LDA cnt
        SUB one
        STA cnt
        JZ done
        JMP loop
done:   LDA prod
        OUT
        HLT
prod:   .byte 0
x:      .byte 7
cnt:    .byte 6
one:    .byte 1
""", [42]),
    "Countdown": ("""; counts 10 down to 0
loop:   LDA n
        OUT
        SUB one      ; sets Z when n reaches 0
        STA n
        JZ done
        JMP loop
done:   LDA n
        OUT
        HLT
n:      .byte 10
one:    .byte 1
""", [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
    "XOR from NAND": ("""; x XOR y built from four NANDs (the CPU has no XOR instruction)
        LDA x
        NAND y       ; n = NAND(x, y)
        STA n
        NAND x       ; p = NAND(n, x)
        STA p
        LDA n
        NAND y       ; NAND(n, y)
        NAND p       ; NAND(NAND(n, y), p) = x XOR y
        OUT
        HLT
x:      .byte 0b11001010
y:      .byte 0b10100110
n:      .byte 0
p:      .byte 0
""", [0b11001010 ^ 0b10100110]),
}
