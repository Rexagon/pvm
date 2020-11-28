#!/usr/bin/env python3

import argparse
import re

from typing import Dict, List, Callable, Optional, Tuple
from enum import Enum


class Number(Enum):
    @classmethod
    def from_str(cls, ty: str):
        if ty == '':
            return cls.BYTE
        elif ty == 's':
            return cls.SHORT
        elif ty == 'i':
            return cls.INT
        else:
            raise RuntimeError(f'unknown number type - {ty}')

    def convert(self, value: int) -> bytes:
        return value.to_bytes(self.value, 'big', signed=False)

    def to_str(self) -> str:
        if self == Number.INT:
            return 'i'
        elif self == Number.SHORT:
            return 's'
        else:
            return ''

    def mask(self) -> int:
        return int(self.value)

    BYTE = 1
    SHORT = 2
    INT = 4


number_types = (Number.BYTE, Number.SHORT, Number.INT)


class CpuFlag(Enum):
    @classmethod
    def from_str(cls, flag: str):
        if flag == 'z':
            return cls.ZERO
        elif flag == 's':
            return cls.SIGN
        elif flag == 'v':
            return cls.OVERFLOW
        elif flag == 'c':
            return cls.CARRY
        else:
            raise RuntimeError(f'unknown flag - {flag}')

    def to_str(self) -> str:
        if self == CpuFlag.ZERO:
            return 'z'
        elif self == CpuFlag.SIGN:
            return 's'
        elif self == CpuFlag.OVERFLOW:
            return 'v'
        else:
            return 'c'

    def mask(self) -> int:
        return int(self.value)

    ZERO = 0b00
    SIGN = 0b01
    OVERFLOW = 0b10
    CARRY = 0b11


cpu_flags = (CpuFlag.ZERO, CpuFlag.SIGN, CpuFlag.OVERFLOW, CpuFlag.CARRY)


# (less = 00, greater = 01, equal = 10)
class Comparison(Enum):
    def mask(self) -> int:
        return int(self.value)

    LT = 0b000  # <
    GE = 0b111  # >=

    GT = 0b010  # >
    LE = 0b101  # <=

    EQ = 0b100  # ==
    NEQ = 0b011  # !=

    # 001 -> used for lshift
    # 110 -> used for rshift


class CompilationContext:
    def __init__(self):
        self.code: bytearray = bytearray()
        self.pos = 0
        self.labels: Dict[str, int] = {}
        self.label_usages: Dict[str, List[int]] = {}

    def add_label(self, label: str):
        if label not in self.labels:
            self.labels[label] = self.pos
        else:
            raise RuntimeError(f"found duplicated label - {label}")

    def push_label(self, label: str):
        if label in self.label_usages:
            self.label_usages[label] += [self.pos]
        else:
            self.label_usages[label] = [self.pos]
        self.push(Number.INT, 0)

    def push(self, ty: Number, value: int):
        b = ty.convert(value)
        self.pos += len(b)
        self.code.extend(b)

    def fill_labels(self):
        for (label, usages) in self.label_usages.items():
            label_addr = self.labels[label]
            if label_addr is None:
                raise RuntimeError(f"unknown label found - {label}")

            addr_bytes = Number.INT.convert(label_addr)

            for addr in usages:
                for (i, byte) in enumerate(addr_bytes):
                    self.code[addr + i] = byte
        self.label_usages = {}


class ArgumentValue:
    def into_bytecode(self, ctx: CompilationContext):
        raise NotImplementedError()


class ValueRaw(ArgumentValue):
    @classmethod
    def from_str(cls, ty: Number, line: str) -> ArgumentValue:
        if line.startswith('0b'):
            return ValueRaw(ty, int(line[2:], 2))
        elif line.startswith('0x'):
            return ValueRaw(ty, int(line[2:], 16))
        else:
            return ValueRaw(ty, int(line, 10))

    def __init__(self, ty: Number, value: int):
        self.ty = ty
        self.value = value

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(self.ty, self.value)


class ValueLabel(ArgumentValue):
    def __init__(self, name: str):
        self.name = name

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push_label(self.name)


class OpCode:
    def into_bytecode(self, ctx: CompilationContext):
        raise NotImplementedError()


class OpCodeTrieNode(object):
    def __init__(self, char: str):
        self.char: str = char
        self.factory: Optional[Callable[[str], OpCode]] = None
        self.children: Dict[str, OpCodeTrieNode] = {}


class OpCodeTrie(object):
    def __init__(self):
        self.root = OpCodeTrieNode('')

    def insert(self, word: str, factory: Callable[[str], OpCode]):
        node = self.root

        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                new_node = OpCodeTrieNode(char)
                node.children[char] = new_node
                node = new_node

        node.factory = factory

    def query(self, line: str) -> OpCode:
        node = self.root

        i = 0
        for char in line:
            if char.isspace():
                break

            if char in node.children:
                node = node.children[char]
                i += 1
            else:
                break

        if node.factory is not None:
            return node.factory(line[i:])
        else:
            raise RuntimeError(f'command not found - {line}')


# 00000000
class CmdNop(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        storage.insert("nop", lambda _: CmdNop())

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0)


# 00000001
class CmdHlt(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        storage.insert("hlt", lambda _: CmdHlt())

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 1)


# 000001xx
class CmdPush(OpCode):
    regex = re.compile(
        r"^\s*(?P<label>(?:[a-z][a-z0-9_]*)|(?:0b[01]+)|(?:\d+)|(?:0x[0-9A-Fa-f]+))$")

    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert("push" + ty.to_str(), lambda args, t=ty: CmdPush.parse(t, args))

    @classmethod
    def parse(cls, ty: Number, line: str) -> OpCode:
        match = next(re.finditer(cls.regex, line))
        if match.group(1).isdigit():
            return CmdPush(ty, ValueRaw.from_str(ty, match.group(1)))
        else:
            return CmdPush(ty, ValueLabel(match.group(1)))

    def __init__(self, ty: Number, value: ArgumentValue):
        self.ty = ty
        self.value = value

    def into_bytecode(self, ctx: CompilationContext):
        self.value.into_bytecode(ctx)
        ctx.push(Number.BYTE, 0b00000100 | self.ty.mask())


# 000010xx
class CmdPop(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert("pop" + ty.to_str(), lambda _, t=ty: CmdPop(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00001000 | self.ty.mask())


# 000011xx
class CmdSwap(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert("swap" + ty.to_str(), lambda _, t=ty: CmdSwap(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00001100 | self.ty.mask())


# 00010dxx, d - direction (0 - right, 1 - left)
class CmdRotate(OpCode):
    class Direction(Enum):
        def mask(self) -> int:
            return int(self.value)

        RIGHT = 0
        LEFT = 1

    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert("ror" + ty.to_str(), lambda _, t=ty: CmdRotate(t, CmdRotate.Direction.RIGHT))
            storage.insert("rol" + ty.to_str(), lambda _, t=ty: CmdRotate(t, CmdRotate.Direction.LEFT))

    def __init__(self, ty: Number, direction: Direction):
        self.ty = ty
        self.direction = direction

    def into_bytecode(self, ctx: CompilationContext):
        left_mask = self.direction.mask() << 2
        ctx.push(Number.BYTE, 0b00010000 | left_mask | self.ty.mask())


# 00100000
class CmdCall(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        storage.insert('call', lambda _: CmdCall())

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00100000)


# 00100001
class CmdJmp(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        storage.insert('jmp', lambda _: CmdJmp())

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00100001)


# 00100010
class CmdRet(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        storage.insert('ret', lambda _: CmdRet())

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00100010)


# 00101ixx, i - invert
class CmdBranch(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('branch' + ty.to_str(), lambda _, t=ty: CmdBranch(t, inv=False))
            storage.insert('branchnot' + ty.to_str(), lambda _, t=ty: CmdBranch(t, inv=True))

    def __init__(self, ty: Number, inv: bool):
        self.ty = ty
        self.inv = inv

    def into_bytecode(self, ctx: CompilationContext):
        inv_mask = int(self.inv) << 2
        ctx.push(Number.BYTE, 0b00101000 | inv_mask | self.ty.mask())


# 00110iff, i - invert, f - flag
class CmdBranchFlag(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for flag in cpu_flags:
            storage.insert('fbranch' + flag.to_str(), lambda _, f=flag: CmdBranchFlag(f, inv=False))
            storage.insert('fbranchnot' + flag.to_str(), lambda _, f=flag: CmdBranchFlag(f, inv=True))

    def __init__(self, flag: CpuFlag, inv: bool):
        self.inv = inv
        self.flag = flag

    def into_bytecode(self, ctx: CompilationContext):
        inv_mask = int(self.inv) << 2
        ctx.push(Number.BYTE, 0b00110000 | inv_mask | self.flag.mask())


# 001110xx
class CmdDup(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('dup' + ty.to_str(), lambda _, t=ty: CmdDup(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00111000 | self.ty.mask())


# 01rrrrxx, r - rel-1, x - number
class CmdPeek(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('peek' + ty.to_str(), lambda arg, t=ty: CmdPeek(t, int(arg.strip())))

    def __init__(self, ty: Number, rel: int):
        self.ty = ty
        self.rel = rel

    def into_bytecode(self, ctx: CompilationContext):
        rel_mask = ((self.rel - 1) & 0b1111) << 2
        ctx.push(Number.BYTE, 0b01000000 | rel_mask | self.ty.mask())


# 10000000
class CmdClrFlags(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        storage.insert('clrflags', lambda _: CmdClrFlags())

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b10000000)


# 1000lcxx, l - store long result, c - carry, x - number
class CmdAdd(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('add' + ty.to_str(), lambda _, t=ty: CmdAdd(t, carry=False, store_long=False))
            storage.insert('addl' + ty.to_str(), lambda _, t=ty: CmdAdd(t, carry=False, store_long=True))
            storage.insert('addc' + ty.to_str(), lambda _, t=ty: CmdAdd(t, carry=True, store_long=False))
            storage.insert('addcl' + ty.to_str(), lambda _, t=ty: CmdAdd(t, carry=True, store_long=True))

    def __init__(self, ty: Number, carry: bool, store_long: bool):
        self.ty = ty
        self.carry = carry
        self.store_long = store_long

    def into_bytecode(self, ctx: CompilationContext):
        carry_mask = int(self.carry) << 2
        store_long_mask = int(self.store_long) << 3
        ctx.push(Number.BYTE, 0b10000000 | store_long_mask | carry_mask | self.ty.mask())


# 10010cxx, c - carry, x - number
class CmdSub(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('sub' + ty.to_str(), lambda _, t=ty: CmdSub(t, carry=False))
            storage.insert('subc' + ty.to_str(), lambda _, t=ty: CmdSub(t, carry=True))

    def __init__(self, ty: Number, carry: bool):
        self.ty = ty
        self.carry = carry

    def into_bytecode(self, ctx: CompilationContext):
        carry_mask = int(self.carry) << 2
        ctx.push(Number.BYTE, 0b10010000 | carry_mask | self.ty.mask())


# 100110xx
class CmdInc(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('inc' + ty.to_str(), lambda _, t=ty: CmdInc(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b10011000 | self.ty.mask())


# 100111xx
class CmdDec(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('dec' + ty.to_str(), lambda _, t=ty: CmdDec(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b10011100 | self.ty.mask())


# 10100lxx
class CmdMul(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('mul' + ty.to_str(), lambda _, t=ty: CmdMul(t, store_long=False))
            storage.insert('mull' + ty.to_str(), lambda _, t=ty: CmdMul(t, store_long=True))

    def __init__(self, ty: Number, store_long: bool):
        self.ty = ty
        self.store_long = store_long

    def into_bytecode(self, ctx: CompilationContext):
        store_long_mask = int(self.store_long) << 2
        ctx.push(Number.BYTE, 0b10100000 | store_long_mask | self.ty.mask())


# 101100xx, x - number
class CmdNegate(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('neg' + ty.to_str(), lambda _, t=ty: CmdNegate(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b10110000 | self.ty.mask())


# 1011dmxx, dm != 00
class CmdDiv(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('div' + ty.to_str(), lambda _, t=ty: CmdDiv(t, div=True, mod=False))
            storage.insert('mod' + ty.to_str(), lambda _, t=ty: CmdDiv(t, div=False, mod=True))
            storage.insert('divmod' + ty.to_str(), lambda _, t=ty: CmdDiv(t, div=True, mod=True))

    def __init__(self, ty: Number, div: bool, mod: bool):
        self.ty = ty
        self.div = div
        self.mod = mod

    def into_bytecode(self, ctx: CompilationContext):
        div_mask = int(self.div) << 3
        mod_mask = int(self.mod) << 2
        ctx.push(Number.BYTE, 0b10110000 | div_mask | mod_mask | self.ty.mask())


# 110000xx
class CmdAnd(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('and' + ty.to_str(), lambda _, t=ty: CmdAnd(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11000000 | self.ty.mask())


# 110001xx
class CmdOr(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('or' + ty.to_str(), lambda _, t=ty: CmdOr(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11000100 | self.ty.mask())


# 110010xx
class CmdXor(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('xor' + ty.to_str(), lambda _, t=ty: CmdXor(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11001000 | self.ty.mask())


# 110011xx
class CmdNot(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('not' + ty.to_str(), lambda _, t=ty: CmdNot(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11001100 | self.ty.mask())


# 110100xx
class CmdMin(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('min' + ty.to_str(), lambda _, t=ty: CmdMin(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11010000 | self.ty.mask())


# 110101xx
class CmdMax(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('max' + ty.to_str(), lambda _, t=ty: CmdMax(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11010100 | self.ty.mask())


# 110110xx
class CmdAbs(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('abs' + ty.to_str(), lambda _, t=ty: CmdAbs(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11011000 | self.ty.mask())


# 110111xx
class CmdSign(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('sign' + ty.to_str(), lambda _, t=ty: CmdSign(t))

    def __init__(self, ty: Number):
        self.ty = ty

    def into_bytecode(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11011100 | self.ty.mask())


# 11100dxx, d - direction: 0 - right, 1 - left
class CmdShift(OpCode):
    class Direction(Enum):
        def mask(self) -> int:
            return int(self.value)

        RIGHT = 0
        LEFT = 1

    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('rshift' + ty.to_str(), lambda _, t=ty: CmdShift(t, CmdShift.Direction.RIGHT))
            storage.insert('lshift' + ty.to_str(), lambda _, t=ty: CmdShift(t, CmdShift.Direction.LEFT))

    def __init__(self, ty: Number, direction: Direction):
        self.ty = ty
        self.direction = direction

    def into_bytecode(self, ctx: CompilationContext):
        direction_mask = self.direction.mask() << 2
        ctx.push(Number.BYTE, 0b11100000 | direction_mask | self.ty.mask())


# 111oooxx, o - operation, ooo not in [001, 110]
class CmdCompare(OpCode):
    @classmethod
    def register(cls, storage: OpCodeTrie):
        for ty in number_types:
            storage.insert('lt' + ty.to_str(), lambda _, t=ty: CmdCompare(t, Comparison.LT))
            storage.insert('le' + ty.to_str(), lambda _, t=ty: CmdCompare(t, Comparison.LE))
            storage.insert('gt' + ty.to_str(), lambda _, t=ty: CmdCompare(t, Comparison.GT))
            storage.insert('ge' + ty.to_str(), lambda _, t=ty: CmdCompare(t, Comparison.GE))
            storage.insert('eq' + ty.to_str(), lambda _, t=ty: CmdCompare(t, Comparison.EQ))
            storage.insert('neq' + ty.to_str(), lambda _, t=ty: CmdCompare(t, Comparison.NEQ))

    def __init__(self, ty: Number, cmp: Comparison):
        self.ty = ty
        self.cmp = cmp

    def into_bytecode(self, ctx: CompilationContext):
        cmp_mask = self.cmp.mask() << 2
        ctx.push(Number.BYTE, 0b11100000 | cmp_mask | self.ty.mask())


asm_line_regex = re.compile(
    r"^\s*(?P<label>[a-z][a-z0-9_]*:)?\s*(?P<cmd>[a-z]+[a-z0-9_ \t]*[a-z0-9_])?\s*(?:;.*)?$")


def parse_asm(storage: OpCodeTrie, text: str) -> List[Tuple[List[str], OpCode]]:
    opcodes: List[Tuple[List[str], OpCode]] = []

    current_labels: List[str] = []
    for (i, line) in enumerate(text.splitlines()):
        line = line.lower()

        matches = re.finditer(asm_line_regex, line)

        try:
            match = next(matches)

            label = match.group(1)
            opcode = match.group(2)

            if label is not None:
                current_labels.extend([label.replace(':', '')])

            if opcode is not None:
                opcodes.append((current_labels.copy(), storage.query(opcode)))
                current_labels.clear()

        except StopIteration:
            print(f"invalid syntax found at line {i} - {line}")

    return opcodes


def read_file(path: str) -> str:
    result: str
    with open(path) as file:
        result = file.read()
    return result


def create_opcodes_storage() -> OpCodeTrie:
    trie = OpCodeTrie()
    [cls.register(trie) for cls in OpCode.__subclasses__()]
    return trie


def compile_asm(opcodes_storage: OpCodeTrie, text: str) -> bytes:
    opcodes = parse_asm(opcodes_storage, text)

    ctx = CompilationContext()
    for (labels, opcode) in opcodes:
        for label in labels:
            ctx.add_label(label)

        opcode.into_bytecode(ctx)
    ctx.fill_labels()

    return ctx.code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Path to input file')
    parser.add_argument('-o', '--output', help="Path to output file", required=True)
    args = parser.parse_args()

    opcodes_storage = create_opcodes_storage()
    text = read_file(args.input_file)
    code = compile_asm(opcodes_storage, text)

    with open(args.output, 'wb') as file:
        file.write(code)


if __name__ == "__main__":
    main()
