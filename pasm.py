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
        return value.to_bytes(self.len(), 'big', signed=False)

    def to_str(self) -> str:
        if self == Number.INT:
            return 'i'
        elif self == Number.SHORT:
            return 's'
        else:
            return ''

    def mask(self) -> int:
        return int(self.value)

    def len(self) -> int:
        if self == Number.INT:
            return 4
        elif self == Number.SHORT:
            return 2
        else:
            return 1

    BYTE = 1
    SHORT = 2
    INT = 3


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


class ExecutionContext:
    def __init__(self, code: bytes, memory_size: int):
        if memory_size < len(code):
            raise RuntimeError('memory is not enough to execute code')

        self.code_len = len(code)
        self.PC = 0
        self.SP = self.code_len
        self.R = 0
        self.flag_z = False
        self.flag_s = False
        self.flag_v = False
        self.flag_c = False
        self.is_running = True

        self.memory = bytearray(memory_size)
        self.memory[:self.code_len] = code

        print(f'Initialized VM with PC={self.PC}, SP={self.SP}')

    def check_flag(self, flag: CpuFlag) -> bool:
        if flag == CpuFlag.ZERO:
            return self.flag_z
        elif flag == CpuFlag.SIGN:
            return self.flag_s
        elif flag == CpuFlag.OVERFLOW:
            return self.flag_v
        else:
            return self.flag_c

    def read_arg(self, ty: Number) -> int:
        value = int.from_bytes(self.memory[self.PC:(self.PC + ty.len())], byteorder='big', signed=False)
        self.PC += ty.len()
        return value

    def push(self, ty: Number, value: int):
        b = ty.convert(value)
        for byte in b:
            self.memory[self.SP] = byte
            self.SP += 1

    def push_raw(self, value: bytes):
        for byte in value:
            self.memory[self.SP] = byte
            self.SP += 1

    def pop(self, ty: Number) -> int:
        value = int.from_bytes(self.memory[(self.SP - ty.len()):self.SP], byteorder='big', signed=False)
        self.SP -= ty.len()
        return value

    def peek(self, ty: Number, rel: int):
        addr = self.SP - rel - ty.len()
        return int.from_bytes(self.memory[addr:(addr + ty.len())], byteorder='big', signed=False)

    def jump(self, addr: int):
        self.PC = addr

    def set_ret(self):
        self.R = self.PC

    def do_ret(self):
        self.PC = self.R

    def stop(self):
        self.is_running = False

    def print_memory(self, column: int = 16):
        for addr in range(0, self.SP):
            print('{:02x}'.format(self.memory[addr]), end=(' ' if addr == 0 or addr % column != (column - 1) else '\n'))
        if self.SP % column != 0:
            print()

    def print_result(self):
        for addr in range(self.code_len, self.SP):
            print('{:02x}'.format(self.memory[addr]), end=' ')
        print()


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
    def compile(self, ctx: CompilationContext):
        raise NotImplementedError()


class OpCodeTrieNode(object):
    def __init__(self, char: str):
        self.char: str = char
        self.factory: Optional[Callable[[str], OpCode]] = None
        self.children: Dict[str, OpCodeTrieNode] = {}


class OpCodeStorage(object):
    def __init__(self):
        self.root = OpCodeTrieNode('')
        self.lut: List[Optional[Callable[[ExecutionContext], None]]] = [None] * 256
        [cls.register(self) for cls in OpCode.__subclasses__()]

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
    def register(cls, storage: OpCodeStorage):
        storage.insert("nop", lambda _: CmdNop())
        storage.lut[0b00000000] = lambda _: None

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0)


# 00000001
class CmdHlt(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        storage.insert("hlt", lambda _: CmdHlt())
        storage.lut[0b00000001] = CmdHlt.execute

    @classmethod
    def execute(cls, ctx: ExecutionContext):
        ctx.stop()

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 1)


# 000001xx
class CmdPush(OpCode):
    regex = re.compile(
        r"^\s*(?P<label>(?:[a-z][a-z0-9_]*)|(?:0b[01]+)|(?:\d+)|(?:0x[0-9A-Fa-f]+))$")

    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert("push" + ty.to_str(), lambda args, t=ty: CmdPush.parse(t, args))
            storage.lut[0b00000100 | ty.mask()] = lambda ctx, t=ty: CmdPush.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        print(f"Pushing {ty.len()} bytes")
        value = ctx.read_arg(ty)
        ctx.push(ty, value)

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

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00000100 | self.ty.mask())
        self.value.into_bytecode(ctx)


# 000010xx
class CmdPop(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert("pop" + ty.to_str(), lambda _, t=ty: CmdPop(t))
            storage.lut[0b00001000 | ty.mask()] = lambda ctx, t=ty: CmdPop.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        print(f"Popping {ty.len()} bytes")
        ctx.pop(ty)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00001000 | self.ty.mask())


# 000011xx
class CmdSwap(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert("swap" + ty.to_str(), lambda _, t=ty: CmdSwap(t))
            storage.lut[0b00001100 | ty.mask()] = lambda ctx, t=ty: CmdSwap.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        b = ctx.pop(ty)
        a = ctx.pop(ty)
        ctx.push(ty, b)
        ctx.push(ty, a)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00001100 | self.ty.mask())


# 0001ldxx, l - long (4 items), d - direction (0 - right, 1 - left)
class CmdRotate(OpCode):
    class Direction(Enum):
        def mask(self) -> int:
            return int(self.value)

        RIGHT = 0
        LEFT = 1

    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert("ror" + ty.to_str(), lambda _, t=ty: CmdRotate(t, CmdRotate.Direction.RIGHT, extended=False))
            storage.insert("rol" + ty.to_str(), lambda _, t=ty: CmdRotate(t, CmdRotate.Direction.LEFT, extended=False))
            storage.insert("eror" + ty.to_str(), lambda _, t=ty: CmdRotate(t, CmdRotate.Direction.RIGHT, extended=True))
            storage.insert("erol" + ty.to_str(), lambda _, t=ty: CmdRotate(t, CmdRotate.Direction.LEFT, extended=True))

            storage.lut[0b00010000 | ty.mask()] = lambda ctx, t=ty: CmdRotate.execute(ctx, t, CmdRotate.Direction.RIGHT,
                                                                                      extended=False)
            storage.lut[0b00010100 | ty.mask()] = lambda ctx, t=ty: CmdRotate.execute(ctx, t, CmdRotate.Direction.LEFT,
                                                                                      extended=False)
            storage.lut[0b00011000 | ty.mask()] = lambda ctx, t=ty: CmdRotate.execute(ctx, t, CmdRotate.Direction.RIGHT,
                                                                                      extended=True)
            storage.lut[0b00011100 | ty.mask()] = lambda ctx, t=ty: CmdRotate.execute(ctx, t, CmdRotate.Direction.LEFT,
                                                                                      extended=True)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number, direction: Direction, extended: bool):
        if extended:
            d = ctx.pop(ty)
            c = ctx.pop(ty)
            b = ctx.pop(ty)
            a = ctx.pop(ty)
            if direction == CmdRotate.Direction.RIGHT:
                ctx.push(ty, d)
                ctx.push(ty, a)
                ctx.push(ty, b)
                ctx.push(ty, c)
            else:
                ctx.push(ty, b)
                ctx.push(ty, c)
                ctx.push(ty, d)
                ctx.push(ty, a)
        else:
            c = ctx.pop(ty)
            b = ctx.pop(ty)
            a = ctx.pop(ty)
            if direction == CmdRotate.Direction.RIGHT:
                ctx.push(ty, c)
                ctx.push(ty, a)
                ctx.push(ty, b)
            else:
                ctx.push(ty, b)
                ctx.push(ty, c)
                ctx.push(ty, a)

    def __init__(self, ty: Number, direction: Direction, extended: bool):
        self.ty = ty
        self.direction = direction
        self.extended = extended

    def compile(self, ctx: CompilationContext):
        left_mask = self.direction.mask() << 2
        extended_mask = int(self.extended) << 3
        ctx.push(Number.BYTE, 0b00010000 | extended_mask | left_mask | self.ty.mask())


# 00100000
class CmdCall(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        storage.insert('call', lambda _: CmdCall())
        storage.lut[0b00100000] = CmdCall.execute

    @classmethod
    def execute(cls, ctx: ExecutionContext):
        addr = ctx.pop(Number.INT)
        ctx.set_ret()
        ctx.jump(addr)

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00100000)


# 00100001
class CmdJmp(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        storage.insert('jmp', lambda _: CmdJmp())
        storage.lut[0b00100001] = CmdJmp.execute

    @classmethod
    def execute(cls, ctx: ExecutionContext):
        addr = ctx.pop(Number.INT)
        ctx.jump(addr)

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00100001)


# 00100010
class CmdRet(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        storage.insert('ret', lambda _: CmdRet())
        storage.lut[0b00100010] = CmdRet.execute

    @classmethod
    def execute(cls, ctx: ExecutionContext):
        ctx.do_ret()

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00100010)


# 001001xx
class CmdLoad(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('load' + ty.to_str(), lambda _, t=ty: CmdLoad(t))
            storage.lut[0b00100100 | ty.mask()] = lambda ctx, t=ty: CmdLoad.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        rel = ctx.pop(ty)
        value = ctx.peek(ty, rel)
        ctx.push(ty, value)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00100100 | self.ty.mask())


# 00101ixx, i - invert
class CmdBranch(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('branch' + ty.to_str(), lambda _, t=ty: CmdBranch(t, inv=False))
            storage.insert('branchnot' + ty.to_str(), lambda _, t=ty: CmdBranch(t, inv=True))
            storage.lut[0b00101000 | ty.mask()] = lambda ctx, t=ty: CmdBranch.execute(ctx, t, inv=False)
            storage.lut[0b00101100 | ty.mask()] = lambda ctx, t=ty: CmdBranch.execute(ctx, t, inv=True)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number, inv: bool):
        addr = ctx.pop(Number.INT)
        value = ctx.pop(ty)
        if inv ^ (value != 0):
            ctx.jump(addr)

    def __init__(self, ty: Number, inv: bool):
        self.ty = ty
        self.inv = inv

    def compile(self, ctx: CompilationContext):
        inv_mask = int(self.inv) << 2
        ctx.push(Number.BYTE, 0b00101000 | inv_mask | self.ty.mask())


# 00110iff, i - invert, f - flag
class CmdBranchFlag(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for flag in cpu_flags:
            storage.insert('fbranch' + flag.to_str(), lambda _, f=flag: CmdBranchFlag(f, inv=False))
            storage.insert('fbranchnot' + flag.to_str(), lambda _, f=flag: CmdBranchFlag(f, inv=True))
            storage.lut[0b00110000 | flag.mask()] = lambda ctx, f=flag: CmdBranchFlag.execute(ctx, f, inv=False)
            storage.lut[0b00110100 | flag.mask()] = lambda ctx, f=flag: CmdBranchFlag.execute(ctx, f, inv=True)

    @classmethod
    def execute(cls, ctx: ExecutionContext, flag: CpuFlag, inv: bool):
        addr = ctx.pop(Number.INT)
        if inv ^ ctx.check_flag(flag):
            ctx.jump(addr)

    def __init__(self, flag: CpuFlag, inv: bool):
        self.inv = inv
        self.flag = flag

    def compile(self, ctx: CompilationContext):
        inv_mask = int(self.inv) << 2
        ctx.push(Number.BYTE, 0b00110000 | inv_mask | self.flag.mask())


# 001110xx
class CmdDup(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('dup' + ty.to_str(), lambda _, t=ty: CmdDup(t))
            storage.lut[0b00111000 | ty.mask()] = lambda ctx, t=ty: CmdDup.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        value = ctx.peek(ty, 0)
        ctx.push(ty, value)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b00111000 | self.ty.mask())


# 01rrrrxx, r - rel-1, x - number
class CmdPeek(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('peek' + ty.to_str(), lambda arg, t=ty: CmdPeek(t, int(arg.strip())))
            for rel in range(16):
                rel_mask = rel << 2
                storage.lut[0b01000000 | rel_mask | ty.mask()] = lambda ctx, t=ty, r=rel: CmdPeek.execute(ctx, t, r)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number, rel: int):
        value = ctx.peek(ty, rel + 1)
        ctx.push(ty, value)

    def __init__(self, ty: Number, rel: int):
        self.ty = ty
        self.rel = rel

    def compile(self, ctx: CompilationContext):
        rel_mask = ((self.rel - 1) & 0b1111) << 2
        ctx.push(Number.BYTE, 0b01000000 | rel_mask | self.ty.mask())


# 10000000
class CmdClrFlags(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        storage.insert('clrflags', lambda _: CmdClrFlags())
        storage.lut[0b10000000] = CmdClrFlags.execute

    @classmethod
    def execute(cls, ctx: ExecutionContext):
        ctx.flag_z = False
        ctx.flag_s = False
        ctx.flag_v = False
        ctx.flag_c = False

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b10000000)


# 1000lcxx, l - store long result, c - carry, x - number
class CmdAdd(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('add' + ty.to_str(), lambda _, t=ty: CmdAdd(t, carry=False, store_long=False))
            storage.insert('addl' + ty.to_str(), lambda _, t=ty: CmdAdd(t, carry=False, store_long=True))
            storage.insert('addc' + ty.to_str(), lambda _, t=ty: CmdAdd(t, carry=True, store_long=False))
            storage.insert('addcl' + ty.to_str(), lambda _, t=ty: CmdAdd(t, carry=True, store_long=True))

            storage.lut[0b10000000 | ty.mask()] = lambda ctx, t=ty: CmdAdd.execute(ctx, t, carry=False,
                                                                                   store_long=False)
            storage.lut[0b10001000 | ty.mask()] = lambda ctx, t=ty: CmdAdd.execute(ctx, t, carry=False,
                                                                                   store_long=True)
            storage.lut[0b10000100 | ty.mask()] = lambda ctx, t=ty: CmdAdd.execute(ctx, t, carry=True,
                                                                                   store_long=False)
            storage.lut[0b10001100 | ty.mask()] = lambda ctx, t=ty: CmdAdd.execute(ctx, t, carry=True,
                                                                                   store_long=True)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number, carry: bool, store_long: bool):
        b = ctx.pop(ty)
        a = ctx.pop(ty)
        s = a + b
        if carry and ctx.flag_z:
            s += 1
        result = s.to_bytes(ty.len() * 2, byteorder='big', signed=False)

        if s == 0:
            ctx.flag_z = True
        ctx.flag_s = not store_long and result[ty.len() - 1] & 0x80
        ctx.flag_v = False
        ctx.flag_c = result[ty.len()] > 0

        if store_long:
            ctx.push_raw(result)
        else:
            ctx.push_raw(result[ty.len():])

    def __init__(self, ty: Number, carry: bool, store_long: bool):
        self.ty = ty
        self.carry = carry
        self.store_long = store_long

    def compile(self, ctx: CompilationContext):
        carry_mask = int(self.carry) << 2
        store_long_mask = int(self.store_long) << 3
        ctx.push(Number.BYTE, 0b10000000 | store_long_mask | carry_mask | self.ty.mask())


# 10010cxx, c - carry, x - number
class CmdSub(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('sub' + ty.to_str(), lambda _, t=ty: CmdSub(t, carry=False))
            storage.insert('subc' + ty.to_str(), lambda _, t=ty: CmdSub(t, carry=True))
            storage.lut[0b10010000 | ty.mask()] = lambda ctx, t=ty: CmdSub.execute(ctx, t, carry=False)
            storage.lut[0b10010100 | ty.mask()] = lambda ctx, t=ty: CmdSub.execute(ctx, t, carry=True)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number, carry: bool):
        b = ctx.pop(ty)
        a = ctx.pop(ty)
        # TODO: improve substraction
        s = a - b
        if carry and ctx.flag_z:
            s -= 1
        result = s.to_bytes(ty.len() * 2, byteorder='big', signed=False)

        if s == 0:
            ctx.flag_z = True
        ctx.flag_s = result[ty.len() - 1] & 0x80
        ctx.flag_v = False
        ctx.flag_c = result[ty.len()] > 0
        ctx.push_raw(result[ty.len():])

    def __init__(self, ty: Number, carry: bool):
        self.ty = ty
        self.carry = carry

    def compile(self, ctx: CompilationContext):
        carry_mask = int(self.carry) << 2
        ctx.push(Number.BYTE, 0b10010000 | carry_mask | self.ty.mask())


# 100110xx
class CmdInc(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('inc' + ty.to_str(), lambda _, t=ty: CmdInc(t))
            storage.lut[0b10011000 | ty.mask()] = lambda ctx, t=ty: CmdInc.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        x = ctx.pop(ty)
        x += 1
        ctx.push(ty, x)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b10011000 | self.ty.mask())


# 100111xx
class CmdDec(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('dec' + ty.to_str(), lambda _, t=ty: CmdDec(t))
            storage.lut[0b10011100 | ty.mask()] = lambda ctx, t=ty: CmdDec.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        x = ctx.pop(ty)
        x -= 1
        ctx.push(ty, x)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b10011100 | self.ty.mask())


# 10100lxx
class CmdMul(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('mul' + ty.to_str(), lambda _, t=ty: CmdMul(t, store_long=False))
            storage.insert('mull' + ty.to_str(), lambda _, t=ty: CmdMul(t, store_long=True))
            storage.lut[0b10100000 | ty.mask()] = lambda ctx, t=ty: CmdMul.execute(ctx, t, store_long=False)
            storage.lut[0b10100100 | ty.mask()] = lambda ctx, t=ty: CmdMul.execute(ctx, t, store_long=True)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number, store_long: bool):
        b = ctx.pop(ty)
        a = ctx.pop(ty)
        s = a * b
        result = s.to_bytes(ty.len() * 2, byteorder='big', signed=False)

        if s == 0:
            ctx.flag_z = True
        ctx.flag_s = not store_long and result[ty.len() - 1] & 0x80
        ctx.flag_v = False
        ctx.flag_c = result[ty.len()] > 0

        if store_long:
            ctx.push_raw(result)
        else:
            ctx.push_raw(result[ty.len():])

    def __init__(self, ty: Number, store_long: bool):
        self.ty = ty
        self.store_long = store_long

    def compile(self, ctx: CompilationContext):
        store_long_mask = int(self.store_long) << 2
        ctx.push(Number.BYTE, 0b10100000 | store_long_mask | self.ty.mask())


# 101100xx, x - number
class CmdNegate(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('neg' + ty.to_str(), lambda _, t=ty: CmdNegate(t))
            storage.lut[0b10110000 | ty.mask()] = lambda ctx, t=ty: CmdNegate.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        a = ctx.pop(ty)
        # TODO: improve negation
        a = -a
        ctx.push(ty, a)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b10110000 | self.ty.mask())


# 1011dmxx, dm != 00
class CmdDiv(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('div' + ty.to_str(), lambda _, t=ty: CmdDiv(t, div=True, mod=False))
            storage.insert('mod' + ty.to_str(), lambda _, t=ty: CmdDiv(t, div=False, mod=True))
            storage.insert('divmod' + ty.to_str(), lambda _, t=ty: CmdDiv(t, div=True, mod=True))

            storage.lut[0b10111000 | ty.mask()] = lambda ctx, t=ty: CmdDiv.execute(ctx, t, div=True, mod=False)
            storage.lut[0b10110100 | ty.mask()] = lambda ctx, t=ty: CmdDiv.execute(ctx, t, div=False, mod=True)
            storage.lut[0b10111100 | ty.mask()] = lambda ctx, t=ty: CmdDiv.execute(ctx, t, div=True, mod=True)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number, div: bool, mod: bool):
        b = ctx.pop(ty)
        a = ctx.pop(ty)
        if div:
            div_result = a // b
            ctx.push(ty, div_result)
        if mod:
            mod_result = a % b
            ctx.push(ty, mod_result)

    def __init__(self, ty: Number, div: bool, mod: bool):
        self.ty = ty
        self.div = div
        self.mod = mod

    def compile(self, ctx: CompilationContext):
        div_mask = int(self.div) << 3
        mod_mask = int(self.mod) << 2
        ctx.push(Number.BYTE, 0b10110000 | div_mask | mod_mask | self.ty.mask())


# 110000xx
class CmdAnd(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('and' + ty.to_str(), lambda _, t=ty: CmdAnd(t))
            storage.lut[0b11000000 | ty.mask()] = lambda ctx, t=ty: CmdAnd.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        b = ctx.pop(ty)
        a = ctx.pop(ty)
        res = a & b
        ctx.push(ty, res)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11000000 | self.ty.mask())


# 110001xx
class CmdOr(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('or' + ty.to_str(), lambda _, t=ty: CmdOr(t))
            storage.lut[0b11000100 | ty.mask()] = lambda ctx, t=ty: CmdOr.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        b = ctx.pop(ty)
        a = ctx.pop(ty)
        res = a | b
        ctx.push(ty, res)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11000100 | self.ty.mask())


# 110010xx
class CmdXor(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('xor' + ty.to_str(), lambda _, t=ty: CmdXor(t))
            storage.lut[0b11001000 | ty.mask()] = lambda ctx, t=ty: CmdXor.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        b = ctx.pop(ty)
        a = ctx.pop(ty)
        res = a ^ b
        ctx.push(ty, res)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11001000 | self.ty.mask())


# 110011xx
class CmdNot(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('not' + ty.to_str(), lambda _, t=ty: CmdNot(t))
            storage.lut[0b11001100 | ty.mask()] = lambda ctx, t=ty: CmdNot.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        a = ctx.pop(ty)
        a = ~a
        ctx.push(ty, a)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11001100 | self.ty.mask())


# 110100xx
class CmdMin(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('min' + ty.to_str(), lambda _, t=ty: CmdMin(t))
            storage.lut[0b11010000 | ty.mask()] = lambda ctx, t=ty: CmdMin.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        b = ctx.pop(ty)
        a = ctx.pop(ty)
        res = min(a, b)
        ctx.push(ty, res)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11010000 | self.ty.mask())


# 110101xx
class CmdMax(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('max' + ty.to_str(), lambda _, t=ty: CmdMax(t))
            storage.lut[0b11010100 | ty.mask()] = lambda ctx, t=ty: CmdMax.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        b = ctx.pop(ty)
        a = ctx.pop(ty)
        res = max(a, b)
        ctx.push(ty, res)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11010100 | self.ty.mask())


# 110110xx
class CmdAbs(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('abs' + ty.to_str(), lambda _, t=ty: CmdAbs(t))
            storage.lut[0b11011000 | ty.mask()] = lambda ctx, t=ty: CmdAbs.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        a = ctx.pop(ty)
        a = abs(a)
        ctx.push(ty, a)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11011000 | self.ty.mask())


# 110111xx
class CmdSign(OpCode):
    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('sign' + ty.to_str(), lambda _, t=ty: CmdSign(t))
            storage.lut[0b11011100 | ty.mask()] = lambda ctx, t=ty: CmdSign.execute(ctx, t)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number):
        a = ctx.pop(ty)
        if a < 0:
            res = -1
        elif a == 0:
            res = 0
        else:
            res = 1
        ctx.push(ty, res)

    def __init__(self, ty: Number):
        self.ty = ty

    def compile(self, ctx: CompilationContext):
        ctx.push(Number.BYTE, 0b11011100 | self.ty.mask())


# 11100dxx, d - direction: 0 - right, 1 - left
class CmdShift(OpCode):
    class Direction(Enum):
        def mask(self) -> int:
            return int(self.value)

        RIGHT = 0
        LEFT = 1

    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            storage.insert('rshift' + ty.to_str(), lambda _, t=ty: CmdShift(t, CmdShift.Direction.RIGHT))
            storage.insert('lshift' + ty.to_str(), lambda _, t=ty: CmdShift(t, CmdShift.Direction.LEFT))
            storage.lut[0b11100000 | ty.mask()] = lambda ctx, t=ty: CmdShift.execute(ctx, t, CmdShift.Direction.RIGHT)
            storage.lut[0b11100100 | ty.mask()] = lambda ctx, t=ty: CmdShift.execute(ctx, t, CmdShift.Direction.LEFT)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number, direction: Direction):
        n = ctx.pop(ty)
        a = ctx.pop(ty)
        if direction == CmdShift.Direction.RIGHT:
            res = a >> n
        else:
            res = a << n
        ctx.push(ty, res)

    def __init__(self, ty: Number, direction: Direction):
        self.ty = ty
        self.direction = direction

    def compile(self, ctx: CompilationContext):
        direction_mask = self.direction.mask() << 2
        ctx.push(Number.BYTE, 0b11100000 | direction_mask | self.ty.mask())


# 111oooxx, o - operation, ooo not in [001, 110]
class CmdCompare(OpCode):
    class Comparison(Enum):
        def to_str(self) -> str:
            if self == CmdCompare.Comparison.LT:
                return "lt"
            elif self == CmdCompare.Comparison.GE:
                return "ge"
            elif self == CmdCompare.Comparison.GT:
                return "gt"
            elif self == CmdCompare.Comparison.LE:
                return "le"
            elif self == CmdCompare.Comparison.EQ:
                return "eq"
            elif self == CmdCompare.Comparison.NEQ:
                return "neq"
            else:
                raise RuntimeError("unknown comparison variant")

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

    comparison_types = [Comparison.LT, Comparison.GE, Comparison.GT, Comparison.LE, Comparison.EQ, Comparison.NEQ]

    @classmethod
    def register(cls, storage: OpCodeStorage):
        for ty in number_types:
            for cmp in CmdCompare.comparison_types:
                storage.insert(cmp.to_str() + ty.to_str(), lambda _, t=ty, c=cmp: CmdCompare(t, c))

                cmp_mask = cmp.mask() << 2
                storage.lut[0b11100000 | cmp_mask | ty.mask()] = lambda ctx, t=ty, c=cmp: CmdCompare.execute(ctx, t, c)

    @classmethod
    def execute(cls, ctx: ExecutionContext, ty: Number, comparison: Comparison):
        b = ctx.pop(ty)
        a = ctx.pop(ty)
        res = (comparison == CmdCompare.Comparison.LT and a < b) or (
                comparison == CmdCompare.Comparison.GE and a >= b) or (
                      comparison == CmdCompare.Comparison.GT and a > b) or (
                      comparison == CmdCompare.Comparison.LE and a <= b) or (
                      comparison == CmdCompare.Comparison.EQ and a == b) or (
                      comparison == CmdCompare.Comparison.NEQ and a != b)
        ctx.push(Number.BYTE, res)

    def __init__(self, ty: Number, cmp: Comparison):
        self.ty = ty
        self.cmp = cmp

    def compile(self, ctx: CompilationContext):
        cmp_mask = self.cmp.mask() << 2
        ctx.push(Number.BYTE, 0b11100000 | cmp_mask | self.ty.mask())


asm_line_regex = re.compile(
    r"^\s*(?P<label>[a-z][a-z0-9_]*:)?\s*(?P<cmd>[a-z]+[a-z0-9_ \t]*[a-z0-9_])?\s*(?:;.*)?$")


def parse_asm(storage: OpCodeStorage, text: str) -> List[Tuple[List[str], OpCode]]:
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


def compile_asm(opcodes_storage: OpCodeStorage, text: str) -> bytes:
    opcodes = parse_asm(opcodes_storage, text)

    ctx = CompilationContext()
    for (labels, opcode) in opcodes:
        for label in labels:
            ctx.add_label(label)

        opcode.compile(ctx)
    ctx.fill_labels()

    return ctx.code


def run(opcodes_storage: OpCodeStorage, ctx: ExecutionContext):
    while ctx.is_running:
        ctx.print_result()

        opcode = ctx.memory[ctx.PC]
        f = opcodes_storage.lut[opcode]
        print("[{:03x}][{:08b}]: {}".format(ctx.PC, opcode, f))
        ctx.PC += 1
        if f is not None:
            f(ctx)
        else:
            print(f"FOUND UNKNOWN OPCODE: {bin(opcode)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30))
    parser.add_argument('input_file', type=str, help='path to the input file')
    parser.add_argument('-o', '--output', help="path to the output file")
    parser.add_argument('-m', '--memory', type=int, help='memory size allocated for execution')
    args = parser.parse_args()

    opcodes_storage = OpCodeStorage()
    text = read_file(args.input_file)
    code = compile_asm(opcodes_storage, text)

    if args.output is not None:
        with open(args.output, 'wb') as file:
            file.write(code)

    if args.memory is not None:
        print(f'Memory size is: {args.memory}')
        execution_context = ExecutionContext(code, args.memory)
        execution_context.print_memory()

        print("Running VM")
        run(opcodes_storage, execution_context)
        print("Done")
