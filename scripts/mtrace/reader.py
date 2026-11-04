"""Streaming reader for the MTRB binary MTRACE format.

The on-disk layout is documented in src/trace/mtrace/FORMAT.md. The reader
uses only the Python standard library and yields events one at a time, so it
can process traces larger than memory.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

MAGIC = b"MTRB"
VERSION = 1
HEADER_SIZE = 16

_KINDS = ("mem", "tile", "acc", "vec32", "imm")
_PATTERNS = ("none", "linear", "block")
_REGIONS = ("ram", "serial", "mmio", "unknown")
_DESCRIPTORS = ("", "tm", "am", "vm")

_REC_OP_DEF = 1
_REC_PART_DEF = 2
_REC_EVENT = 3
_REC_END = 4

_EVENT_COMPUTE = 1 << 0
_EVENT_HAS_MEM = 1 << 1
_EVENT_HAS_MACS = 1 << 2


@dataclass(frozen=True)
class Header:
    magic: str
    version: int


@dataclass(frozen=True)
class Endpoint:
    kind: str
    index: int
    bytes: int
    elem: int
    shape: tuple[int, ...]
    part: str


@dataclass(frozen=True)
class Memory:
    pattern: str
    region: str
    addr: int
    elem: int
    count: int = 0
    stride: int = 0
    rows: int = 0
    cols: int = 0
    row_stride: int = 0
    col_stride: int = 0
    order: str = "row"
    desc: str = ""


@dataclass(frozen=True)
class Event:
    seq: int
    instr: int
    pc: int
    type: str
    op: str
    src: tuple[Endpoint, ...]
    dst: tuple[Endpoint, ...]
    mem: Memory | None
    macs: int


class _Cursor:
    __slots__ = ("data", "pos")

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def u8(self) -> int:
        value = self.data[self.pos]
        self.pos += 1
        return value

    def take(self, size: int) -> bytes:
        end = self.pos + size
        if end > len(self.data):
            raise ValueError("truncated MTRB record")
        value = self.data[self.pos:end]
        self.pos = end
        return value

    def varint(self) -> int:
        value = 0
        shift = 0
        while True:
            byte = self.u8()
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7

    def svarint(self) -> int:
        value = self.varint()
        return (value >> 1) ^ -(value & 1)


def _read_exact(fp: BinaryIO, size: int) -> bytes:
    data = fp.read(size)
    if len(data) != size:
        raise ValueError("truncated MTRB file")
    return data


def _read_varint(fp: BinaryIO) -> int:
    value = 0
    shift = 0
    while True:
        raw = fp.read(1)
        if not raw:
            raise ValueError("truncated MTRB varint")
        byte = raw[0]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value
        shift += 7


class TraceReader:
    """Context-manager reader; ``header`` is available after construction."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._fp: BinaryIO = open(path, "rb")
        try:
            self.header = self._read_header()
        except Exception:
            self._fp.close()
            raise
        self.op_names: dict[int, str] = {}
        self.part_names: dict[int, str] = {0: ""}
        self._prev_instr = 0
        self._prev_pc = 0
        self._prev_addr = 0
        self._seq = 0

    def _read_header(self) -> Header:
        magic, version, _flags, header_size, _reserved = struct.unpack(
            "<4sHHII", _read_exact(self._fp, HEADER_SIZE)
        )
        if magic != MAGIC:
            raise ValueError(f"{self.path}: bad MTRB magic {magic!r}")
        if version != VERSION:
            raise ValueError(f"{self.path}: unsupported MTRB version {version}")
        if header_size != HEADER_SIZE:
            raise ValueError(f"{self.path}: unsupported MTRB header size {header_size}")
        return Header(magic=magic.decode("ascii"), version=version)

    def __enter__(self) -> "TraceReader":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def close(self) -> None:
        self._fp.close()

    def __iter__(self) -> Iterator[Event]:
        while True:
            raw = self._fp.read(1)
            if not raw:
                return
            record_type = raw[0]
            payload = _read_exact(self._fp, _read_varint(self._fp))
            if record_type == _REC_OP_DEF:
                cursor = _Cursor(payload)
                op_id = cursor.varint()
                name = cursor.take(cursor.varint()).decode("utf-8")
                self.op_names[op_id] = name
            elif record_type == _REC_PART_DEF:
                cursor = _Cursor(payload)
                part_id = cursor.varint()
                name = cursor.take(cursor.varint()).decode("utf-8")
                self.part_names[part_id] = name
            elif record_type == _REC_EVENT:
                event = self._read_event(payload)
                self._seq += 1
                yield event
            elif record_type == _REC_END:
                cursor = _Cursor(payload)
                expected = cursor.varint()
                if expected != self._seq:
                    raise ValueError(
                        f"{self.path}: END says {expected} events, read {self._seq}"
                    )
                return
            # Unknown record types are skipped by length for forward compatibility.

    def _read_endpoint(self, cursor: _Cursor) -> Endpoint:
        kind_index = cursor.u8()
        kind = _KINDS[kind_index & 0x7]
        index = (kind_index >> 3) & 0x1F
        size = cursor.varint()
        elem = cursor.u8()
        dims = cursor.u8()
        shape = tuple(cursor.varint() for _ in range(dims))
        part = self.part_names[cursor.varint()]
        return Endpoint(
            kind=kind,
            index=index,
            bytes=size,
            elem=elem,
            shape=shape,
            part=part,
        )

    def _read_memory(self, cursor: _Cursor) -> Memory:
        flags = cursor.u8()
        pattern_code = flags & 0x3
        region = _REGIONS[(flags >> 2) & 0x3]
        descriptor = _DESCRIPTORS[(flags >> 4) & 0x3]
        order = "col" if flags & 0x40 else "row"
        desc_index = cursor.u8() & 0x1F
        addr = self._prev_addr + cursor.svarint()
        self._prev_addr = addr
        elem = cursor.u8()
        count = stride = rows = cols = row_stride = col_stride = 0
        if pattern_code == 1:
            count = cursor.varint()
            stride = cursor.svarint()
        elif pattern_code == 2:
            rows = cursor.u8()
            cols = cursor.u8()
            row_stride = cursor.svarint()
            col_stride = cursor.svarint()
        return Memory(
            pattern=_PATTERNS[pattern_code],
            region=region,
            addr=addr,
            elem=elem,
            count=count,
            stride=stride,
            rows=rows,
            cols=cols,
            row_stride=row_stride,
            col_stride=col_stride,
            order=order,
            desc=f"{descriptor}{desc_index}" if descriptor else "",
        )

    def _read_event(self, payload: bytes) -> Event:
        cursor = _Cursor(payload)
        flags = cursor.u8()
        instr = self._prev_instr + cursor.svarint()
        pc = self._prev_pc + cursor.svarint()
        self._prev_instr = instr
        self._prev_pc = pc
        op = self.op_names[cursor.varint()]
        src_count = cursor.u8()
        dst_count = cursor.u8()
        src = tuple(self._read_endpoint(cursor) for _ in range(src_count))
        dst = tuple(self._read_endpoint(cursor) for _ in range(dst_count))
        mem = self._read_memory(cursor) if flags & _EVENT_HAS_MEM else None
        macs = cursor.varint() if flags & _EVENT_HAS_MACS else 0
        return Event(
            seq=self._seq,
            instr=instr,
            pc=pc,
            type="compute" if flags & _EVENT_COMPUTE else "move",
            op=op,
            src=src,
            dst=dst,
            mem=mem,
            macs=macs,
        )


def iter_events(path: str | Path) -> Iterator[Event]:
    """Yield events from ``path`` without exposing the file header."""
    with TraceReader(path) as trace:
        yield from trace
