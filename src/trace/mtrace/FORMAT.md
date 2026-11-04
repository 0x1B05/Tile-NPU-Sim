# MTRACE binary format (MTRB)

`--mtrace` writes the logical MTRACE event stream as `mtrace.mtrb` in the
versioned MTRB format. JSONL is no longer produced by the simulator; use
`scripts/mtrace/analyze.py --dump-jsonl` when a text stream is needed for
debugging.

The format is little-endian, streaming, and self-describing. Integers use
LEB128 varints; signed values use zigzag LEB128. There is no compression
layer; an external `zstd mtrace.mtrb` is allowed but not required.

## File header

```text
magic[4]       "MTRB"
version        u16    currently 1
flags          u16    bit 0: little-endian
header_size    u32    16
reserved       u32    0
```

## Records

Each record starts with a one-byte type and a varint payload length. Readers
must skip unknown record types using the length.

```text
type           u8
payload_len    varint
payload        payload_len bytes
```

| Type | Name       | Payload                                 |
| ---- | ---------- | --------------------------------------- |
| 1    | `OP_DEF`   | `id:varint`, `len:varint`, `name:bytes` |
| 2    | `PART_DEF` | `id:varint`, `len:varint`, `name:bytes` |
| 3    | `EVENT`    | see below                               |
| 4    | `END`      | `event_count:varint`                    |

`OP_DEF` and `PART_DEF` are emitted lazily, before the first event that uses
the string. Op ids start at 0. Part id 0 is reserved for the empty string and
has no definition record.

The event sequence number is not stored: it is the zero-based ordinal of the
`EVENT` record in the file. If a future writer samples or drops events, the
header version must change and an optional sequence delta field can be added.

## Event payload

```text
flags          u8       bit 0: 1 = compute, 0 = move
                        bit 1: has_mem
                        bit 2: has_macs
instr_delta    svarint  relative to the previous event
pc_delta       svarint  relative to the previous event
op_id          varint
src_count      u8
dst_count      u8
src[src_count] endpoint
dst[dst_count] endpoint
mem            memory   only if has_mem
macs           varint   only if has_macs
```

The first event uses `prev_instr = prev_pc = 0`.

## Endpoint payload

```text
kind_index     u8       bits 0-2: kind, bits 3-7: register index
bytes          varint
elem           u8
dims           u8       0, 1, or 2
shape[dims]    varint
part_id        varint   0 = empty string
```

| Kind | Meaning   |
| ---- | --------- |
| 0    | memory    |
| 1    | Tile      |
| 2    | Acc       |
| 3    | Vec32     |
| 4    | immediate |

Memory and immediate endpoints use index 0.

## Memory payload

```text
flags          u8
  bits 0-1     pattern: 0 = none, 1 = linear, 2 = block
  bits 2-3     region: 0 = ram, 1 = serial, 2 = mmio, 3 = unknown
  bits 4-5     descriptor: 0 = none, 1 = tm, 2 = am, 3 = vm
  bit  6       order: 0 = row, 1 = column
desc_index     u8       bits 0-4
addr_delta     svarint  relative to the previous memory address
elem           u8

linear:
  count        varint
  stride       svarint

block:
  rows         u8
  cols         u8
  row_stride   svarint
  col_stride   svarint
```

The first memory address uses `prev_addr = 0`. `addr_delta` is updated only
when a memory payload is present.

## End record

`END` carries the number of `EVENT` records written. A reader may use it to
reject truncated files. A file without `END` is still readable up to EOF, but
indicates that the writer did not shut down normally.
