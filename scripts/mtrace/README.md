# MTRACE analysis

`--mtrace` writes the simulator data path to `mtrace.mtrb`, a versioned binary
stream documented in [`src/trace/mtrace/FORMAT.md`](../../src/trace/mtrace/FORMAT.md).
The reader and analyzer use only the Python standard library.

## Commands

```sh
# Overall summary and top ops
python3 scripts/mtrace/analyze.py path/to/mtrace.mtrb

# Group by op / type / pc / instr / endpoint / memory
python3 scripts/mtrace/analyze.py mtrace.mtrb --by op --top 30
python3 scripts/mtrace/analyze.py mtrace.mtrb --by pc --sort bytes
python3 scripts/mtrace/analyze.py mtrace.mtrb --by endpoint

# Filter
python3 scripts/mtrace/analyze.py mtrace.mtrb --filter-op 'mma|vmma'
python3 scripts/mtrace/analyze.py mtrace.mtrb --filter-type compute --pc-start 0x80004000

# Machine-readable output
python3 scripts/mtrace/analyze.py mtrace.mtrb --json > summary.json
python3 scripts/mtrace/analyze.py mtrace.mtrb --by op --csv > ops.csv
```

`--limit N` stops after N matched events. `--top 0` prints all rows.

## Python API

`analyze.py` only uses the summary helpers; custom analysis can stream events
directly:

```python
from scripts.mtrace.reader import iter_events

for event in iter_events("mtrace.mtrb"):
    print(event.seq, event.op, event.pc, event.macs)
```

Or keep the header and dictionaries in scope:

```python
from scripts.mtrace.reader import TraceReader

with TraceReader("mtrace.mtrb") as trace:
    print(trace.header.version)
    for event in trace:
        for endpoint in event.src + event.dst:
            print(endpoint.kind, endpoint.index, endpoint.bytes)
```

`Event` fields: `seq`, `instr`, `pc`, `type`, `op`, `src`, `dst`, `mem`, `macs`.
`Endpoint` fields: `kind`, `index`, `bytes`, `elem`, `shape`, `part`.
`Memory` fields: `pattern`, `region`, `addr`, `elem`, `count`, `stride`,
`rows`, `cols`, `row_stride`, `col_stride`, `order`, `desc`.

`seq` is the zero-based event ordinal reconstructed by the reader; it is not
stored in the file.
