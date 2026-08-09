"""Pull a few real machine traces out of the Alibaba 2018 cluster trace.

The lab handout points at

    https://github.com/alibaba/clusterdata/raw/master/cluster-trace-v2018/machine_usage.csv

which returns 404 -- that CSV is not in the git repository. The data lives in an
OSS bucket referenced by the repo's own ``fetchData.sh``, and ``machine_usage``
is a 1.77 GB tarball holding a 9.0 GB CSV of ~246 M rows (4 034 machines x 8
days). Downloading all of it to plot one machine would be silly.

Rows in the CSV are *grouped by machine*, so the first few tens of megabytes of
the compressed stream already contain several hundred complete machine traces.
This script streams the tarball, decompresses on the fly, keeps the first
``--machines`` machine ids it meets, and closes the connection as soon as they
are complete -- about 60 MB of transfer instead of 1.77 GB.

    python scripts/fetch_alibaba.py --machines 24 --out data/alibaba_raw.csv

Schema (from the trace's own schema.txt):
    machine_id, time_stamp, cpu_util_percent, mem_util_percent,
    mem_gps, mkpi, net_in, net_out, disk_io_percent

``time_stamp`` is seconds since the (undisclosed) start of the trace window, not
a wall-clock time. See ``scripts/prepare_data.py`` for how that is handled.
"""
from __future__ import annotations

import argparse
import csv
import io
import tarfile
import time

import requests

URL = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz"


class _Capped(io.RawIOBase):
    """A read-only stream that stops after `budget` compressed bytes."""

    def __init__(self, response, budget_mb: int):
        self._chunks = response.iter_content(chunk_size=1 << 20)
        self._buf = b""
        self.read_bytes = 0
        self._budget = budget_mb * 1024 * 1024

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self._buf:
            if self.read_bytes >= self._budget:
                return 0
            try:
                chunk = next(self._chunks)
            except StopIteration:
                return 0
            self.read_bytes += len(chunk)
            self._buf = chunk
        n = min(len(b), len(self._buf))
        b[: n] = self._buf[: n]
        self._buf = self._buf[n:]
        return n


def fetch(out_path: str, machines: int, budget_mb: int) -> None:
    wanted: dict[str, list[list[str]]] = {}
    order: list[str] = []
    started = time.time()

    with requests.get(URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = resp.headers.get("Content-Length", "?")
        print(f"GET {URL}\n  HTTP {resp.status_code}, {int(total)/1e9:.2f} GB compressed")

        capped = _Capped(resp, budget_mb)
        tar = tarfile.open(fileobj=io.BufferedReader(capped), mode="r|gz")
        member = tar.next()
        print(f"  member {member.name}: {member.size/1e9:.2f} GB uncompressed")
        handle = tar.extractfile(member)

        rows = 0
        try:
            for line in handle:
                rows += 1
                parts = line.decode("utf-8", "replace").rstrip("\n").split(",")
                mid = parts[0]
                if mid not in wanted:
                    if len(wanted) >= machines:
                        # A new machine id after the quota is full means every
                        # machine we kept is complete: the file is grouped.
                        break
                    wanted[mid] = []
                    order.append(mid)
                wanted[mid].append(parts)
        except tarfile.ReadError:
            # Expected: we hang up mid-stream, so the tar trailer never arrives.
            print("  (stream closed early -- this is the intended shortcut)")

    kept = sum(len(v) for v in wanted.values())
    print(f"  read {rows:,} rows / {capped.read_bytes/1e6:.0f} MB in "
          f"{time.time()-started:.0f}s; kept {kept:,} rows for {len(wanted)} machines")

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["machine_id", "time_stamp", "cpu_util_percent",
                         "mem_util_percent", "mem_gps", "mkpi", "net_in",
                         "net_out", "disk_io_percent"])
        for mid in order:
            writer.writerows(sorted(wanted[mid], key=lambda r: float(r[1])))
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machines", type=int, default=24)
    ap.add_argument("--budget-mb", type=int, default=120,
                    help="give up after this many compressed MB")
    ap.add_argument("--out", default="data/alibaba_raw.csv")
    args = ap.parse_args()
    fetch(args.out, args.machines, args.budget_mb)


if __name__ == "__main__":
    main()
