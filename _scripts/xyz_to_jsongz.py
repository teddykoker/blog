# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ase",
# ]
# ///

import json
import gzip
from ase.io import read
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", type=str, nargs="?", default="images/nvt.xyz")
    parser.add_argument("out_path", type=str, nargs="?", default=None)
    parser.add_argument("--index", type=str, default=":1000")
    parser.add_argument("--precision", type=int, default=3)
    args = parser.parse_args()
    out_path = args.out_path or args.in_path.replace(".xyz", ".json.gz")
    traj = read(args.in_path, index=args.index, format="extxyz")
    frames = []
    for atoms in traj:
        cell = atoms.get_cell()
        lengths = [round(x, args.precision) for x in cell.lengths()]
        positions = [[round(x, args.precision) for x in p] for p in atoms.get_positions()]
        frames.append({
            "cell": lengths,
            "symbols": atoms.get_chemical_symbols(),
            "positions": positions,
            "pbc": [bool(x) for x in atoms.get_pbc()],
        })

    data = json.dumps(frames)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        f.write(data)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
