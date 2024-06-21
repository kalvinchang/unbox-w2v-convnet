from pathlib import Path

import numpy as np
import pandas as pd
from textgrids import TextGrid
from tqdm import tqdm
import argparse


def _get_args():
    # example: python data_prep.py --dataset_path data/voxangeles --textgrid_path data/voxangeles/data/audited_aligned --dataset_type voxangeles --output_path data/voxangeles
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to dataset")
    parser.add_argument("--textgrid_path", type=Path, help="Path to TextGrids")
    parser.add_argument("--dataset_type", type=str, choices=list(_SUPPORTED_DATASETS.keys()))
    parser.add_argument("--output_path", type=Path, help="Output csv folder")

    return parser.parse_args()


def _voxangeles(dataset_path: Path, textgrid_path: Path):
    rows = []
    # each file is one word
    for p in tqdm(textgrid_path.glob("*/*.TextGrid")):
        grid = TextGrid(p)
        phone_key = "phone" if "phone" in grid else ("phones" if "phones" in grid else "Narrow")
        for phone in grid[phone_key]:
            if phone and phone.text:
                rows.append({
                    "text": phone.text,
                    "start": phone.xmin,
                    "finish": phone.xmax,
                    "path": str((dataset_path / p.relative_to(p.parents[3]).with_suffix(".flac")).absolute()),
                })

    return pd.DataFrame(rows)


_SUPPORTED_DATASETS = {
    "voxangeles": _voxangeles,
}


if __name__ == "__main__":
    args = _get_args()
    parser = _SUPPORTED_DATASETS[args.dataset_type]
    df = parser(dataset_path=args.dataset_path, textgrid_path=args.textgrid_path)
    df.to_csv((args.output_path / args.dataset_type).with_suffix('.csv'))
    df.to_pickle((args.output_path / args.dataset_type).with_suffix('.pkl'))
