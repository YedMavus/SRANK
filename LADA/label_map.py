import os
import csv
import argparse

from utils.config import _C as cfg
from scenario_datasets.build_functions import build_TAIL_testloader


def dump_label_map(merged_classnames, indices, path="label_map.csv"):
    rows = []
    for i, name in enumerate(merged_classnames):
        task_id = sum(i >= idx for idx in indices) - 1
        rows.append({"label": i, "task_id": task_id, "classname": name})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "task_id", "classname"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main(args):
    cfg.defrost()
    cfg.merge_from_file(os.path.join("./configs/data", args.data + ".yaml"))
    cfg.merge_from_file(os.path.join("./configs/model", args.model + ".yaml"))
    cfg.merge_from_list(args.opts)

    # Build only the TAIL test loader to get merged class names and indices
    _, merged_classnames, indices = build_TAIL_testloader(
        root=cfg.root,
        dataset_sequence=cfg.dataset_sequence,
        transform_test=None,   # Not needed for just class metadata
        batch_size=1,
        num_workers=0,
        collate_fn=lambda x: x  # Dummy
    )

    dump_label_map(merged_classnames, indices, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", required=True, type=str, help="data config file (e.g. TAIL)")
    parser.add_argument("--model", "-m", required=True, type=str, help="model config file (e.g. clip_vit_b16)")
    parser.add_argument("--output", "-o", default="label_map.csv", help="output CSV path")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="override config options, e.g. dataset aircraft")
    args = parser.parse_args()
    main(args)