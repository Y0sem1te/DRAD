import argparse
import json
import os
from typing import List, Set

import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm


def sample_raw_frames(video_path: str, num_frames: int) -> np.ndarray:
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
    except Exception as e:
        raise RuntimeError(f"Failed to open video: {video_path} | {e}")

    if total_frames <= 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    if total_frames <= num_frames:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    raw = vr.get_batch(indices).asnumpy()
    return raw


def collect_video_relpaths(json_paths: List[str]) -> List[str]:
    relpaths: Set[str] = set()
    for json_path in json_paths:
        with open(json_path, "r") as f:
            items = json.load(f)
        for item in items:
            if "video" in item:
                relpaths.add(item["video"])
    return sorted(relpaths)


def cache_path_for(output_dir: str, video_rel_path: str, num_frames: int) -> str:
    rel = video_rel_path.lstrip("/")
    return os.path.join(output_dir, f"{rel}.nf{num_frames}.npy")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and cache sampled video frames as .npy files")
    parser.add_argument("--json-paths", nargs="+", required=True, help="One or more dataset JSON files")
    parser.add_argument("--video-dir", required=True, help="Root directory for videos")
    parser.add_argument("--output-dir", default="./preprocess/npys", help="Cache output directory")
    parser.add_argument("--num-frames", type=int, default=12, help="Number of sampled frames per video")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_relpaths = collect_video_relpaths(args.json_paths)  # ('video0001', 'video0002', ...)
    print(f"Collected {len(video_relpaths)} unique videos from {len(args.json_paths)} json files")
    print(f"Output dir: {args.output_dir}")
    print(f"Num frames: {args.num_frames}")

    processed = 0
    skipped = 0
    failed = 0

    for video_rel in tqdm(video_relpaths, desc="Caching frames", ncols=120):
        video_path = os.path.join(args.video_dir, video_rel)
        out_path = cache_path_for(args.output_dir, video_rel, args.num_frames)

        if (not args.overwrite) and os.path.exists(out_path):
            skipped += 1
            continue

        if not os.path.exists(video_path):
            failed += 1
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            raw = sample_raw_frames(video_path, args.num_frames)
            tmp_path = f"{out_path}.tmp.{os.getpid()}"
            with open(tmp_path, "wb") as tmp_writer:
                np.save(tmp_writer, raw)
            os.replace(tmp_path, out_path)
            processed += 1
        except Exception:
            failed += 1

    print("=" * 80)
    print(f"Done. processed={processed}, skipped={skipped}, failed={failed}")
    print("=" * 80)


if __name__ == "__main__":
    main()
