import os
import configparser
import shutil
from pathlib import Path
import random

SOCCERNET_DIR = Path("D:/FootballAI/soccernet/tracking/test")
OUTPUT_DIR = Path("D:/FootballAI/dataset_v3")
FRAME_STEP = 3          # every 5th frame
CLIP_STEP = 1           # every other clip (alternating)
VAL_SPLIT = 0.15        # 15% of images go to validation

# Class mapping: 0=Player, 1=Goalkeeper, 2=Referee, 3=Ball
def classify_tracklet(desc):
    d = desc.lower()
    if "ball" in d:
        return 3
    if "goalkeeper" in d:
        return 1
    if "referee" in d:
        return 2
    if "player" in d:
        return 0
    return None

def parse_gameinfo(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    mapping = {}
    for key in cfg["Sequence"]:
        if key.startswith("trackletid_"):
            tid = int(key.split("_")[1])
            desc = cfg["Sequence"][key]
            cls = classify_tracklet(desc)
            if cls is not None:
                mapping[tid] = cls
    return mapping

def parse_seqinfo(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    w = int(cfg["Sequence"]["imWidth"])
    h = int(cfg["Sequence"]["imHeight"])
    return w, h

def process_clip(clip_dir, img_out, lbl_out, all_entries):
    gt_file = clip_dir / "gt" / "gt.txt"
    gameinfo_file = clip_dir / "gameinfo.ini"
    seqinfo_file = clip_dir / "seqinfo.ini"
    img_dir = clip_dir / "img1"

    if not (gt_file.exists() and gameinfo_file.exists() and seqinfo_file.exists()):
        print(f"Skipping {clip_dir.name}, missing files")
        return

    tracklet_map = parse_gameinfo(gameinfo_file)
    img_w, img_h = parse_seqinfo(seqinfo_file)

    frame_boxes = {}
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            cls = tracklet_map.get(tid)
            if cls is None:
                continue
            frame_boxes.setdefault(frame, []).append((cls, x, y, w, h))

    for frame, boxes in frame_boxes.items():
        if frame % FRAME_STEP != 0:
            continue
        img_name = f"{frame:06d}.jpg"
        src_img = img_dir / img_name
        if not src_img.exists():
            continue

        new_name = f"{clip_dir.name}_{img_name}"
        dst_img = img_out / new_name
        dst_lbl = lbl_out / new_name.replace(".jpg", ".txt")

        shutil.copy(src_img, dst_img)
        with open(dst_lbl, "w") as lf:
            for cls, x, y, w, h in boxes:
                xc = (x + w / 2) / img_w
                yc = (y + h / 2) / img_h
                wn = w / img_w
                hn = h / img_h
                lf.write(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        all_entries.append(new_name)

def main():
    clips = sorted([d for d in SOCCERNET_DIR.iterdir() if d.is_dir()])
    selected_clips = clips[::CLIP_STEP]
    print(f"Total clips: {len(clips)}, using: {len(selected_clips)}")

    tmp_img = OUTPUT_DIR / "all_images"
    tmp_lbl = OUTPUT_DIR / "all_labels"
    tmp_img.mkdir(parents=True, exist_ok=True)
    tmp_lbl.mkdir(parents=True, exist_ok=True)

    all_entries = []
    for clip in selected_clips:
        print(f"Processing {clip.name}...")
        process_clip(clip, tmp_img, tmp_lbl, all_entries)

    print(f"Total images extracted: {len(all_entries)}")

    random.seed(42)
    random.shuffle(all_entries)
    val_count = int(len(all_entries) * VAL_SPLIT)
    val_set = set(all_entries[:val_count])

    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for name in all_entries:
        split = "val" if name in val_set else "train"
        shutil.move(str(tmp_img / name), OUTPUT_DIR / "images" / split / name)
        lbl_name = name.replace(".jpg", ".txt")
        shutil.move(str(tmp_lbl / lbl_name), OUTPUT_DIR / "labels" / split / lbl_name)

    shutil.rmtree(tmp_img)
    shutil.rmtree(tmp_lbl)

    yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/val
names:
  0: Player
  1: Goalkeeper
  2: Referee
  3: Ball
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print("Done. Dataset ready at:", OUTPUT_DIR)

if __name__ == "__main__":
    main()