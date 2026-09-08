"""Package existing comparison renders for the Static Space (no inference)."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


CASES = {
    "HOT3D": ["002736", "002026", "002223", "002904", "002945", "003052"],
    "ARCTIC": ["300510", "300514", "300591", "300595"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Release checkout containing outputs/hot3d and outputs/arctic")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "static_space")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    media = args.output / "media"
    media.mkdir(parents=True, exist_ok=True)
    cases = []
    provenance = []
    for dataset, ids in CASES.items():
        for clip_id in ids:
            clip = f"clip-{clip_id}"
            source = args.source / "outputs" / dataset.lower() / f"{clip}_world3d.mp4"
            if not source.is_file():
                raise FileNotFoundError(source)
            probe = subprocess.run([args.ffmpeg, "-hide_banner", "-i", str(source)], capture_output=True, text=True).stderr
            duration_match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe)
            fps_match = re.search(r"([\d.]+) fps", probe)
            if not duration_match or not fps_match:
                raise ValueError(f"Cannot read video timing: {source}")
            hh, mm, ss = map(float, duration_match.groups())
            duration = hh * 3600 + mm * 60 + ss
            fps = float(fps_match[1])
            stem = dataset.lower() + "-" + clip_id
            base = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
            # Remove only the render's burned-in title/footer bands. The three
            # panels stay in their original order and remain one synchronized video.
            subprocess.run(base + ["-i", str(source), "-vf", "crop=iw:ih-64:0:32,scale=1800:-2",
                           "-an", "-c:v", "libx264", "-crf", "21", "-preset", "fast", "-threads", "2",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(media / f"{stem}.mp4")], check=True)
            video = media / f"{stem}.mp4"
            subprocess.run(base + ["-ss", "2", "-i", str(video), "-frames:v", "1", "-q:v", "3",
                           str(media / f"{stem}-poster.jpg")], check=True)
            subprocess.run(base + ["-ss", "2", "-i", str(video), "-vf", "crop=iw/3:ih:0:0,scale=480:-2",
                           "-frames:v", "1", "-q:v", "3", str(media / f"{stem}-thumb.jpg")], check=True)
            cases.append({"id": clip, "dataset": dataset, "video": f"media/{stem}.mp4",
                          "poster": f"media/{stem}-poster.jpg", "thumbnail": f"media/{stem}-thumb.jpg",
                          "duration": duration, "fps": fps, "frames": round(duration * fps)})
            provenance.append({"id": clip, "dataset": dataset,
                               "source": f"outputs/{dataset.lower()}/{source.name}",
                               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                               "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest()})
            print(f"Prepared {dataset} {clip}: {duration:.1f}s, {video.stat().st_size / 1024:.0f} KiB", flush=True)
    # A script also works when index.html is opened directly using file://.
    (args.output / "cases.js").write_text("window.STABLEHAND_CASES = " + json.dumps(cases, indent=2) + ";\n")
    (args.output / "media-provenance.json").write_text(json.dumps({
        "description": "Precomputed released-checkpoint comparison renders; no live inference.",
        "processing": "Title/footer crop, H264/yuv420p encoding, no temporal changes.",
        "panels": ["Input video", "Ground truth", "StableHand"], "cases": provenance}, indent=2) + "\n")


if __name__ == "__main__":
    main()
