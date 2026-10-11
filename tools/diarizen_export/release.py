"""Publish an export as the hosted build `--diarize` downloads, and print its manifest entry.

Two repositories, one per device, because a traced graph keeps the device it was traced on in
both directions -- the CUDA build raises inside TorchScript on CPU and the CPU build on CUDA -- and
`snapshot_download` fetches a whole repository (owner 2026-09-21: host on Hugging Face).

    python tools/diarizen_export/export_diarizen.py --device cuda --out tmp/diarizen-export
    python tools/diarizen_export/export_diarizen.py --device cpu  --out tmp/diarizen-export-cpu
    python tools/diarizen_export/release.py cuda tmp/diarizen-export              # stage only
    python tools/diarizen_export/release.py cuda tmp/diarizen-export --upload     # needs a write token

⚠ Uploading publishes a CC-BY-NC 4.0 derivative under the logged-in account. The model card
(`release_card.md.tmpl`) states the licence and credits every component; do not publish without it.
After an upload, paste the printed entry into `src/finesub_bootstrap/model-manifest.json`
(`diarizen-ts-<device>`): the revision pins the commit, the digests are what `--diarize`
verifies against.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NAMES = ("diarizen_segmentation.ts", "diarizen_embedding.ts", "plda/plda.npz", "plda/xvec_transform.npz")
ACCOUNT = "spurtcarl"


def stage(device: str, source: Path, torch_version: str) -> Path:
    target = REPO / "tmp" / "diarizen-release" / device
    (target / "plda").mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        shutil.copy2(source / name, target / name)
    other = "cpu" if device == "cuda" else "cuda"
    card = (HERE / "release_card.md.tmpl").read_text(encoding="utf-8")
    card = (card.replace("{VARIANT}", device.upper()).replace("{DEVICE}", device.upper())
            .replace("{OTHER}", other.upper())
            .replace("{OTHER_REPO_URL}", f"https://huggingface.co/{ACCOUNT}/finesub-diarizen-{other}")
            .replace("{OTHER_REPO}", f"{ACCOUNT}/finesub-diarizen-{other}")
            .replace("{TORCH}", torch_version))
    (target / "README.md").write_text(card, encoding="utf-8")
    return target


def manifest_entry(folder: Path, repo: str, revision: str) -> dict:
    files = []
    for name in NAMES:
        path = folder / name
        files.append({
            "name": name,
            "url": f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return {"repo": repo, "revision": revision, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("device", choices=("cuda", "cpu"))
    parser.add_argument("source", type=Path, help="the export directory for that device")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    import torch

    folder = stage(args.device, args.source, torch.__version__)
    repo = f"{ACCOUNT}/finesub-diarizen-{args.device}"
    print(f"staged {folder}")
    if not args.upload:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    info = api.upload_folder(repo_id=repo, folder_path=str(folder),
                             commit_message=f"DiariZen md-v2 as TorchScript, {args.device.upper()} build (CC BY-NC 4.0)")
    print(f"uploaded {info.commit_url}")
    print(json.dumps({f"diarizen-ts-{args.device}": manifest_entry(folder, repo, info.oid)}, indent=2))


if __name__ == "__main__":
    main()
