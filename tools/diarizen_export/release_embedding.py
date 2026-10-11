"""Publish the voiceprint model and PLDA the default backend downloads, and print its manifest entry.

The Nemotron-3 backend clusters with DiariZen's second half -- WeSpeaker ResNet34 and a PLDA -- but
not with DiariZen's weights, so it must not fetch them from the CC BY-NC DiariZen repositories.
Both pieces are CC BY 4.0 on their own: the TorchScript is our conversion of
`pyannote/wespeaker-voxceleb-resnet34-LM`, and the PLDA files are byte-identical to
`pyannote/speaker-diarization-community-1`'s (sha256 checked below). One repository holds both
device builds.

    python tools/diarizen_export/release_embedding.py <cuda export dir> <cpu export dir>            # stage
    python tools/diarizen_export/release_embedding.py <cuda export dir> <cpu export dir> --upload   # write token

The two export directories are `export_diarizen.py` output, or the hosted DiariZen builds (whose
`diarizen_embedding.ts` is that output). ⚠ Uploading publishes under the logged-in account; paste
the printed entry into `src/finesub_bootstrap/model-manifest.json` as `speaker-embedding`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ACCOUNT = "spurtcarl"
NAME = "finesub-speaker-embedding"
PLDA = ("plda/plda.npz", "plda/xvec_transform.npz")
#: pyannote/speaker-diarization-community-1 @ 3533c8cf, from the Hub's LFS metadata.
COMMUNITY_PLDA_SHA256 = {
    "plda/plda.npz": "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",
    "plda/xvec_transform.npz": "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
}
FILES = ("wespeaker-cuda.ts", "wespeaker-cpu.ts", *PLDA)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(cuda: Path, cpu: Path, torch_version: str) -> Path:
    target = REPO / "tmp" / "speaker-embedding-release"
    # Emptied first: `upload_folder` publishes whatever is in the folder, so a file left by an
    # earlier staging would land in the repository without being in `FILES` or the manifest.
    shutil.rmtree(target, ignore_errors=True)
    (target / "plda").mkdir(parents=True)
    shutil.copy2(cuda / "diarizen_embedding.ts", target / "wespeaker-cuda.ts")
    shutil.copy2(cpu / "diarizen_embedding.ts", target / "wespeaker-cpu.ts")
    for name in PLDA:
        if sha256(cuda / name) != COMMUNITY_PLDA_SHA256[name]:
            raise SystemExit(f"{cuda / name} is not community-1's {name}; refusing to relicense it")
        shutil.copy2(cuda / name, target / name)
    card = (HERE / "embedding_card.md.tmpl").read_text(encoding="utf-8").replace("{TORCH}", torch_version)
    (target / "README.md").write_text(card, encoding="utf-8")
    return target


def manifest_entry(folder: Path, repo: str, revision: str) -> dict:
    return {"repo": repo, "revision": revision, "files": [
        {"name": name, "url": f"https://huggingface.co/{repo}/resolve/{revision}/{name}",
         "size": (folder / name).stat().st_size, "sha256": sha256(folder / name)}
        for name in FILES]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cuda", type=Path)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    import torch

    folder = stage(args.cuda, args.cpu, torch.__version__.split("+")[0])
    repo = f"{ACCOUNT}/{NAME}"
    print(f"staged {folder}")
    if not args.upload:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    info = api.upload_folder(repo_id=repo, folder_path=str(folder),
                             commit_message="WeSpeaker ResNet34 as TorchScript (CUDA and CPU) and pyannote's PLDA, CC BY 4.0")
    print(json.dumps(manifest_entry(folder, repo, info.oid), indent=2))


if __name__ == "__main__":
    main()
