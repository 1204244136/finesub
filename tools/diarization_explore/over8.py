"""Past eight speakers with Nemotron-3: candidate methods on one bench.

Nemotron-3-Diarization has eight output slots and an arrival-order speaker cache; a ninth voice
has nowhere to go. The methods tried here all keep Nemotron's frame-level activity (its strength:
boundaries, overlap, local discrimination) and add identity from somewhere else:

  n3          the model alone (baseline)
  vbx         Nemotron as the LOCAL segmentation inside DiariZen's own chain: slice its whole-file
              tracks into W-second windows, embed each (window, track) with WeSpeaker, VBx-cluster
              with PLDA, reassemble. A slot that hosts two people in different windows splits;
              the within-window must-not-link comes from `constrained_argmax`.

Datasets (all audio 16 kHz mono):

  vox-<id>    VoxConverse test files with >= 9 reference speakers. ⚠ Nemotron-3 AND DiariZen were
              trained on VoxConverse: compare methods on one model family, never models to models.
  mocai-hN    one hour of 魔裁 input1 starting at N h (galgame voices, 11-15 characters an hour,
              script-derived truth; cue spans are upper bounds, the streamer is not in the truth).

    python tools/diarization_explore/over8.py prep                 # cut / extract the audio
    python tools/diarization_explore/over8.py infer vox mocai      # Nemotron probs, GPU
    python tools/diarization_explore/over8.py run vbx vox mocai    # one method, GPU for embeddings
    python tools/diarization_explore/over8.py score vox mocai
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nemotron3_probe as n3  # noqa: E402

sys.path.insert(0, str(n3.WT / "src"))

OUT = n3.WT / "tmp" / "over8"
VOX_TEST = n3.PUBLIC / "voxconverse" / "voxconverse-master" / "test"
MOCAI_SOURCE = n3.MAIN / "out" / "mocai-in1" / "mocai-in1-vocal.ogg"
MOCAI_HOURS = (1, 3, 5, 7)
SR = 16000


# --- datasets ---------------------------------------------------------------------------------

def vox_ids(min_speakers=9):
    out = []
    for rttm in sorted(VOX_TEST.glob("*.rttm")):
        speakers = {line.split()[7] for line in rttm.read_text().splitlines() if line.strip()}
        if len(speakers) >= min_speakers:
            out.append(f"vox-{rttm.stem}")
    return out


def idents(dataset):
    if dataset == "vox":
        return vox_ids()
    if dataset == "voxs":
        # Screening subset: every 5th of the >= 9-speaker files, spread over 9-21 speakers.
        return vox_ids()[::5]
    if dataset == "mocai":
        return [f"mocai-h{h}" for h in MOCAI_HOURS]
    if dataset == "ali":
        # AliMeeting Eval, 2-4 people: the one clean set where DiariZen beat Nemotron-3.
        return [i for i in n3.ALI]
    if dataset == "onetonton":
        return [f"onetonton-{i:02d}" for i in range(1, 16)]
    if dataset == "hina":
        frozen = json.loads((HERE / "results" / "diarizen-v3-20260917.json").read_text(encoding="utf-8"))
        return frozen["groups"]["heldout_all"]
    raise SystemExit(f"unknown dataset {dataset}")


def is_ours(ident):
    """姐姐咚咚 / hina: the W2 material, scored with the edit-cost ruler, audio in the main checkout."""
    return ident.startswith(("onetonton-", "hina-"))


def is_ali(ident):
    return ident in n3.ALI


def audio_file(ident):
    if is_ours(ident):
        import nemotron3_cost as nc
        return nc.source(ident)
    if is_ali(ident):
        return n3.audio_path(ident)
    return OUT / "audio" / f"{ident}.flac"


def prep():
    import soundfile as sf

    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(n3.PUBLIC / "voxconverse_test_wav.zip") as archive:
        for ident in vox_ids():
            target = audio_file(ident)
            if target.exists():
                continue
            raw = OUT / "audio" / f"{ident}.wav"
            raw.write_bytes(archive.read(f"voxconverse_test_wav/{ident[4:]}.wav"))
            wave, sr = sf.read(raw, dtype="float32", always_2d=True)
            assert sr == SR, (ident, sr)
            sf.write(target, wave.mean(axis=1), SR, subtype="PCM_24")
            raw.unlink()
    info = sf.info(MOCAI_SOURCE)
    assert info.samplerate == SR
    for hour in MOCAI_HOURS:
        target = audio_file(f"mocai-h{hour}")
        if not target.exists():
            wave, _ = sf.read(MOCAI_SOURCE, start=hour * 3600 * SR, frames=3600 * SR,
                              dtype="float32", always_2d=True)
            sf.write(target, wave.mean(axis=1), SR, subtype="PCM_24")
    print("prepared", len(list((OUT / "audio").glob("*.flac"))), "files")


def load_audio(ident):
    import soundfile as sf

    wave, sr = sf.read(audio_file(ident), dtype="float32", always_2d=True)
    assert sr == SR
    return wave.mean(axis=1)


def reference(ident):
    """(start, end, speaker) rows, seconds relative to the file."""
    if is_ali(ident):
        # The rows frozen in the earlier rounds' cached runs, byte-identical truth for every arm.
        return [tuple(r) for r in n3.cached(ident, "high")["reference"]]
    if ident.startswith("vox-"):
        rows = []
        for line in (VOX_TEST / f"{ident[4:]}.rttm").read_text().splitlines():
            f = line.split()
            if f:
                rows.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
        return rows
    import mocai_diar as MD

    hour = int(ident.rsplit("h", 1)[1])
    offset = hour * 3600.0
    return [(a - offset, b - offset, w) for a, b, w in MD.truth_cues("lower1")
            if b > offset and a < offset + 3600]


# --- inference ---------------------------------------------------------------------------------

def probs_path(ident):
    if is_ours(ident) or is_ali(ident):
        return n3.OUT / ident / "probs.npz"
    return OUT / "n3" / f"{ident}.npz"


def infer(datasets):
    import vram_guard
    vram_guard.install()
    import torch

    model = n3.load_model()
    for dataset in datasets:
        for ident in idents(dataset):
            if probs_path(ident).exists():
                continue
            wave = load_audio(ident)
            t = time.perf_counter()
            _, probs = n3.infer(model, wave, SR)
            probs_path(ident).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(probs_path(ident), probs=probs)
            torch.cuda.empty_cache()
            print(f"{ident} {len(wave) / SR / 60:.1f} min {time.perf_counter() - t:.1f}s "
                  f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)


# --- methods: each returns (activity (frames, tracks) 0/1 or prob, frame_seconds) ----------------

def smoothed_raster(probs, frame_in, frame_out, frames_out):
    """Production smoothing at the model's rate, rasterised onto another grid."""
    from finesub.speech.diarization.segments import segments_from_activity

    grid = np.zeros((frames_out, probs.shape[1]), dtype=np.float32)
    for t in segments_from_activity(probs.tolist(), frame_seconds=frame_in):
        grid[int(round(t.start / frame_out)):int(round(t.end / frame_out)), t.track] = 1.0
    return grid


def method_n3(ident, probs, wave, **_):
    return probs, 0.01


_EMBED = {}


def diarizen_handle():
    if "handle" not in _EMBED:
        from finesub.speech.diarization.diarizen_backend import ensure
        _EMBED["handle"] = Path(ensure(None, "cuda"))
    return _EMBED["handle"]


def local_from_tracks(binary20, window, step):
    """(chunks, frames_per_chunk, tracks) windows cut from a whole-file 20 ms 0/1 map."""
    from finesub.speech.diarization import diarizen_assemble as da

    per_chunk = int(round(window / da.FRAME_STEP))
    total = binary20.shape[0]
    duration = total * da.FRAME_STEP
    chunks = max(1, int(np.ceil(max(0.0, duration - window) / step)) + 1)
    local = np.zeros((chunks, per_chunk, binary20.shape[1]), dtype=np.float32)
    for c in range(chunks):
        start = da.closest_frame(c * step + 0.5 * da.FRAME_DURATION)
        piece = binary20[max(0, start):start + per_chunk]
        local[c, :len(piece)] = piece
    return local


def embed_local(wave, local, window, step, batch=16, prod=False):
    """(chunks, tracks, dim) WeSpeaker embeddings; NaN where a track is silent in the window.

    ⚠ The default is the FAST setting every reported vbx number used: batch 16, TF32 left on.
    `prod` switches to what `diarizen_embed.embed` ships (batch 2, TF32 off) -- the chain turns
    last-bit differences into label flips (findings 5.2), so an integration must re-measure.
    """
    import contextlib

    import torch
    from finesub.speech.diarization import diarizen_embed as de
    from finesub.speech.diarization.diarizen_features import compute_fbank
    from finesub.speech.diarization.diarizen_runtime import jit_fusion_disabled, loaded_model, tf32_disabled

    if prod:
        batch = de.EMBEDDING_BATCH_SIZE

    masks = de.chunk_masks(local)
    chunks, per_chunk, tracks = masks.shape
    window_size, step_size = int(window * SR), int(round(step * SR))
    min_frames = int(np.ceil(per_chunk * de.MIN_NUM_SAMPLES / (window * SR)))
    audio = torch.from_numpy(wave).unsqueeze(0)
    out = None
    todo = [(c, k) for c in range(chunks) for k in range(tracks) if masks[c, :, k].sum() > min_frames]
    with loaded_model(diarizen_handle() / "diarizen_embedding.ts", "cuda") as model, \
            jit_fusion_disabled(), (tf32_disabled() if prod else contextlib.nullcontext()), \
            torch.inference_mode():
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            pieces = torch.stack([de.crop_chunk(audio, c, window_size=window_size, step_size=step_size)
                                  for c, _ in part]).to("cuda")
            weights = torch.from_numpy(np.stack([masks[c, :, k] for c, k in part])).to("cuda")
            vectors = model(compute_fbank(pieces), weights).cpu().numpy()
            if out is None:
                out = np.full((chunks, tracks, vectors.shape[1]), np.nan, dtype=np.float32)
            for (c, k), v in zip(part, vectors):
                out[c, k] = v
    if out is None:
        out = np.full((chunks, tracks, 256), np.nan, dtype=np.float32)
    return out


def method_vbx(ident, probs, wave, window=16.0, step=1.6, ahc=None, fa=None, fb=None, prod=0.0, **_):
    from finesub.speech.diarization import diarizen_assemble as da
    from finesub.speech.diarization import diarizen_cluster as dc
    from finesub.speech.diarization import diarizen_runner as dr

    from finesub.speech.diarization import diarizen_embed as de

    frames20 = int(np.ceil(len(wave) / SR / da.FRAME_STEP))
    binary20 = smoothed_raster(probs, 0.01, da.FRAME_STEP, frames20)
    local = local_from_tracks(binary20, window, step)
    # assemble and chunk_masks read the window geometry from their module globals. diarizen_embed's
    # SEGMENTATION_STEP_RATIO is left alone on purpose: embed_local passes step_size to crop_chunk itself.
    saved = dr.SEGMENTATION_DURATION, dr.SEGMENTATION_STEP_RATIO
    try:
        da.SEGMENTATION_DURATION, da.SEGMENTATION_STEP_RATIO = window, step / window
        de.SEGMENTATION_DURATION = window
        embeddings = embed_local(wave, local, window, step, prod=bool(prod))
        knobs = {k: v for k, v in (("ahc_threshold", ahc), ("fa", fa), ("fb", fb)) if v is not None}
        clustered = dc.cluster(embeddings, local, diarizen_handle() / "plda", **knobs)
        native = da.assemble(local, clustered.hard).astype(np.float32)
    finally:
        da.SEGMENTATION_DURATION, da.SEGMENTATION_STEP_RATIO = saved
        de.SEGMENTATION_DURATION = saved[0]
    return native, da.FRAME_STEP


class Embedder:
    """WeSpeaker (DiariZen's embedding export) held open for one method run."""

    def __init__(self):
        import contextlib
        from finesub.speech.diarization.diarizen_runtime import jit_fusion_disabled, loaded_model

        self.stack = contextlib.ExitStack()
        self.model = self.stack.enter_context(loaded_model(diarizen_handle() / "diarizen_embedding.ts", "cuda"))
        self.stack.enter_context(jit_fusion_disabled())

    def close(self):
        self.stack.close()

    def track(self, wave, mask10, *, window=16.0, min_active=1.6):
        """L2-normalised mean embedding of one track, from the 16 s chunks where it speaks cleanly.

        `mask10` is that track's clean (single-speaker) 0/1 activity at 10 ms over `wave`.
        """
        raw = self.raw(wave, mask10, window=window, min_active=min_active)
        if raw is None:
            return None
        vectors = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        mean = vectors.mean(axis=0)
        return mean / np.linalg.norm(mean)

    def raw(self, wave, mask10, *, window=16.0, min_active=1.6):
        """(n, dim) raw embeddings, one per 16 s chunk with enough clean speech, or None."""
        import torch
        from finesub.speech.diarization.diarizen_features import compute_fbank

        per = int(window * 100)
        vectors = []
        with torch.inference_mode():
            for start in range(0, max(1, len(mask10) - per // 4), per):
                m = mask10[start:start + per]
                if m.sum() * 0.01 < min_active:
                    continue
                piece = np.zeros(int(window * SR), dtype=np.float32)
                audio = wave[start * 160:start * 160 + len(piece)]
                piece[:len(audio)] = audio
                weights = np.zeros(int(window / 0.02), dtype=np.float32)
                even = m[: (min(len(m), 2 * len(weights)) // 2) * 2]
                m20 = even.reshape(-1, 2).max(axis=1)
                weights[:len(m20)] = m20
                # (batch, channel, samples), as `diarizen_embed.embed` stacks its crops.
                v = self.model(compute_fbank(torch.from_numpy(piece)[None, None].cuda()),
                               torch.from_numpy(weights)[None].cuda()).cpu().numpy()[0]
                if np.all(np.isfinite(v)):
                    vectors.append(v)
        return np.stack(vectors) if vectors else None


def plda_space():
    """Raw WeSpeaker vectors -> DiariZen's PLDA features (the space findings 8 measured distances in)."""
    if "plda" not in _EMBED:
        from finesub.speech.diarization import diarizen_cluster as dc
        xvec_tf, plda_tf, _ = dc.load_plda(diarizen_handle() / "plda")
        _EMBED["plda"] = lambda raw: plda_tf(xvec_tf(raw), lda_dim=dc.LDA_DIM)
    return _EMBED["plda"]


def constrained_merge(centroids, weights, cannot, threshold):
    """Greedy average-linkage AHC on cosine distance, never joining groups that share a cannot-link.

    `centroids` (n, d) in PLDA space, `weights` (n,) seconds, `cannot` (n, n) bool.
    Returns a group id per input.
    """
    unit = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    groups = {i: [i] for i in range(len(unit))}
    while True:
        best, pair = threshold, None
        keys = sorted(groups)
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                a, b = groups[keys[x]], groups[keys[y]]
                if cannot[np.ix_(a, b)].any():
                    continue
                ca = np.average(unit[a], axis=0, weights=weights[a])
                cb = np.average(unit[b], axis=0, weights=weights[b])
                d = 1 - ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb))
                if d < best:
                    best, pair = d, (keys[x], keys[y])
        if pair is None:
            break
        groups[pair[0]] += groups.pop(pair[1])
    out = np.zeros(len(unit), dtype=int)
    for g, members in enumerate(groups.values()):
        out[members] = g
    return out


def merge_tracks(wave, activity, embedder, threshold, *, min_speech=2.0, co_seconds=1.0):
    """Merge tracks of a 10 ms activity map whose voices match; co-active tracks never merge.

    Tracks with too little clean speech to embed are left alone.
    """
    active, single = clean_masks(activity)
    to_plda = plda_space()
    keep, cents, weights = [], [], []
    for k in range(activity.shape[1]):
        if single[:, k].sum() * 0.01 < min_speech:
            continue
        raw = embedder.raw(wave, single[:, k].astype(np.float32))
        if raw is None:
            continue
        keep.append(k)
        cents.append(to_plda(raw).mean(axis=0))
        weights.append(single[:, k].sum() * 0.01)
    if len(keep) < 2:
        return activity
    a = active[:, keep].astype(np.float32)
    cannot = (a.T @ a) * 0.01 >= co_seconds
    np.fill_diagonal(cannot, False)
    group = constrained_merge(np.stack(cents), np.array(weights), cannot, threshold)
    columns = [activity[:, k] for k in range(activity.shape[1]) if k not in keep]
    for g in range(group.max() + 1):
        members = [keep[i] for i in np.flatnonzero(group == g)]
        columns.append(activity[:, members].max(axis=1))
    return np.stack(columns, axis=1)


def method_recur(ident, probs, wave, merge=0.4, min_parent=10.0, **_):
    """Recursive split (findings 7) with Nemotron, then a constrained PLDA merge over all leaves.

    Each track's clean speech is concatenated and run through the model again, so up to eight voices
    that shared a slot get slots of their own. Frames a parent holds in overlap go to the child of the
    nearest clean frame. The merge pass undoes over-splitting: children of one parent never co-occur,
    so only the embeddings keep them apart, and leaves that ever speak together cannot merge.
    """
    model = _EMBED.get("n3") or _EMBED.setdefault("n3", n3.load_model())
    embedder = Embedder()
    active, single = clean_masks(probs)
    frames = probs.shape[0]
    leaves = []
    try:
        for k in range(probs.shape[1]):
            if not active[:, k].any():
                continue
            clean = np.flatnonzero(single[:, k])
            if len(clean) * 0.01 < min_parent:
                leaves.append(probs[:, k] * active[:, k])
                continue
            samples = (clean[:, None] * 160 + np.arange(160)).ravel()
            audio = wave[samples[samples < len(wave)]]
            _, child = n3.infer(model, audio, SR)
            child = child[:len(clean)]
            owner_clean = np.where(child.max(axis=1) >= 0.5, child.argmax(axis=1), -1)
            # every frame the parent holds -> the child of the nearest clean frame
            held = np.flatnonzero(active[:, k])
            nearest = np.clip(np.searchsorted(clean, held), 0, len(clean) - 1)
            before = np.clip(nearest - 1, 0, len(clean) - 1)
            pick = np.where(np.abs(clean[before] - held) < np.abs(clean[nearest] - held), before, nearest)
            owner = owner_clean[pick]
            for c in np.unique(owner):
                col = np.zeros(frames, dtype=np.float32)
                col[held[owner == c]] = probs[held[owner == c], k]
                if (col > 0.5).sum() * 0.01 >= 0.3:
                    leaves.append(col)
        activity = np.stack(leaves, axis=1) if leaves else np.zeros((frames, 0), dtype=np.float32)
        if merge > 0:
            activity = merge_tracks(wave, activity, embedder, merge)
    finally:
        embedder.close()
    return activity, 0.01


def method_recurvbx(ident, probs, wave, **params):
    """Recursive split WITHOUT its merge pass, then vbx over the leaves: the split separates what the
    model itself can hear apart, VBx re-links across windows with WeSpeaker + PLDA."""
    leaves, _ = method_recur(ident, probs, wave, merge=0.0)
    return method_vbx(ident, leaves, wave, **params)


def method_enrollmerge(ident, probs, wave, merge=0.4, **params):
    """Prefix enrolment, then the constrained PLDA merge to fold the re-registered duplicates back."""
    activity, frame = method_enroll(ident, probs, wave, **params)
    embedder = Embedder()
    try:
        return merge_tracks(wave, activity, embedder, merge), frame
    finally:
        embedder.close()


def clean_masks(probs, threshold=0.5):
    """(frames, tracks) 0/1 activity, and the same restricted to single-speaker frames."""
    active = probs > threshold
    single = active & (active.sum(axis=1, keepdims=True) == 1)
    return active, single


def enrolment_clip(wave, single_col, seconds=6.0, min_run=0.5):
    """Longest clean runs of one track, concatenated to about `seconds`, 0.3 s of silence between."""
    runs, start = [], None
    for i, on in enumerate(np.append(single_col, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if (i - start) * 0.01 >= min_run:
                runs.append((i - start, start, i))
            start = None
    runs.sort(reverse=True)
    pieces, total = [], 0.0
    gap = np.zeros(int(0.3 * SR), dtype=np.float32)
    for length, a, b in runs:
        pieces += [wave[a * 160:b * 160], gap]
        total += length * 0.01
        if total >= seconds:
            break
    return np.concatenate(pieces) if pieces else None


def method_enroll(ident, probs, wave, window=300.0, max_enrol=6.0, min_sim=0.2, clip=6.0, **_):
    """Prefix enrolment: known voices go FIRST in each window, so arrival order hands them slots.

    Per window: (1) a plain pass gives local tracks; (2) their embeddings pick the closest registered
    speakers (cosine >= min_sim, at most `max_enrol`, leaving slots free for newcomers); (3) those
    speakers' clips are prepended and the model runs again. A window slot that the model also used for
    enrolled clip j IS registered speaker j -- the linking decision is the model's own, not a threshold
    on embeddings. Unclaimed slots become new speakers and are registered with a clip of their own.
    """
    model = _EMBED.get("n3") or _EMBED.setdefault("n3", n3.load_model())
    embedder = Embedder()
    per = int(window * SR)
    total_frames = probs.shape[0]
    registry = []            # dicts: clip (np), vec (np)
    columns = []             # per global speaker: (frames,) activity
    try:
        for w0 in range(0, len(wave), per):
            seg = wave[w0:w0 + per]
            if len(seg) < SR:
                break
            f0 = w0 // 160
            _, local = n3.infer(model, seg, SR)
            active, single = clean_masks(local)
            slots = [k for k in range(local.shape[1]) if active[:, k].sum() * 0.01 >= 1.0]
            vecs = {k: embedder.track(seg, single[:, k].astype(np.float32)) for k in slots}
            chosen = []
            if registry:
                scored = []
                for k, v in vecs.items():
                    if v is None:
                        continue
                    sims = [float(r["vec"] @ v) for r in registry]
                    best = int(np.argmax(sims))
                    if sims[best] >= min_sim:
                        scored.append((sims[best], best))
                for _, g in sorted(scored, reverse=True):
                    if g not in chosen and len(chosen) < int(max_enrol):
                        chosen.append(g)
            mapping = {}
            if chosen:
                pad = np.zeros(int(0.5 * SR), dtype=np.float32)
                prefix, spans = [], []
                cursor = 0
                for g in chosen:
                    c = registry[g]["clip"]
                    spans.append((cursor // 160, (cursor + len(c)) // 160))
                    prefix += [c, pad]
                    cursor += len(c) + len(pad)
                prefix.append(np.zeros(SR, dtype=np.float32))
                cursor += SR
                _, joint = n3.infer(model, np.concatenate(prefix + [seg]), SR)
                owner = {}
                for g, (a, b) in zip(chosen, spans):
                    means = joint[a:b].mean(axis=0)
                    k = int(means.argmax())
                    if means[k] >= 0.5 and (k not in owner or means[k] > owner[k][1]):
                        owner[k] = (g, float(means[k]))
                mapping = {k: g for k, (g, _) in owner.items()}
                local = joint[cursor // 160:cursor // 160 + len(seg) // 160 + 1]
                active, single = clean_masks(local)
            for k in range(local.shape[1]):
                if active[:, k].sum() * 0.01 < 0.3:
                    continue
                if k not in mapping:
                    c = enrolment_clip(seg, single[:, k], seconds=clip)
                    v = embedder.track(seg, single[:, k].astype(np.float32))
                    if c is None or v is None:
                        # Too little clean speech to enrol: a speaker for this window only.
                        registry.append({"clip": None, "vec": np.zeros(256, dtype=np.float32)})
                    else:
                        registry.append({"clip": c, "vec": v})
                    columns.append(np.zeros(total_frames, dtype=np.float32))
                    mapping[k] = len(registry) - 1
                g = mapping[k]
                n = min(len(local), total_frames - f0)
                columns[g][f0:f0 + n] = np.maximum(columns[g][f0:f0 + n], local[:n, k])
            # Unenrollable speakers have no clip; keep them out of later candidate lists.
            for r in registry:
                if r["clip"] is None:
                    r["vec"] = np.zeros_like(r["vec"])
    finally:
        embedder.close()
    activity = np.stack(columns, axis=1) if columns else np.zeros((total_frames, 0), dtype=np.float32)
    return activity, 0.01


def production(backend_name):
    """A shipped backend end to end on the same waveform -- the reference arms."""
    def method(ident, probs, wave, **_):
        import torch
        from finesub.speech.diarization.stage import resolve_backend

        backend = resolve_backend(backend_name)
        handle = backend.ensure(None, "cuda")
        out = backend.activity(torch.from_numpy(wave).unsqueeze(0), handle=handle, device="cuda",
                               max_speakers=None).activity
        return out.float().cpu().numpy(), backend.frame_seconds
    return method


METHODS = {"n3": method_n3, "vbx": method_vbx, "enroll": method_enroll, "recur": method_recur,
           "recurvbx": method_recurvbx, "enrollmerge": method_enrollmerge,
           "diarizen": production("diarizen"), "sortformer": production("sortformer")}


def result_path(method, ident):
    return OUT / "methods" / method / f"{ident}.npz"


def run(method, datasets, **params):
    import vram_guard
    vram_guard.install()

    name = method + "".join(f"-{k}{v}" for k, v in sorted(params.items()))
    for dataset in datasets:
        for ident in idents(dataset):
            path = result_path(name, ident)
            if path.exists():
                continue
            probs = np.load(probs_path(ident))["probs"]
            wave = load_audio(ident)
            t = time.perf_counter()
            activity, frame = METHODS[method](ident, probs, wave, **params)
            elapsed = time.perf_counter() - t
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, activity=activity, frame=frame, elapsed=elapsed)
            print(f"{name} {ident} tracks {activity.shape[1]} {elapsed:.1f}s", flush=True)


# --- scoring -----------------------------------------------------------------------------------

def hypothesis(activity, frame):
    from finesub.speech.diarization.segments import segments_from_activity

    return [(t.start, t.end, str(t.track)) for t in segments_from_activity(activity.tolist(), frame_seconds=frame)]


def cue_scores(reference_rows, hyp, duration):
    """Mocai: per cue, the most active track; many-to-one owner and one-to-one (Hungarian) accuracy."""
    from scipy.optimize import linear_sum_assignment

    people = sorted({w for *_, w in reference_rows})
    tracks = sorted({k for *_, k in hyp})
    if not tracks:
        return dict(many=0.0, one=0.0, tracks=0, people=len(people), owned=0)
    ti = {k: i for i, k in enumerate(tracks)}
    step = 0.08
    frames = int(duration / step) + 2
    act = np.zeros((frames, len(tracks)), dtype=bool)
    for a, b, k in hyp:
        act[int(a / step):int(np.ceil(b / step)), ti[k]] = True
    lead, weight, truth = [], [], []
    overlap = np.zeros((len(people), len(tracks)))
    for a, b, w in reference_rows:
        lo, hi = max(0, int(a / step)), min(frames, max(int(a / step) + 1, int(np.ceil(b / step))))
        span = act[lo:hi].sum(axis=0)
        overlap[people.index(w)] += span * step
        lead.append(int(span.argmax()) if span.any() else -1)
        weight.append(b - a)
        truth.append(people.index(w))
    owner = overlap.argmax(axis=0)                       # many-to-one: each track to its best person
    rows, cols = linear_sum_assignment(-overlap)         # one-to-one
    one = {c: r for r, c in zip(rows, cols)}
    weight = np.array(weight)
    many = sum(wt for l, t, wt in zip(lead, truth, weight) if l >= 0 and owner[l] == t) / weight.sum()
    oto = sum(wt for l, t, wt in zip(lead, truth, weight) if l >= 0 and one.get(l) == t) / weight.sum()
    owned = len({owner[k] for k in range(len(tracks)) if act[:, k].sum() * step >= 20})
    return dict(many=many, one=oto, tracks=len(tracks), people=len(people), owned=owned)


def score_one(method, ident):
    from accuracy_overlap import score_intervals

    data = np.load(result_path(method, ident))
    hyp = hypothesis(data["activity"], float(data["frame"]))
    rows = reference(ident)
    duration = float(np.load(probs_path(ident))["probs"].shape[0]) * 0.01
    out = dict(tracks=len({k for *_, k in hyp}), people=len({w for *_, w in rows}), elapsed=float(data["elapsed"]))
    if ident.startswith("vox-") or is_ali(ident):
        m = score_intervals(rows, hyp, duration)
        out.update(der=m["der"], miss=m["miss"], fa=m["false_alarm"], conf=m["confusion"],
                   ovl_recall=m["overlap_recall"], ovl_all=m["overlap_all_speakers_recall"])
    else:
        out.update(cue_scores(rows, hyp, duration))
    return out


def score_ours(methods, dataset):
    """The W2 / W2b ruler: edit cost (primary), fixed, B-cubed F1, tracks leading >= 10 rows.

    Both prices of an unlabelled row are reported with the abstention count (findings 11.2 makes the count
    part of the number): `edit` is the owner's labelled-rows-only price, `priced` also charges the rows a
    timeline left silent (`price_unlabelled`), which is the setting that compares backends of different
    coverage. hina is also split by genre (星穹铁道 前瞻 vs 原神), whose casts and pacing differ."""
    # ⚠ nemotron3_cost first: it points lab.WT at the main checkout, and backend_cost freezes
    # V2_OUT from lab.WT at import -- the other order silently finds no cached DiariZen run.
    import nemotron3_cost as nc
    import backend_cost as bc
    import cost_metrics as cm
    import refined_multi as rm
    from accuracy_identity import identity_score
    from accuracy_report import bootstrap_mean_ci
    from finesub.speech.diarization.segments import segments_from_activity

    keys = nc.corpus_keys(dataset)
    titles = json.loads(Path(rm.IDS).read_text(encoding="utf-8")) if dataset == "hina" else {}
    table = {m: {} for m in ["dz-cached"] + methods}
    for key in keys:
        loaded = nc.load_episode(dataset, key)
        if loaded is None:
            continue
        ident, stable, grid, people, starts, arms = loaded
        rasters = {"dz-cached": arms["diarizen"]}
        for m in methods:
            if result_path(m, ident).exists():
                data = np.load(result_path(m, ident))
                tracks = [{"start": t.start, "end": t.end, "track": t.track} for t in
                          segments_from_activity(data["activity"].tolist(), frame_seconds=float(data["frame"]))]
                rasters[m] = bc.timeline_probs(tracks, len(grid))
        for m, probs in rasters.items():
            probs = probs[: len(grid)]
            free = cm.edit_cost(probs, stable, grid, people, starts)
            table[m][ident] = dict(
                cost=free["total"], abstained=free["abstained"],
                priced=float(cm.edit_cost(probs, stable, grid, people, starts, price_unlabelled=True)["total"]),
                fixed=float(lab_score(probs, stable, grid, people)["fixed"]),
                f1=float(identity_score(probs, stable, grid)["f1"]),
                lead10=int(cm.leading_counts(probs, stable, grid)[10]), people=len(people),
                genre=("hsr" if "星穹铁道" in titles.get(ident, {}).get("title", "") else "genshin") if titles else "")
    groups = [("all", None)] + ([("星穹铁道", "hsr"), ("原神", "genshin")] if titles else [])
    for label, genre in groups:
        print(f"\n== {dataset} {label} (edit = labelled rows only, priced = silent rows charged; paired vs {methods[0]})")
        print(f"{'method':<22}{'n':>4}{'edit':>9}{'priced':>9}{'abst':>7}{'fixed':>8}{'B3 F1':>8}{'lead10':>8}"
              f"   paired edit (95% CI)      paired priced (95% CI)")
        base = table[methods[0]]
        for m, rows in table.items():
            common = [i for i in rows if i in base and (genre is None or rows[i]["genre"] == genre)]
            if not common:
                continue
            cells = []
            for k in ("cost", "priced"):
                diffs = [rows[i][k] - base[i][k] for i in common]
                ci = bootstrap_mean_ci(diffs) if m != methods[0] else None
                cells.append(f"{np.mean(diffs):+8.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}] {sum(d < 0 for d in diffs)}/{len(diffs)}"
                             if ci else "")
            mean = lambda k: np.mean([rows[i][k] for i in common])  # noqa: E731
            print(f"{m:<22}{len(common):>4}{mean('cost'):>9.1f}{mean('priced'):>9.1f}{mean('abstained'):>7.1f}"
                  f"{100*mean('fixed'):>8.2f}{100*mean('f1'):>8.2f}{mean('lead10'):>8.2f}   {cells[0]:<26}{cells[1]}")
    (OUT / f"score-{dataset}.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


def lab_score(probs, stable, grid, people):
    import accuracy_lab as lab
    return lab.score(probs, stable, grid, people)


def score(methods, datasets):
    for dataset in datasets:
        if dataset in ("hina", "onetonton"):
            score_ours(methods, dataset)
            continue
        ids = idents(dataset)
        print(f"\n== {dataset} ({len(ids)} files)")
        table = {m: {i: score_one(m, i) for i in ids if result_path(m, i).exists()} for m in methods}
        if dataset in ("vox", "voxs", "ali"):
            print(f"{'method':<22}{'n':>4}{'DER':>8}{'miss':>7}{'FA':>7}{'conf':>7}{'|trk-ref|':>10}"
                  f"{'ovl rec':>9}{'ovl all':>9}{'sec/file':>10}")
            for m, rows in table.items():
                v = list(rows.values())
                if not v:
                    continue
                print(f"{m:<22}{len(v):>4}{100*np.mean([r['der'] for r in v]):>8.2f}"
                      f"{100*np.mean([r['miss'] for r in v]):>7.2f}{100*np.mean([r['fa'] for r in v]):>7.2f}"
                      f"{100*np.mean([r['conf'] for r in v]):>7.2f}"
                      f"{np.mean([abs(r['tracks'] - r['people']) for r in v]):>10.2f}"
                      f"{100*np.mean([r['ovl_recall'] for r in v]):>9.2f}"
                      f"{100*np.mean([r['ovl_all'] for r in v]):>9.2f}"
                      f"{np.mean([r['elapsed'] for r in v]):>10.1f}")
        else:
            print(f"{'method':<22}" + "".join(f"{i:>26}" for i in ids))
            for m, rows in table.items():
                print(f"{m:<22}" + "".join(
                    f"   many {100*r['many']:5.1f} one {100*r['one']:5.1f} t{r['tracks']:>2}" if r else f"{'-':>26}"
                    for r in (rows.get(i) for i in ids)))
                v = list(rows.values())
                if v:
                    print(f"{'':<22}   mean many {100*np.mean([r['many'] for r in v]):5.1f} "
                          f"one {100*np.mean([r['one'] for r in v]):5.1f}   "
                          f"people {[r['people'] for r in v]}  owned {[r['owned'] for r in v]}  "
                          f"sec/file {np.mean([r['elapsed'] for r in v]):.1f}")
        base = table.get(methods[0], {})
        for m in methods[1:]:
            common = [i for i in table[m] if i in base]
            if dataset in ("vox", "voxs", "ali") and common:
                d = [table[m][i]["der"] - base[i]["der"] for i in common]
                print(f"  {m} - {methods[0]}: DER {100*np.mean(d):+.2f} pp, better on "
                      f"{sum(x < 0 for x in d)}/{len(d)}")
        (OUT / f"score-{dataset}.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    sub.add_parser("infer").add_argument("datasets", nargs="+")
    r = sub.add_parser("run")
    r.add_argument("method", choices=sorted(METHODS))
    r.add_argument("datasets", nargs="+")
    r.add_argument("--param", action="append", default=[], help="key=value, floats")
    s = sub.add_parser("score")
    s.add_argument("--methods", nargs="+", default=["n3"])
    s.add_argument("datasets", nargs="+")
    args = p.parse_args()
    if args.cmd == "prep":
        prep()
    elif args.cmd == "infer":
        infer(args.datasets)
    elif args.cmd == "run":
        run(args.method, args.datasets, **{k: float(v) for k, v in (x.split("=") for x in args.param)})
    else:
        score(args.methods, args.datasets)


if __name__ == "__main__":
    main()
