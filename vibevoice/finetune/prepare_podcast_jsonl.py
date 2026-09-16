#!/usr/bin/env python3
"""Build multi-speaker (podcast) fine-tuning rows from diarized, transcribed episodes.

Input: one JSON object per episode (a JSONL file, or a directory of ``*.json`` files)::

    {"id": "ep01", "audio": "/data/ep01.mp3",
     "turns": [{"speaker": "A", "start": 0.4, "end": 3.9, "text": "...", "confidence": 0.95}, ...],
     "prompts": {"A": "/data/ep01_A.wav"}}          # optional fixed voice prompts per speaker

Output (``--out``): ``train.jsonl`` / ``validation.jsonl`` rows the trainer reads directly with
``--normalize_speaker_ids True``::

    {"text": "Speaker 0: ...\\nSpeaker 1: ...", "audio": ".../windows/ep01/000123.wav",
     "voice_prompts": [".../prompts/ep01/A_120.40.wav", ".../prompts/ep01/B_88.10.wav"], ...}

Each row is a window of consecutive turns cut from the original recording (so the gaps, overlaps
and reactions between speakers are real), with one voice prompt per speaker taken from the same
episode *outside* the window. Speakers are numbered 0..N-1 by first appearance inside the window
and ``voice_prompts`` is in that order, which is how the processor labels prompts at inference.

Turns without text that last at least ``--barrier-textless`` seconds, and turns below
``--min-confidence``, end a window: audio the model would have to produce without text hurts
alignment. Shorter textless turns (back-channels) are simply dropped and stay inside the window.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ----------------------------------------------------------------------------- pure logic

def merge_turns(turns: Sequence[Dict[str, Any]], merge_gap: float, min_turn: float,
                barrier_textless: float, min_confidence: float) -> List[Dict[str, Any]]:
    """Sorted turns with consecutive same-speaker turns merged across short gaps.

    Returns dicts with keys speaker/start/end/text/confidence/barrier. ``barrier`` marks a stretch
    of audio that must not be inside a window (long textless speech or low-confidence transcript).
    """
    clean = []
    for t in sorted(turns, key=lambda t: (float(t["start"]), float(t["end"]))):
        start, end = float(t["start"]), float(t["end"])
        if end <= start:
            continue
        text = " ".join(str(t.get("text") or "").split())
        conf = t.get("confidence")
        conf = float(conf) if conf is not None else 1.0
        if not text:
            if end - start >= barrier_textless:
                clean.append(dict(speaker=t["speaker"], start=start, end=end, text="", confidence=conf, barrier=True))
            continue  # short back-channel without text: ignore, stays inside whatever window covers it
        if end - start < min_turn:
            continue
        clean.append(dict(speaker=t["speaker"], start=start, end=end, text=text, confidence=conf,
                          barrier=conf < min_confidence))
    merged: List[Dict[str, Any]] = []
    for t in clean:
        prev = merged[-1] if merged else None
        if (prev is not None and not prev["barrier"] and not t["barrier"] and prev["speaker"] == t["speaker"]
                and t["start"] - prev["end"] <= merge_gap):
            prev["end"] = max(prev["end"], t["end"])
            prev["text"] = (prev["text"] + " " + t["text"]).strip()
            prev["confidence"] = min(prev["confidence"], t["confidence"])
        else:
            merged.append(dict(t))
    return merged


def make_windows(turns: Sequence[Dict[str, Any]], *, min_seconds: float, max_seconds: float, max_gap: float,
                 max_speakers: int, require_speaker_change: bool, overlap_turns: int = 0) -> List[Tuple[int, int]]:
    """Index ranges ``(i, j)`` (inclusive) of merged turns forming one training window each."""
    windows: List[Tuple[int, int]] = []
    n = len(turns)
    i = 0
    while i < n:
        if turns[i]["barrier"]:
            i += 1
            continue
        speakers = {turns[i]["speaker"]}
        j = i
        while j + 1 < n:
            nxt = turns[j + 1]
            if nxt["barrier"] or nxt["start"] - turns[j]["end"] > max_gap:
                break
            if nxt["end"] - turns[i]["start"] > max_seconds:
                break
            if nxt["speaker"] not in speakers and len(speakers) >= max_speakers:
                break
            speakers.add(nxt["speaker"])
            j += 1
        duration = turns[j]["end"] - turns[i]["start"]
        if duration >= min_seconds and (len(speakers) > 1 or not require_speaker_change):
            windows.append((i, j))
            i = max(i + 1, j + 1 - overlap_turns) if overlap_turns else j + 1
        else:
            i += 1
    return windows


def window_text(turns: Sequence[Dict[str, Any]]) -> Tuple[str, List[Any]]:
    """Script with speakers renumbered 0..N-1 by first appearance, plus that speaker order."""
    order: List[Any] = []
    for t in turns:
        if t["speaker"] not in order:
            order.append(t["speaker"])
    lines = [f"Speaker {order.index(t['speaker'])}: {t['text']}" for t in turns]
    return "\n".join(lines), order


def choose_prompt(candidates: Sequence[Dict[str, Any]], window: Tuple[float, float], key: str,
                  prompt_min: float, prompt_max: float) -> Optional[Dict[str, Any]]:
    """A same-speaker turn outside ``window`` to use as voice prompt, chosen deterministically.

    Prefers turns whose duration is within [prompt_min, prompt_max]; otherwise the longest turn
    (clipped to prompt_max at cutting time). Returns None when the speaker never speaks outside
    the window.
    """
    w_start, w_end = window
    outside = [c for c in candidates if c["text"] and not c["barrier"] and (c["end"] <= w_start or c["start"] >= w_end)]
    if not outside:
        return None
    in_range = [c for c in outside if prompt_min <= c["end"] - c["start"] <= prompt_max]
    if in_range:
        h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        return in_range[h % len(in_range)]
    best = max(outside, key=lambda c: c["end"] - c["start"])
    return best if best["end"] - best["start"] >= 1.0 else None


def choose_split(episode_ids: Iterable[str], validation_fraction: float, seed: int) -> Dict[str, str]:
    ordered = sorted(episode_ids, key=lambda e: hashlib.sha256(f"{seed}:{e}".encode()).hexdigest())
    held = 0
    if validation_fraction > 0 and len(ordered) >= 2:
        held = max(1, round(len(ordered) * validation_fraction))
    return {e: ("validation" if k < held else "train") for k, e in enumerate(ordered)}


# ----------------------------------------------------------------------------- audio

def cut_audio(source: str, start: float, end: float, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".tmp.wav")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", source,
                    "-t", f"{end - start:.3f}", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp)], check=True)
    tmp.replace(dest)


# ----------------------------------------------------------------------------- driver

def load_episodes(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]
    episodes = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def build_rows(episode: Dict[str, Any], out: Path, a: argparse.Namespace, rejected: Counter) -> List[Dict[str, Any]]:
    ident = str(episode["id"])
    turns = merge_turns(episode["turns"], a.merge_gap, a.min_turn, a.barrier_textless, a.min_confidence)
    windows = make_windows(turns, min_seconds=a.min_seconds, max_seconds=a.max_seconds, max_gap=a.max_gap,
                           max_speakers=a.max_speakers, require_speaker_change=a.require_speaker_change,
                           overlap_turns=a.overlap_turns)
    by_speaker: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for t in turns:
        by_speaker[t["speaker"]].append(t)
    fixed_prompts = episode.get("prompts") or {}
    rows = []
    for i, j in windows:
        w = turns[i:j + 1]
        text, order = window_text(w)
        w_start, w_end = w[0]["start"], w[-1]["end"]
        prev_end = turns[i - 1]["end"] if i > 0 else 0.0
        next_start = turns[j + 1]["start"] if j + 1 < len(turns) else w_end + a.pad
        start = max(prev_end, w_start - a.pad)
        end = min(next_start, w_end + a.pad)
        wid = f"{i:06d}"
        prompts: List[str] = []
        prompt_meta = []
        ok = True
        for spk in order:
            if str(spk) in fixed_prompts:
                prompts.append(str(fixed_prompts[str(spk)]))
                prompt_meta.append(dict(speaker=spk, fixed=True))
                continue
            c = choose_prompt(by_speaker[spk], (w_start, w_end), f"{ident}:{wid}:{spk}", a.prompt_min, a.prompt_max)
            if c is None:
                ok = False
                break
            p_start = max(0.0, c["start"] - a.prompt_pad)
            p_end = min(c["end"] + a.prompt_pad, p_start + a.prompt_max)
            dest = out / "prompts" / ident / f"{spk}_{p_start:.2f}.wav"
            prompts.append(str(dest.resolve()))
            prompt_meta.append(dict(speaker=spk, start=p_start, end=p_end, source=episode["audio"], dest=str(dest)))
        if not ok:
            if a.allow_missing_prompts:
                prompts = []
                prompt_meta = []
            else:
                rejected["no_prompt_for_speaker"] += 1
                continue
        rows.append(dict(
            id=f"{ident}_{wid}", episode=ident, text=text,
            audio=str((out / "windows" / ident / f"{wid}.wav").resolve()),
            voice_prompts=prompts or None,
            start=start, end=end, duration=end - start, num_turns=len(w), num_speakers=len(order),
            speakers=[str(s) for s in order], speaker_changes=sum(1 for k in range(1, len(w)) if w[k]["speaker"] != w[k - 1]["speaker"]),
            min_confidence=min(t["confidence"] for t in w),
            _source=episode["audio"], _prompt_meta=prompt_meta,
        ))
    if not windows:
        rejected["episode_without_windows"] += 1
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", required=True, type=Path, help="JSONL file or directory of *.json episode files")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--min-seconds", type=float, default=8.0)
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--max-gap", type=float, default=2.0, help="Longer silence between turns ends a window")
    p.add_argument("--merge-gap", type=float, default=0.6, help="Same-speaker turns closer than this are one line")
    p.add_argument("--min-turn", type=float, default=0.4)
    p.add_argument("--barrier-textless", type=float, default=1.0, help="Textless turns at least this long end a window")
    p.add_argument("--min-confidence", type=float, default=0.0, help="Turns below this transcript confidence end a window")
    p.add_argument("--max-speakers", type=int, default=4)
    p.add_argument("--require-speaker-change", action="store_true", help="Keep only windows with 2+ speakers")
    p.add_argument("--overlap-turns", type=int, default=0, help="Start the next window this many turns before the end of the last")
    p.add_argument("--pad", type=float, default=0.15, help="Audio kept before/after the window, bounded by neighbouring turns")
    p.add_argument("--prompt-min", type=float, default=3.0)
    p.add_argument("--prompt-max", type=float, default=12.0)
    p.add_argument("--prompt-pad", type=float, default=0.1)
    p.add_argument("--allow-missing-prompts", action="store_true", help="Keep windows whose speaker has no prompt candidate (trained without prompts)")
    p.add_argument("--validation-fraction", type=float, default=0.1, help="Fraction of EPISODES held out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(4, min(32, os.cpu_count() or 8)))
    p.add_argument("--dry-run", action="store_true", help="Report counts and hours; cut nothing, write nothing")
    a = p.parse_args(argv)

    episodes = load_episodes(a.episodes)
    if not episodes:
        print(f"No episodes found at {a.episodes}", file=sys.stderr)
        return 2
    rejected: Counter = Counter()
    rows: List[Dict[str, Any]] = []
    for ep in episodes:
        if not Path(ep["audio"]).exists():
            rejected["missing_audio"] += 1
            continue
        rows.extend(build_rows(ep, a.out, a, rejected))
    splits = choose_split({r["episode"] for r in rows}, a.validation_fraction, a.seed)
    for r in rows:
        r["split"] = splits[r["episode"]]
    summary = {
        "episodes": len(episodes),
        "rejected": dict(rejected),
        "splits": {sp: dict(rows=sum(r["split"] == sp for r in rows),
                            hours=round(sum(r["duration"] for r in rows if r["split"] == sp) / 3600, 3),
                            multi_speaker_rows=sum(r["split"] == sp and r["num_speakers"] > 1 for r in rows),
                            speaker_changes=sum(r["speaker_changes"] for r in rows if r["split"] == sp))
                   for sp in ("train", "validation")},
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not rows:
        print("No windows produced; relax --min-seconds/--max-gap or check the turns.", file=sys.stderr)
        return 1
    if a.dry_run:
        return 0

    a.out.mkdir(parents=True, exist_ok=True)
    jobs: Dict[str, Tuple[str, float, float]] = {}
    for r in rows:
        jobs[r["audio"]] = (r["_source"], r["start"], r["end"])
        for m in r["_prompt_meta"]:
            if not m.get("fixed"):
                jobs[m["dest"]] = (m["source"], m["start"], m["end"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(cut_audio, src, s, e, Path(dest)) for dest, (src, s, e) in jobs.items()]
        for k, f in enumerate(concurrent.futures.as_completed(futures), 1):
            f.result()
            if k % 500 == 0:
                print(f"cut {k}/{len(futures)} files", flush=True)
    for sp in ("train", "validation"):
        with (a.out / f"{sp}.jsonl").open("w") as f:
            for r in sorted((r for r in rows if r["split"] == sp), key=lambda r: r["id"]):
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
    (a.out / "preparation.json").write_text(json.dumps(dict(summary, episode_splits=splits), indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {a.out}/train.jsonl and validation.jsonl. Listen to a few windows/*.wav against their text before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
