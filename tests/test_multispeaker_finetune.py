"""Multi-speaker (podcast) fine-tuning: speaker normalization, window building, and the
training/inference token contract of the collator."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vibevoice.finetune import prepare_podcast_jsonl as prep
from vibevoice.finetune.speakers import count_speakers, normalize_script, parse_speaker_lines


# ----------------------------------------------------------------------------- speakers.py

def test_parse_speaker_lines_handles_continuations_and_plain_text():
    text = "hello there\nSpeaker 2: first\ncontinued here\n\nSpeaker 1: reply"
    assert parse_speaker_lines(text) == [(0, "hello there"), (2, "first continued here"), (1, "reply")]


def test_normalize_script_renumbers_by_first_appearance_and_orders_prompts():
    text = "Speaker 2: אַ גוטן טאָג\nSpeaker 1: ברוך השם\nSpeaker 2: וואָס הערט זיך?"
    script, prompts, order = normalize_script(text, ["two.wav", "one.wav"])
    assert order == [2, 1]
    assert script == "Speaker 0: אַ גוטן טאָג\nSpeaker 1: ברוך השם\nSpeaker 0: וואָס הערט זיך?"
    assert prompts == ["two.wav", "one.wav"]
    script_d, prompts_d, _ = normalize_script(text, {"1": "one.wav", "2": "two.wav"})
    assert script_d == script and prompts_d == ["two.wav", "one.wav"]
    assert count_speakers(script) == 2


def test_normalize_script_rejects_too_few_prompts():
    with pytest.raises(ValueError):
        normalize_script("Speaker 1: a\nSpeaker 2: b", ["only.wav"])
    with pytest.raises(ValueError):
        normalize_script("Speaker 1: a\nSpeaker 2: b", {"1": "one.wav"})
    with pytest.raises(ValueError):
        normalize_script("Speaker 1: a\nSpeaker 2: b", "single.wav")
    script, prompts, _ = normalize_script("Speaker 1: a", "single.wav")
    assert (script, prompts) == ("Speaker 0: a", ["single.wav"])
    script, prompts, _ = normalize_script("Speaker 1: a", ["first.wav", "spare.wav"])
    assert prompts == ["first.wav"]  # extra prompts are ignored, as the processor does


# ----------------------------------------------------------------------------- windows

def turns_fixture():
    A, B = "SPEAKER_00", "SPEAKER_01"
    return [
        dict(speaker=A, start=0.0, end=4.0, text="a1"),
        dict(speaker=A, start=4.3, end=7.0, text="a2"),          # merged into a1 (gap .3)
        dict(speaker=B, start=7.2, end=7.5, text=""),            # short back-channel: dropped, stays inside
        dict(speaker=B, start=7.6, end=12.0, text="b1"),
        dict(speaker=A, start=12.4, end=16.0, text="a3"),
        dict(speaker=B, start=16.2, end=19.0, text="b2", confidence=0.4),  # low confidence: barrier
        dict(speaker=A, start=19.5, end=25.0, text="a4"),
        dict(speaker=B, start=25.2, end=31.0, text="b3"),
        dict(speaker=A, start=36.0, end=40.0, text="a5"),        # gap 5 s ends the window
        dict(speaker=B, start=40.5, end=43.0, text=""),          # long textless: barrier
        dict(speaker=A, start=43.5, end=52.0, text="a6"),
    ]


def test_merge_turns_and_windows():
    turns = prep.merge_turns(turns_fixture(), merge_gap=0.6, min_turn=0.4, barrier_textless=1.0, min_confidence=0.6)
    assert [t["text"] for t in turns] == ["a1 a2", "b1", "a3", "b2", "a4", "b3", "a5", "", "a6"]
    assert [t["barrier"] for t in turns] == [False, False, False, True, False, False, False, True, False]
    windows = prep.make_windows(turns, min_seconds=8, max_seconds=60, max_gap=2.0, max_speakers=4, require_speaker_change=True)
    assert windows == [(0, 2), (4, 5)]  # barrier b2 ends the first; the 5 s gap ends the second; a5/a6 have no partner
    text, order = prep.window_text(turns[0:3])
    assert order == ["SPEAKER_00", "SPEAKER_01"]
    assert text == "Speaker 0: a1 a2\nSpeaker 1: b1\nSpeaker 0: a3"
    windows_any = prep.make_windows(turns, min_seconds=8, max_seconds=60, max_gap=2.0, max_speakers=4, require_speaker_change=False)
    assert (8, 8) in windows_any  # a6 alone is 8.5 s, single speaker


def test_choose_prompt_prefers_outside_window_in_range():
    turns = prep.merge_turns(turns_fixture(), merge_gap=0.6, min_turn=0.4, barrier_textless=1.0, min_confidence=0.6)
    a_turns = [t for t in turns if t["speaker"] == "SPEAKER_00"]
    c = prep.choose_prompt(a_turns, (0.0, 16.0), "k", prompt_min=3.0, prompt_max=12.0)
    assert c is not None and c["start"] >= 16.0 and c["text"]
    assert prep.choose_prompt(a_turns, (0.0, 60.0), "k", 3.0, 12.0) is None


def test_choose_split_holds_out_whole_episodes():
    splits = prep.choose_split([f"ep{i}" for i in range(10)], 0.1, seed=3)
    assert sum(v == "validation" for v in splits.values()) == 1
    assert prep.choose_split(["only"], 0.1, seed=0) == {"only": "train"}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_prepare_end_to_end(tmp_path):
    import soundfile as sf
    sr = 24000
    t = np.arange(int(60 * sr)) / sr
    audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wav = tmp_path / "ep.wav"
    sf.write(wav, audio, sr)
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps(dict(id="ep", audio=str(wav), turns=turns_fixture())) + "\n")
    out = tmp_path / "out"
    rc = prep.main(["--episodes", str(episodes), "--out", str(out), "--require-speaker-change",
                    "--min-confidence", "0.6", "--validation-fraction", "0"])
    assert rc == 0
    rows = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and not (out / "validation.jsonl").read_text().strip()
    first = rows[0]
    assert first["text"] == "Speaker 0: a1 a2\nSpeaker 1: b1\nSpeaker 0: a3"
    assert first["num_speakers"] == 2 and len(first["voice_prompts"]) == 2
    for path in [first["audio"], *first["voice_prompts"]]:
        data, rate = sf.read(path)
        assert rate == 24000 and len(data) > 24000
    win, _ = sf.read(first["audio"])
    assert abs(len(win) / 24000 - first["duration"]) < 0.05
    # prompts come from outside the window
    for m in first["voice_prompts"]:
        start = float(Path(m).stem.split("_", 1)[1])
        assert start >= 16.0
    assert (out / "preparation.json").exists()


# ----------------------------------------------------------------------------- collator contract

@pytest.fixture(scope="module")
def processor():
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    try:
        return VibeVoiceProcessor.from_pretrained("vibevoice/VibeVoice-1.5B")
    except Exception as e:  # pragma: no cover - offline
        pytest.skip(f"processor unavailable: {e}")


def test_collator_multispeaker_matches_inference_prefix(processor):
    """A two-speaker training row is [inference prefix for the same script+prompts] + target
    placeholders + speech_end + eos; only the target latents get loss; prompts are input only."""
    from vibevoice.finetune.data_vibevoice import VibeVoiceCollator, VibeVoiceDataset

    sr = 24000
    p0 = np.random.RandomState(0).randn(4 * sr).astype(np.float32) * 0.05
    p1 = np.random.RandomState(1).randn(6 * sr).astype(np.float32) * 0.05
    target = np.random.RandomState(2).randn(20 * sr).astype(np.float32) * 0.05
    raw_text = "Speaker 1: אַ גוטן טאָג, וואָס הערט זיך?\nSpeaker 2: ברוך השם, אַלץ איז גוט.\nSpeaker 1: שיין."
    ds = VibeVoiceDataset([dict(text=raw_text, audio=target, voice_prompts=[p0, p1])], normalize_speaker_ids=True)
    item = ds[0]
    assert item["text"].startswith("Speaker 0:") and "Speaker 1:" in item["text"] and "Speaker 2" not in item["text"]
    assert len(item["voice_prompts"]) == 2

    class _StubSemanticTokenizer:  # token layout does not depend on the semantic features' values
        def encode(self, w):
            return np.zeros((int(np.ceil(len(w) / 3200)), 128), dtype=np.float32)

    processor.semantic_tokenizer = _StubSemanticTokenizer()
    collator = VibeVoiceCollator(processor=processor, compute_semantics=True, voice_prompt_drop_rate=0.0)
    batch = collator([item])
    ids = batch["input_ids"][0].tolist()
    ain = batch["acoustic_input_mask"][0]
    aloss = batch["acoustic_loss_mask"][0]

    ref = processor(text=[item["text"]], voice_samples=[item["voice_prompts"]], padding=False, return_tensors="pt")
    prefix = ref["input_ids"][0].tolist()
    assert ids[: len(prefix)] == prefix, "training prefix must equal the inference prompt"
    assert ref["all_speakers_list"][0] and sorted(ref["all_speakers_list"][0]) == [0, 1]

    tok = processor.tokenizer
    prompt_latents = sum(int(np.ceil(len(w) / 3200)) for w in (p0, p1))
    # target gets 0.25 s + 0.75 s of silence padding in the collator
    target_latents = int(np.ceil((len(target) + sr) / 3200))
    assert int(ain[: len(prefix)].sum()) == prompt_latents
    assert int(aloss[: len(prefix)].sum()) == 0
    assert int(aloss.sum()) == target_latents
    assert ids[len(prefix): len(prefix) + target_latents] == [tok.speech_diffusion_id] * target_latents
    assert ids[len(prefix) + target_latents] == tok.speech_end_id
    assert ids[len(prefix) + target_latents + 1] == tok.eos_id
    assert batch["speeches_loss_input"].shape[0] == 3 and batch["speeches_loss_input"][:2].sum() == 0
    assert bool(batch["speeches_loss_input"][2].any())
    # the 'Voice input' section labels prompts Speaker 0 and Speaker 1, same ids as the text lines
    decoded = tok.decode([i for i in prefix if i not in (tok.speech_diffusion_id,)])
    assert " Speaker 0:" in decoded.split(" Text input:")[0] and " Speaker 1:" in decoded.split(" Text input:")[0]


def test_dataset_multispeaker_without_prompts_trains_promptless(processor):
    from vibevoice.finetune.data_vibevoice import VibeVoiceDataset
    target = np.zeros(24000 * 5, dtype=np.float32)
    ds = VibeVoiceDataset([dict(text="Speaker 1: a\nSpeaker 2: b", audio=target)], normalize_speaker_ids=True)
    with pytest.warns(UserWarning):
        item = ds[0]
    assert item["voice_prompts"] is None
    ds_bad = VibeVoiceDataset([dict(text="Speaker 1: a\nSpeaker 2: b", audio=target, voice_prompts=["x.wav"])])
    with pytest.raises(ValueError):
        ds_bad[0]


# ----------------------------------------------------------------------------- LM cross-entropy labels

def test_mask_for_ce_supervises_continue_and_skips_prefix():
    import torch
    from vibevoice.finetune.train_vibevoice import mask_for_ce
    # tokens: [sys, sys, P, P, txt, txt, start, T, T, T, end, eos, pad]  (P = prompt latents, T = target latents)
    ids = torch.tensor([[10, 11, 99, 99, 12, 13, 50, 99, 99, 99, 51, 52, 0]])
    attn = torch.tensor([[1] * 12 + [0]])
    ain = torch.tensor([[False, False, True, True, False, False, False, True, True, True, False, False, False]])
    aloss = torch.tensor([[False] * 7 + [True] * 3 + [False] * 3])

    default = mask_for_ce(ids, attn, ain)[0].tolist()
    assert default == [11, -100, -100, 12, 13, 50, -100, -100, -100, 51, 52, -100]  # upstream: no 'continue' labels

    cont = mask_for_ce(ids, attn, ain, acoustic_loss_mask=aloss, include_speech_tokens=True)[0].tolist()
    assert cont == [11, -100, -100, 12, 13, 50, 99, 99, 99, 51, 52, -100]  # target placeholders supervised, prompt ones not

    both = mask_for_ce(ids, attn, ain, acoustic_loss_mask=aloss, include_speech_tokens=True, skip_text_prefix=True)[0].tolist()
    assert both == [-100] * 6 + [99, 99, 99, 51, 52, -100]  # only continue decisions, speech_end and eos remain

    skip_only = mask_for_ce(ids, attn, ain, acoustic_loss_mask=aloss, skip_text_prefix=True)[0].tolist()
    assert skip_only == [-100] * 9 + [51, 52, -100]

    # a row without any target keeps the upstream labels under skip_text_prefix
    none_loss = torch.zeros_like(aloss)
    assert mask_for_ce(ids, attn, ain, acoustic_loss_mask=none_loss, skip_text_prefix=True)[0].tolist() == default
