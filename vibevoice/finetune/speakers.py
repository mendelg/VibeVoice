"""Speaker-line handling shared by fine-tuning data code and inference helpers.

VibeVoice scripts are lines of the form ``Speaker N: text``. The processor labels the voice
prompts it is given ``Speaker 0``, ``Speaker 1``, ... *by position*, so the text lines and the
prompt list agree only when speaker ids are 0-based in order of first appearance. These helpers
renumber a script that way and align the voice prompts with it, so multi-speaker (podcast)
training rows are built exactly the way inference builds them.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

SPEAKER_LINE = re.compile(r"^\s*Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)


def parse_speaker_lines(text: str, default_speaker: int = 0) -> List[Tuple[int, str]]:
    """Split a script into ``(speaker_id, text)`` lines.

    Lines without a ``Speaker N:`` prefix continue the previous line; leading plain text goes to
    ``default_speaker``. Empty lines are ignored. Consecutive lines of the same speaker are kept
    separate (they are separate lines in the script the processor sees).
    """
    lines: List[Tuple[int, str]] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = SPEAKER_LINE.match(line)
        if m:
            lines.append((int(m.group(1)), m.group(2).strip()))
        elif lines:
            prev_id, prev_text = lines[-1]
            lines[-1] = (prev_id, (prev_text + " " + line).strip())
        else:
            lines.append((default_speaker, line))
    return [(sid, t) for sid, t in lines if t]


def speaker_order(lines: Sequence[Tuple[int, Any]]) -> List[int]:
    """Distinct speaker ids in order of first appearance."""
    order: List[int] = []
    for sid, _ in lines:
        if sid not in order:
            order.append(sid)
    return order


def normalize_script(
    text: str,
    voice_prompts: Optional[Union[Any, Sequence[Any], Dict[Any, Any]]] = None,
) -> Tuple[str, Optional[List[Any]], List[int]]:
    """Renumber speakers to ``0..N-1`` by first appearance and align ``voice_prompts`` with them.

    ``voice_prompts`` may be ``None``, a single prompt (path or waveform), a list ordered by first
    appearance of the speakers in ``text``, or a dict keyed by the *original* speaker id
    (``1`` or ``"1"``). Returns ``(script, prompts, original_ids)`` where ``prompts`` is a list
    with one entry per speaker in the new order, or ``None``. A list shorter than the number of
    speakers, or a dict missing a speaker, raises ``ValueError``: the processor would otherwise
    silently leave some speakers without a voice.
    """
    lines = parse_speaker_lines(text)
    if not lines:
        raise ValueError("Script has no text")
    order = speaker_order(lines)
    remap = {sid: k for k, sid in enumerate(order)}
    script = "\n".join(f"Speaker {remap[sid]}: {t}" for sid, t in lines)

    prompts: Optional[List[Any]]
    if voice_prompts is None:
        prompts = None
    elif isinstance(voice_prompts, dict):
        picked = []
        for sid in order:
            value = voice_prompts.get(sid, voice_prompts.get(str(sid)))
            if value is None:
                raise ValueError(f"voice_prompts has no entry for Speaker {sid} (speakers in text: {order})")
            picked.append(value)
        prompts = picked
    elif isinstance(voice_prompts, (list, tuple)):
        items = [v for v in voice_prompts if v is not None]
        if len(items) < len(order):
            raise ValueError(
                f"{len(items)} voice prompt(s) for {len(order)} speakers {order}; give one prompt per speaker "
                "in order of first appearance, or a dict keyed by speaker id"
            )
        prompts = list(items[: len(order)])
    else:
        if len(order) > 1:
            raise ValueError(f"A single voice prompt was given for {len(order)} speakers {order}")
        prompts = [voice_prompts]
    return script, prompts, order


def count_speakers(text: str) -> int:
    return len(speaker_order(parse_speaker_lines(text)))
