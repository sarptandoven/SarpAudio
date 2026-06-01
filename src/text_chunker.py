from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional







_PREFIX_RE = re.compile(r'^([^"\']{3,}?)(,\s*)(?=["\'])', re.DOTALL)


@dataclass
class PromptChunk:
    text: str
    est_duration_s: float


def extract_speaker_prefix(prompt: str) -> tuple[Optional[str], str]:
    m = _PREFIX_RE.match(prompt)
    if not m:
        return None, prompt
    return m.group(1).strip(), prompt[m.end():]


def split_sentences_outside_quotes(text: str) -> List[str]:
    sentences: List[str] = []
    buf: List[str] = []
    in_double = False
    in_single = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)

        if ch == '"' and not in_single:
            was_inside = in_double
            in_double = not in_double


            if was_inside and len(buf) >= 2 and buf[-2] in ".!?":

                if i + 1 >= n or text[i + 1].isspace():
                    sentence = "".join(buf).strip()
                    if sentence:
                        sentences.append(sentence)
                    buf = []
                    i += 1
                    continue

        elif ch == "'" and not in_double:

            prev = text[i - 1] if i > 0 else " "
            nxt = text[i + 1] if i + 1 < n else " "
            if not (prev.isalpha() and nxt.isalpha()):
                in_single = not in_single

        elif ch in ".!?" and not in_double and not in_single:

            j = i + 1
            while j < n and text[j] in '."\')]':
                buf.append(text[j])
                if text[j] == '"':
                    in_double = not in_double
                j += 1
            if j >= n or text[j].isspace():
                sentence = "".join(buf).strip()
                if sentence:
                    sentences.append(sentence)
                buf = []
                i = j
                continue
        i += 1

    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _assemble(prefix: Optional[str], sentences: List[str]) -> str:
    body = " ".join(s.strip() for s in sentences if s.strip())
    if not prefix:
        return body



    if body.lstrip().startswith(("'", '"')):
        return f"{prefix}, {body}"
    return f"{prefix}. {body}"


def chunk_prompt_for_duration(
    prompt: str,
    max_duration_s: float = 45.0,
    target_duration_s: float = 37.0,
    duration_multiplier: float = 1.1,
) -> List[PromptChunk]:
    from duration_estimator import estimate_speech_duration

    def _est(t: str) -> float:
        return estimate_speech_duration(t) * duration_multiplier

    total = _est(prompt)
    if total <= max_duration_s:
        return [PromptChunk(text=prompt, est_duration_s=total)]

    prefix, body = extract_speaker_prefix(prompt)
    sentences = split_sentences_outside_quotes(body)
    if not sentences:


        sentences = body.split()

    chunks: List[PromptChunk] = []
    current: List[str] = []
    current_dur = 0.0

    for sent in sentences:
        candidate = _assemble(prefix, current + [sent])
        cand_dur = _est(candidate)

        if current and cand_dur > target_duration_s:

            assembled = _assemble(prefix, current)
            chunks.append(PromptChunk(text=assembled, est_duration_s=_est(assembled)))
            current = [sent]
            current_dur = _est(_assemble(prefix, current))
        else:
            current.append(sent)
            current_dur = cand_dur





        if len(current) == 1 and current_dur > max_duration_s:
            solo = _assemble(prefix, current)
            chunks.append(PromptChunk(text=solo, est_duration_s=current_dur))
            current = []
            current_dur = 0.0

    if current:
        assembled = _assemble(prefix, current)
        chunks.append(PromptChunk(text=assembled, est_duration_s=_est(assembled)))

    return chunks
