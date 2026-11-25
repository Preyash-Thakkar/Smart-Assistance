"""Single source of truth for complaint -> intent taxonomy.

The original ``clean_data.py`` carried this map inline.  Keeping it in one
module means the data builder, the audit script and the semantic checker all
agree on the label space.

The source database has *questions* (the canonical complaint text written by
the fitting team) and *topics* (Loudness, Noise, ...).  The intent codes below
are the acoustic concepts the DSP payloads are grouped by.

Notes
-----
* Keep the keys exactly as they appear in ``Real Data/test.complains.json``
  after :func:`normalize_text` (lower-case, collapsed whitespace, trailing dot
  removed).
* ``INTENT_MERGE_MAP`` is intentionally empty for P0.  Add entries here only
  after a domain/taxonomy decision, e.g.
  ``"NOISE_SOUND_UNCOMFORTABLE": "NOISE_SUDDEN_UNCOMFORTABLE"``.
"""

from __future__ import annotations

import html
import re
from difflib import get_close_matches
from typing import Iterable

# ---------------------------------------------------------------------------
# Canonical question -> intent map (derived from the fitting team's database)
# ---------------------------------------------------------------------------

QUESTION_TO_INTENT: dict[str, str] = {
    # Loudness
    "all soft sound are too soft": "LOUDNESS_SOFT_TOO_SOFT",
    "all soft sounds too soft": "LOUDNESS_SOFT_TOO_SOFT",
    "all loud sounds are too loud": "LOUDNESS_LOUD_TOO_LOUD",
    "all loud sounds too loud": "LOUDNESS_LOUD_TOO_LOUD",
    "loud sounds are distorted": "LOUDNESS_LOUD_DISTORTED",
    "loud sounds distorted": "LOUDNESS_LOUD_DISTORTED",
    "nearby sounds are loud": "LOUDNESS_NEARBY_TOO_LOUD",
    "nearby sounds too loud": "LOUDNESS_NEARBY_TOO_LOUD",
    "alarm/childrencrying are loud": "LOUDNESS_ALARMS_CRYING_LOUD",
    "alarms/children crying too loud": "LOUDNESS_ALARMS_CRYING_LOUD",
    "alarms/childr en crying too loud": "LOUDNESS_ALARMS_CRYING_LOUD",
    "sudden sounds/door slamming sounds are loud": "LOUDNESS_SUDDEN_SOUNDS_LOUD",
    "sudden sounds (e.g., door slams) too loud": "LOUDNESS_SUDDEN_SOUNDS_LOUD",
    "own chewing sounds are loud": "LOUDNESS_CHEWING_TOO_LOUD",
    "own chewing sounds too loud": "LOUDNESS_CHEWING_TOO_LOUD",
    "running water sound feels loud": "LOUDNESS_WATER_FAN_LOUD",
    "running fan sounds are loud": "LOUDNESS_WATER_FAN_LOUD",
    "running water/fan sounds too loud": "LOUDNESS_WATER_FAN_LOUD",
    # Own voice
    "own voice muffled/dull": "OWN_VOICE_MUFFLED_DULL",
    "own voice hollow": "OWN_VOICE_HOLLOW",
    "own voice too sharp/metallic": "OWN_VOICE_SHARP_METALLIC",
    "own voice thin": "OWN_VOICE_THIN",
    "own voice boomy/loud": "OWN_VOICE_BOOMY_LOUD",
    # Speech
    "speech too soft": "SPEECH_TOO_SOFT",
    "whisper/distant speech too soft": "SPEECH_WHISPER_DISTANT_SOFT",
    "whisper/dista nt speech too soft": "SPEECH_WHISPER_DISTANT_SOFT",
    "speech too loud/boomy": "SPEECH_TOO_LOUD_BOOMY",
    "speech echoic": "SPEECH_ECHOIC",
    "poor speech quality": "SPEECH_POOR_QUALITY",
    # General sound
    "sounds feel tiny & metallic & sharp": "SOUND_TINY_METALLIC_SHARP",
    "sounds tiny/metallic/sharp": "SOUND_TINY_METALLIC_SHARP",
    "sounds tiny/metallic/s harp": "SOUND_TINY_METALLIC_SHARP",
    "sounds tiny/metallic/ sharp": "SOUND_TINY_METALLIC_SHARP",
    "sounds are echoic": "SOUND_ECHOIC",
    "sounds echoic": "SOUND_ECHOIC",
    "sounds are muffled or dull": "SOUND_MUFFLED_DULL",
    "sounds muffled/dull": "SOUND_MUFFLED_DULL",
    "sounds are mechanical or artificial": "SOUND_ARTIFICIAL_MECHANICAL",
    "sounds artificial/mechanical": "SOUND_ARTIFICIAL_MECHANICAL",
    "sounds artificial/mech anical": "SOUND_ARTIFICIAL_MECHANICAL",
    "too much bass": "SOUND_TOO_MUCH_BASS",
    "sounds are distorted": "SOUND_DISTORTED",
    "sounds distorted": "SOUND_DISTORTED",
    # Noise
    "soft ambience noise is too noticeable": "NOISE_AMBIENT_NOTICEABLE",
    "ambient noise too noticeable": "NOISE_AMBIENT_NOTICEABLE",
    "sound in noise is uncomfortable": "NOISE_SOUND_UNCOMFORTABLE",
    "sudden noise is uncomfortable": "NOISE_SUDDEN_UNCOMFORTABLE",
    "sudden noise uncomfortable": "NOISE_SUDDEN_UNCOMFORTABLE",
    "background noise is too loud": "NOISE_BACKGROUND_RESTAURANT",
    "restaurant noise is loud": "NOISE_BACKGROUND_RESTAURANT",
    "street&#x2f;traffic noise is too loud": "NOISE_BACKGROUND_RESTAURANT",
    "background noise (e.g., restaurants) too loud": "NOISE_BACKGROUND_RESTAURANT",
    "wind noise is loud": "NOISE_WIND_TOO_LOUD",
    "wind noise too loud": "NOISE_WIND_TOO_LOUD",
    "crowd noise is loud": "NOISE_CROWD_TOO_LOUD",
    "crowd noise too loud": "NOISE_CROWD_TOO_LOUD",
    # Feedback
    "hearing aid whistles": "FEEDBACK_WHISTLING",
    "hearing aid whistles (feedback)": "FEEDBACK_WHISTLING",
}

# P0: keep the full source taxonomy.  Add merges only after a domain decision.
INTENT_MERGE_MAP: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Intent metadata
# ---------------------------------------------------------------------------

INTENT_TO_TOPIC: dict[str, str] = {
    "LOUDNESS_SOFT_TOO_SOFT": "Loudness",
    "LOUDNESS_LOUD_TOO_LOUD": "Loudness",
    "LOUDNESS_LOUD_DISTORTED": "Loudness",
    "LOUDNESS_NEARBY_TOO_LOUD": "Loudness",
    "LOUDNESS_ALARMS_CRYING_LOUD": "Loudness",
    "LOUDNESS_SUDDEN_SOUNDS_LOUD": "Loudness",
    "LOUDNESS_CHEWING_TOO_LOUD": "Loudness",
    "LOUDNESS_WATER_FAN_LOUD": "Loudness",
    "OWN_VOICE_MUFFLED_DULL": "Own Voice",
    "OWN_VOICE_HOLLOW": "Own Voice",
    "OWN_VOICE_SHARP_METALLIC": "Own Voice",
    "OWN_VOICE_THIN": "Own Voice",
    "OWN_VOICE_BOOMY_LOUD": "Own Voice",
    "SPEECH_TOO_SOFT": "Speech",
    "SPEECH_WHISPER_DISTANT_SOFT": "Speech",
    "SPEECH_TOO_LOUD_BOOMY": "Speech",
    "SPEECH_ECHOIC": "Speech",
    "SPEECH_POOR_QUALITY": "Speech",
    "SOUND_TINY_METALLIC_SHARP": "General Sound",
    "SOUND_ECHOIC": "General Sound",
    "SOUND_MUFFLED_DULL": "General Sound",
    "SOUND_ARTIFICIAL_MECHANICAL": "General Sound",
    "SOUND_TOO_MUCH_BASS": "General Sound",
    "SOUND_DISTORTED": "General Sound",
    "NOISE_AMBIENT_NOTICEABLE": "Noise",
    "NOISE_SOUND_UNCOMFORTABLE": "Noise",
    "NOISE_SUDDEN_UNCOMFORTABLE": "Noise",
    "NOISE_BACKGROUND_RESTAURANT": "Noise",
    "NOISE_WIND_TOO_LOUD": "Noise",
    "NOISE_CROWD_TOO_LOUD": "Noise",
    "FEEDBACK_WHISTLING": "Feedback",
}

# Short descriptions used by the optional semantic audit.  These are *not*
# labels; they only help a human spot content/source mismatches.
INTENT_DESCRIPTIONS: dict[str, str] = {
    "LOUDNESS_SOFT_TOO_SOFT": "soft quiet sounds are too soft and faint",
    "LOUDNESS_LOUD_TOO_LOUD": "loud sounds are too loud and hurt my ears",
    "LOUDNESS_LOUD_DISTORTED": "loud sounds are distorted and unclear",
    "LOUDNESS_NEARBY_TOO_LOUD": "nearby sounds are too loud",
    "LOUDNESS_ALARMS_CRYING_LOUD": "alarms and children crying are too loud",
    "LOUDNESS_SUDDEN_SOUNDS_LOUD": "sudden sounds like door slams are too loud",
    "LOUDNESS_CHEWING_TOO_LOUD": "my own chewing sounds are too loud",
    "LOUDNESS_WATER_FAN_LOUD": "running water and fan sounds are too loud",
    "OWN_VOICE_MUFFLED_DULL": "my own voice sounds muffled and dull",
    "OWN_VOICE_HOLLOW": "my own voice sounds hollow like in a barrel",
    "OWN_VOICE_SHARP_METALLIC": "my own voice sounds too sharp and metallic",
    "OWN_VOICE_THIN": "my own voice sounds thin",
    "OWN_VOICE_BOOMY_LOUD": "my own voice sounds boomy and too loud",
    "SPEECH_TOO_SOFT": "people speaking and speech is too soft",
    "SPEECH_WHISPER_DISTANT_SOFT": "whispers and distant speech are too soft",
    "SPEECH_TOO_LOUD_BOOMY": "speech is too loud and boomy",
    "SPEECH_ECHOIC": "speech echoes",
    "SPEECH_POOR_QUALITY": "poor speech quality and unclear words",
    "SOUND_TINY_METALLIC_SHARP": "sounds feel tiny metallic and sharp",
    "SOUND_ECHOIC": "sounds are echoic and bounce around",
    "SOUND_MUFFLED_DULL": "sounds are muffled or dull",
    "SOUND_ARTIFICIAL_MECHANICAL": "sounds are mechanical or artificial",
    "SOUND_TOO_MUCH_BASS": "there is too much bass",
    "SOUND_DISTORTED": "general sounds are distorted",
    "NOISE_AMBIENT_NOTICEABLE": "soft ambient background noise is too noticeable",
    "NOISE_SOUND_UNCOMFORTABLE": "sound in noise is uncomfortable",
    "NOISE_SUDDEN_UNCOMFORTABLE": "sudden noise is uncomfortable",
    "NOISE_BACKGROUND_RESTAURANT": "background noise in restaurants or traffic is too loud",
    "NOISE_WIND_TOO_LOUD": "wind noise is too loud",
    "NOISE_CROWD_TOO_LOUD": "crowd noise is too loud",
    "FEEDBACK_WHISTLING": "hearing aid whistles and has feedback",
}

GARBAGE_QUESTIONS: set[str] = {"sdc", "afse", "sdfgsd", "sdfghj"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str | None) -> str:
    """Normalise source text for exact lookup.

    Handles the HTML entities that leaked out of the old export
    (``Street&#x2F;Traffic``) and the occasional leading/trailing spaces.
    """

    if not text:
        return ""
    text = html.unescape(str(text))
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().rstrip(".").lower()


def final_intent(raw_intent: str) -> str:
    """Apply :data:`INTENT_MERGE_MAP` to a raw intent label."""

    return INTENT_MERGE_MAP.get(raw_intent, raw_intent)


def question_to_intent(question: str | None, fuzzy: bool = True) -> str | None:
    """Map a canonical complaint question to an intent code.

    Returns ``None`` when the question is empty/garbage/unknown.  A fuzzy
    fallback is available because the source export has inconsistent spacing
    (``"childr en"``, ``"dista nt"``); callers should log when it is used so
    the taxonomy can be extended.
    """

    key = normalize_text(question)
    if not key or key in GARBAGE_QUESTIONS:
        return None

    direct = QUESTION_TO_INTENT.get(key)
    if direct is not None:
        return final_intent(direct)

    if not fuzzy:
        return None

    matches = get_close_matches(key, QUESTION_TO_INTENT.keys(), n=1, cutoff=0.88)
    if matches:
        return final_intent(QUESTION_TO_INTENT[matches[0]])
    return None


def all_intents(include_merged: bool = False) -> list[str]:
    """Return the sorted label space."""

    intents = set(QUESTION_TO_INTENT.values()) | set(INTENT_MERGE_MAP.values())
    if include_merged:
        intents |= set(QUESTION_TO_INTENT.values())
    return sorted(intents)


def intent_topic(intent: str) -> str:
    return INTENT_TO_TOPIC.get(intent, "Unknown")


def topic_of_questions(questions: Iterable[str]) -> dict[str, str]:
    """Convenience helper for reports: question -> topic title."""

    out: dict[str, str] = {}
    for q in questions:
        intent = question_to_intent(q)
        out[q] = intent_topic(intent) if intent else "Unknown"
    return out
