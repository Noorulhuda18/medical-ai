"""
utils.py
--------
Small, dependency-free helper functions used across the app - mainly the
"never let bad JSON crash the app" safe parser.
"""

import json
import re
from typing import Optional, Tuple


def strip_code_fences(text: str) -> str:
    """
    Remove accidental ```json ... ``` or ``` ... ``` fences (and any stray
    leading/trailing text outside the outermost { } ) that a model
    sometimes adds even when told not to.
    """
    text = text.strip()

    # Remove ```json / ``` fences if present.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # If there's still leading/trailing junk, grab the outermost { ... }.
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace: last_brace + 1]

    return text.strip()


def safe_parse_json(raw_text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Try to parse `raw_text` (the model's raw output) into a dict matching
    the MediGuide AI JSON schema.

    Returns a tuple: (parsed_dict_or_None, error_message_or_None)
    - On success: (dict, None)
    - On failure: (None, "human readable error message")

    This function must NEVER raise - the caller can rely on it always
    returning safely, per the assignment's reliability requirement.
    """
    if not raw_text or not raw_text.strip():
        return None, "The model returned an empty response."

    cleaned = strip_code_fences(raw_text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"Could not parse the model's JSON output ({exc})."

    if not isinstance(parsed, dict):
        return None, "The model's JSON output was not a JSON object."

    # Fill in any missing expected keys with safe defaults rather than
    # crashing, so the dashboard can still render partial results.
    defaults = {
        "summary": "",
        "possible_conditions": [],
        "urgency_level": "LOW",
        "recommended_next_steps": [],
        "questions_for_doctor": [],
        "warning_signs": [],
    }
    for key, default_value in defaults.items():
        parsed.setdefault(key, default_value)

    # Normalise urgency level to one of the four expected values.
    urgency = str(parsed.get("urgency_level", "LOW")).strip().upper()
    if urgency not in {"LOW", "MEDIUM", "HIGH", "EMERGENCY"}:
        urgency = "LOW"
    parsed["urgency_level"] = urgency

    return parsed, None


def format_symptom_list(selected_symptoms, free_text: str) -> str:
    """Combine the multiselect symptoms and optional free-text into one
    readable comma-separated string for the prompt."""
    items = list(selected_symptoms) if selected_symptoms else []
    if free_text and free_text.strip():
        items.append(free_text.strip())
    return ", ".join(items) if items else "None reported"
