"""
config.py
---------
Beginner-friendly settings module.

This file holds:
1. Helper to load the OpenAI API key (from .env, or from the user via the
   Streamlit "enter your key" page).
2. All the static dropdown / multiselect options used by the form in app.py.

Keeping these in one place makes the rest of the app easier to read.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file (if one exists) into the environment.
# This lets a developer running the app locally skip the "enter API key" page.
load_dotenv()


def get_env_api_key() -> str:
    """
    Return the OpenAI API key found in the environment (.env file), or an
    empty string if none is set. The Streamlit app uses this to decide
    whether it can skip straight past the "enter your API key" page.
    """
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_default_model() -> str:
    """Return the default model name, falling back to gpt-4o-mini."""
    return os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"


def is_valid_looking_key(key: str) -> bool:
    """
    Very light sanity check on an OpenAI key format. This is NOT a
    verification that the key actually works (that only happens when we
    call the API) -- it just avoids obviously-empty or malformed input.
    """
    key = (key or "").strip()
    return key.startswith("sk-") and len(key) >= 20


# ---------------------------------------------------------------------------
# Static options used to populate the Streamlit form widgets
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo",
]

CACHE_OPTIONS = [
    "No caching",
    "In-memory cache",
    "SQLite cache",
]

GENDER_OPTIONS = [
    "Female",
    "Male",
    "Non-binary",
    "Prefer not to say",
]

DURATION_OPTIONS = [
    "Less than a day",
    "1-3 days",
    "4-7 days",
    "1-2 weeks",
    "2-4 weeks",
    "More than a month",
]

SYMPTOM_OPTIONS = [
    "Fever",
    "Cough",
    "Sore throat",
    "Runny / blocked nose",
    "Headache",
    "Fatigue",
    "Nausea",
    "Vomiting",
    "Diarrhea",
    "Abdominal pain",
    "Chest pain",
    "Shortness of breath",
    "Dizziness",
    "Muscle aches",
    "Rash",
    "Joint pain",
    "Loss of appetite",
    "Sore eyes",
]

LANGUAGE_OPTIONS = [
    "English",
    "Urdu",
    "Spanish",
    "French",
    "Arabic",
    "Hindi",
]

URGENCY_COLORS = {
    "LOW": "green",
    "MEDIUM": "orange",
    "HIGH": "red",
    "EMERGENCY": "red",
}
