"""
app.py
------
Streamlit entry point for MediGuide AI.

Flow:
  1. API KEY PAGE - shown first. The user pastes their OpenAI API key (or the
     app auto-skips this if a key is already present in .env). The key is
     kept only in st.session_state for this browser session - never written
     to disk.
  2. MAIN APP PAGE - the patient form, sidebar configuration, and the
     results dashboard (summary, streamed narrative, urgency, next steps,
     doctor questions, warning signs).

Run with:  streamlit run app.py
"""

import streamlit as st

from src import config
from src.prompts import ASSESSMENT_CHAT_TEMPLATE  # noqa: F401 (kept for reference / grading visibility)
from src.chains import build_llm, build_assessment_chain, run_assessment_chain, run_raw_message_demo, stream_narrative
from src.cache_manager import set_cache, CACHE_EXPLANATION
from src.utils import safe_parse_json, format_symptom_list

st.set_page_config(page_title="MediGuide AI", page_icon="🩺", layout="wide")

DISCLAIMER_TEXT = (
    "**MediGuide AI is an educational prototype only.** It is **not** a "
    "licensed doctor, does not provide medical diagnoses, and is not a "
    "substitute for professional medical care. If this is a medical "
    "emergency, call your local emergency number or go to the nearest "
    "emergency room immediately."
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "api_key" not in st.session_state:
    # Auto-fill from .env if a developer has one set up, so local testing
    # doesn't force you through the key-entry page every time.
    st.session_state.api_key = config.get_env_api_key()

if "page" not in st.session_state:
    st.session_state.page = "api_key" if not st.session_state.api_key else "main"


def go_to_main():
    st.session_state.page = "main"


def log_out_key():
    st.session_state.api_key = ""
    st.session_state.page = "api_key"


# ---------------------------------------------------------------------------
# PAGE 1: API KEY ENTRY
# ---------------------------------------------------------------------------
def render_api_key_page():
    st.title("🩺 MediGuide AI")
    st.caption("AI-Powered Medical Symptom Assessment and Patient Guidance Assistant")

    st.warning(DISCLAIMER_TEXT)

    st.subheader("Step 1 - Connect your OpenAI API key")
    st.write(
        "MediGuide AI uses your own OpenAI API key to run the assessment "
        "model. Your key is kept only in this browser session's memory - "
        "it is never saved to disk or sent anywhere except directly to "
        "OpenAI's API."
    )

    with st.form("api_key_form"):
        key_input = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="Get a key at platform.openai.com under API keys.",
        )
        submitted = st.form_submit_button("Save & Continue →")

    if submitted:
        if not config.is_valid_looking_key(key_input):
            st.error(
                "That doesn't look like a valid OpenAI API key. Keys start "
                "with 'sk-' and are reasonably long. Please check and try again."
            )
        else:
            st.session_state.api_key = key_input.strip()
            go_to_main()
            st.rerun()

    with st.expander("Don't have a key yet?"):
        st.markdown(
            "1. Create an account at [platform.openai.com](https://platform.openai.com).\n"
            "2. Open **API keys** and generate a new secret key.\n"
            "3. Paste it above. For local development you can instead copy "
            "`.env.example` to `.env` and set `OPENAI_API_KEY` there."
        )


# ---------------------------------------------------------------------------
# PAGE 2: MAIN APPLICATION (sidebar + form + dashboard)
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.title("🩺 MediGuide AI")
        st.caption("Educational symptom-guidance prototype (LangChain + Streamlit)")

        st.warning(DISCLAIMER_TEXT)

        st.markdown("---")
        st.subheader("Model configuration")
        model_name = st.selectbox("OpenAI model", config.AVAILABLE_MODELS, index=0)
        cache_choice = st.selectbox("Caching strategy", config.CACHE_OPTIONS, index=0)

        with st.expander("What's the difference between the caches?"):
            st.markdown(CACHE_EXPLANATION)

        st.markdown("---")
        language = st.selectbox("Answer language", config.LANGUAGE_OPTIONS, index=0)

        st.markdown("---")
        if st.button("Log out / change API key"):
            log_out_key()
            st.rerun()

        return model_name, cache_choice, language


def render_patient_form():
    st.subheader("Step 2 - Tell us about your symptoms")

    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.text_input("Patient age", placeholder="e.g. 34")
        with col2:
            gender = st.selectbox("Gender", config.GENDER_OPTIONS)

        symptoms = st.multiselect("Symptoms (select all that apply)", config.SYMPTOM_OPTIONS)
        symptoms_free_text = st.text_area(
            "Other symptoms not listed above (optional)",
            placeholder="Describe any additional symptoms here...",
        )

        col3, col4 = st.columns(2)
        with col3:
            duration = st.selectbox("Duration of symptoms", config.DURATION_OPTIONS)
        with col4:
            severity = st.slider("Severity (1 = mild, 10 = severe)", min_value=1, max_value=10, value=3)

        existing_conditions = st.text_area(
            "Existing medical conditions (optional)",
            placeholder="e.g. asthma, diabetes, none",
        )
        medications = st.text_area(
            "Current medications (optional)",
            placeholder="e.g. metformin, ibuprofen as needed, none",
        )
        notes = st.text_area(
            "Additional notes (optional)",
            placeholder="Anything else you'd like to mention...",
        )

        submitted = st.form_submit_button("Get guidance", use_container_width=True)

    form_data = {
        "age": age,
        "gender": gender,
        "symptoms": symptoms,
        "symptoms_free_text": symptoms_free_text,
        "duration": duration,
        "severity": severity,
        "existing_conditions": existing_conditions or "None reported",
        "medications": medications or "None reported",
        "notes": notes or "None",
    }
    return submitted, form_data


def render_dashboard(parsed: dict, raw_text: str, narrative_placeholder_used: bool):
    urgency = parsed["urgency_level"]

    st.markdown("### Results dashboard")
    st.warning(DISCLAIMER_TEXT)

    tab_overview, tab_conditions, tab_steps, tab_raw = st.tabs(
        ["Overview", "Possible Conditions", "Next Steps & Questions", "Raw Output"]
    )

    with tab_overview:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Urgency level", urgency)
        with m2:
            st.metric("Warning signs identified", len(parsed.get("warning_signs", [])))

        if urgency == "EMERGENCY":
            st.error(
                "🚨 EMERGENCY: Seek immediate medical attention. Call your "
                "local emergency number or go to the nearest emergency room now."
            )
        elif urgency == "HIGH":
            st.error("⚠️ HIGH urgency: please see a healthcare professional promptly.")
        elif urgency == "MEDIUM":
            st.warning("🟠 MEDIUM urgency: consider seeing a healthcare professional soon.")
        else:
            st.success("🟢 LOW urgency: monitor symptoms and practice general self-care.")

        st.markdown("**Patient symptom summary**")
        st.info(parsed.get("summary", "No summary provided."))

        if parsed.get("warning_signs"):
            with st.expander("⚠️ Warning signs requiring immediate attention"):
                for sign in parsed["warning_signs"]:
                    st.write(f"- {sign}")

    with tab_conditions:
        st.caption("For general education only - not a diagnosis.")
        possible = parsed.get("possible_conditions", [])
        if not possible:
            st.write("No specific conditions were suggested.")
        for item in possible:
            name = item.get("name", "Unnamed") if isinstance(item, dict) else str(item)
            reason = item.get("reason", "") if isinstance(item, dict) else ""
            with st.expander(f"🔎 {name}"):
                st.write(reason or "No further detail provided.")

    with tab_steps:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Recommended next steps**")
            steps = parsed.get("recommended_next_steps", [])
            if steps:
                for step in steps:
                    st.write(f"✅ {step}")
            else:
                st.write("No specific steps provided.")
        with col_b:
            st.markdown("**Questions to ask your doctor**")
            questions = parsed.get("questions_for_doctor", [])
            if questions:
                for q in questions:
                    st.write(f"❓ {q}")
            else:
                st.write("No specific questions provided.")

    with tab_raw:
        st.caption("Useful for debugging the model's structured output.")
        st.code(raw_text, language="json")


def render_main_app():
    model_name, cache_choice, language = render_sidebar()

    st.title("🩺 MediGuide AI")
    st.warning(DISCLAIMER_TEXT)

    cache_status = set_cache(cache_choice)
    st.caption(f"Cache status: {cache_status}")

    submitted, form_data = render_patient_form()

    if not submitted:
        return

    # --- Requirement: empty symptoms should warn the user and skip the API call ---
    combined_symptoms = format_symptom_list(form_data["symptoms"], form_data["symptoms_free_text"])
    if combined_symptoms == "None reported":
        st.warning(
            "Please select at least one symptom, or describe your symptoms in "
            "the free-text box, before requesting guidance."
        )
        return

    if not form_data["age"].strip():
        st.warning("Please enter the patient's age before continuing.")
        return

    chain_inputs = {
        "age": form_data["age"],
        "gender": form_data["gender"],
        "symptoms": combined_symptoms,
        "duration": form_data["duration"],
        "severity": form_data["severity"],
        "existing_conditions": form_data["existing_conditions"],
        "medications": form_data["medications"],
        "notes": form_data["notes"],
        "language": language,
    }

    api_key = st.session_state.api_key

    try:
        with st.spinner("Analysing symptoms and preparing your guidance..."):
            # Non-streaming LLM + LLMChain for the structured JSON assessment.
            json_llm = build_llm(api_key, model_name, streaming=False)
            chain = build_assessment_chain(json_llm)
            raw_output = run_assessment_chain(chain, chain_inputs)

        parsed, error = safe_parse_json(raw_output)

        if error or parsed is None:
            st.error(f"We couldn't read the model's structured response: {error}")
            with st.expander("Raw model output (for debugging)"):
                st.code(raw_output or "(empty response)")
            return

        st.markdown("### AI-generated narrative")
        streaming_llm = build_llm(api_key, model_name, streaming=True)
        st.write_stream(stream_narrative(streaming_llm, chain_inputs))

        render_dashboard(parsed, raw_output, narrative_placeholder_used=True)

        with st.expander("🔧 Developer demo: raw System/Human/AI message exchange"):
            st.caption(
                "Demonstrates building a conversation directly from "
                "SystemMessage / HumanMessage / AIMessage objects."
            )
            demo_messages = run_raw_message_demo(json_llm, parsed.get("summary", combined_symptoms))
            for msg in demo_messages:
                st.write(f"**{msg['role']}**: {msg['content']}")

    except Exception as exc:  # noqa: BLE001 - top-level safety net per requirements
        st.error(
            "Something went wrong while contacting the AI model. Please "
            "check your API key and try again."
        )
        with st.expander("Technical details"):
            st.code(str(exc))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if st.session_state.page == "api_key":
    render_api_key_page()
else:
    render_main_app()
