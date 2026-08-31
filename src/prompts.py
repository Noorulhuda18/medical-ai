"""
prompts.py
----------
All prompt engineering lives here: the safety-first system prompt, the JSON
schema instruction, a plain PromptTemplate, and a ChatPromptTemplate built
from System/Human messages.

Two templates are exposed:
- ASSESSMENT_PROMPT_TEMPLATE  -> a single-string PromptTemplate (legacy style,
  used with LLMChain to produce the structured JSON).
- ASSESSMENT_CHAT_TEMPLATE    -> a ChatPromptTemplate (System + Human) used
  for the same JSON task but in "chat message" form.
- NARRATIVE_CHAT_TEMPLATE     -> a ChatPromptTemplate used purely for the
  human-readable narrative that gets streamed live into the UI.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# ---------------------------------------------------------------------------
# 1. The safety rules. These are injected into EVERY call to the model so
#    the model itself is constrained, in addition to the UI-level warnings.
# ---------------------------------------------------------------------------

SAFETY_SYSTEM_PROMPT = """You are MediGuide AI, an educational medical-information
assistant embedded in a prototype application. You are NOT a doctor and you do
NOT provide medical diagnoses.

Non-negotiable rules you must always follow:
1. You must never state or imply a confirmed diagnosis. Use language such as
   "this may be associated with" or "one possibility to discuss with a doctor
   is", never "you have X".
2. Always recommend the user consult a licensed healthcare professional for
   any real diagnosis or treatment decision.
3. If the described symptoms could indicate a medical emergency (for example:
   severe chest pain, difficulty breathing, signs of stroke, severe bleeding,
   suicidal thoughts, loss of consciousness), you must set the urgency level
   to "EMERGENCY" and clearly instruct the user to seek emergency care
   immediately (e.g. call local emergency services or go to the nearest ER).
4. Keep your tone calm, clear, supportive, and non-alarmist, while still being
   honest about risk.
5. Base your response only on the information given. Do not invent patient
   history that was not provided.
6. Respond in the language requested by the user for all patient-facing text.
"""

# ---------------------------------------------------------------------------
# 2. The exact JSON schema the model must return for the structured
#    assessment. Keeping this as its own constant means it can be reused by
#    both the PromptTemplate and the ChatPromptTemplate below.
# ---------------------------------------------------------------------------

JSON_SCHEMA_INSTRUCTION = """Return ONLY a single valid JSON object and nothing
else - no markdown code fences, no preamble, no explanation outside the JSON.
The JSON object must match this exact structure:

{{
  "summary": "<one paragraph summary of the patient-reported symptoms>",
  "possible_conditions": [
    {{ "name": "<condition name, for education only>", "reason": "<why this is mentioned>" }}
  ],
  "urgency_level": "<one of: LOW, MEDIUM, HIGH, EMERGENCY>",
  "recommended_next_steps": ["<step 1>", "<step 2>"],
  "questions_for_doctor": ["<question 1>", "<question 2>"],
  "warning_signs": ["<warning sign 1>", "<warning sign 2>"]
}}

Respond in {language}. Keep "urgency_level" in uppercase English exactly as
one of LOW, MEDIUM, HIGH, EMERGENCY regardless of the response language, since
the app uses that field for color-coded display logic.
"""

# ---------------------------------------------------------------------------
# 3. A plain single-string PromptTemplate (assignment requirement: it must
#    demonstrate PromptTemplate with variables).
# ---------------------------------------------------------------------------

ASSESSMENT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "age", "gender", "symptoms", "duration", "severity",
        "existing_conditions", "medications", "notes", "language",
    ],
    template=(
        SAFETY_SYSTEM_PROMPT
        + "\n\nPatient information:\n"
        + "- Age: {age}\n"
        + "- Gender: {gender}\n"
        + "- Reported symptoms: {symptoms}\n"
        + "- Duration of symptoms: {duration}\n"
        + "- Self-reported severity (1-10): {severity}\n"
        + "- Existing medical conditions: {existing_conditions}\n"
        + "- Current medications: {medications}\n"
        + "- Additional notes: {notes}\n\n"
        + JSON_SCHEMA_INSTRUCTION
    ),
)

# ---------------------------------------------------------------------------
# 4. A ChatPromptTemplate built from System + Human messages, used for the
#    same structured-JSON task (assignment requirement: ChatPromptTemplate).
# ---------------------------------------------------------------------------

ASSESSMENT_CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SAFETY_SYSTEM_PROMPT),
        (
            "human",
            "Patient information:\n"
            "- Age: {age}\n"
            "- Gender: {gender}\n"
            "- Reported symptoms: {symptoms}\n"
            "- Duration of symptoms: {duration}\n"
            "- Self-reported severity (1-10): {severity}\n"
            "- Existing medical conditions: {existing_conditions}\n"
            "- Current medications: {medications}\n"
            "- Additional notes: {notes}\n\n" + JSON_SCHEMA_INSTRUCTION,
        ),
    ]
)

# ---------------------------------------------------------------------------
# 5. A second ChatPromptTemplate, used only to stream a friendly, readable
#    narrative version of the guidance (not JSON) for the "typing" effect.
# ---------------------------------------------------------------------------

NARRATIVE_CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SAFETY_SYSTEM_PROMPT),
        (
            "human",
            "Based on this patient information, write a short, warm, "
            "easy-to-read paragraph (4-6 sentences) in {language} summarising "
            "what these symptoms might mean in general terms, reminding the "
            "patient this is not a diagnosis, and encouraging them to review "
            "the detailed dashboard below. Do not use JSON here - plain "
            "readable prose only.\n\n"
            "- Age: {age}\n"
            "- Gender: {gender}\n"
            "- Reported symptoms: {symptoms}\n"
            "- Duration of symptoms: {duration}\n"
            "- Self-reported severity (1-10): {severity}\n"
            "- Existing medical conditions: {existing_conditions}\n"
            "- Current medications: {medications}\n"
            "- Additional notes: {notes}\n",
        ),
    ]
)
