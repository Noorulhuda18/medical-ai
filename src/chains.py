"""
chains.py
---------
Everything related to talking to the LLM lives here:
- build_llm(...)              -> constructs a ChatOpenAI instance.
- build_assessment_chain(...) -> a reusable LLMChain around the JSON prompt.
- run_raw_message_demo(...)   -> shows SystemMessage / HumanMessage / AIMessage
  used directly (assignment requirement, separate from the chain/template
  approach) - useful for the "how LangChain messages work" teaching moment.
- stream_narrative(...)       -> a generator that yields narrative text chunks
  for st.write_stream().
"""

from typing import Any, Iterator, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.prompts import (
    SAFETY_SYSTEM_PROMPT,
    ASSESSMENT_PROMPT_TEMPLATE,
    NARRATIVE_CHAT_TEMPLATE,
)


def build_llm(api_key: str, model_name: str, streaming: bool = False, temperature: float = 0.3) -> ChatOpenAI:
    """Construct a ChatOpenAI chat model instance."""
    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        streaming=streaming,
    )


def build_assessment_chain(llm: ChatOpenAI) -> Any:
    """
    Build the reusable runnable that turns patient-form inputs into the
    structured JSON assessment, using the single-string PromptTemplate.
    """
    return ASSESSMENT_PROMPT_TEMPLATE | llm


def run_assessment_chain(chain: Any, inputs: dict) -> str:
    """Run the assessment chain once and return the raw text output."""
    result = chain.invoke(inputs)
    # Modern LangChain runnables return an AIMessage rather than an LLMChain dict.
    return result.content if hasattr(result, "content") else str(result)


def run_raw_message_demo(llm: ChatOpenAI, user_summary: str) -> List[dict]:
    """
    A small standalone demonstration (not used for the main assessment) of
    building a conversation directly from SystemMessage / HumanMessage /
    AIMessage objects, showing how a follow-up AIMessage fits into history.

    Returns a list of {"role": ..., "content": ...} dicts for easy display
    in the Streamlit UI (e.g. inside an st.expander).
    """
    messages = [
        SystemMessage(content=SAFETY_SYSTEM_PROMPT),
        HumanMessage(content=f"In one sentence, acknowledge this patient note: {user_summary}"),
    ]

    response = llm.invoke(messages)

    # Manually append the model's reply as an AIMessage to show how a
    # multi-turn conversation history is represented in LangChain.
    messages.append(AIMessage(content=response.content))

    return [{"role": m.type, "content": m.content} for m in messages]


def stream_narrative(llm: ChatOpenAI, inputs: dict) -> Iterator[str]:
    """
    Yield the human-readable narrative guidance chunk by chunk, so the
    Streamlit UI can render it live with st.write_stream().
    """
    messages = NARRATIVE_CHAT_TEMPLATE.format_messages(**inputs)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
