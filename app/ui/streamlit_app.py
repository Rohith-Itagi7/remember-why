"""Local Streamlit interface for Remember Why."""

from __future__ import annotations

import re
import streamlit as st

from app.observability.sentry import initialize_sentry

initialize_sentry()

from app.ui.presentation import answer_question, existing_screenshot_path


st.set_page_config(page_title="Remember Why", page_icon="ðŸ§­", layout="centered")
st.markdown(
    """
    <style>
    .block-container { max-width: 860px; padding-top: 3rem; padding-bottom: 3rem; }
    .rw-tagline { color: #667085; font-size: 1.1rem; margin-top: -0.7rem; }
    .rw-muted { color: #667085; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Remember Why")
st.markdown('<p class="rw-tagline">You saved it for a reason. I help you remember why.</p>', unsafe_allow_html=True)

with st.form("remember_why_question"):
    question = st.text_area(
        "What do you want to remember?",
        placeholder="Why did I save MCP_Architecture.png?",
        height=100,
    )
    submitted = st.form_submit_button("Remember", type="primary", use_container_width=True)

if submitted:
    st.session_state["remember_why_result"] = answer_question(question)


def memory_card(memory: dict[str, object], *, target: bool = False) -> None:
    """Show concise metadata and keep the full OCR text expandable."""
    filename = str(memory.get("filename") or "Screenshot memory")
    with st.container(border=True):
        st.text(filename)
        timestamp = memory.get("timestamp")
        if timestamp:
            st.caption(f"Timestamp: {timestamp}")
        similarity = memory.get("similarity")
        if similarity:
            st.caption(f"Semantic similarity: {similarity}")
        distance = memory.get("time_distance")
        if distance:
            st.caption(f"Time distance: {distance}")
        local_path = existing_screenshot_path(memory.get("path"))
        if local_path:
            if target:
                st.image(local_path, caption=filename, width="stretch")
            else:
                with st.expander("View screenshot"):
                    st.image(local_path, caption=filename, width="stretch")
        elif memory.get("path"):
            st.caption("Screenshot image is unavailable.")
        text = str(memory.get("ocr") or "")
        preview = re.sub(r"\s+", " ", text).strip()
        if preview:
            st.caption("OCR evidence")
            st.text(preview if len(preview) <= 260 else preview[:257].rstrip() + "...")
            with st.expander("View OCR evidence"):
                st.text(text)
        elif target:
            st.caption("No OCR text is available for this memory.")


def render_result(result: dict[str, object]) -> None:
    """Render either context reconstruction or semantic-search evidence."""
    kind = result.get("kind")
    if kind == "message":
        level = result.get("level")
        message = str(result.get("message", "Remember Why couldn't complete that request."))
        if level == "error":
            st.error(message)
        else:
            st.info(message)
        return

    if kind == "context":
        st.subheader("Context evidence")
        target_records = result.get("target", [])
        if target_records:
            st.markdown("#### Target memory")
            memory_card(target_records[0], target=True)
        related = result.get("related", [])
        st.markdown("#### Related memories")
        if related:
            for memory in related:
                memory_card(memory)
        else:
            st.caption("No related memories were found.")
        st.markdown("#### Possible context")
        with st.container(border=True):
            st.write(str(result.get("possible_context") or "There are no related records here to suggest context."))
        return

    memories = result.get("memories", [])
    st.subheader("Memories found")
    if memories:
        for memory in memories:
            memory_card(memory)
    else:
        st.info("I couldn't find any matching memories. Try another question or build the screenshot index.")


if "remember_why_result" in st.session_state:
    render_result(st.session_state["remember_why_result"])

st.markdown("---")
st.markdown('<p class="rw-muted">ðŸ”’ Your memories stay on your machine.</p>', unsafe_allow_html=True)
