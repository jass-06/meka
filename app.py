"""
Multimodal Enterprise Knowledge Assistant - Streamlit Application

UI notes:
- All visual styling lives in `inject_css()` and the small `status_dot` /
  `panel` helpers below. Nothing else in the file should need raw HTML.
- We deliberately avoid Streamlit's default st.error/st.warning/st.success
  boxes (loud colored backgrounds + built-in icons) in favor of quiet
  inline status indicators, to keep the interface looking like a real
  product rather than a demo.
"""
import streamlit as st
import base64
from typing import Optional, Any
import time

from rag.retrieval import RAGPipeline
from rag.config import Config

# ── Design tokens ──────────────────────────────────────────────────────────
INK = "#12161C"
INK_SOFT = "#5B6472"
SURFACE = "#FFFFFF"
CANVAS = "#F6F7F9"
BORDER = "#E4E7EC"
ACCENT = "#3538CD"
ACCENT_SOFT = "#EEF0FD"
GOOD = "#12A150"
NEUTRAL_OFF = "#98A2B3"
WARN = "#B54708"


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {INK};
        }}

        code, .stCode, pre {{
            font-family: 'JetBrains Mono', ui-monospace, monospace !important;
        }}

        /* Hide default Streamlit chrome */
        #MainMenu, footer, header {{ visibility: hidden; }}

        .stApp {{ background-color: {CANVAS}; }}

        section[data-testid="stSidebar"] {{
            background-color: {SURFACE};
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] .block-container {{ padding-top: 2rem; }}

        /* Eyebrow section labels */
        .eyebrow {{
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {INK_SOFT};
            margin: 1.75rem 0 0.6rem 0;
        }}
        .eyebrow:first-child {{ margin-top: 0; }}

        /* Brand mark in sidebar */
        .brand {{
            font-size: 15px;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: {INK};
        }}
        .brand-sub {{
            font-size: 12px;
            color: {INK_SOFT};
            margin-top: 2px;
        }}

        hr.thin {{
            border: none;
            border-top: 1px solid {BORDER};
            margin: 1.25rem 0;
        }}

        /* Status rows */
        .status-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            padding: 5px 0;
            color: {INK};
        }}
        .dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .dot-good {{ background-color: {GOOD}; }}
        .dot-off {{ background-color: {NEUTRAL_OFF}; }}
        .dot-warn {{ background-color: {WARN}; }}
        .status-label {{ color: {INK_SOFT}; }}
        .status-value {{ font-weight: 500; margin-left: auto; }}

        /* Neutral bordered panel (replaces colored alert boxes) */
        .panel {{
            background-color: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1.4rem 1.5rem;
            margin: 0.5rem 0 1.25rem 0;
        }}
        .panel-accent {{
            border-left: 3px solid {ACCENT};
        }}
        .panel-title {{
            font-size: 14px;
            font-weight: 600;
            color: {INK};
            margin-bottom: 4px;
        }}
        .panel-body {{
            font-size: 13.5px;
            color: {INK_SOFT};
            line-height: 1.6;
        }}
        .panel-body code {{
            background-color: {ACCENT_SOFT};
            color: {ACCENT};
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 12.5px;
        }}

        /* Main header */
        .app-title {{
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.015em;
            color: {INK};
            margin-bottom: 2px;
        }}
        .app-subtitle {{
            font-size: 14px;
            color: {INK_SOFT};
            margin-bottom: 1.5rem;
        }}

        /* Chat bubbles */
        div[data-testid="stChatMessage"] {{
            border-radius: 10px;
            padding: 4px 2px;
        }}

        /* Sidebar radio -> quiet pill style */
        div[role="radiogroup"] label {{
            font-size: 13px !important;
        }}

        /* Source citation chip */
        .source-chip {{
            display: inline-block;
            background-color: {CANVAS};
            border: 1px solid {BORDER};
            color: {INK_SOFT};
            font-size: 12px;
            padding: 2px 9px;
            border-radius: 6px;
            margin: 2px 4px 2px 0;
        }}

        .caption-row {{
            font-size: 12px;
            color: {INK_SOFT};
            margin-top: 2px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_dot(label: str, state: str, value: Optional[str] = None):
    """state: 'good' | 'off' | 'warn'"""
    cls = {"good": "dot-good", "off": "dot-off", "warn": "dot-warn"}.get(state, "dot-off")
    value_html = f'<span class="status-value">{value}</span>' if value else ""
    st.markdown(
        f'<div class="status-row"><span class="dot {cls}"></span>'
        f'<span class="status-label">{label}</span>{value_html}</div>',
        unsafe_allow_html=True,
    )


def panel(title: str, body_html: str, accent: bool = False):
    accent_cls = " panel-accent" if accent else ""
    st.markdown(
        f'<div class="panel{accent_cls}"><div class="panel-title">{title}</div>'
        f'<div class="panel-body">{body_html}</div></div>',
        unsafe_allow_html=True,
    )


# ── App state / logic ──────────────────────────────────────────────────────

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    if "config_validated" not in st.session_state:
        st.session_state.config_validated = False
    if "use_groq" not in st.session_state:
        st.session_state.use_groq = False


def validate_config():
    if not st.session_state.config_validated:
        hf_token = getattr(Config, "HF_API_TOKEN", None)
        if not hf_token:
            panel(
                "Configuration required",
                "<code>HF_API_TOKEN</code> is not set. Embeddings and image "
                "understanding won't work until it's added to your <code>.env</code> file.",
                accent=True,
            )
        st.session_state.config_validated = True
    return True


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="brand">Enterprise Knowledge Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-sub">Internal document intelligence</div>', unsafe_allow_html=True)

        st.markdown('<div class="eyebrow">Model</div>', unsafe_allow_html=True)
        backend_choice = st.radio(
            "Backend",
            ["Ollama + Groq assist", "Ollama only"],
            index=0,
            label_visibility="collapsed",
        )
        st.session_state.use_groq = backend_choice.startswith("Ollama + Groq")

        st.markdown('<div class="eyebrow">Status</div>', unsafe_allow_html=True)
        stats = st.session_state.rag_pipeline.get_stats()
        is_empty = st.session_state.rag_pipeline.is_empty()
        vector_stats = stats.get("vector_store", {})
        doc_count = vector_stats.get("document_count", 0) if "error" not in vector_stats else "—"
        llm_status = stats.get("llm_backends", {})

        status_dot("Documents indexed", "off" if is_empty else "good", str(doc_count))
        status_dot("Ollama", "good" if llm_status.get("ollama") else "off",
                    "connected" if llm_status.get("ollama") else "unavailable")
        status_dot("Groq", "good" if llm_status.get("groq") else "off",
                    "connected" if llm_status.get("groq") else "not configured")

        st.markdown('<div class="eyebrow">Getting started</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-body">'
            '1. Add files to <code>./data/</code><br>'
            '2. Run <code>python offline_ingest.py</code><br>'
            '3. Refresh this page'
            '</div>',
            unsafe_allow_html=True,
        )


def render_chat_interface():
    st.markdown('<div class="app-title">Enterprise Knowledge Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Ask questions against your organization\'s documents.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.rag_pipeline.is_empty():
        panel(
            "No documents indexed yet",
            "Add <code>.txt</code>, <code>.md</code>, <code>.pdf</code>, or <code>.docx</code> "
            "files to the <code>data/</code> folder, then run "
            "<code>python offline_ingest.py</code> and refresh this page.",
        )
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("sources"):
                chips = "".join(f'<span class="source-chip">{s}</span>' for s in msg["sources"])
                st.markdown(chips, unsafe_allow_html=True)

            if msg["role"] == "assistant":
                used_llm = msg.get("used_llm", "")
                latency = msg.get("latency")
                bits = []
                if used_llm and used_llm != "none":
                    bits.append(f"answered by {used_llm}")
                if latency is not None:
                    bits.append(f"{latency}s")
                if bits:
                    st.markdown(f'<div class="caption-row">{" · ".join(bits)}</div>', unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.chat_input("Ask a question about your documents...")
        with col2:
            uploaded_image = st.file_uploader(
                "Image",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
                key="chat_image",
            )

    if question:
        process_user_question(question, uploaded_image)


def process_user_question(question: str, uploaded_image: Optional[Any] = None):
    user_content = question
    if uploaded_image:
        user_content += f"  ({uploaded_image.name})"

    st.session_state.messages.append({"role": "user", "content": user_content})

    with st.chat_message("user"):
        st.markdown(user_content)
        if uploaded_image:
            st.image(uploaded_image)

    image_base64 = None
    if uploaded_image:
        image_base64 = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        use_groq = st.session_state.get("use_groq", False)
        start = time.time()
        try:
            result = st.session_state.rag_pipeline.process_question(
                question=question, image_base64=image_base64, use_groq=use_groq
            )
        except Exception as e:
            placeholder.markdown(f"Something went wrong generating a response: {e}")
            st.session_state.messages.append(
                {"role": "assistant", "content": "Something went wrong generating a response.",
                 "sources": [], "used_llm": "none", "latency": None}
            )
            return

        latency = round(time.time() - start, 2)
        answer = (result.get("answer") or "").strip() or "Something went wrong generating a response."
        sources = result.get("sources", []) or []
        used_llm = result.get("used_llm", "none")

        placeholder.markdown(answer)
        if sources:
            chips = "".join(f'<span class="source-chip">{s}</span>' for s in sources)
            st.markdown(chips, unsafe_allow_html=True)
        st.markdown(f'<div class="caption-row">answered by {used_llm} · {latency}s</div>', unsafe_allow_html=True)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources,
             "used_llm": used_llm, "latency": latency}
        )


def render_main():
    st.set_page_config(
        page_title="Enterprise Knowledge Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    initialize_session_state()
    if not validate_config():
        st.stop()

    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    render_main()
