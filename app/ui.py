"""Streamlit UI kept intentionally thin; all RAG work is handled by the API."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
EXAMPLES = [
    "图书馆的开放时间是什么？",
    "学生证丢失后应该如何补办？",
    "申请奖学金需要满足哪些条件？",
]

st.set_page_config(
    page_title="校园问答 · Campus Guide",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# A restrained editorial theme. Component selectors are scoped to Streamlit's
# test ids so the markup stays semantic and the Python UI remains easy to read.
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600&family=Playfair+Display:wght@600&display=swap');

        :root {
            --paper: #f7f6f2;
            --ink: #20231f;
            --muted: #74786f;
            --line: #dedfd8;
            --sage: #687466;
            --sage-soft: #e8ece5;
            --white: #fffefa;
        }

        html, body, [class*="css"] {
            font-family: "Noto Sans SC", "PingFang SC", sans-serif;
            color: var(--ink);
        }

        .stApp { background: var(--paper); }

        [data-testid="stHeader"] { background: transparent; }

        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 820px;
            padding-top: 4.2rem;
            padding-bottom: 8rem;
        }

        [data-testid="stSidebar"] {
            background: #eeeee8;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] .block-container {
            padding: 3rem 1.6rem 2rem;
        }

        .sidebar-brand {
            min-height: calc(100vh - 7rem);
            display: flex;
            flex-direction: column;
        }

        .sidebar-eyebrow {
            color: var(--sage);
            font-size: .61rem;
            font-weight: 600;
            letter-spacing: .2em;
            margin-bottom: 1.8rem;
            text-transform: uppercase;
        }

        .sidebar-mark {
            align-items: center;
            background: var(--ink);
            border-radius: 50%;
            color: var(--paper);
            display: flex;
            font-family: "Playfair Display", serif;
            font-size: 1.1rem;
            height: 2.7rem;
            justify-content: center;
            margin-bottom: 1.4rem;
            width: 2.7rem;
        }

        .sidebar-name {
            color: var(--ink);
            font-family: "Songti SC", "Noto Sans SC", serif;
            font-size: 2.35rem;
            font-weight: 600;
            letter-spacing: -.06em;
            line-height: 1.08;
            margin: 0;
        }

        .sidebar-name span {
            color: var(--sage);
            display: block;
            font-size: .72rem;
            font-weight: 500;
            letter-spacing: .18em;
            margin-bottom: .65rem;
        }

        .sidebar-rule {
            background: var(--line);
            height: 1px;
            margin: 2rem 0 1.7rem;
            width: 100%;
        }

        .sidebar-copy {
            color: var(--muted);
            font-size: .72rem;
            line-height: 1.9;
            margin: 0;
        }

        .sidebar-meta {
            border-top: 1px solid var(--line);
            color: #8c9088;
            font-size: .58rem;
            letter-spacing: .14em;
            margin-top: auto;
            padding-top: 1.2rem;
            text-transform: uppercase;
        }

        .brand-kicker {
            color: var(--sage);
            font-size: .72rem;
            font-weight: 600;
            letter-spacing: .18em;
            margin-bottom: .55rem;
            text-transform: uppercase;
        }

        .brand-title {
            color: var(--ink);
            font-family: "Playfair Display", "Songti SC", serif;
            font-size: 2rem;
            line-height: 1.12;
            margin: 0;
        }

        .brand-copy {
            color: var(--muted);
            font-size: .84rem;
            line-height: 1.75;
            margin: 1rem 0 2.2rem;
        }

        .section-label {
            color: var(--muted);
            font-size: .7rem;
            font-weight: 600;
            letter-spacing: .14em;
            margin: 1.35rem 0 .75rem;
            text-transform: uppercase;
        }

        .hero { margin-bottom: 3rem; }

        .hero-rule {
            background: var(--sage);
            height: 2px;
            margin-bottom: 1.4rem;
            width: 44px;
        }

        .hero h1 {
            color: var(--ink);
            font-family: "Playfair Display", "Songti SC", serif;
            font-size: clamp(2.7rem, 7vw, 4.7rem);
            font-weight: 600;
            letter-spacing: -.04em;
            line-height: 1.03;
            margin: 0 0 1.2rem;
        }

        .hero p {
            color: var(--muted);
            font-size: .96rem;
            line-height: 1.8;
            margin: 0;
            max-width: 590px;
        }

        [data-testid="stHorizontalBlock"] .stButton > button {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 3px;
            color: #454a43;
            font-size: .82rem;
            line-height: 1.5;
            min-height: 3.6rem;
            padding: .65rem .8rem;
            transition: border-color .18s ease, color .18s ease, transform .18s ease;
        }

        [data-testid="stHorizontalBlock"] .stButton > button:hover {
            background: var(--white);
            border-color: var(--sage);
            color: var(--sage);
            transform: translateY(-1px);
        }

        [data-testid="stChatMessage"] {
            background: transparent;
            border-bottom: 1px solid var(--line);
            border-radius: 0;
            gap: .8rem;
            padding: 1.25rem .15rem 1.45rem;
        }

        [data-testid="stChatMessage"] p { line-height: 1.8; }

        [data-testid="stChatMessage"] p { color: var(--ink); }

        [data-testid="stChatMessageAvatarUser"] {
            background: var(--ink);
        }

        [data-testid="stChatMessageAvatarAssistant"] {
            background: var(--sage-soft);
            color: var(--sage);
        }

        [data-testid="stExpander"] {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 3px;
            margin-top: .7rem;
        }

        /* Streamlit's fixed input area inherits the system theme through
           several wrapper layers. Paint every layer so dark mode cannot leave
           a black frame around the light input. */
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"],
        [data-testid="stBottomBlockContainer"] > div {
            background: var(--paper) !important;
        }

        [data-testid="stBottom"]::before,
        [data-testid="stBottom"]::after {
            background: none !important;
        }

        [data-testid="stChatInput"] {
            background: var(--paper) !important;
            border-top: 1px solid var(--line);
            padding-bottom: 1.15rem;
            padding-top: 1rem;
        }

        [data-testid="stChatInput"] > div {
            background: var(--white);
            border: 1px solid #cfd2c9;
            border-radius: 2px;
            box-shadow: 0 10px 30px rgba(32, 35, 31, .06);
        }

        [data-testid="stChatInput"] textarea { font-size: .92rem; }

        [data-testid="stChatInput"] textarea {
            background: var(--white);
            caret-color: var(--sage);
            color: var(--ink);
        }

        [data-testid="stChatInput"] textarea::placeholder { color: #8b8f86; }

        [data-testid="stChatInputSubmitButton"] { color: var(--sage); }

        .stAlert { border-radius: 2px; }
        .stAlert p { color: var(--ink); }

        @media (max-width: 640px) {
            [data-testid="stAppViewContainer"] > .main .block-container {
                padding-top: 2.2rem;
            }
            .hero { margin-bottom: 2rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown(
        """
        <section class="sidebar-brand">
            <div class="sidebar-eyebrow">CUHK · Shenzhen</div>
            <div class="sidebar-mark">中</div>
            <p class="sidebar-name"><span>港中深</span>校园助手</p>
            <div class="sidebar-rule"></div>
            <p class="sidebar-copy">
                查阅校园规章、学生手册与学校官网，<br>
                为你的问题提供清晰、可追溯的回答。
            </p>
            <div class="sidebar-meta">Campus Knowledge Desk · 2026</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <section class="hero">
        <div class="hero-rule"></div>
        <div class="brand-kicker">Student Affairs / Knowledge Desk</div>
        <h1>Ask the<br>Campus.</h1>
        <p>把复杂的校园规定，变成清晰、可追溯的答案。输入你的问题，助手会检索现有资料并附上对应页码。</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            """
### 你好，我是港中深校园助手 👋

你可以问我关于港中深的：

- **专业与课程**：专业设置、培养方案和课程信息
- **校园事务**：学生证、奖学金、社会实践等办事流程
- **校园服务**：图书馆开放时间、部门联系方式和服务指南

我会从已收录的学校资料和官网内容中查找答案，并附上**资料来源、页码或原文链接**，方便你进一步核对。
            """
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("grounded") is False:
            st.info("现有资料暂时不足，你可以补充更具体的信息，我再帮你找找。")
        for source in message.get("sources", []):
            location = f"第 {source['page']} 页" if source.get("page") else source.get("section") or "学校官网"
            with st.expander(f"资料来源 · {source['document']} · {location} · 相关度 {source['score']:.2f}"):
                st.write(source["excerpt"])
                if source.get("url"):
                    st.link_button("查看学校官网原文", source["url"])
                if source.get("crawled_at"):
                    st.caption(f"最后采集：{source['crawled_at'][:10]}")

st.markdown(
    f'<div class="section-label">{"你可以从这些问题开始" if not st.session_state.messages else "继续探索"}</div>',
    unsafe_allow_html=True,
)
selected_example = None
example_columns = st.columns(len(EXAMPLES))
for index, (column, example) in enumerate(zip(example_columns, EXAMPLES, strict=True), start=1):
    with column:
        if st.button(example, key=f"example_{index}", use_container_width=True):
            selected_example = example

# Always render the input. Using ``selected_example or st.chat_input(...)`` here
# makes Python skip the widget whenever an example is clicked, which is why the
# input used to disappear until the next browser refresh.
typed_question = st.chat_input("询问校园规章、办事流程或学生服务……")
question = selected_example if selected_example is not None else typed_question

if question:
    request_history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[-6:]
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("正在查阅校园资料……"):
                api_response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question, "history": request_history},
                    timeout=90,
                )
                if api_response.status_code == 503:
                    detail = api_response.json().get("detail", "知识库尚未就绪")
                    raise RuntimeError(f"问答服务已连接，但暂不可用：{detail}")
                api_response.raise_for_status()
                result = api_response.json()
            st.markdown(result["answer"])
            if not result["grounded"]:
                st.info("现有资料暂时不足，你可以补充更具体的信息，我再帮你找找。")
            for source in result["sources"]:
                location = f"第 {source['page']} 页" if source.get("page") else source.get("section") or "学校官网"
                with st.expander(f"资料来源 · {source['document']} · {location} · 相关度 {source['score']:.2f}"):
                    st.write(source["excerpt"])
                    if source.get("url"):
                        st.link_button("查看学校官网原文", source["url"])
                    if source.get("crawled_at"):
                        st.caption(f"最后采集：{source['crawled_at'][:10]}")
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], **result})
        except RuntimeError as exc:
            error = str(exc)
            st.warning(error)
            st.session_state.messages.append({"role": "assistant", "content": error, "grounded": False, "sources": []})
        except (requests.RequestException, ValueError):
            error = "无法连接问答服务，请确认 API 已启动且索引已建立。"
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error, "grounded": False, "sources": []})

    # Persist the completed turn before accepting another quick question. This
    # rerun redraws the full history from session state and prevents a second
    # example click from replacing the answer that was rendered provisionally.
    st.rerun()
