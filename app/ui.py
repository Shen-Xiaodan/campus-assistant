"""Streamlit UI kept intentionally thin; all RAG work is handled by the API."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
COPY = {
    "zh": {
        "switch": "English",
        "sidebar_name": "<span>港中深</span>校园助手",
        "sidebar_copy": "查阅校园规章、学生手册与学校官网，<br>为你的问题提供清晰、可追溯的回答。",
        "hero_copy": "把复杂的校园规定，变成清晰、可追溯的答案。输入你的问题，助手会检索现有资料并附上对应页码。",
        "welcome": """### 你好，我是港中深校园助手 👋

你可以问我关于港中深的：

- **专业与课程**：专业设置、培养方案和课程信息
- **校园事务**：学生证、奖学金、社会实践等办事流程
- **校园服务**：图书馆开放时间、部门联系方式和服务指南

我会从已收录的学校资料和官网内容中查找答案，并附上**资料来源、页码或原文链接**，方便你进一步核对。""",
        "examples": ["图书馆的开放时间是什么？", "学生证丢失后应该如何补办？", "申请奖学金需要满足哪些条件？"],
        "start": "你可以从这些问题开始",
        "continue": "继续探索",
        "placeholder": "询问校园规章、办事流程或学生服务……",
        "insufficient": "现有资料暂时不足，你可以补充更具体的信息，我再帮你找找。",
        "source": "资料来源",
        "page": "第 {page} 页",
        "website": "学校官网",
        "relevance": "相关度",
        "department": "发布部门",
        "original": "查看学校官网原文",
        "crawled": "最后采集",
        "loading": "正在查阅校园资料……",
        "not_ready": "知识库尚未就绪",
        "unavailable": "问答服务已连接，但暂不可用：{detail}",
        "connection_error": "无法连接问答服务，请确认 API 已启动且索引已建立。",
        "selected_version": "已选择：{option}",
        "transcript_title": "成绩单匹配",
        "transcript_intro": "上传 PDF 成绩单，临时解析课程并按培养方案核对。文件不会加入知识库。",
        "upload_transcript": "上传并解析成绩单",
        "choose_file": "选择成绩单 PDF",
        "start_parse": "开始解析",
        "parsing": "正在读取成绩单，请稍候……",
        "parsed_summary": "解析摘要",
        "course_count": "识别课程数",
        "parsed_completed": "已完成课程",
        "parsed_in_progress": "在修课程",
        "parsed_credits": "已识别完成学分",
        "confirm_programme": "专业（请确认）",
        "confirm_year": "入学年份（请确认）",
        "check_report": "生成匹配报告",
        "checking": "正在核对培养方案，请稍候……",
        "reset_transcript": "重新上传",
        "close": "关闭",
        "parse_failed": "成绩单解析失败：{detail}",
        "report_failed": "报告生成失败：{detail}",
        "file_too_large": "文件超过 10 MB，请选择较小的 PDF。",
        "choose_hint": "请选择 PDF 文件后继续。",
        "course_details": "查看识别到的课程",
        "term": "学期",
        "current_status": "当前状态",
        "credits_progress": "已获得学分 / 要求",
        "completed_courses": "已完成",
        "in_progress_courses": "在修",
        "missing_courses": "缺少",
        "none": "无",
        "manual_review": "以下项目需要人工核验：",
    },
    "en": {
        "switch": "中文",
        "sidebar_name": "<span>CUHK-Shenzhen</span>Campus Guide",
        "sidebar_copy": (
            "Explore campus policies, student handbooks and official webpages<br>with clear, traceable answers."
        ),
        "hero_copy": (
            "Turn complex campus policies into clear, traceable answers. Ask a question and the guide will search "
            "available sources with page-level citations."
        ),
        "welcome": """### Hi, I'm your CUHK-Shenzhen Campus Guide 👋

You can ask me about:

- **Programmes and courses**: programme requirements, study schemes and course information
- **Campus affairs**: student cards, scholarships and administrative procedures
- **Campus services**: library hours, department contacts and service guides

I'll search the available university documents and official webpages, then provide **source documents, page numbers or
original links** for you to verify.""",
        "examples": [
            "What are the library opening hours?",
            "How do I replace a lost student card?",
            "What are the scholarship requirements?",
        ],
        "start": "QUESTIONS TO GET YOU STARTED",
        "continue": "KEEP EXPLORING",
        "placeholder": "Ask about campus policies, procedures or student services…",
        "insufficient": "The available sources are not quite enough. Add a little more detail and I'll look again.",
        "source": "Source",
        "page": "Page {page}",
        "website": "Official website",
        "relevance": "Relevance",
        "department": "Department",
        "original": "View the original webpage",
        "crawled": "Last collected",
        "loading": "Searching campus sources…",
        "not_ready": "The knowledge base is not ready",
        "unavailable": "The service is connected but temporarily unavailable: {detail}",
        "connection_error": "Unable to reach the service. Check that the API is running and the index is available.",
        "selected_version": "Selected: {option}",
        "transcript_title": "Transcript matching",
        "transcript_intro": (
            "Upload a PDF transcript to temporarily parse courses and check your study scheme. "
            "The file is never added to the knowledge base."
        ),
        "upload_transcript": "Upload and parse transcript",
        "choose_file": "Choose transcript PDF",
        "start_parse": "Start parsing",
        "parsing": "Reading your transcript…",
        "parsed_summary": "Parsing summary",
        "course_count": "Courses detected",
        "parsed_completed": "Completed courses",
        "parsed_in_progress": "Courses in progress",
        "parsed_credits": "Detected completed credits",
        "confirm_programme": "Programme (confirm)",
        "confirm_year": "Admission year (confirm)",
        "check_report": "Generate matching report",
        "checking": "Checking your study scheme…",
        "reset_transcript": "Upload another",
        "close": "Close",
        "parse_failed": "Transcript parsing failed: {detail}",
        "report_failed": "Report generation failed: {detail}",
        "file_too_large": "The file exceeds 10 MB. Please choose a smaller PDF.",
        "choose_hint": "Choose a PDF file to continue.",
        "course_details": "View detected courses",
        "term": "Term",
        "current_status": "Current status",
        "credits_progress": "Earned / required credits",
        "completed_courses": "Completed",
        "in_progress_courses": "In progress",
        "missing_courses": "Missing",
        "none": "None",
        "manual_review": "Manual review required: ",
    },
}

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

        /* Hide Streamlit's deploy and overflow controls. */
        [data-testid="stMainMenu"],
        [data-testid="stStatusWidget"],
        [data-testid="stDecoration"],
        [data-testid="stAppDeployButton"],
        .stDeployButton,
        button[title="View app in Streamlit Community Cloud"],
        button[aria-label="Main menu"] {
            display: none !important;
        }

        /* Keep Streamlit's sidebar reopen control available after collapse.
           It shares the header area with the controls hidden above. */
        [data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            position: relative;
            z-index: 1000;
        }

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
            padding: 2rem 1.6rem 1.5rem;
        }

        .sidebar-brand {
            display: flex;
            flex-direction: column;
        }

        .sidebar-eyebrow {
            color: var(--sage);
            font-size: .61rem;
            font-weight: 600;
            letter-spacing: .2em;
            margin-bottom: 1rem;
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
            height: 2.35rem;
            justify-content: center;
            margin-bottom: 1rem;
            width: 2.35rem;
        }

        .sidebar-name {
            color: var(--ink);
            font-family: "Songti SC", "Noto Sans SC", serif;
            font-size: 2rem;
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
            margin: 1.25rem 0 1rem;
            width: 100%;
        }

        .sidebar-copy {
            color: var(--muted);
            font-size: .72rem;
            line-height: 1.65;
            margin: 0;
        }

        .sidebar-meta {
            border-top: 1px solid var(--line);
            color: #8c9088;
            font-size: .58rem;
            letter-spacing: .14em;
            margin-top: 1rem;
            padding-top: .8rem;
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

        .model-status {
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: .72rem;
            line-height: 1.7;
            margin-top: 1.2rem;
            padding-top: 1rem;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: rgba(255, 254, 250, .55);
            margin-top: 1.6rem;
        }

        [data-testid="stSidebar"] .stButton > button {
            border-color: var(--line);
            border-radius: 3px;
        }

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
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "zh"
if "transcript_state" not in st.session_state:
    st.session_state.transcript_state = "idle"
if "transcript_upload_nonce" not in st.session_state:
    st.session_state.transcript_upload_nonce = 0

language = st.session_state.ui_language
copy = COPY[language]


def render_source(source: dict) -> None:
    location = (
        copy["page"].format(page=source["page"])
        if source.get("page")
        else source.get("section") or copy["website"]
    )
    label = f'{copy["source"]} · {source["document"]} · {location} · {copy["relevance"]} {source["score"]:.2f}'
    with st.expander(label):
        department = source.get("department_en") if language == "en" else source.get("department_zh")
        alternate = source.get("department_zh") if language == "en" else source.get("department_en")
        if department:
            if alternate and alternate != department:
                department += f" / {alternate}"
            st.caption(f'{copy["department"]}: {department}')
        st.write(source["excerpt"])
        if source.get("url"):
            st.link_button(copy["original"], source["url"])
        if source.get("crawled_at"):
            st.caption(f'{copy["crawled"]}: {source["crawled_at"][:10]}')


def _api_detail(response: requests.Response, fallback: str) -> str:
    try:
        return str(response.json().get("detail", fallback))
    except ValueError:
        return fallback


def _clear_transcript() -> None:
    analysis_id = st.session_state.get("transcript_parse", {}).get("analysis_id")
    if analysis_id:
        try:
            requests.delete(f"{API_URL}/transcripts/{analysis_id}", timeout=10)
        except requests.RequestException:
            pass
    for key in (
        "transcript_file", "transcript_parse", "graduation_report", "transcript_programme",
        "transcript_year", "transcript_error", "transcript_error_stage",
    ):
        st.session_state.pop(key, None)
    st.session_state.transcript_upload_nonce += 1
    st.session_state.transcript_state = "idle"


@st.dialog("成绩单匹配 / Transcript matching")
def transcript_dialog() -> None:
    st.caption(copy["transcript_intro"])
    state = st.session_state.transcript_state

    if state in {"idle", "file_selected"}:
        upload = st.file_uploader(
            copy["choose_file"], type=["pdf"],
            key=f"transcript_file_picker_{st.session_state.transcript_upload_nonce}",
        )
        if upload:
            if upload.size > 10 * 1024 * 1024:
                st.error(copy["file_too_large"])
                st.session_state.transcript_file = None
            else:
                st.session_state.transcript_file = upload
                st.session_state.transcript_state = "file_selected"
        file = st.session_state.get("transcript_file")
        if file:
            st.caption(f"{file.name} · {file.size / 1024 / 1024:.2f} MB")
            if st.button(copy["start_parse"], type="primary", use_container_width=True):
                st.session_state.transcript_state = "parsing"
                st.session_state.pop("transcript_error", None)
                try:
                    with st.spinner(copy["parsing"]):
                        response = requests.post(
                            f"{API_URL}/transcripts/parse",
                            files={"upload": (file.name, file.getvalue(), "application/pdf")},
                            timeout=60,
                        )
                        response.raise_for_status()
                        st.session_state.transcript_parse = response.json()
                    st.session_state.transcript_state = "parsed"
                except (requests.RequestException, ValueError) as exc:
                    detail = _api_detail(response, str(exc)) if "response" in locals() else str(exc)
                    st.session_state.transcript_error = copy["parse_failed"].format(detail=detail)
                    st.session_state.transcript_error_stage = "parse"
                    st.session_state.transcript_state = "file_selected"
                st.rerun(scope="fragment")
        else:
            st.info(copy["choose_hint"])

    if st.session_state.get("transcript_error"):
        st.error(st.session_state.transcript_error)

    if st.session_state.transcript_state in {"parsed", "completed"}:
        parsed = st.session_state.transcript_parse
        st.subheader(copy["parsed_summary"])
        st.write(f"{copy['course_count']}：{len(parsed.get('courses', []))}")
        completed_courses = [course for course in parsed.get("courses", []) if course.get("status") == "passed"]
        in_progress_courses = [
            course for course in parsed.get("courses", []) if course.get("status") == "in_progress"
        ]
        completed_credits = sum(course.get("credits") or 0 for course in completed_courses)
        summary_columns = st.columns(3)
        summary_columns[0].metric(copy["parsed_completed"], len(completed_courses))
        summary_columns[1].metric(copy["parsed_in_progress"], len(in_progress_courses))
        summary_columns[2].metric(copy["parsed_credits"], f"{completed_credits:g}")
        if parsed.get("warnings"):
            st.warning("；".join(parsed["warnings"]))
        if parsed.get("courses"):
            with st.expander(copy["course_details"]):
                for course in parsed["courses"]:
                    code = course.get("course_code") or "—"
                    name = course.get("course_name") or "—"
                    credits = course.get("credits")
                    grade = course.get("grade") or "—"
                    status = course.get("status") or "unknown"
                    term = course.get("term") or "—"
                    credit_text = f"{credits:g}" if isinstance(credits, int | float) else "—"
                    st.text(f"{code} · {name}")
                    st.caption(
                        f"{copy['term']}: {term} · Credits: {credit_text} · Grade: {grade} · Status: {status}"
                    )
        programme = st.text_input(
            copy["confirm_programme"], value=parsed.get("programme") or "", key="transcript_programme"
        )
        admission_year = st.number_input(
            copy["confirm_year"], min_value=2000, max_value=2100,
            value=parsed.get("admission_year") or 2023, key="transcript_year"
        )
        if st.button(
            copy["check_report"], type="primary", use_container_width=True,
            disabled=not programme.strip(),
        ):
            st.session_state.transcript_state = "checking"
            st.session_state.pop("transcript_error", None)
            try:
                with st.spinner(copy["checking"]):
                    response = requests.post(
                        f"{API_URL}/graduation/check",
                        json={
                            "analysis_id": parsed["analysis_id"],
                            "programme": programme,
                            "admission_year": admission_year,
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    st.session_state.graduation_report = response.json()
                st.session_state.transcript_state = "completed"
            except (requests.RequestException, ValueError) as exc:
                detail = _api_detail(response, str(exc)) if "response" in locals() else str(exc)
                st.session_state.transcript_error = copy["report_failed"].format(detail=detail)
                st.session_state.transcript_error_stage = "report"
                st.session_state.transcript_state = "parsed"
            st.rerun(scope="fragment")

    if st.session_state.transcript_state == "completed":
        report = st.session_state.graduation_report
        credits = report.get("credits", {})
        st.metric(copy["credits_progress"], f"{credits.get('earned', 0):g} / {credits.get('required', 0):g}")
        st.write(f"{copy['current_status']}：{report.get('overall_status', 'manual_review_required')}")
        for group in report.get("requirement_groups", []):
            with st.expander(group.get("name", group.get("id", "要求"))):
                completed = ", ".join(group.get("completed_courses", [])) or copy["none"]
                in_progress = ", ".join(group.get("in_progress_courses", [])) or copy["none"]
                missing = ", ".join(group.get("missing_courses", [])) or copy["none"]
                st.write(f"{copy['completed_courses']}：{completed}")
                st.write(f"{copy['in_progress_courses']}：{in_progress}")
                st.write(f"{copy['missing_courses']}：{missing}")
        if report.get("manual_review_items"):
            st.warning(copy["manual_review"] + "；".join(report["manual_review_items"]))
        st.caption(report.get("disclaimer", "结果仅供规划参考，以教务处最终审核为准。"))

    if (
        st.session_state.transcript_state in {"file_selected", "parsed", "completed"}
        and st.button(copy["reset_transcript"], use_container_width=True)
    ):
        _clear_transcript()
        st.rerun(scope="fragment")

with st.sidebar:
    st.markdown(
        f"""
        <section class="sidebar-brand">
            <div class="sidebar-eyebrow">CUHK · Shenzhen</div>
            <div class="sidebar-mark">{"中" if language == "zh" else "EN"}</div>
            <p class="sidebar-name">{copy["sidebar_name"]}</p>
            <div class="sidebar-rule"></div>
            <p class="sidebar-copy">{copy["sidebar_copy"]}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if st.button(copy["switch"], key="language_switch", use_container_width=True):
        st.session_state.ui_language = "en" if language == "zh" else "zh"
        st.rerun()
    st.divider()
    st.markdown(f"### {copy['transcript_title']}")
    st.caption(copy["transcript_intro"])
    if st.button(copy["upload_transcript"], type="primary", use_container_width=True):
        transcript_dialog()
    st.markdown('<div class="sidebar-meta">Campus Knowledge Desk · 2026</div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <section class="hero">
        <div class="hero-rule"></div>
        <div class="brand-kicker">Student Affairs / Knowledge Desk</div>
        <h1>Ask the<br>Campus.</h1>
        <p>{copy["hero_copy"]}</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if current_model_config() is None:
    st.info("准备好后，请先在左侧填写你自己的模型 API。配置只用于当前会话，不会保存到项目中。")

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(copy["welcome"])

selected_clarification = None
selected_scope = None
for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("scope_notice"):
            st.caption(f'◌ {message["scope_notice"]}')
        st.markdown(message["content"])
        if message.get("disclaimer"):
            st.caption(message["disclaimer"])
        if message.get("grounded") is False and not message.get("needs_clarification"):
            st.info(copy["insufficient"])
        if message.get("needs_clarification"):
            if message.get("clarification_resolved"):
                st.caption(copy["selected_version"].format(option=message["clarification_resolved"]))
            else:
                options = message.get("clarification_options") or []
                option_columns = st.columns(len(options))
                for option_index, (column, option) in enumerate(zip(option_columns, options, strict=True)):
                    with column:
                        if st.button(
                            option,
                            key=f"clarification_{message_index}_{option_index}",
                            use_container_width=True,
                        ):
                            message["clarification_resolved"] = option
                            selected_clarification = option
        if message.get("scope_options"):
            if message.get("scope_resolved"):
                st.caption(copy["selected_version"].format(option=message["scope_resolved"]))
            else:
                scope_options = message.get("scope_options") or []
                scope_columns = st.columns(len(scope_options))
                for option_index, (column, option) in enumerate(
                    zip(scope_columns, scope_options, strict=True)
                ):
                    with column:
                        if st.button(
                            option,
                            key=f"scope_{message_index}_{option_index}",
                            use_container_width=True,
                        ):
                            message["scope_resolved"] = option
                            selected_scope = option
        for source in message.get("sources", []):
            render_source(source)

st.markdown(
    f'<div class="section-label">{copy["start"] if not st.session_state.messages else copy["continue"]}</div>',
    unsafe_allow_html=True,
)
selected_example = None
example_columns = st.columns(len(copy["examples"]))
for index, (column, example) in enumerate(zip(example_columns, copy["examples"], strict=True), start=1):
    with column:
        if st.button(example, key=f"example_{index}", use_container_width=True):
            selected_example = example

# Always render the input. Using ``selected_example or st.chat_input(...)`` here
# makes Python skip the widget whenever an example is clicked, which is why the
# input used to disappear until the next browser refresh.
typed_question = st.chat_input(copy["placeholder"])
question = selected_scope or selected_clarification or selected_example or typed_question

if question:
    model_config = current_model_config()
    if model_config is None:
        st.info("开始提问前，请先在左侧配置你自己的模型服务。密钥仅用于当前会话中的模型请求。")
        st.stop()
    request_history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[-6:]
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner(copy["loading"]):
                api_response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question, "history": request_history, "model": model_config},
                    timeout=90,
                )
                if api_response.status_code == 503:
                    detail = api_response.json().get("detail", copy["not_ready"])
                    raise RuntimeError(copy["unavailable"].format(detail=detail))
                api_response.raise_for_status()
                result = api_response.json()
            if result.get("scope_notice"):
                st.caption(f'◌ {result["scope_notice"]}')
            st.markdown(result["answer"])
            if result.get("disclaimer"):
                st.caption(result["disclaimer"])
            if (
                not result["grounded"]
                and result.get("answer_mode", "official_fact") == "official_fact"
                and not result.get("needs_clarification")
            ):
                st.info(copy["insufficient"])
            for source in result["sources"]:
                render_source(source)
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], **result})
        except RuntimeError as exc:
            error = str(exc)
            st.warning(error)
            st.session_state.messages.append({"role": "assistant", "content": error, "grounded": False, "sources": []})
        except (requests.RequestException, ValueError):
            error = copy["connection_error"]
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error, "grounded": False, "sources": []})

    # Persist the completed turn before accepting another quick question. This
    # rerun redraws the full history from session state and prevents a second
    # example click from replacing the answer that was rendered provisionally.
    st.rerun()
