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

st.set_page_config(page_title="校园知识问答助手", page_icon="🎓", layout="centered")
st.title("🎓 校园知识问答助手")
st.caption("基于已导入的校园规章、学生手册和办事指南回答，并展示对应资料与页码。")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("使用说明")
    st.write("回答仅依据当前校园资料。来源不足时，助手会明确提示无法确定。")
    st.subheader("示例问题")
    selected_example = None
    for index, example in enumerate(EXAMPLES):
        if st.button(example, key=f"example_{index}", use_container_width=True):
            selected_example = example

if not st.session_state.messages:
    st.info("尚无对话。可以点击左侧示例问题，或在下方输入你的问题。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("grounded") is False:
            st.warning("证据不足：当前资料不能支持确定答案。")
        for source in message.get("sources", []):
            with st.expander(f"📄 {source['document']} · 第 {source['page']} 页 · 相关度 {source['score']:.2f}"):
                st.write(source["excerpt"])

question = selected_example or st.chat_input("请输入关于校园规章或办事流程的问题")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("正在检索校园资料并组织答案……"):
                api_response = requests.post(f"{API_URL}/chat", json={"question": question}, timeout=90)
                if api_response.status_code == 503:
                    detail = api_response.json().get("detail", "知识库尚未就绪")
                    raise RuntimeError(f"问答服务已连接，但暂不可用：{detail}")
                api_response.raise_for_status()
                result = api_response.json()
            st.markdown(result["answer"])
            if not result["grounded"]:
                st.warning("证据不足：当前资料不能支持确定答案。")
            for source in result["sources"]:
                with st.expander(f"📄 {source['document']} · 第 {source['page']} 页 · 相关度 {source['score']:.2f}"):
                    st.write(source["excerpt"])
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], **result})
        except RuntimeError as exc:
            error = str(exc)
            st.warning(error)
            st.session_state.messages.append(
                {"role": "assistant", "content": error, "grounded": False, "sources": []}
            )
        except (requests.RequestException, ValueError):
            error = "无法连接问答服务，请确认 API 已启动且索引已建立。"
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error, "grounded": False, "sources": []})
