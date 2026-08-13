# 港中深校园助手

一个面向香港中文大学（深圳）校园资料的 RAG 问答项目。系统从培养方案、学生手册和学校官网中检索证据，再生成带页码或原文链接的回答。

![港中深校园助手界面](docs/ui-preview.jpg)

## 主要功能

- PDF 与学校官网增量索引
- 向量与关键词混合检索
- 汇总问题的多文档召回与专业资料精确过滤
- 只展示答案实际引用的来源
- 最近对话上下文，支持“还有呢”“详细一点”等追问
- FastAPI 接口与 Streamlit 对话界面

回答仅基于已收录资料；证据不足时会提示用户补充信息，不使用模型知识猜测校园事实。

## 技术栈

Python、FastAPI、Streamlit、Chroma、Sentence Transformers、pypdf，以及 OpenAI-compatible Chat API。

## 本地运行

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写模型服务配置：

```env
LLM_PROVIDER=siliconflow
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL_ID=Qwen/Qwen3.6-35B-A3B
```

`.env` 已加入忽略列表，请勿提交真实密钥。

### 2. 导入资料

将 PDF 放入 `data/`，然后构建索引：

```bash
python -m app.ingest
```

如需采集已配置的学校官网页面：

```bash
python -m app.ingest_web
```

官网来源配置见 `data/web_sources.example.json`。

### 3. 启动服务

分别运行：

```bash
uvicorn app.api:app --reload
```

```bash
streamlit run app/ui.py
```

- 前端：<http://localhost:8501>
- API 文档：<http://127.0.0.1:8000/docs>

## API 示例

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "还有哪些？",
    "history": [
      {"role": "user", "content": "金融学有哪些必修课？"},
      {"role": "assistant", "content": "目前能确认的是……"}
    ]
  }'
```

`history` 可省略。前端默认发送最近 6 条消息用于理解追问，校园事实仍须由检索证据支持。

## 测试

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

## 项目结构

```text
app/
├── api.py             # FastAPI 接口
├── ui.py              # Streamlit 前端
├── retrieval.py       # 检索、过滤与多文档召回
├── generation.py      # 提示词、回答与引用校验
├── documents.py       # PDF 解析与切分
├── web_documents.py   # 官网采集与正文提取
├── index.py           # 向量索引
└── service.py         # 问答流程编排
tests/                 # 单元测试
data/                  # 校园资料与官网配置
evaluation/            # 离线评测数据
```

## 当前限制

- 扫描版 PDF 需要预先 OCR。
- 复杂表格和多栏 PDF 的效果取决于文本提取质量。
- JavaScript 动态渲染的官网页面暂不支持。
- 回答不能替代学校正式通知，请通过引用来源核对重要信息。

本项目基于 [`DeepShah1406/College_RAG_Chatbot`](https://github.com/DeepShah1406/College_RAG_Chatbot) 重构，采用 [MIT License](LICENSE)。
