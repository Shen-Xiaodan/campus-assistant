from app.documents import ParsedPage, split_page


def test_split_page_preserves_metadata_and_heading():
    page = ParsedPage(
        text="第一章 学生证管理\n\n学生证丢失后，应当向学生事务中心提交补办申请。\n\n办理时需要携带身份证明。",
        document="学生手册.pdf",
        source_path="/docs/学生手册.pdf",
        page=12,
        modified_at="2026-01-01T00:00:00+00:00",
    )
    chunks = split_page(page, chunk_size=45, overlap=8)

    assert chunks
    assert chunks[0].text.startswith("第一章 学生证管理")
    assert chunks[0].metadata == {
        "document": "学生手册.pdf",
        "source_path": "/docs/学生手册.pdf",
        "page": 12,
        "chunk_index": 1,
        "modified_at": "2026-01-01T00:00:00+00:00",
    }
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(1, len(chunks) + 1))


def test_long_paragraph_is_split_with_overlap():
    page = ParsedPage("标题\n\n" + "校园规定内容。" * 30, "规定.pdf", "/规定.pdf", 2, None)
    chunks = split_page(page, chunk_size=60, overlap=10)
    assert len(chunks) >= 3
    assert all(len(chunk.text) <= 60 for chunk in chunks)
