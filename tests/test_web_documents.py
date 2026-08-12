import json

import pytest

from app.web_documents import (
    WebSource,
    extract_structured_fields,
    extract_web_page,
    is_allowed_url,
    load_web_sources,
    split_web_page,
)


@pytest.fixture
def source() -> WebSource:
    return WebSource(
        name="学生事务处",
        department="学生事务处",
        start_urls=("https://student.example.edu.cn/services/",),
        allowed_domains=("example.edu.cn",),
        allowed_paths=("/services/",),
    )


def test_url_allowlist_restricts_domain_and_path(source):
    assert is_allowed_url("https://student.example.edu.cn/services/contact", source)
    assert not is_allowed_url("https://student.example.edu.cn/news/contact", source)
    assert not is_allowed_url("https://external.example.com/services/contact", source)
    assert not is_allowed_url("https://student.example.edu.cn/services/logo.png", source)


def test_extracts_headings_lists_tables_and_contact_fields(source):
    html = """
    <html><head><title>学生服务指南</title></head><body>
      <nav>首页 新闻 联系我们</nav><main>
      <h1>奖学金申请</h1><h2>申请材料</h2>
      <p>申请人需要提交以下材料。</p><ul><li>申请表</li><li>成绩单</li></ul>
      <h2>办理信息</h2>
      <table><tr><th>事项</th><th>说明</th></tr><tr><td>受理方式</td><td>现场办理</td></tr></table>
      <p>办公时间：周一至周五 09:00-17:00</p>
      <p>办公地址：行政楼 201 室</p><p>邮箱：student@example.edu.cn</p>
      </main><footer>版权信息</footer>
    </body></html>
    """
    page = extract_web_page(html, "https://student.example.edu.cn/services/scholarship", source)

    assert page.title == "学生服务指南"
    assert any(section.endswith("申请材料") and "申请表" in text for section, text in page.sections)
    assert any("事项：受理方式" in text and "说明：现场办理" in text for _, text in page.sections)
    assert page.fields["emails"] == ["student@example.edu.cn"]
    assert "办公时间：周一至周五 09:00-17:00" in page.fields["office_hours"]
    assert "首页 新闻" not in " ".join(text for _, text in page.sections)


def test_web_chunks_preserve_source_metadata(source):
    page = extract_web_page(
        "<html><title>办事指南</title><main><h2>办理流程</h2><p>先提交申请，再等待审核。</p></main></html>",
        "https://student.example.edu.cn/services/process",
        source,
    )
    chunks = split_web_page(page, chunk_size=200, overlap=20)

    assert chunks
    assert chunks[0].metadata["source_type"] == "web"
    assert chunks[0].metadata["source_url"].endswith("/services/process")
    assert chunks[0].metadata["section"] == "办理流程"
    assert "章节：办理流程" in chunks[0].text


def test_load_web_sources_requires_explicit_allowlist(tmp_path):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"sources": [{"name": "官网", "start_urls": ["https://example.edu.cn/"]}]}))
    with pytest.raises(ValueError, match="allowed_domains"):
        load_web_sources(path)


def test_structured_field_extraction_is_optional():
    assert extract_structured_fields("这是一个没有联系方式的奖学金办理流程说明。") == {}


def test_robots_4xx_means_no_rules_but_5xx_stops_crawl(monkeypatch):
    from app.web_documents import OfficialSiteCrawler

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code
            self.text = ""

    crawler = OfficialSiteCrawler()
    monkeypatch.setattr(crawler, "_get", lambda *_args, **_kwargs: Response(403))
    assert crawler._robot_allowed("https://example.edu.cn/public") is True

    crawler = OfficialSiteCrawler()
    monkeypatch.setattr(crawler, "_get", lambda *_args, **_kwargs: Response(503))
    assert crawler._robot_allowed("https://example.edu.cn/public") is False
