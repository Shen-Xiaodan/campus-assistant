from app.crawl_study_schemes import applies_to_year, choose_document, parse_documents, parse_index


def test_applies_to_admission_year() -> None:
    assert applies_to_year("经济学_适用于2023至24年度入学学生.pdf", 2023)
    assert applies_to_year("临床医学_适用于2023至24年度及以后入学学生.pdf", 2025)
    assert applies_to_year("专业_适用于2021至22及2022至23年度入学学生.pdf", 2022)
    assert applies_to_year("材料科学与工程_适用于2023-24年度入学学生.pdf", 2023)
    assert applies_to_year("翻译_适用于2022至23年度及之后入学学生.pdf", 2023)
    assert applies_to_year("应用心理学_适用于2020至21年度至2024至25年度入学学生.pdf", 2023)
    assert not applies_to_year("专业_适用于2024至25年度入学学生.pdf", 2023)


def test_choose_prefers_exact_version() -> None:
    documents = [
        ("https://example/after.pdf", "专业_适用于2023至24年度及以后入学学生.pdf"),
        ("https://example/exact.pdf", "专业_适用于2023至24年度入学学生.pdf"),
    ]
    assert choose_document(documents, 2023) == documents[1]


def test_parse_index_only_uses_print_content() -> None:
    html = """
    <a href="/page/999">导航噪声</a>
    <div id="print-content"><a href="/page/1">普通内容</a>
    <h2>主修课程规定</h2><a href="/page/65">经济学</a><a href="/page/57">计算机科学与技术</a>
    <h3>双主修课程</h3><a href="/page/66">经济学双主修</a></div>
    """
    assert parse_index(html, "https://registry.cuhk.edu.cn/page/22") == [
        ("经济学", "https://registry.cuhk.edu.cn/page/65"),
        ("计算机科学与技术", "https://registry.cuhk.edu.cn/page/57"),
    ]


def test_parse_documents() -> None:
    html = """
    <div id="print-content"><h2 class="page-title">经济学</h2>
      <div class="list-content"><a href="/files/econ.pdf">下载</a>
      <div class="list-title">经济学_适用于2023至24年度入学学生.pdf</div></div>
    </div>
    """
    title, documents = parse_documents(html, "https://registry.cuhk.edu.cn/page/65")
    assert title == "经济学"
    assert documents == [("https://registry.cuhk.edu.cn/files/econ.pdf", "经济学_适用于2023至24年度入学学生.pdf")]
