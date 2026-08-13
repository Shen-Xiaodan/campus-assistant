from app.language import bilingual_department, detect_language


def test_detects_dominant_question_language():
    assert detect_language("奖学金 scholarship 怎么申请？") == "zh"
    assert detect_language("Where is the 教务处 office?") == "en"


def test_department_names_are_bilingual_without_guessing_unknown_names():
    assert bilingual_department("学生事务处") == ("学生事务处", "Office of Student Affairs")
    assert bilingual_department("Library") == ("图书馆", "Library")
    assert bilingual_department("未知部门") == ("未知部门", None)
