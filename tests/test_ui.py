from pathlib import Path


def test_transcript_course_list_does_not_use_arrow_dataframe():
    """Avoid a native PyArrow crash seen on Python 3.13 during dialog reruns."""
    source = (Path(__file__).parents[1] / "app" / "ui.py").read_text(encoding="utf-8")

    assert "st.dataframe(" not in source
    assert "st.table(" not in source
