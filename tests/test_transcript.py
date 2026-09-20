from app.transcript import parse_transcript_pages

PAGE_ONE = """
Admitted in: Sep 2023
Major/Programme: Computer Science and Engineering
Issue Date: 13 Sep 2026 Page 1 of 3

2023-24 Term 1
Course Code Course Title Units Grade % of A- and above
CHI1000 Chinese 3.0 B+ 37.3
CSC1003 Introduction to Computer Science and Java Programming 3.0 A- 38.8
ENG1001 English Bridge Program (EBP) 3.0 A- 30.4
GEW1001 Ethics and the Rule of Law 3.0 PA N/A
MAT1001 Calculus I 3.0 B+ 38.7
MAT2041 Linear Algebra and Applications 3.0 B+ 39.1
PED1001 Physical Education 1.0 A 38.5
Units Passed = 19.0 Term GPA = 3.494
Cumulative Units Passed = 19.0 Cumulative GPA = 3.494

2023-24 Term 2
Course Code Course Title Units Grade % of A- and above
CSC1004 Computational Laboratory Using Java 1.0 A 74.0
DDA2001 Introduction to Data Science 3.0 A 39.1
ENG1002 English for Academic Purposes I 3.0 A- 35.9
GEW2001 Introduction to Marxism 3.0 DI N/A
GFN1000 In Dialogue with Nature 3.0 B+ 38.7
ITE1000 Information Technology 1.0 DI N/A
MAT1002 Calculus II 3.0 A- 35.3
PED1002 Fitness and Health 1.0 A- 38.5
PHY1001 Mechanics 3.0 A 49.6
Units Passed = 21.0 Term GPA = 3.753
Cumulative Units Passed = 40.0 Cumulative GPA = 3.627
Honour(s)/Award(s): Dean's List 2023-24

2024-25 Term 1
Course Code Course Title Units Grade % of A- and above
BIO1008 Chemistry and Life Sciences 3.0 A- 39.5
CSC3001 Discrete Mathematics 3.0 A 42.6
CSC3002 C/C++ Programming 3.0 B 38.5
ENG2001 English for Academic Purposes II 3.0 B+ 34.1
GFH1000 In Dialogue with Humanity 3.0 A- 36.3
STA2001H Honours Probability and Statistics I 3.0 A- 38.9
Units Passed = 18.0 Term GPA = 3.567
Cumulative Units Passed = 58.0 Cumulative GPA = 3.606
"""


PAGE_TWO = """
Issue Date: 13 Sep 2026 Page 2 of 3
2024-25 Term 2
Course Code Course Title Units Grade % of A- and above
CSC3050 Computer Architecture 3.0 A- 39.1
CSC3100 Data Structures 3.0 A- 38.4
ECE2050 Digital Logic and Systems 3.0 B+ 40.0
ENG2002S English for Science and Engineering Communication 3.0 B+ 34.4
GEA2000 Modern Chinese History and Culture 3.0 A- 36.2
Units Passed = 15.0 Term GPA = 3.540
Cumulative Units Passed = 73.0 Cumulative GPA = 3.591
Honour(s)/Award(s): Dean's List 2024-25

2025-26 Term 1
Course Code Course Title Units Grade % of A- and above
CSC3150 Operating System 3.0 A- 37.5
CSC3170 Database System 3.0 A 38.6
DDA3020 Machine Learning 3.0 A 37.9
GEC2118 Healing Empires: Medicine, Power, and Colonialism (1600– 3.0 A- 35.7
                             1914)
MAT3007 Optimization 3.0 B+ 38.9
Units Passed = 15.0 Term GPA = 3.740
Cumulative Units Passed = 88.0 Cumulative GPA = 3.619

2025-26 Term 2
Courses taken elsewhere and accepted for transfer
Honour(s)/Award(s): Dean's List 2025-26

2026-27 Term 1
Course Code Course Title Units Grade % of A- and above
CSC3160 Fundamentals of Speech and Language Processing 3.0 IP 0.0
CSC4120 Design and Analysis of Algorithms 3.0 IP 0.0
CSC4140 Computer Graphics 3.0 IP 0.0
CSC4160 Cloud Computing 3.0 IP 0.0
GEB2602 Stuff Matters - Amazing Material World 3.0 IP 0.0
Units Passed = 0.0 Term GPA = 0.000
Cumulative Units Passed = 88.0 Cumulative GPA = 3.619
Summary
"""


PAGE_THREE = """
Issue Date: 13 Sep 2026 Page 3 of 3
Attended an exchange university in Academic Year 2025-26 Term 2 in partial fulfillment of the degree.
End of Transcript
"""


def test_cuhksz_layout_parser_extracts_only_complete_course_rows():
    record = parse_transcript_pages([PAGE_ONE, PAGE_TWO, PAGE_THREE, ""])

    assert len(record.courses) == 37
    assert sum(course.status == "passed" for course in record.courses) == 32
    assert sum(course.status == "in_progress" for course in record.courses) == 5
    assert sum(course.credits or 0 for course in record.courses if course.status == "passed") == 88
    assert sum(course.credits or 0 for course in record.courses if course.status == "in_progress") == 15
    assert record.total_credits_reported == 88
    assert record.programme == "Computer Science and Engineering"
    assert record.admission_year == 2023
    assert record.warnings == []


def test_metadata_dates_and_headings_are_not_courses():
    record = parse_transcript_pages([PAGE_ONE, PAGE_TWO, PAGE_THREE])
    codes = {course.course_code for course in record.courses}

    assert not {"EP2026", "SEP2023", "LIST2023", "LIST2024", "LIST2025", "YEAR2025"} & codes


def test_course_fields_statuses_terms_and_wrapped_title_are_preserved():
    record = parse_transcript_pages([PAGE_ONE, PAGE_TWO])
    courses = {course.course_code: course for course in record.courses}

    assert courses["CHI1000"].course_name == "Chinese"
    assert courses["CHI1000"].credits == 3
    assert courses["CHI1000"].grade == "B+"
    assert courses["CHI1000"].status == "passed"
    assert courses["CHI1000"].term == "2023-24 Term 1"
    assert courses["GEW1001"].status == "passed"
    assert courses["GEW2001"].status == "passed"
    assert courses["CSC3160"].status == "in_progress"
    assert courses["GEC2118"].course_name == (
        "Healing Empires: Medicine, Power, and Colonialism (1600–1914)"
    )


def test_incomplete_course_row_is_reported_but_not_added_or_merged():
    text = """
2026-27 Term 1
Course Code Course Title Units Grade % of A- and above
CSC1001 Complete Course 3.0 A 50.0
CSC9999 Missing Grade And Other Columns
Units Passed = 3.0
"""

    record = parse_transcript_pages([text])

    assert [course.course_code for course in record.courses] == ["CSC1001"]
    assert record.courses[0].course_name == "Complete Course"
    assert record.warnings == ["第 1 页课程 CSC9999 字段不完整，已跳过"]
