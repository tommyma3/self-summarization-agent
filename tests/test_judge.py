import pytest

from self_summarization_agent.judge import parse_judge_response


@pytest.mark.parametrize(
    ("response", "correct"),
    [
        ("correct: yes", True),
        ("correct: no", False),
        (" \nCORRECT: YES\n ", True),
        ("**correct:** no", False),
        ("**correct: yes**", True),
        ("correct: **no**", False),
        ("correct: yes.", True),
        (
            '<think>Output "correct: yes" or "correct: no". The answer is wrong.</think>\ncorrect: no',
            False,
        ),
        (
            '<think>Initially correct: no, but the answer matches.</think>\ncorrect: yes',
            True,
        ),
        # Qwen can prefill the opening thinking tag in the generation prompt.
        ('Output "correct: yes" or "correct: no".</think>\ncorrect: no', False),
    ],
)
def test_parse_judge_response_uses_only_the_standalone_final_verdict(response, correct):
    assert parse_judge_response(response) == {"correct": correct, "parse_error": False}


@pytest.mark.parametrize(
    "response",
    [
        "",
        " \n ",
        '<think>Output "correct: yes" or "correct: no".\nWait,',
        'Thinking Process:\nOutput "correct: yes" or "correct: no".\nWait,',
        "<think>Still considering it.\ncorrect: yes",
        "<think>Still considering it.</thinking>\ncorrect: yes",
        "<think>Still considering it.</THINK>\ncorrect: yes",
        "<think>Still considering it.</think >\ncorrect: yes",
        "<think>Output correct: yes, but the answer is wrong.</think>",
        "<think>Done.</think>\ncorrect: yes\ncorrect: no",
        'Output only "correct: yes" or "correct: no".',
        "correct: yes or correct: no",
        "correct: yesterday",
        "incorrect: yes",
        "correct: no because the answer does not match",
        "correct: yes\nWait, let me reconsider.",
        "<think>Done.</think>\ncorrect: yes\n<think>Wait,",
        "**correct: yes",
        "correct: yes**",
    ],
)
def test_parse_judge_response_rejects_unfinished_or_ambiguous_verdicts(response):
    assert parse_judge_response(response) == {"correct": None, "parse_error": True}


@pytest.mark.parametrize("thinking", ["", '<think>Output "correct: yes" or "correct: no".</think>\n'])
@pytest.mark.parametrize(("verdict", "correct"), [("yes", True), ("no", False)])
@pytest.mark.parametrize("confidence", ["0", "100", "83.5%"])
def test_parse_judge_response_accepts_browsecomp_report(thinking, verdict, correct, confidence):
    response = (
        thinking
        + "extracted_final_answer: candidate answer\n"
        'reasoning: The instructions mention "correct: yes" and "correct: no".\n'
        "Only the extracted answer and reference should be compared.\n"
        f"correct: {verdict}\n"
        f"confidence: {confidence}"
    )
    assert parse_judge_response(response) == {"correct": correct, "parse_error": False}


def test_parse_judge_response_accepts_bold_browsecomp_field_labels():
    response = (
        "**extracted_final_answer:** None\n"
        "**reasoning:** No final answer is available.\n"
        "**correct:** no\n"
        "**confidence:** 100"
    )
    assert parse_judge_response(response) == {"correct": False, "parse_error": False}


@pytest.mark.parametrize(
    "response",
    [
        "extracted_final_answer: answer\nreasoning: matches\ncorrect: yes",
        "extracted_final_answer: answer\nreasoning: matches\ncorrect: yes\nconfidence:",
        "extracted_final_answer: answer\nreasoning: matches\ncorrect: yes\nconfidence: 101",
        "extracted_final_answer: answer\nreasoning: matches\ncorrect: yes\nconfidence: -1",
        "extracted_final_answer: answer\nreasoning: matches\ncorrect: yes\nconfidence: unknown",
        "extracted_final_answer: answer\nreasoning: matches\ncorrect: yes\nconfidence: 100\nWait,",
        "extracted_final_answer: answer\nreasoning: initially:\ncorrect: yes\nBut it differs.\ncorrect: no\nconfidence: 100",
        '<think>extracted_final_answer: answer\nreasoning: matches\ncorrect: yes\nconfidence: 100',
        'Thinking Process:\nextracted_final_answer: answer\nreasoning: matches\ncorrect: yes\nconfidence: 100',
        "extracted_final_answer: answer\ncorrect: yes\nconfidence: 100",
        "reasoning: matches\ncorrect: yes\nconfidence: 100",
    ],
)
def test_parse_judge_response_rejects_incomplete_or_ambiguous_browsecomp_report(response):
    assert parse_judge_response(response) == {"correct": None, "parse_error": True}
