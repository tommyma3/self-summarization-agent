from __future__ import annotations

import re
from dataclasses import dataclass

from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.generation import TextGenerator
from self_summarization_agent.models import Message
from self_summarization_agent.prompts import ConversationPrompt


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise
and unambiguous [correct_answer] below.
[question]: {question}
[response]: {response}
Your judgement must be in the format and criteria specified below:
extracted_final_answer: The final exact answer extracted from the [response]. Put the
extracted answer as 'None' if there is no exact, final answer to extract from the response.
[correct_answer]: {correct_answer}
focus only on whether the answers match.
reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer],
focusing only on if there are meaningful differences between [correct_answer]
and the extracted_final_answer. Do not comment on any background to the problem, do not
attempt to solve the problem, do not argue for any answer different than [correct_answer],
correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above,
or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if
there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is
incorrect.
confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put
100 if there is no confidence score available.
""".strip()


_FINAL_VERDICT_RE = re.compile(
    r"(?:correct:\s*(yes|no)"
    r"|\*\*correct:\*\*\s*(yes|no)"
    r"|\*\*correct:\s*(yes|no)\*\*"
    r"|correct:\s*\*\*(yes|no)\*\*)[.!]?",
    re.IGNORECASE,
)
_JUDGE_REPORT_RE = re.compile(
    r"(?:extracted_final_answer:|\*\*extracted_final_answer:\*\*)[ \t]*(?P<answer>.+?)\n\s*"
    r"(?:reasoning:|\*\*reasoning:\*\*)[ \t]*(?P<reasoning>.+?)\n\s*"
    r"(?P<verdict>[^\n]+)\n\s*"
    r"(?:confidence:|\*\*confidence:\*\*)[ \t]*(?P<confidence>\d+(?:\.\d+)?%?)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True)
class JudgeDecision:
    outcome: str
    judge_prompt: str | None
    judge_response: str | None
    parse_error: bool


def create_judge_prompt(question: str, response: str, correct_answer: str) -> ConversationPrompt:
    instructions = GRADER_TEMPLATE.format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )
    return ConversationPrompt(
        [
            Message(role="system", content=instructions),
            # The project-owned Qwen template requires a user turn.
            Message(role="user", content="Please provide your judgement."),
        ]
    )


def parse_judge_response(judge_response: str) -> dict[str, object]:
    """Read the final verdict or BrowseComp report outside completed thinking.

    Qwen may prefill the opening thinking tag, so a closing tag can appear
    without an opening one in the sampled response. Without a closing tag,
    only a response consisting entirely of a verdict or complete report is accepted.
    The complete raw response remains available in the JudgeDecision.
    """
    parsed = {"correct": None, "parse_error": False}
    if not judge_response:
        parsed["parse_error"] = True
        return parsed
    verdict = judge_response.rsplit("</think>", 1)[-1].strip()
    match = _FINAL_VERDICT_RE.fullmatch(verdict)
    if not match:
        report = _JUDGE_REPORT_RE.fullmatch(verdict)
        if (
            report is None
            or not report.group("answer").strip()
            or not report.group("reasoning").strip()
            or not 0 <= float(report.group("confidence").removesuffix("%")) <= 100
            or sum(_FINAL_VERDICT_RE.fullmatch(line.strip()) is not None for line in verdict.splitlines()) != 1
        ):
            parsed["parse_error"] = True
            return parsed
        match = _FINAL_VERDICT_RE.fullmatch(report.group("verdict").strip())
    if not match:
        parsed["parse_error"] = True
        return parsed
    parsed["correct"] = next(group for group in match.groups() if group is not None).lower() == "yes"
    return parsed


@dataclass(slots=True)
class RewardJudge:
    generator: TextGenerator

    def _decision_from_judge_response(self, judge_prompt: str, judge_response: str) -> JudgeDecision:
        parsed = parse_judge_response(judge_response)
        if parsed["parse_error"]:
            return JudgeDecision(
                outcome="wrong_answer",
                judge_prompt=judge_prompt,
                judge_response=judge_response,
                parse_error=True,
            )
        outcome = "correct_answer" if parsed["correct"] else "wrong_answer"
        return JudgeDecision(
            outcome=outcome,
            judge_prompt=judge_prompt,
            judge_response=judge_response,
            parse_error=False,
        )

    def evaluate(self, example: QueryExample, status: str, response: str) -> JudgeDecision:
        if status != "completed":
            return JudgeDecision(
                outcome="budget_exhausted",
                judge_prompt=None,
                judge_response=None,
                parse_error=False,
            )
        if not response.strip():
            return JudgeDecision(
                outcome="wrong_answer",
                judge_prompt=None,
                judge_response=None,
                parse_error=False,
            )
        if example.answer is None:
            raise ValueError(f"Query {example.query_id} is missing an answer for judging")
        judge_prompt = create_judge_prompt(example.query, response, example.answer)
        judge_response = self.generator.generate(judge_prompt)
        return self._decision_from_judge_response(judge_prompt, judge_response)

    def evaluate_batch(
        self,
        items: list[tuple[QueryExample, str, str]],
    ) -> list[JudgeDecision]:
        decisions: list[JudgeDecision | None] = [None] * len(items)
        prompt_items: list[tuple[int, str]] = []
        for index, (example, status, response) in enumerate(items):
            if status != "completed":
                decisions[index] = JudgeDecision(
                    outcome="budget_exhausted",
                    judge_prompt=None,
                    judge_response=None,
                    parse_error=False,
                )
                continue
            if not response.strip():
                decisions[index] = JudgeDecision(
                    outcome="wrong_answer",
                    judge_prompt=None,
                    judge_response=None,
                    parse_error=False,
                )
                continue
            if example.answer is None:
                raise ValueError(f"Query {example.query_id} is missing an answer for judging")
            prompt_items.append((index, create_judge_prompt(example.query, response, example.answer)))

        if prompt_items:
            generate_batch = getattr(self.generator, "generate_batch", None)
            prompts = [prompt for _, prompt in prompt_items]
            responses = generate_batch(prompts) if generate_batch is not None else [self.generator.generate(prompt) for prompt in prompts]
            if len(responses) != len(prompt_items):
                raise ValueError(f"Batch judge returned {len(responses)} outputs for {len(prompt_items)} prompts")
            for (index, prompt), response in zip(prompt_items, responses):
                decisions[index] = self._decision_from_judge_response(prompt, response)

        return [decision for decision in decisions if decision is not None]
