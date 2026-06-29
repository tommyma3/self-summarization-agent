from __future__ import annotations

import re
from dataclasses import dataclass

from self_summarization_agent.dataset import QueryExample
from self_summarization_agent.generation import TextGenerator


GRADER_TEMPLATE = """
Judge whether the answer is correct.

Question: {question}

Answer: {response}

Correct answer: {correct_answer}

Output only one line:
correct: yes
or
correct: no
""".strip()


@dataclass(slots=True)
class JudgeDecision:
    outcome: str
    judge_prompt: str | None
    judge_response: str | None
    parse_error: bool


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )


def parse_judge_response(judge_response: str) -> dict[str, object]:
    parsed = {"correct": None, "parse_error": False}
    if not judge_response:
        parsed["parse_error"] = True
        return parsed
    match = re.search(r"correct:\s*(yes|no)", judge_response, re.IGNORECASE)
    if not match:
        match = re.search(r"\*\*correct:\*\*\s*(yes|no)", judge_response, re.IGNORECASE)
    if not match:
        parsed["parse_error"] = True
        return parsed
    parsed["correct"] = match.group(1).lower() == "yes"
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
