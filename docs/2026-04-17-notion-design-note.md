# Self-Summarization Agent Design for BrowseComp-Plus

## Overview

This project is a research prototype for an AI agent with self-summarization, built on top of a Qwen3.5-9B base model and evaluated on the BrowseComp-Plus benchmark. The main goal is to improve long-horizon reasoning by teaching the model to compress its own past interaction history into useful summaries when the context becomes too large.

The design keeps retrieval fixed and focuses on the agent's ability to preserve important information across long trajectories. This makes it easier to study whether self-summarization alone improves downstream task performance.

## Overall Structure

The system is split into two layers.

The first layer is the BrowseComp-Plus benchmark backend. It provides the fixed environment, including the benchmark questions and the retrieval tools used by the agent. In this project, the agent uses the benchmark's search and document access tools directly rather than adding new tools or training a separate retriever.

The second layer is a custom RL runtime built specifically for this project. This runtime is responsible for running episodes, tracking context length, injecting summarization turns, assigning rewards, logging trajectories, and training the model. Keeping this runtime separate from the benchmark code makes the research setup cleaner and easier to control.

## Runtime Behavior

At the beginning of each episode, the model sees the system prompt, the user question, and any previous summary state. It then interacts with the benchmark using tool calls to search for relevant documents and inspect retrieved evidence.

The runtime monitors the packed context length itself. When the context becomes too large, the runtime does not interrupt the agent in the middle of a tool interaction. Instead, it waits until the current tool round is complete and then injects a synthetic summarization step. The model writes a new summary that captures the essential information from earlier context, and the runtime replaces the older history with that summary plus a small recent raw tail.

This means the agent can continue reasoning with a compact state while preserving the most useful evidence and next-step information.

## Training Design

The RL setup is intentionally narrow. The project does not train the model on tool-calling behavior. Instead, RL is applied only to:

- summary outputs
- final answer outputs

This means the training objective focuses on two questions:

- can the model learn to write better summaries that support future reasoning?
- can the model produce better final answers from compressed context?

Tool calls are still part of the rollout, but they are not included in the policy loss for the first version of the system.

## Reward Design

The reward structure is simple. If the final answer is correct, the run receives positive reward. If the final answer is incorrect, or if the rollout ends because the budget is exhausted, the run receives negative reward. Those task-level rewards are used to train the summary turns and the final answer turn.

Malformed tool calls are handled separately. If the model emits a bad tool call format, the rollout terminates immediately. That penalty is treated as a local failure for that step rather than something that should be propagated backward through earlier summaries.

## Core Components

The main components of the system are:

- a task sampler for selecting benchmark queries
- a benchmark backend adapter for search and document retrieval
- an agent runtime that manages episodes and state
- a context packer that decides what to keep and what to summarize
- a summary injector that creates synthetic summary turns
- a reward engine that applies the training rules
- a trajectory store for logging rollouts
- a GRPO trainer for multi-GPU full-parameter RL
- an evaluator that exports benchmark-compatible results

Together, these components form a clean pipeline for running, training, and evaluating the self-summarization agent.

## Why This Design

This design keeps the research question focused. It avoids changing too many variables at once by leaving retrieval fixed and by not trying to optimize every part of the agent policy. Instead, it isolates the effect of summary quality and compressed-context answering.

That makes the system easier to analyze and easier to compare against baselines such as:

- no summarization
- summarization without RL training
- summarization plus RL-trained summaries and final answers

## Expected Outcome

If the design works as intended, the trained model should learn to preserve the most important evidence, reasoning state, and unresolved questions in its summaries. That should help it continue solving long tasks more effectively under limited context, leading to better final performance on BrowseComp-Plus.
