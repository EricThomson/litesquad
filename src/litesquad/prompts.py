"""Prompt builders for each stage. Plain strings, no output parsing."""

# Appended to every system prompt so the models write ASCII by choice. The hard
# guarantee is llm.to_ascii; this just keeps the prose natural (the model uses a
# hyphen or comma on purpose, rather than us mechanically swapping an em-dash).
ASCII_RULE = (
    " Write in plain ASCII only: straight quotes, a hyphen instead of em/en dashes, "
    "and no curly quotes, ellipsis characters, or other non-ASCII symbols."
)

PM_SYSTEM = (
    "You are the PM of a small squad solving planning and design problems. "
    "You are concrete, practical, and allergic to overengineering."
) + ASCII_RULE
WORKER_SYSTEM = (
    "You are a worker on a small squad. You propose one concrete, actionable plan "
    "or solution. Be specific and realistic; prefer the simplest thing that works."
) + ASCII_RULE
CRITIC_SYSTEM = (
    "You are a Critic. You stress-test proposals, focusing on weak assumptions, "
    "overengineering, missing steps, and practical risks. Be direct and useful, not pedantic. "
    "If a proposal is genuinely strong, say so plainly instead of inventing problems."
) + ASCII_RULE


def frame_prompt(task: str, history: str = "") -> str:
    context = f"\n\nPrior context from this session:\n{history}\n" if history else ""
    return (
        f"The user's task:\n{task}\n{context}\n"
        "Frame this task for the squad. Restate the real goal in your own words, "
        "list the key sub-questions, constraints, and the criteria a good answer must meet. "
        "Keep it tight."
    )


def propose_prompt(task: str, framing: str) -> str:
    return (
        f"The user's task:\n{task}\n\n"
        f"The PM's framing:\n{framing}\n\n"
        "Propose one concrete plan or solution that meets the framing. "
        "Be specific about the steps and the reasoning behind them."
    )


def critique_prompt(task: str, framing: str, proposal: str) -> str:
    return (
        f"The user's task:\n{task}\n\n"
        f"The PM's framing:\n{framing}\n\n"
        f"The proposal to review:\n{proposal}\n\n"
        "Critique this proposal on its own terms. Call out weak assumptions, overengineering, "
        "missing steps, and practical risks. If it is already strong, say so plainly rather than "
        "manufacturing problems. Otherwise, end with the few concrete changes that would most "
        "improve it, specific enough that the author can act on them directly."
    )


def revise_prompt(task: str, framing: str, own_proposal: str, critique: str) -> str:
    return (
        f"The user's task:\n{task}\n\n"
        f"The PM's framing:\n{framing}\n\n"
        f"Your earlier proposal:\n{own_proposal}\n\n"
        f"The Critic's review:\n{critique}\n\n"
        "Revise your proposal once in light of the critique. Keep what holds up, fix what doesn't, "
        "and stay concrete. If the critique found little or nothing to change, light edits or "
        "leaving it as is are both fine."
    )


def synthesize_prompt(task: str, framing: str, proposals: list[str]) -> str:
    blocks = "\n\n".join(f"Proposal {i + 1} (revised):\n{p}" for i, p in enumerate(proposals))
    return (
        f"The user's task:\n{task}\n\n"
        f"The PM's framing:\n{framing}\n\n"
        f"{blocks}\n\n"
        "Each proposal above was independently critiqued and then revised by its author. "
        "Synthesize the single best answer for the user: resolve any disagreements between them, "
        "take the strongest ideas from each, and deliver a clear, actionable result. This is what "
        "the user reads, so make it self-contained."
    )
