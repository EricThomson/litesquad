"""Prompt builders for each stage. Plain strings, no output parsing."""

PM_SYSTEM = (
    "You are the PM of a small squad solving planning and design problems. "
    "You are concrete, practical, and allergic to overengineering."
)
WORKER_SYSTEM = (
    "You are a worker on a small squad. You propose one concrete, actionable plan "
    "or solution. Be specific and realistic; prefer the simplest thing that works."
)
CRITIC_SYSTEM = (
    "You are a Critic. You stress-test proposals, focusing on weak assumptions, "
    "overengineering, missing steps, and practical risks. Be direct and useful, not pedantic."
)


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


def critique_prompt(task: str, framing: str, proposals: list[str]) -> str:
    blocks = "\n\n".join(f"Proposal {i + 1}:\n{p}" for i, p in enumerate(proposals))
    return (
        f"The user's task:\n{task}\n\n"
        f"The PM's framing:\n{framing}\n\n"
        f"{blocks}\n\n"
        "Review the proposals. Where do they agree and conflict? Call out weak assumptions, "
        "overengineering, missing steps, and practical risks. End with the few changes that "
        "would most improve the final answer."
    )


def revise_prompt(task: str, framing: str, own_proposal: str, critique: str) -> str:
    return (
        f"The user's task:\n{task}\n\n"
        f"The PM's framing:\n{framing}\n\n"
        f"Your earlier proposal:\n{own_proposal}\n\n"
        f"The Critic's review:\n{critique}\n\n"
        "Revise your proposal once in light of the critique. Keep what holds up, fix what doesn't, "
        "and stay concrete."
    )


def synthesize_prompt(task: str, framing: str, proposals: list[str], critique: str) -> str:
    blocks = "\n\n".join(f"Proposal {i + 1}:\n{p}" for i, p in enumerate(proposals))
    return (
        f"The user's task:\n{task}\n\n"
        f"The PM's framing:\n{framing}\n\n"
        f"{blocks}\n\n"
        f"The Critic's review:\n{critique}\n\n"
        "Synthesize the single best answer for the user. Resolve the disagreements, take the "
        "strongest ideas from each proposal, heed the critique, and deliver a clear, actionable "
        "result. This is what the user reads, so make it self-contained."
    )
