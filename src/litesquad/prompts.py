"""Prompt builders for each stage. Plain strings, no output parsing.

The tool is a general ensemble: whatever the user brings (a question, a problem,
an idea to explore, something to write or think through), several independent
minds respond, a critic gives each one feedback, they revise, and a judge
renders the single best answer. Nothing here should assume the task is an
engineering or design problem.
"""

# Appended to every system prompt so the models write ASCII by choice. The hard
# guarantee is llm.to_ascii; this just keeps the prose natural (the model uses a
# hyphen or comma on purpose, rather than us mechanically swapping an em-dash).
ASCII_RULE = (
    " Write in plain ASCII only: straight quotes, a hyphen instead of em/en dashes, "
    "and no curly quotes, ellipsis characters, or other non-ASCII symbols."
)

WORKER_SYSTEM = (
    "You are one worker in a small ensemble of independent minds answering a user's "
    "question. Whatever the user brings you -- a question, a problem, a plan to make, an "
    "idea to explore, something to write or think through -- give your own unique best and "
    "complete response to it, in whatever form fits the problem. Do not pad the answer, but also"
    "do not answer in bullet-point list dumps."
) + ASCII_RULE

# The critic sees one worker's response at a time (never the others), and its
# feedback goes straight back to that worker for a single revision. So the prompt
# gives it the one thing it can act on, make the feedback revisable, rather than
# ensemble backstory it cannot use.
CRITIC_SYSTEM = (
    "You give one response a single round of honest feedback, which its author will use "
    "to revise it once. Judge it on its own terms and by what would actually help the "
    "user, whatever the subject and whatever form the response takes. Say what is strong, "
    "and where it is weak, shallow, generic, wrong, or missing something that matters. If "
    "it is genuinely strong, say so plainly instead of inventing problems. When you do want "
    "changes, be specific enough that the author can act on them directly. Be direct and "
    "brief, never pedantic."
) + ASCII_RULE

JUDGE_SYSTEM = (
    "You are the judge who renders the ensemble's final answer for the user. Several "
    "independent workers have each responded, and each response has been critiqued and "
    "revised. Hear them all, weigh them, and deliver the single best answer, balancing "
    "everything into a reasoned whole. Where they conflict, decide what serves the user "
    "best rather than splitting the difference; where they read the user differently, use "
    "your judgment about what best answers the user's actual query, and when the disagreement itself is illuminating, let that into the "
    "answer. This is what the user reads, so make it self-contained and worth their time."
) + ASCII_RULE

QUICK_SYSTEM = (
    "You are answering the user directly, one on one. Give a substantive, honest, useful "
    "response in whatever form the message calls for. Think it through, but never pad."
) + ASCII_RULE


def _message(task: str, history: str) -> str:
    if history:
        return f"Earlier in this conversation:\n{history}\n\nThe user's message now:\n{task}"
    return f"The user's message:\n{task}"


def propose_prompt(task: str, history: str = "") -> str:
    return (
        _message(task, history) + "\n\n"
        "Give your own best response. Answer in whatever form the message calls for, and "
        "commit to your own angle."
    )


def critique_prompt(task: str, proposal: str) -> str:
    return (
        f"The user's message:\n{task}\n\n"
        f"One response from the ensemble:\n{proposal}\n\n"
        "Critique this response on its own terms. What is strong, and what is weak, shallow, "
        "wrong, unsupported, or missing? If it is already strong, say so plainly rather than "
        "manufacturing problems. Otherwise end with the few concrete changes that would most "
        "improve it, specific enough that the author can act on them directly."
    )


def revise_prompt(task: str, own_proposal: str, critique: str) -> str:
    return (
        f"The user's message:\n{task}\n\n"
        f"Your earlier response:\n{own_proposal}\n\n"
        f"The critic's feedback:\n{critique}\n\n"
        "Revise your response in light of the feedback. Keep what holds up, fix what does not. "
        "If the critique found little or nothing to change, light edits or leaving it as is are "
        "both fine."
    )


def synthesize_prompt(task: str, responses: list[str], history: str = "") -> str:
    blocks = "\n\n".join(f"Response {i + 1}:\n{r}" for i, r in enumerate(responses))
    return (
        _message(task, history) + "\n\n"
        f"{blocks}\n\n"
        "Each response above came from an independent member of the ensemble and has already "
        "been critiqued and revised. Render the single best answer for the user: weigh the "
        "responses, take the strongest of each, and reconcile where they conflict. Where they "
        "read the message differently, judge what serves the user best. Make it self-contained."
    )


def quick_prompt(task: str, history: str = "") -> str:
    return _message(task, history) + "\n\nRespond directly and well."
