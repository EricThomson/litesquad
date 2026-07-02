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

# The extractor de-stylizes one response into content units; the clusterer groups equivalent
# units across responses and flags conflicts. Both output JSON, so no ASCII_RULE (their text is
# parsed, not shown; the final answer, which is shown, goes through the judge and to_ascii).
EXTRACT_SYSTEM = (
    "You are a neutral analyst reducing one response to its distinct reusable content units so "
    "the judge can build a new answer from them. Strip all voice, formatting, and phrasing; keep "
    "only the content. Output ONLY JSON."
)

CLUSTER_SYSTEM = (
    "You map content: you group units by what they say. You are CONSERVATIVE -- you merge only "
    "units that are genuinely the same idea or the same option, and you keep distinct variants "
    "separate so variety is never collapsed. You do NOT judge quality, importance, or priority; "
    "that is the judge's job. You only group equivalent content and flag where merged units "
    "disagree. Output ONLY JSON."
)

JUDGE_SYSTEM = (
    "You are the judge who renders the ensemble's final answer for the user. You receive a "
    "de-stylized content map: clusters of the same idea gathered from several independent "
    "responses, each tagged with how many responses back it and any conflict between them (not "
    "their prose). The map gives you the content and the facts; every judgment is yours. Write "
    "the single best answer for the user FROM this map, in the form the question calls for. If it "
    "calls for one integrated answer (a plan, an argument, an analysis), weave the clusters into "
    "one coherent whole organized by theme, never response by response. If it calls for a set of "
    "distinct options (e.g. 'give me N ideas'), curate the best distinct options and never blend "
    "genuinely distinct options into one. You decide what matters most for the user's actual "
    "goal, what to emphasize, and what to cut: cross-response support is a signal, but a strong "
    "idea from a single response can be the most important one. Where it would genuinely help the "
    "user, distinguish what is essential to their goal from what is optional. Resolve conflicts "
    "by deciding, not by listing both sides. This is what the user reads, so make it "
    "self-contained and worth their time."
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


def extract_prompt(task: str, response: str) -> str:
    return (
        f"The user's question:\n{task}\n\n"
        f"One response:\n{response}\n\n"
        "Reduce this response to its distinct content UNITS, discarding all wording and style. "
        "A unit is one of:\n"
        "- claim: an atomic point, fact, recommendation, step, or judgment;\n"
        "- move: a structural or organizing choice that makes the response good;\n"
        "- option: one discrete proposal, when the response offers a set of alternatives "
        "(e.g. distinct designs, names, or ideas).\n"
        "Keep each unit self-contained and plain. Do NOT merge distinct options together. "
        "Extract at most about 20 substantive units: combine trivial or repetitive sub-points "
        "into the meaningful ones rather than atomizing every sentence.\n\n"
        'Return ONLY JSON: {"units": [{"kind": "claim|move|option", "text": "..."}]}'
    )


def cluster_prompt(task: str, lines: list[str]) -> str:
    body = "\n".join(lines)
    return (
        f"The user's question:\n{task}\n\n"
        "A pooled list of content units from several independent responses (sources hidden, "
        f"order shuffled):\n{body}\n\n"
        "Group units that express GENUINELY THE SAME idea or option into one cluster. Be "
        "CONSERVATIVE: merge only true equivalents; keep distinct variants separate; when in "
        "doubt, keep them separate. Do not judge quality or importance. For each cluster give: "
        "label (a short neutral statement of the shared idea/option), member_ids, and conflict "
        "(note it if members disagree, else null).\n\n"
        'Return ONLY JSON: {"clusters": [{"label": "...", "member_ids": ["u3","u7"], '
        '"conflict": null}]}'
    )


def judge_prompt(task: str, content_map: str, history: str = "") -> str:
    return (
        _message(task, history) + "\n\n"
        "De-stylized content map (clusters, with cross-response support and any conflicts as "
        f"metadata):\n{content_map}\n\n"
        "Write the single best answer to the user's message, in the form it calls for."
    )


def quick_prompt(task: str, history: str = "") -> str:
    return _message(task, history) + "\n\nRespond directly and well."
