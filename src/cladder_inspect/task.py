"""CLadder: Assessing Causal Reasoning in Language Models.

Jin, Chen, Leeb, Gresele, Kamal, Lyu, Blin, Gonzalez Adauto, Kleiman-Weiner,
Sachan, Schölkopf. NeurIPS 2023. https://arxiv.org/abs/2312.04350

About 10k yes/no questions generated from formal causal graphs, spread over
the three rungs of Pearl's ladder (association, intervention, counterfactual).

Prompts and answer parsing follow the authors' evaluation script,
`causalbenchmark/eval/run_gpt_v1.5.py` in https://github.com/causalNLP/cladder
(commit 3d2d1169b4b939a09048a6a75956c8972a93cc38). Deviations are listed in
the README.
"""

import re
from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    grouped,
    scorer,
    stderr,
)
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    generate,
    solver,
    system_message,
)

DATASET_PATH = "causalnlp/CLadder"
DATASET_REVISION = "18d684e444ed07cc57ae0f3a91ee983a79b8119a"

# Each file is a full ~10k-question test set. `full_v1.5_default` is the
# balanced v1.5 set used for the paper's main results; the others are the
# paper's alternative samplings.
DATA_FILES = {
    "default": "data/full_v1.5_default.csv",
    "easy": "data/test-easy-v1.5.csv",
    "hard": "data/test-hard-v1.5.csv",
    "commonsense": "data/test-commonsense-v1.5.csv",
    "anticommonsense": "data/test-anticommonsense-v1.5.csv",
    "noncommonsense": "data/test-noncommonsense-v1.5.csv",
}
Subset = Literal[
    "default", "easy", "hard", "commonsense", "anticommonsense", "noncommonsense"
]

SYSTEM_PROMPT = (
    "You are an expert in causal inference. The following question is not a "
    "typical commonsense query, but rather a meticulously designed question "
    "created by a professor specializing in causal inference, intended to "
    "assess the students' mastery of the course content."
)

DIRECT_SUFFIX = (
    'Start your answer with "Yes" or "No", followed by additional reasoning '
    "or evidence to support your explanation."
)

# Display names of the ten query types, from `FewShotExamples.long_query_types`
# upstream (spelling corrected for "Explaining").
QUERY_TYPE_NAMES = {
    "correlation": "correlation",
    "marginal": "marginal distribution",
    "exp_away": "explaining away effect",
    "ate": "average treatment effect",
    "backadj": "backdoor adjustment set",
    "collider_bias": "collider bias",
    "ett": "effect of the treatment on the treated",
    "nde": "natural direct effect",
    "nie": "natural indirect effect",
    "det-counterfactual": "counterfactual",
}

_QUERY_TYPE_OPTIONS = ", ".join(f'"{name}"' for name in QUERY_TYPE_NAMES.values())

COT_STEPS = [
    (
        "Extract the causal graph",
        "Identify the causal graph that depicts the relationships in the "
        "scenario. {var_notions}The diagram should simply consist of edges "
        'denoted in "var1 -> var2" format, separated by commas. Answer concisely.',
    ),
    (
        "Determine the query type",
        "Identify the type of query implied by the main question. Choices "
        f"include {_QUERY_TYPE_OPTIONS}. Your answer should only be a term "
        "from the list above, enclosed in quotation marks. Answer concisely.",
    ),
    (
        "Formalize the query",
        "Translate the query into its formal mathematical expression based on "
        'its type, utilizing the "do(·)" notation or counterfactual notations '
        "as needed. Answer concisely.",
    ),
    (
        "Gather all relevant data",
        "Extract all the available data. Your answer should contain nothing "
        'but marginal probabilities and conditional probabilities in the form '
        '"P(...)=..." or "P(...|...)=...", each probability being separated '
        "by a semicolon. Stick to the previously mentioned denotations for "
        "the variables. Answer concisely.",
    ),
    (
        "Deduce the estimand using causal inference",
        "Given all the information above, deduce the estimand using skills "
        "such as do-calculus, counterfactual prediction, and the basics of "
        "probabilities. Answer step by step.",
    ),
    (
        "Calculate the estimate",
        "Insert the relevant data into the estimand, perform basic arithmetic "
        "calculations, and derive the final answer. Answer step by step.",
    ),
]

COT_FINAL = (
    "Based on all the reasoning above, output one word to answer the initial "
    'question with just "Yes" or "No".'
)

# Upstream treats answers starting with these as refusals (scored incorrect).
REFUSAL_PREFIXES = ("as a", "i'm sorry ", "neither ", "none ")

_VARIABLES_LINE = re.compile(r"^Let (.+)\.$")


@task
def cladder(
    prompting: Literal["direct", "causal_cot"] = "direct",
    subset: Subset = "default",
) -> Task:
    """CLadder causal reasoning benchmark.

    Args:
        prompting: `direct` asks for a yes/no answer straight away (the
            paper's baseline). `causal_cot` runs the paper's CausalCoT prompt:
            six guided reasoning steps, then a follow-up turn asking for the
            one-word answer.
        subset: Which v1.5 test file to use. `default` is the balanced set
            behind the paper's main results.
    """
    return Task(
        dataset=cladder_dataset(subset),
        solver=[
            system_message(SYSTEM_PROMPT),
            causal_cot() if prompting == "causal_cot" else direct(),
        ],
        scorer=yes_no_prefix(),
        metrics=[
            accuracy(),
            stderr(),
            grouped(accuracy(), "rung", all=False, name_template="rung_{group_name}"),
            grouped(accuracy(), "query_type", all=False),
            grouped(accuracy(), "question_property", all=False),
        ],
        version="1.0.0",
    )


def cladder_dataset(subset: Subset = "default") -> Dataset:
    if subset not in DATA_FILES:
        raise ValueError(f"Unknown subset {subset!r}. Choose from {list(DATA_FILES)}.")
    return hf_dataset(
        path=DATASET_PATH,
        data_files=DATA_FILES[subset],
        split="train",
        revision=DATASET_REVISION,
        sample_fields=record_to_sample,
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=int(record["id"]),
        input=record["prompt"],
        target=record["label"],
        metadata={
            "rung": int(record["rung"]),
            "query_type": record["query_type"],
            "graph_id": record["graph_id"],
            "story_id": record["story_id"],
            # Only the default file carries this column.
            "question_property": record.get("question_property") or "unknown",
            "formal_form": record["formal_form"],
            "var_notions": variable_notions(record.get("reasoning")),
        },
    )


def variable_notions(reasoning: str | None) -> str:
    """Build the 'Use "X" to represent "husband".' hints for CausalCoT step 1.

    Upstream builds these from a variable mapping that the Hugging Face files
    do not include. The same mapping appears as the first line of the
    reference reasoning ("Let X = husband; V2 = wife; Y = alarm clock."), so
    it is recovered from there. Backdoor-adjustment questions have no
    reference reasoning and get no hints.
    """
    if not isinstance(reasoning, str):
        return ""
    match = _VARIABLES_LINE.match(reasoning.split("\n", 1)[0].strip())
    if not match:
        return ""
    notions = []
    for pair in match.group(1).split(";"):
        symbol, _, text = pair.partition("=")
        if symbol.strip() and text.strip():
            notions.append(f'Use "{symbol.strip()}" to represent "{text.strip()}".')
    return " ".join(notions)


@solver
def direct() -> Solver:
    """Paper baseline: the question followed by the yes/no instruction."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = f"{state.user_prompt.text}\n\n{DIRECT_SUFFIX}"
        return await generate(state)

    return solve


def causal_cot_prompt(question: str, var_notions: str) -> str:
    hints = f"{var_notions} " if var_notions else ""
    steps = [
        f"Step {i}) {title}: {instruction.format(var_notions=hints)}"
        for i, (title, instruction) in enumerate(COT_STEPS, start=1)
    ]
    guidance = "\n\n".join(
        ["Guidance: Address the question by following the steps below:", *steps]
    )
    return f"{question}\n\n{guidance}"


@solver
def causal_cot() -> Solver:
    """CausalCoT: guided six-step reasoning, then a one-word answer turn."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = causal_cot_prompt(
            state.user_prompt.text, state.metadata.get("var_notions", "")
        )
        state = await generate(state)
        state.messages.append(ChatMessageUser(content=COT_FINAL))
        return await generate(state)

    return solve


def parse_answer(completion: str) -> str:
    """Return 'yes', 'no', or 'invalid' from the start of a completion.

    Upstream lowercases, strips whitespace and double quotes, then matches by
    prefix. This also strips leading markdown emphasis and single quotes so
    that "**Yes**" is read as yes.
    """
    text = completion.lower().strip().lstrip("\"'*_` \n\t")
    if text.startswith(REFUSAL_PREFIXES):
        return "invalid"
    for answer in ("yes", "no"):
        if text.startswith(answer):
            return answer
    return "invalid"


@scorer(metrics=[accuracy(), stderr()])
def yes_no_prefix() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = parse_answer(state.output.completion)
        return Score(
            value=CORRECT if answer == target.text.lower() else INCORRECT,
            answer=answer,
            explanation=state.output.completion,
        )

    return score
