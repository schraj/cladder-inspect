import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from cladder_inspect.task import (
    COT_FINAL,
    DIRECT_SUFFIX,
    causal_cot_prompt,
    cladder,
    cladder_dataset,
    parse_answer,
    record_to_sample,
    variable_notions,
)

RECORD = {
    "id": 4,
    "prompt": "Husband has a direct effect on wife and alarm clock. Would husband positively affect alarm clock?",
    "label": "yes",
    "reasoning": "Let X = husband; V2 = wife; Y = alarm clock.\nX->V2,X->Y,V2->Y",
    "graph_id": "mediation",
    "story_id": "alarm",
    "rung": 3,
    "query_type": "nde",
    "question_property": "easy",
    "formal_form": "E[Y_{X=1, V2=0} - Y_{X=0, V2=0}]",
}


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("Yes, because the effect is positive.", "yes"),
        ("No.", "no"),
        ("  yes", "yes"),
        ('"No", the estimate is negative.', "no"),
        ("**Yes** — the direct effect is 0.32.", "yes"),
        ("NO", "no"),
        ("The answer is yes.", "invalid"),
        ("As a language model, I cannot say.", "invalid"),
        ("None of the above.", "invalid"),
        ("Neither holds here.", "invalid"),
        ("", "invalid"),
    ],
)
def test_parse_answer(completion: str, expected: str) -> None:
    assert parse_answer(completion) == expected


def test_variable_notions() -> None:
    assert variable_notions(RECORD["reasoning"]) == (
        'Use "X" to represent "husband". Use "V2" to represent "wife". '
        'Use "Y" to represent "alarm clock".'
    )
    # Backdoor-adjustment rows have no reference reasoning (NaN in the CSV).
    assert variable_notions(None) == ""
    assert variable_notions(float("nan")) == ""  # type: ignore[arg-type]
    assert variable_notions("X->Y") == ""


def test_record_to_sample() -> None:
    sample = record_to_sample(RECORD)
    assert sample.id == 4
    assert sample.input == RECORD["prompt"]
    assert sample.target == "yes"
    assert sample.metadata is not None
    assert sample.metadata["rung"] == 3
    assert sample.metadata["query_type"] == "nde"
    assert sample.metadata["question_property"] == "easy"


def test_record_without_question_property() -> None:
    record = {k: v for k, v in RECORD.items() if k != "question_property"}
    sample = record_to_sample(record)
    assert sample.metadata is not None
    assert sample.metadata["question_property"] == "unknown"


def test_causal_cot_prompt() -> None:
    prompt = causal_cot_prompt("Q?", 'Use "X" to represent "husband".')
    assert prompt.startswith(
        "Q?\n\nGuidance: Address the question by following the steps below:\n\n"
        "Step 1) Extract the causal graph: Identify the causal graph that depicts "
        'the relationships in the scenario. Use "X" to represent "husband". '
        "The diagram should"
    )
    assert "Step 6) Calculate the estimate:" in prompt
    assert "Step 7)" not in prompt
    assert "{var_notions}" not in causal_cot_prompt("Q?", "")
    assert "scenario. The diagram" in causal_cot_prompt("Q?", "")


def _run(prompting: str, outputs: list[str], limit: int):
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content("mockllm/model", text) for text in outputs
        ],
    )
    [log] = eval(cladder(prompting=prompting), model=model, limit=limit, display="none")
    assert log.status == "success", log.error
    assert log.samples is not None
    return log


@pytest.mark.dataset_download
def test_dataset_shape() -> None:
    dataset = cladder_dataset()
    assert len(dataset) == 10112
    assert {s.target for s in dataset} == {"yes", "no"}
    assert {s.metadata["rung"] for s in dataset if s.metadata} == {1, 2, 3}
    assert len({s.id for s in dataset}) == len(dataset)


@pytest.mark.dataset_download
def test_direct_end_to_end() -> None:
    limit = 4
    targets = [s.target for s in cladder_dataset()[:limit]]
    # Answer the first two correctly and the last two wrongly.
    wrong = {"yes": "No", "no": "Yes"}
    outputs = [
        t.capitalize() if i < 2 else wrong[t] for i, t in enumerate(targets)
    ]
    log = _run("direct", [f"{o}, because of the data." for o in outputs], limit)

    assert [s.scores["yes_no_prefix"].value for s in log.samples] == ["C", "C", "I", "I"]
    first = log.samples[0]
    assert first.messages[0].role == "system"
    assert first.messages[1].text.endswith(DIRECT_SUFFIX)
    metrics = log.results.scores[0].metrics
    assert metrics["accuracy"].value == 0.5


@pytest.mark.dataset_download
def test_causal_cot_end_to_end() -> None:
    target = cladder_dataset()[0].target
    log = _run("causal_cot", ["Step 1) X->Y ... the estimate is 0.3.", target.capitalize()], 1)

    sample = log.samples[0]
    roles = [m.role for m in sample.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert "Guidance: Address the question" in sample.messages[1].text
    assert sample.messages[3].text == COT_FINAL
    # Scored on the final one-word turn, not on the reasoning turn.
    assert sample.scores["yes_no_prefix"].value == "C"
