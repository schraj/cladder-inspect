# CLadder for Inspect

An [Inspect](https://inspect.aisi.org.uk/) implementation of **CLadder**
(Jin et al., NeurIPS 2023, [arXiv:2312.04350](https://arxiv.org/abs/2312.04350)),
a benchmark of about 10,000 yes/no causal reasoning questions generated from
formal causal graphs. Questions cover the three rungs of Pearl's ladder of
causation: association, intervention and counterfactual.

- Original code: <https://github.com/causalNLP/cladder> (MIT)
- Dataset: <https://huggingface.co/datasets/causalnlp/CLadder>, pinned to
  revision `18d684e444ed07cc57ae0f3a91ee983a79b8119a`

## Usage

```bash
git clone https://github.com/schraj/cladder-inspect && cd cladder-inspect
uv sync

# Paper baseline: direct yes/no answer
uv run inspect eval src/cladder_inspect/task.py@cladder --model openai/gpt-4o-mini

# CausalCoT: the paper's guided six-step reasoning prompt
uv run inspect eval src/cladder_inspect/task.py@cladder -T prompting=causal_cot --model openai/gpt-4o-mini

# A quick look at 50 samples
uv run inspect eval src/cladder_inspect/task.py@cladder --limit 50 --model openai/gpt-4o-mini
```

### Task parameters

| Parameter   | Values                                                                       | Default   |
| ----------- | ---------------------------------------------------------------------------- | --------- |
| `prompting` | `direct`, `causal_cot`                                                       | `direct`  |
| `subset`    | `default`, `easy`, `hard`, `commonsense`, `anticommonsense`, `noncommonsense` | `default` |

`default` is the balanced v1.5 test set (10,112 questions) behind the paper's
main results. The other subsets are the paper's alternative samplings, each
about 10,000 questions.

## Scoring

The answer is read from the start of the model's completion: it must begin
with "yes" or "no" (case-insensitive). Anything else, including refusals,
scores as incorrect. With `causal_cot`, only the final one-word turn is scored.

Reported metrics:

- overall accuracy and standard error
- accuracy by rung (`rung_1`, `rung_2`, `rung_3`)
- accuracy by query type (`ate`, `nde`, `backadj`, ...)
- accuracy by commonsense alignment (`commonsense`, `anticommonsense`,
  `nonsense`, plus the `easy` and `hard` labels in the default file)

## Fidelity to the paper

Prompts, the system message and answer parsing follow the authors' script
`causalbenchmark/eval/run_gpt_v1.5.py` at commit
`3d2d1169b4b939a09048a6a75956c8972a93cc38`. Known differences:

1. **No output-token cap.** The original caps the direct answer at 1 token
   and the CausalCoT reasoning turn at 1,024. This implementation leaves the
   cap to the caller (`--max-tokens`), because a 1-token cap breaks reasoning
   models.
2. **Lenient answer prefix.** Leading markdown emphasis and single quotes are
   stripped before matching, so `**Yes**` counts as yes. The original strips
   only whitespace and double quotes.
3. **CausalCoT query-type list.** Step 2 lists the ten query types by the
   display names in the original code (`long_query_types`). The original
   reads its list from a config file that is not in the public repository,
   so the exact wording may differ.
4. **CausalCoT variable hints.** Step 1 tells the model which symbol stands
   for which variable. The original takes this from a variable mapping that
   the Hugging Face files omit; here it is recovered from the first line of
   each question's reference reasoning. Backdoor-adjustment questions
   (1,580 of 10,112) have no reference reasoning and get no hints.

## Results

Full default subset (10,112 questions), November 2026, task version 1.0.0.
Both runs use CausalCoT with model reasoning turned off, so all reasoning is
in the visible transcript:

```bash
uv run inspect eval src/cladder_inspect/task.py@cladder -T prompting=causal_cot \
  --reasoning-effort none --model <model>
```

| Model                        | Accuracy | Stderr | Rung 1 | Rung 2 | Rung 3 |
| ---------------------------- | -------- | ------ | ------ | ------ | ------ |
| `anthropic/claude-haiku-5-5` | 0.907    | 0.003  | 0.942  | 0.942  | 0.848  |
| `openai/gpt-6-luna`          | 0.755    | 0.004  | 0.947  | 0.799  | 0.557  |

Accuracy by query type:

| Query type           | Questions | `claude-haiku-5-5` | `gpt-6-luna` |
| -------------------- | --------- | ------------------ | ------------ |
| `marginal`           | 1,580     | 0.979              | 0.968        |
| `correlation`        | 1,422     | 0.896              | 0.921        |
| `exp_away`           | 158       | 0.987              | 0.981        |
| `collider_bias`      | 158       | 0.506              | 0.759        |
| `ate`                | 1,422     | 0.989              | 0.787        |
| `backadj`            | 1,580     | 0.944              | 0.815        |
| `det-counterfactual` | 1,422     | 0.845              | 0.527        |
| `ett`                | 1,264     | 0.844              | 0.491        |
| `nde`                | 316       | 0.899              | 0.744        |
| `nie`                | 790       | 0.839              | 0.642        |

No samples errored in either run. `gpt-6-luna` gave 115 final answers (1.1%)
that were neither yes nor no (for example "Cannot determine"); these score as
incorrect.

### Comparison with the paper

Table 2 of the paper (arXiv v3) reports these results on the same v1.5 set,
at temperature 0:

| Model (paper)                         | Accuracy | Rung 1 | Rung 2 | Rung 3 |
| ------------------------------------- | -------- | ------ | ------ | ------ |
| Random                                | 0.493    | 0.503  | 0.484  | 0.491  |
| GPT-3.5                               | 0.522    | 0.518  | 0.548  | 0.503  |
| GPT-4 (`gpt-4-1106-preview`), direct  | 0.620    | 0.630  | 0.628  | 0.606  |
| GPT-4 + CausalCoT                     | 0.704    | 0.834  | 0.675  | 0.621  |

The models in the paper are no longer the current ones, so this
implementation has not been checked by reproducing a paper number with the
same model. The paper does not report accuracy by query type.

### Effect of prompting mode and reasoning

On a fixed random sample of 200 questions (`--limit 200 --sample-shuffle 42`):

| Prompting    | Model reasoning | `claude-haiku-5-5` | `gpt-5.4-mini` | `gpt-6-luna` |
| ------------ | --------------- | ------------------ | -------------- | ------------ |
| `direct`     | default (on)    | 0.955              |                |              |
| `direct`     | off             | 0.540              | 0.560          |              |
| `causal_cot` | off             | 0.935              | 0.830          | 0.790        |

With reasoning off, `direct` is close to chance. The prompt requires the
answer to start with "Yes" or "No", so the model commits before working the
problem out, and some transcripts reverse the opening word later in the text.
The scorer reads only the opening word, as the original does.

### Question-text error in 69 `ett` questions

69 `ett` questions ask about the action that was already taken: 40 in the
`firing_employee` story ("For managers who sign termination letters, would it
be more likely to see employee being fired if the manager had signed the
termination letter?") and 29 in `vaccine_kills` ("... if the person had been
vaccinated?"). Every other `ett` question asks about the opposite action
("if the person had not visited England?").

The labels were checked against the sign of the effect in each question's
reference derivation. For 66 of the 69 the label is the answer to the "had
not" version of the question (the other 3 have an effect of exactly zero),
the same rule that holds for the rest of the `ett` questions. So the word
"not" is missing from the question text, and as written these questions are
labelled the opposite of the natural reading. `claude-haiku-5-5` scores 0.232
on these 69 and 0.879 on the other 1,195 `ett` questions; `gpt-6-luna` scores
0.304 and 0.502. This has not been reported to or confirmed by the dataset
authors.

### Ambiguous wording in `collider_bias` questions

All 158 `collider_bias` questions use the graph X -> V3 <- Y and come in two
forms: 79 ask "does it mean that X affects Y?" (label "no") and 79 ask "does
it mean that X does not affect Y?" (label "yes"). `claude-haiku-5-5` answers
"no" to 157 of the 158, which gives 0.506: every un-negated question right
and 78 of the 79 negated questions wrong.

The low score reflects how the negated question is read, not a failure to
recognise collider bias. In all 78 wrong transcripts the reasoning turn both
names the collider (or selection) mechanism and states that X has no causal
effect on Y, by keyword match over the full text. The conclusions of 15 of
them, read by hand, fall into two groups:

- The model answers the un-negated question: "No. The correlation ... does
  not imply that X affects Y ... the true causal effect is zero."
- The model reads the negated question literally: "No. ... 'X does not affect
  Y' is true in this hypothetical world, but it cannot be inferred from the
  conditional correlation alone; it follows from the graph structure."

Both state the fact the "yes" label encodes and still answer "No". The one
negated question scored correct has the same reasoning, with "No" in the
reasoning turn and "Yes" in the final one-word turn. `gpt-6-luna` answers
"yes" to 41 of the 79 negated questions and "no" to all 79 of the others.

## Development

```bash
uv sync
uv run pytest                              # all tests (downloads the dataset once)
uv run pytest -m "not dataset_download"    # offline unit tests only
```

## Citation

```bibtex
@inproceedings{jin2023cladder,
  author = {Zhijing Jin and Yuen Chen and Felix Leeb and Luigi Gresele and Ojasv Kamal and Zhiheng Lyu and Kevin Blin and Fernando Gonzalez Adauto and Max Kleiman-Weiner and Mrinmaya Sachan and Bernhard Sch{\"{o}}lkopf},
  title = {{CL}adder: {A}ssessing Causal Reasoning in Language Models},
  booktitle = {NeurIPS},
  year = {2023},
  url = {https://openreview.net/forum?id=e2wtjx0Yqu},
}
```
