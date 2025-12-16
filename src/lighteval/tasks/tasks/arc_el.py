from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbTokenNorm
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


ARC_EL_SUBSETS = ["ARC-Challenge", "ARC-Easy"]
ARC_SUBSET_MAPPER = {"ARC-Challenge": "challenge", "ARC-Easy": "easy"}


def arc_el_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Ερώτηση: {line['question']}\nΑπάντηση:",
        choices=[f" {c}" for c in line["choices"]["text"]],
        gold_index=line["choices"]["label"].index(line["answerKey"]),
    )


def record_to_sample(record):
    query = record["question"].strip()
    target = record["answerKey"]
    choices = record["choices"]["text"]

    return Sample(input=query, target=target, choices=choices)


class ARCELTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            prompt_function=arc_el_prompt,
            hf_repo="ilsp/arc_greek",
            hf_subset=hf_subset,
            hf_avail_splits=["train", "test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select="random_sampling_from_train",
            generation_size=1,
            metrics=[
                Metrics.loglikelihood_acc,
                # FIXME tentative
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm())
            ],
            stop_sequence=["\n"],
            version=0,
            sample_fields=record_to_sample,
            solver=[multiple_choice(cache=True)],
            scorer=choice(),
        )


ARC_EL_TASKS = [ARCELTask(name=f"arc_el:{ARC_SUBSET_MAPPER[subset]}", hf_subset=subset) for subset in ARC_EL_SUBSETS]

TASKS_TABLE = ARC_EL_TASKS
