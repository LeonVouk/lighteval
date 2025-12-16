import numpy as np

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbTokenNorm
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def medical_mc_qa_prompt_el(line, task_name: str = None):
    mcs = "\n".join(line["multiple_choice_targets"])
    return Doc(
        task_name=task_name,
        query=f"Ερώτηση: {line['inputs']}\n\nΕπιλογές:\n{mcs}\n\nΑπάντηση:",
        choices=[f" {c}" for c in line["multiple_choice_targets"]],
        gold_index=int(np.argmax(np.array(line["multiple_choice_scores"]))),
    )


medical_mc_qa_el_task = LightevalTaskConfig(
    name="medicalmcqa_el",
    prompt_function=medical_mc_qa_prompt_el,
    hf_repo="ilsp/medical_mcqa_greek",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["train"],
    few_shots_split="validation",
    few_shots_select="sequential",
    generation_size=1,
    metrics=[Metrics.loglikelihood_acc, LogLikelihoodAccMetric(normalization=LogProbTokenNorm())],
    stop_sequence=["\n"],
    version=0,
)



TASKS_TABLE = [medical_mc_qa_el_task]
