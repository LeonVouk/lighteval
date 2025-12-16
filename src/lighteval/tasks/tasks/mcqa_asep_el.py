from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def mcqa_asep_prompt_el(line, task_name: str = None):
    mcs = "\n".join(line["choices"])
    return Doc(
        task_name=task_name,
        query=f"Ερώτηση: {line['question']}\n\nΕπιλογές:\n{mcs}\n\nΑπάντηση:",
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["answer"],
    )


mcqa_asep_el_task = LightevalTaskConfig(
    name="mcqa_asep",
    prompt_function=mcqa_asep_prompt_el,
    hf_repo="ilsp/mcqa_greek_asep",
    hf_subset="default",
    hf_avail_splits=["default"],
    evaluation_splits=["default"],
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
        # FIXME add after fixing the result merging
        # Metrics.pass_at_k_letters(sample_params={"k": 1}),
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [mcqa_asep_el_task]
