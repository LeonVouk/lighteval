from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


TEMPLATE_EL = """
Απάντησε τις παρακάτω ερωτήσεις πολλαπλής επιλογής. Η τελευταία γραμμή της απάντησής σου πρέπει να έχει την ακόλουθη μορφή: 'Απάντηση: $ΓΡΑΜΜΑ' (χωρίς εισαγωγικά) όπου ΓΡΑΜΜΑ είναι ένα εκ των ABCDEFGHIJ, όποιο αντιστοιχεί στην σωστή επιλογή. Σκέψου βήμα προς βήμα πριν απαντήσεις. 
{question}

{choices}

Answer:""".strip()


def mmlu_el_pro_prompt_function(line, task_name: str = None):
    choices = "\n".join([f"{letter}: {choice}" for letter, choice in zip(ascii_uppercase, line["options"])])

    query = TEMPLATE_EL.format(
        question=line["question"],
        choices=choices,
    )

    return Doc(
        task_name=task_name,
        query=query,
        choices=ascii_uppercase[: len(choices)],
        gold_index=line["answer_index"],
        instruction=query,
    )


def record_to_sample(record):
    return Sample(input=record["question"], target=record["answer"], choices=record["options"])


mmlu_pro_el = LightevalTaskConfig(
    name="mmlu_pro_el",
    prompt_function=mmlu_el_pro_prompt_function,
    sample_fields=record_to_sample,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="ilsp/MMLU-Pro_greek",
    hf_subset="default",
    evaluation_splits=("test",),
    few_shots_split="test",
    metrics=[Metrics.gpqa_instruct_metric], # its valid since the answer letters haven't been translated, they are still in English
)

TASKS_TABLE = [mmlu_pro_el]
