from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbTokenNorm
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


GREEK_LETTER_INDICES = [
    "Α",
    "Β",
    "Γ",
    "Δ",
    "Ε",
    "Ζ",
    "Η",
    "Θ",
    "Ι",
    "Κ",
    "Λ",
    "Μ",
    "Ν",
    "Ξ",
    "Ο",
    "Π",
    "Ρ",
    "Σ",
    "Τ",
    "Υ",
    "Φ",
    "Χ",
    "Ψ",
    "Ω",
]


def hellaswag_prompt_el(line, task_name: str = None):
    query = "The following are multiple choice questions (with answers) about common sense.\n\n"
    query= "Οι ακόλουθες ερωτήσεις πολλαπλής επιλογής (με τις απαντήσεις τους) εξετάζουν την χρήση κοινής λογικής."
    query += f"Ερώτηση: {line['activity_label']}: {line['ctx_a']} {line['ctx_b'].capitalize()}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(GREEK_LETTER_INDICES, line["endings"])])
    query += "Απάντηση:"

    gold_ix = int(line["label"]) if line["label"] != "" else -1
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + i for i in GREEK_LETTER_INDICES[: len(line["endings"])]],
        gold_index=gold_ix,
        instruction="Οι ακόλουθες ερωτήσεις πολλαπλής επιλογής (με τις απαντήσεις τους) εξετάζουν την χρήση κοινής λογικής.\n\n",
    )


hellaswag_el_task = LightevalTaskConfig(
    name="hellaswag_el",
    prompt_function=hellaswag_prompt_el,
    hf_repo="ilsp/hellaswag_greek",
    hf_subset="default",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[
        Metrics.loglikelihood_acc,
        # FIXME tentative
        LogLikelihoodAccMetric(normalization=LogProbTokenNorm())
        # FIXME EM in hellaswag? why? Metrics.exact_match,
    ],
    stop_sequence=["\n"],
    version=0,
)


TASKS_TABLE = [hellaswag_el_task]
