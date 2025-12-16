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


MMLU_EL_SUBSETS = [
    "all",
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def mmlu_el_prompt(line, task_name: str = None):
    # TODO probably have to change choice labels.
    query = f"Οι ακόλουθες ερωτήσεις πολλαπλής επιλογής (που παρουσιάζονται μαζί με τις απαντήσεις τους) έχουν να κάνουν με {line['subject'].replace('_', ' ')}.\n\n"
    query += f"Ερώτηση: {line['question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(GREEK_LETTER_INDICES, line["choices"])])
    query += "Απάντηση:"

    gold_ix = GREEK_LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" Α", " Β", " Γ", " Δ"],
        gold_index=gold_ix,
        fewshot_sorting_class=line["choices"][gold_ix],
        instruction=f"Οι ακόλουθες ερωτήσεις πολλαπλής επιλογής (που παρουσιάζονται μαζί με τις απαντήσεις τους) έχουν να κάνουν με {line['subject'].replace('_', ' ')}.\n\n",
    )


class MMLUELTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            prompt_function=mmlu_el_prompt,
            hf_repo="ilsp/mmlu_greek",
            hf_subset=hf_subset,
            hf_avail_splits=["test", "dev", "validation"],
            evaluation_splits=["test"],
            few_shots_split="dev",
            few_shots_select=None,
            generation_size=1,
            metrics=[
                Metrics.loglikelihood_acc,
                # FIXME add after fixing the result merging
                # Metrics.pass_at_k_letters(sample_params={"k": 1}),
            ],
            stop_sequence=["\n"],
            version=0,
        )


MMLU_EL_TASKS = [MMLUELTask(name=f"mmlu_el:{subset}", hf_subset=subset) for subset in MMLU_EL_SUBSETS]

TASKS_TABLE = MMLU_EL_TASKS
