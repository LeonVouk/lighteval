from string import ascii_uppercase

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


INCLUDE_BASE_44_SUBSETS = ["Greek"]

def include_base_44_cot_prompt_el(line, task_name: str = None):
    prompt="""Η ακόλουθη ερώτηση πολλαπλής επιλογής παρουσιάζεται μαζί με τις πιθανές απαντήσεις της. Σκέψου βήμα προς βήμα.\n"""
    query = prompt + f"Ερώτηση: {line['question']}\n"
    query += "".join([f"{key}) {choice}\n" for key, choice in zip(GREEK_LETTER_INDICES, [line["option_a"], line["option_b"], line["option_c"], line["option_d"]])])
    query += "Απάντηση:"

    gold_ix = GREEK_LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" Α", " Β", " Γ", " Δ"],
        gold_index=gold_ix,
    )

def include_base_44_cot_prompt_en(line, task_name: str = None):
    prompt="""The following multiple choice question is provided alongside its possible answers. Let's take it step by step"""
    query = prompt + f"Question: {line['question']}\n"
    query += "".join([f"{key}) {choice}\n" for key, choice in zip(ascii_uppercase, [line["option_a"], line["option_b"], line["option_c"], line["option_d"]])])
    query += "Answer:"

    gold_ix = ascii_uppercase.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
    )

def include_base_44_prompt_el(line, task_name: str = None):
    query = f"Ερώτηση: {line['question']}\n"
    query += "".join([f"{key}) {choice}\n" for key, choice in zip(GREEK_LETTER_INDICES, [line["option_a"], line["option_b"], line["option_c"], line["option_d"]])])
    query += "Απάντηση:"

    gold_ix = GREEK_LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" Α", " Β", " Γ", " Δ"],
        gold_index=gold_ix,
    )

def include_base_44_prompt_en(line, task_name: str = None):
    query = f"Question: {line['question']}\n"
    query += "".join([f"{key}) {choice}\n" for key, choice in zip(ascii_uppercase, [line["option_a"], line["option_b"], line["option_c"], line["option_d"]])])
    query += "Answer:"

    gold_ix = ascii_uppercase.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
    )


INCLUDE_BASE_44_PROMPT_MAPPER = {
    'Greek': {
        'cot_el': include_base_44_cot_prompt_el, 
        'cot_en': include_base_44_cot_prompt_en, 
        'fewshot_el': include_base_44_prompt_el, 
        'fewshot_en': include_base_44_prompt_en
    }
}


class IncludeBase44Task(LightevalTaskConfig):
    def __init__(self, name, hf_subset, prompt_fn):
        super().__init__(
            name=name,
            prompt_function=prompt_fn,
            hf_repo="CohereForAI/include-base-44",
            hf_subset=hf_subset,
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            generation_size=1,
            metrics=[
                Metrics.loglikelihood_acc,
                # FIXME tentative
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm())
            ],
            stop_sequence=["\n"],
            version=0,
        )


INCLUDE_BASE_44_TASKS = [IncludeBase44Task(name=f"include_base_44:{subset}:{setting}", hf_subset=subset, prompt_fn=prompt) for subset, mapping in INCLUDE_BASE_44_PROMPT_MAPPER.items() for setting, prompt in mapping.items()]

TASKS_TABLE = INCLUDE_BASE_44_TASKS
