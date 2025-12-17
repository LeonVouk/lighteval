from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbTokenNorm
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


BELEBELE_SPLITS = ["ell_Grek", "eng_Latn"]


def belebele_prompt_el(line, task_name: str = None):
    is_few_shots = line.get("__few_shots", False)
    return Doc(
        task_name=task_name,
        query=f"Απόσπασμα: {line['flores_passage']}\n\nΕρώτηση:\n{line['question']}\n\nΑ: {line['mc_answer1']}\nΒ: {line['mc_answer2']}\nΓ: {line['mc_answer3']}\nΔ: {line['mc_answer4']}\n\nΑπάντηση:",
        choices=[" Α", " Β", " Γ", " Δ"] if is_few_shots else ["Α", "Β", "Γ", "Δ"],
        gold_index=int(line["correct_answer_num"]) - 1,
    )


def belebele_prompt_en(line, task_name: str = None):
    is_few_shots = line.get("__few_shots", False)
    return Doc(
        task_name=task_name,
        query=f"P: {line['flores_passage']}\n\nQ:\n{line['question']}\n\nA: {line['mc_answer1']}\nB: {line['mc_answer2']}\nC: {line['mc_answer3']}\nD: {line['mc_answer4']}\n\nAnswer:",
        choices=[" A", " B", " C", " D"] if is_few_shots else ["A", "B", "C", "D"],
        gold_index=int(line["correct_answer_num"]) - 1,
    )


BELEBELE_SPLIT_MAPPER = {
    "ell_Grek": {"split": "el", "prompt_fn": belebele_prompt_el},
    "eng_Latn": {"split": "en", "prompt_fn": belebele_prompt_en},
}


def greek_civics_qa_prompt(line, task_name: str = None):
    query = "Απάντησε στην παρακάτω ερώτηση που σχετίζεται με το μάθημα της κοινωνικής και πολιτικής αγωγής.\n\n"
    query += f"Ερώτηση:\n{line['question'].strip()}\n\n"
    query += "Απάντηση:\n"
    return Doc(
        task_name=task_name, 
        query=query,
        choices=[line["answer"].strip()], 
        gold_index=0
    )


class BELEBELETask(LightevalTaskConfig):
    def __init__(
            self,
            name,
            hf_subset,
            prompt_fn
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_fn,
            hf_repo="facebook/belebele",
            hf_subset=hf_subset,
            hf_avail_splits=BELEBELE_SPLITS,
            evaluation_splits=["test"],
            few_shots_split="test",
            few_shots_select="sequential",
            generation_size=1,
            metrics=[
                Metrics.loglikelihood_acc,
                LogLikelihoodAccMetric(normalization=LogProbTokenNorm())
            ],
            stop_sequence=["\n"],
            version=0,
        )


BELEBELE_TASKS = [
    BELEBELETask(
        name=f"belebele:{BELEBELE_SPLIT_MAPPER[split]['split']}",
        hf_subset=split,
        prompt_fn=BELEBELE_SPLIT_MAPPER[split]["prompt_fn"],
    )
    for split in BELEBELE_SPLITS
]


TASKS_TABLE = BELEBELE_TASKS
