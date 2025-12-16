from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


AMG_DIRECTIONS = [
    "grc->ell",
    "ell->grc",
]

def amg_grc_to_ell_prompt(line, task_name: str = None):
    query = "Μετάφρασε το κείμενο από τα Αρχαία Ελληνικά στα Νέα Ελληνικά.\n\n"
    query += f"Αρχαία Ελληνικά:\n{line['grc']}\n\n"
    query += "Νέα Ελληνικά:\n"
    return Doc(
        task_name=task_name,
        query=query,
        instruction="Μετάφρασε το κείμενο από τα Αρχαία Ελληνικά στα Νέα Ελληνικά.\n\n",
        choices=[line["ell"]],
        gold_index=0,
    )


def amg_ell_to_grc_prompt(line, task_name: str = None):
    query = "Μετάφρασε το κείμενο από τα Νέα Ελληνικά στα Αρχαία Ελληνικά.\n\n"
    query += f"Νέα Ελληνικά:\n{line['ell']}\n\n"
    query += "Αρχαία Ελληνικά:\n"
    return Doc(
        task_name=task_name,
        query=query,
        instruction="Μετάφρασε το κείμενο από τα Νέα Ελληνικά στα Αρχαία Ελληνικά.\n\n",
        choices=[line["grc"]],
        gold_index=0,
    )


AMG_PROMPT_FN_MAPPER = {
    "grc->ell": amg_grc_to_ell_prompt,    
    "ell->grc": amg_ell_to_grc_prompt,
}


class AMGTask(LightevalTaskConfig):
    def __init__(self, name, prompt_fn):
        super().__init__(
            name=name,
            prompt_function=prompt_fn,
            hf_repo="ilsp/ancient-modern_greek_translations",
            hf_subset="default",
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=100,
            metrics=[Metrics.bleu],
            stop_sequence=["\n"],
            version=0,
        )


AMG_TASKS = [
    AMGTask(name=f"amg:{direction}", prompt_fn=AMG_PROMPT_FN_MAPPER[direction])
    for direction in AMG_DIRECTIONS
]


TASKS_TABLE = AMG_TASKS
