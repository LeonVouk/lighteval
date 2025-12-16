from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


LANG_NAMES = {
    "en": "Αγγλικά",
    "fr": "Γαλλικά",
    "pt": "Πορτογαλικά",
    "de": "Γερμανικά",
    "es": "Ισπανικά",
    "it": "Ιταλικά",
    "el": "Ελληνικά"
}


WMT24_DIRECTIONS = [
    "en->el", 
    "el->en",
]


SUBSET_MAPPING = {
    "en->el": "en-el_GR",
    "el->en": "en-el_GR",
}


def create_wmt24_prompt(src_lang: str, tgt_lang: str):
    def prompt_fn(line, task_name: str = None):
        
        src_col = 'source'
        tgt_col = 'target'
        if src_lang != 'en':
            src_col = 'target'
            tgt_col = 'source'
        
        query = f"Μετάφρασε το κείμενο απο τα {LANG_NAMES[src_lang]} στα {LANG_NAMES[tgt_lang]}.\n\n"
        query += f"{LANG_NAMES[src_lang]}:\n{line[src_col]}\n\n"
        query += f"{LANG_NAMES[tgt_lang]}:\n"
        return Doc(
            task_name=task_name,
            query=query,
            instruction=f"Μετάφρασε το κείμενο απο τα {LANG_NAMES[src_lang]} στα {LANG_NAMES[tgt_lang]}.\n\n",
            choices=[line[tgt_col]],
            gold_index=0,
        )
    return prompt_fn


WMT24_PROMPT_FN_MAPPER = {
    direction: create_wmt24_prompt(direction.split("->")[0], direction.split("->")[1])
    for direction in WMT24_DIRECTIONS
}


class WMT24Task(LightevalTaskConfig):
    def __init__(
            self,
            name,
            subset,
            prompt_fn
    ):
        super().__init__(
            name=name,
            prompt_function=prompt_fn,
            hf_repo="google/wmt24pp",
            hf_subset=subset,
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split="train",
            few_shots_select="sequential",
            generation_size=100,
            metrics=[Metrics.bleu],
            stop_sequence=["\n"],
            version=0,
        )


WMT24_TASKS = [
    WMT24Task(name=f"wmt24:{direction}", subset=SUBSET_MAPPING[direction], prompt_fn=WMT24_PROMPT_FN_MAPPER[direction])
    for direction in WMT24_DIRECTIONS
]


TASKS_TABLE = WMT24_TASKS
