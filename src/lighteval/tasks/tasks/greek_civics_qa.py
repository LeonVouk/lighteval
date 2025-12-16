from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


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

greek_civics_qa_task = LightevalTaskConfig(
    name="greek_civics_qa",
    prompt_function=greek_civics_qa_prompt,
    hf_repo="ilsp/greek_civics_qa",
    hf_subset="default",
    hf_avail_splits=["default"],
    evaluation_splits=["default"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=100,
    metrics=[Metrics.bleu],
    stop_sequence=["\n"],
    version=0,
)



TASKS_TABLE = [greek_civics_qa_task]
