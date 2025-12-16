import re
import numpy as np
from typing import  Callable

from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


class ParsedAnswerAccMGSMel(SampleLevelComputation):
    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = np.mean,
        normalize_gold: Callable[[str], str] | None = None,
        normalize_pred: Callable[[str], str] | None = None
    ):
        """An F1 score class. F1 is computed over the bag of words of the golds and predictions.

        Args:
            aggregation_function (callable, optional): How to aggregate the item results. Defaults to max.
                Used if there are several golds or predictions on which scores were computed.
            normalize_gold (callable, optional): Function to use to normalize the reference strings.
                Defaults to None if no normalization is applied.
            normalize_pred (callable, optional): Function to use to normalize the predicted strings.
                Defaults to None if no normalization is applied.
        """
        if aggregation_function is None:
            aggregation_function = np.mean
        
        self.aggregation_function = aggregation_function
        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred
    
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        """Computes the metric over a list of golds and predictions for one single sample.

        Args:
            doc (Doc): The document containing gold references.
            model_response (ModelResponse): The model's response containing predictions.
            **kwargs: Additional keyword arguments.

        Returns:
            float: Aggregated score over the current sample's items.
        """
        print("Doc:", doc)
        print("Response:", model_response)
        results = []
        golds = doc.get_golds()
        predictions = model_response.final_text
        # We might need to flatten golds if they are a list of lists
        for gold in golds:
            for pred in predictions:
                print("Gold:", gold)
                print("Pred:", pred)
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(self, gold: str, pred: str) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The f1 score over the bag of words, computed using nltk.
        """

        if self.normalize_gold:
            gold = self.normalize_gold(gold)

        if self.normalize_pred:
            pred = self.normalize_pred(pred)

        number_regex = re.compile(r"(\-?(\d*[.,])*\d+)")
        parsed_response = ""
        try:
            for line in pred.split("\n"):
                line = line.strip()
                all_numbers = re.findall(number_regex, line)
                if all_numbers:
                    parsed_response = all_numbers[-1][0]
        except Exception:
            pass
        return parsed_response == gold.strip()


mgsm_el_metric = SampleLevelMetric(
    metric_name="mgsm_el_parsed_exact_match",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=ParsedAnswerAccMGSMel(),
    corpus_level_fn=np.mean,
)


def mgsm_el_prompt(line, task_name: str = None):
    question_key = "Ερώτηση:"
    answer_key = "Απάντηση βήμα προς βήμα:"

    # FIXME go back to return mgsm(line, question_key, answer_key, task_name) when dataset is fixed
    if line["answer"] not in ["nan", "None", None, ""]:
        query = f"{line['question']}\n{answer_key}"
        gold = f" {line['answer'][len(answer_key) + 1:]}"
    else:
        query = f"{question_key} {line['question']}\n{answer_key}"
        gold = f"{str(line['answer_number'])}"
    return Doc(
        task_name=task_name, 
        query=query, 
        choices=[gold], 
        gold_index=0
    )


mgsm_el_task = LightevalTaskConfig(
    name="mgsm:el",
    prompt_function=mgsm_el_prompt,
    hf_repo="ilsp/mgsm_greek",
    hf_subset="default",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=250,
    metrics=[mgsm_el_metric],
    stop_sequence=[],
    version=0,
)



TASKS_TABLE = [mgsm_el_task]
