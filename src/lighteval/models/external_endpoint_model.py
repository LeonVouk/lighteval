# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import asyncio
from typing import Coroutine, List, Optional, Union
import openai
import torch
from huggingface_hub import (
    AsyncInferenceClient,
    InferenceClient,
    InferenceEndpoint,
    InferenceEndpointTimeoutError,
    TextGenerationInputGrammarType,
    TextGenerationOutput,
    create_inference_endpoint,
    get_inference_endpoint,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.logging.hierarchical_logger import hlog, hlog_err, hlog_warn
from lighteval.models.abstract_model import LightevalModel, ModelInfo
from lighteval.models.model_config import ExternalEndpointModelConfig
from lighteval.models.model_output import GenerativeResponse, LoglikelihoodResponse, LoglikelihoodSingleTokenResponse
from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
)
from lighteval.utils.utils import EnvConfig, as_list


BATCH_SIZE = 50


class ExternalEndpointModel(LightevalModel):
    """ExternalEndpointModels can be used to evaluate models from external endpoints e.g OpenAI, Amazon Bedrock
    """

    def __init__(
        self, config: ExternalEndpointModelConfig, env_config: EnvConfig
    ) -> None:

        self.client_name = config.client.lower()
        if self.client_name == "openai":
            # TODO add async_client
            self.client = openai.OpenAI(
                api_key=config.api_key,
                base_url=config.base_url
            )
        else:
            raise NotImplementedError(f"Client type {config.client} not yet implemented")
        
        self.name = config.model
        self.revision = "default"
        self.client_kwargs = config.client_kwargs

        self.use_async = False  # set to False for debug - async use is faster

        self._tokenizer = AutoTokenizer.from_pretrained(self.name)
        self._add_special_tokens = config.add_special_tokens if config.add_special_tokens is not None else False

        self.model_info = ModelInfo(
            model_name=self.model,
            model_sha=self.revision,
            model_dtype="default",
            model_size=-1,
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self):
        return self._add_special_tokens

    @property
    def disable_tqdm(self) -> bool:
        False  # no accelerator = this is the main process

    def max_length(self):
        if self._max_length is not None:
            return self._max_length

        if hasattr(self.tokenizer, "model_max_length"):
            self._max_length = self.tokenizer.model_max_length
        else:
            self._max_length = 2048
        return self._max_length

    # TODO add async request
    # def _async_process_request(
    #     self,
    #     context: str,
    #     stop_tokens: list[str],
    #     max_tokens: int,
    #     grammar: Optional[TextGenerationInputGrammarType] = None,
    # ) -> Coroutine[None, list[TextGenerationOutput], str]:
    #     # Todo: add an option to launch with conversational instead for chat prompts
    #     # https://huggingface.co/docs/huggingface_hub/v0.20.3/en/package_reference/inference_client#huggingface_hub.AsyncInferenceClient.conversational
    #     generated_text = self.async_client.text_generation(
    #         prompt=context,
    #         details=True,
    #         decoder_input_details=True,
    #         grammar=grammar,
    #         max_new_tokens=max_tokens,
    #         stop_sequences=stop_tokens,
    #         # truncate=,
    #     )

    #     return generated_text

    def _process_request(
        self,
        context: str,
        stop_tokens: list[str],
        max_tokens: int,
        grammar: Optional[TextGenerationInputGrammarType] = None,
    ) -> TextGenerationOutput:
       
        if self.client_name == 'openai':
            generated_text = self.client.chat.completions.create(
                model=self.name,
                messages=context,
                logprobs=True,
                # top_logprobs=5,
                max_tokens=max_tokens,
                stop=stop_tokens,
                **self.client_kwargs
            )
        else:
            raise NotImplementedError(f"Client type {self.client_name} not yet implemented")

        return generated_text

    # TODO add async batch
    # async def _async_process_batch_generate(
    #     self,
    #     requests: list[GreedyUntilRequest],
    # ) -> list[TextGenerationOutput]:
    #     return await asyncio.gather(
    #         *[
    #             self._async_process_request(
    #                 context=request.context,
    #                 stop_tokens=as_list(request.stop_sequence),
    #                 max_tokens=request.generation_size,
    #                 grammar=request.generation_grammar,
    #             )
    #             for request in requests
    #         ]
    #     )

    def _process_batch_generate(
        self,
        requests: list[GreedyUntilRequest],
    ) -> list[TextGenerationOutput]:
        return [
            self._process_request(
                context=request.context,
                stop_tokens=as_list(request.stop_sequence),
                max_tokens=request.generation_size,
                grammar=request.generation_grammar,
            )
            for request in requests
        ]

    # TODO add async process logprob
    # async def _async_process_batch_logprob(
    #     self, requests: list[LoglikelihoodRequest], rolling: bool = False
    # ) -> list[TextGenerationOutput]:
    #     return await asyncio.gather(
    #         *[
    #             self._async_process_request(
    #                 context=request.context if rolling else request.context + request.choice,
    #                 stop_tokens=[],
    #                 max_tokens=1,
    #             )
    #             for request in requests
    #         ]
    #     )

    def _process_batch_logprob(
        self, requests: list[LoglikelihoodRequest], rolling: bool = False
    ) -> list[TextGenerationOutput]:
        return [
            self._process_request(
                context=request.context if rolling else request.context + request.choice,
                stop_tokens=[],
                max_tokens=1,
            )
            for request in requests
        ]

    def parse_response(self, response):

        if self.client_name == 'openai':
            return {
                'generated_text': response.choices[0].message.content.strip(),
                'output_tokens': [token.token for token in response.choices[0].logprobs.content],
                'logprobs': [token.logprob for token in response.choices[0].logprobs.content]
            }
        else:
            raise NotImplementedError(f"Client type {self.client_name} not yet implemented")



    def greedy_until(
        self,
        requests: List[GreedyUntilRequest],
        override_bs: Optional[int] = None,
    ) -> List[GenerativeResponse]:
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)
            request.stop_sequence = as_list(request.stop_sequence) + [self.tokenizer.eos_token]

        dataset = GenerativeTaskDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[str] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=self.DATASET_SPLITS,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(
                dataloader, desc="Greedy generation", position=1, leave=False, disable=self.disable_tqdm
            ):
                # the `returns_logits` flag is only used to filter the results, we always request the full details.
                returns_logits = batch[0].use_logits
                num_samples = batch[0].num_samples
                if num_samples > 1:
                    hlog_err(
                        "Inference endpoints does not allow sampling evaluations - this is likely to fail or provide problematic results"
                    )

                if self.use_async:
                    responses = asyncio.run(self._async_process_batch_generate(batch))
                else:
                    responses = self._process_batch_generate(batch)
                for response in responses:
                    response = self.parse_response(response)
                    results.append(
                        GenerativeResponse(
                            result=response['generated_text'],
                            logits=response['logprobs'] if returns_logits else None,
                            truncated_tokens_count=-1,
                            padded_tokens_count=-1,
                        )
                    )

        return dataset.get_original_order(results)

    def loglikelihood(
        self, requests: list[LoglikelihoodRequest], override_bs: Optional[int] = None
    ) -> list[LoglikelihoodResponse]:
        for request in requests:
            request.tokenized_context = self.tok_encode(request.context)
            request.tokenized_continuation = self.tok_encode(request.choice)
        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[str] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=self.DATASET_SPLITS,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(dataloader, desc="Loglikelihoods", position=1, leave=False, disable=self.disable_tqdm):
                if self.use_async:
                    responses = asyncio.run(self._async_process_batch_logprob(batch))
                else:
                    responses = self._process_batch_logprob(batch)
                for cur_request, response in zip(batch, responses):
                    
                    input_tokens = torch.tensor(cur_request.tokenized_context)
                    cont_toks = torch.tensor(cur_request.tokenized_continuation)
                    
                    len_choice = len(cont_toks)
                    response = self.parse_response(response)

                    logits = response['logprobs']

                    greedy_tokens = torch.tensor(logits).argmax(dim=-1)
                    max_equal = (greedy_tokens == cont_toks).all().squeeze(0)
                    results.append(
                        LoglikelihoodResponse(
                            result=(sum(logits), bool(max_equal)),
                            input_tokens=[t for t in input_tokens],
                            generated_tokens=[t for t in cont_toks],
                            truncated_tokens_count=-1,
                            padded_tokens_count=-1,
                        )
                    )

        return dataset.get_original_order(results)

    def loglikelihood_rolling(
        self, requests: list[LoglikelihoodRollingRequest], override_bs=None
    ) -> list[LoglikelihoodResponse]:
        """This function is used to compute the log likelihood of the context for perplexity metrics."""
        for request in requests:
            request.tokenized_context = [self.tokenizer.eos_token_id]
            request.tokenized_continuation = self.tok_encode(request.context)

        dataset = LoglikelihoodDataset(requests=requests, num_dataset_splits=self.DATASET_SPLITS)
        batch_size = override_bs if override_bs is not None else BATCH_SIZE
        results: List[str] = []

        for _, _ in tqdm(
            dataset.splits_start_end_iterator(),
            total=self.DATASET_SPLITS,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: batch)

            for batch in tqdm(
                dataloader, desc="Loglikelihoods, rolling", position=1, leave=False, disable=self.disable_tqdm
            ):
                if self.use_async:
                    responses = asyncio.run(self._async_process_batch_logprob(batch, rolling=True))
                else:
                    responses = self._process_batch_logprob(batch, rolling=True)
                for cur_request, response in zip(batch, responses):
                    logits = self.parse_response(response)['logprobs']
                    input_tokens = torch.tensor(cur_request.tokenized_context)
                    cont_toks = torch.tensor(cur_request.tokenized_continuation)
                    results.append(
                        LoglikelihoodResponse(
                            result=sum(logits),
                            input_tokens=[t for t in input_tokens],
                            generated_tokens=[t for t in cont_toks],
                            truncated_tokens_count=-1,
                            padded_tokens_count=-1,
                        )
                    )

        return dataset.get_original_order(results)

    def loglikelihood_single_token(
        self,
        requests: list[LoglikelihoodSingleTokenRequest],
        override_bs: Optional[int] = None,
    ) -> list[LoglikelihoodSingleTokenResponse]:
        raise ValueError("Endpoint models can't use single token metrics. Change the metric to the standard version")