from collections import OrderedDict
from typing import Optional, Union

import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import pipeline as transformers_pipeline
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.pipelines import PIPELINE_REGISTRY

import model as lm_model
import document_queries_pipeline


PIPELINE_DEFAULTS = {
    "document-queries": "impira/layoutlm-document-qa",
}

DEFAULT_REVISIONS = {
    "impira/layoutlm-document-qa": "ff904df",
}

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("layoutlm", "LayoutLMForQuestionAnswering"),
    ]
)

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
)

class AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING

PIPELINE_REGISTRY.register_pipeline(
    "document-queries",
    pipeline_class=document_queries_pipeline.DocumentQueriesPipeline,
    pt_model=AutoModelForDocumentQuestionAnswering,
)

def pipeline(
    task: str = None,
    model: Optional[str] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    revision: Optional[str] = None,
    device: Optional[Union[int, str, "torch.device"]] = None,
    **pipeline_kwargs
):

    if model is None and task is not None:
        model = PIPELINE_DEFAULTS.get(task)

    if revision is None and model is not None:
        revision = DEFAULT_REVISIONS.get(model)

    config = AutoConfig.from_pretrained(model, revision=revision, **{**pipeline_kwargs})

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            config=config,
            **pipeline_kwargs,
        )

    if any(a == "LayoutLMForQuestionAnswering" for a in config.architectures):
        model = lm_model.LayoutLMForQuestionAnswering.from_pretrained(
            model, config=config, revision=revision, **{**pipeline_kwargs}
        )

    if config.model_type == "vision-encoder-decoder":
        pipeline_kwargs["feature_extractor"] = model

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    return transformers_pipeline(
        task,
        revision=revision,
        model=model,
        tokenizer=tokenizer,
        device=device,
        **pipeline_kwargs,
    )
