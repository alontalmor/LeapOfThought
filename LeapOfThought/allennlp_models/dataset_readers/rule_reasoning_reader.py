from typing import Dict, Any
import json
import logging
import random
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.fields import MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# TagSpanType = ((int, int), str)

@DatasetReader.register("rule_reasoning")
class RuleReasoningReader(DatasetReader):
    """

    Parameters
    ----------
    """

    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 syntax: str = "rulebase",
                 add_prefix: Dict[str, str] = None,
                 skip_id_regex: str = None,
                 scramble_context: bool = False,
                 use_context_full: bool = False,
                 sample: int = -1) -> None:
        super().__init__()
        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model, max_length=max_pieces)
        self._tokenizer_internal = self._tokenizer.tokenizer
        token_indexer = PretrainedTransformerIndexer(pretrained_model)
        self._token_indexers = {'tokens': token_indexer}

        self._max_pieces = max_pieces
        self._add_prefix = add_prefix
        self._scramble_context = scramble_context
        self._use_context_full = use_context_full
        self._sample = sample
        self._syntax = syntax
        self._skip_id_regex = skip_id_regex

    @overrides
    def _read(self, file_path: str):
        instances = self._read_internal(file_path)
        return instances

    def _read_internal(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample + 1
        debug = 5
        is_done = False

        with open(file_path, 'r') as data_file:
            logger.info("Reading instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                if is_done:
                    break
                item_json = json.loads(line.strip())
                item_id = item_json.get("id", "NA")
                if self._skip_id_regex and re.match(self._skip_id_regex, item_id):
                    continue

                if self._syntax == "rulebase":
                    questions = item_json['questions']
                    if self._use_context_full:
                        context = item_json.get('context_full', '')
                    else:
                        context = item_json.get('context', "")
                elif self._syntax == "propositional-meta":
                    questions = item_json['questions'].items()
                    sentences = [x['text'] for x in item_json['triples'].values()] + \
                                [x['text'] for x in item_json['rules'].values()]
                    if self._scramble_context:
                        random.shuffle(sentences)
                    context = " ".join(sentences)
                else:
                    raise ValueError(f"Unknown syntax {self._syntax}")

                for question in questions:
                    counter -= 1
                    debug -= 1
                    if counter == 0:
                        is_done = True
                        break
                    if debug > 0:
                        logger.info(item_json)
                    if self._syntax == "rulebase":
                        text = question['text']
                        q_id = question.get('id')
                        label = None
                        if 'label' in question:
                            label = 1 if question['label'] else 0
                    elif self._syntax == "propositional-meta":
                        text = question[1]['question']
                        q_id = f"{item_id}-{question[0]}"
                        label = question[1].get('propAnswer')
                        if label is not None:
                            label = ["False", "True", "Unknown"].index(label)

                    yield self.text_to_instance(
                        item_id=q_id,
                        question_text=text,
                        context=context,
                        label=label,
                        debug=debug)

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question_text: str,
                         label: int = None,
                         context: str = None,
                         debug: int = -1) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        qa_tokens, segment_ids = self.transformer_features_from_qa(question_text, context)
        qa_field = TextField(qa_tokens, self._token_indexers)
        fields['phrase'] = qa_field

        metadata = {
            "id": item_id,
            "question_text": question_text,
            "tokens": [x.text for x in qa_tokens],
            "context": context
        }

        if label is not None:
            # We'll assume integer labels don't need indexing
            fields['label'] = LabelField(label, skip_indexing=isinstance(label, int))
            metadata['label'] = label

        if debug > 0:
            logger.info(f"qa_tokens = {qa_tokens}")
            logger.info(f"context = {context}")
            logger.info(f"label = {label}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def transformer_features_from_qa(self, question: str, context: str):
        if self._add_prefix is not None:
            question = self._add_prefix.get("q", "") + question
            context = self._add_prefix.get("c", "") + context
        if context is not None:
            tokens = self._tokenizer.tokenize_sentence_pair(question, context)
        else:
            tokens = self._tokenizer.tokenize(question)
        segment_ids = [0] * len(tokens)

        return tokens, segment_ids