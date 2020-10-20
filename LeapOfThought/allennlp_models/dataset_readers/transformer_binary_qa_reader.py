from typing import Dict, Optional
import json
import logging
import gzip
import random

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer, SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer, SpacyTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("transformer_binary_qa")
class TransformerBinaryReader(DatasetReader):
    """
    Supports reading artisets for both transformer models (e.g., pretrained_model="roberta-base" and
    combine_input_fields=True) and NLI models with separate premise and hypothesis (set combine_input_fields=False).
    Empty contexts (premises) are replaced by "N/A" in the NLI case.
    max_pieces and add_prefix only apply to transformer models
    """

    def __init__(self,
                 pretrained_model: str = None,
                 tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_pieces: int = 512,
                 add_prefix: bool = False,
                 combine_input_fields: bool = True,
                 sample: int = -1) -> None:
        super().__init__()

        if pretrained_model != None:
            self._tokenizer = PretrainedTransformerTokenizer(pretrained_model, max_length=max_pieces)
            token_indexer = PretrainedTransformerIndexer(pretrained_model)
            self._token_indexers = {'tokens': token_indexer}
        else:
            self._tokenizer = tokenizer or SpacyTokenizer()
            self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._sample = sample
        self._add_prefix = add_prefix
        self._combine_input_fields = combine_input_fields
        self._debug_prints = -1

    @overrides
    def _read(self, file_path: str):
        self._debug_prints = 5
        cached_file_path = cached_path(file_path)

        if file_path.endswith('.gz'):
            data_file = gzip.open(cached_file_path, 'rb')
        else:
            data_file = open(cached_file_path, 'r')

        logger.info("Reading QA instances from jsonl dataset at: %s", file_path)
        item_jsons = []
        for line in data_file:
            item_jsons.append(json.loads(line.strip()))

        if self._sample != -1:
            item_jsons = random.sample(item_jsons, self._sample)
            logger.info("Sampling %d examples", self._sample)

        for item_json in Tqdm.tqdm(item_jsons, total=len(item_jsons)):
            self._debug_prints -= 1
            if self._debug_prints >= 0:
                logger.info(f"====================================")
                logger.info(f"Input json: {item_json}")
            item_id = item_json["id"]

            statement_text = item_json["phrase"]
            metadata = {} if "metadata" not in item_json else item_json["metadata"]
            context = item_json["context"] if "context" in item_json else None

            yield self.text_to_instance(
                    item_id=item_id,
                    question=statement_text,
                    answer_id=item_json["answer"],
                    context = context,
                    org_metadata =metadata)

        data_file.close()

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question: str,
                         answer_id: int = None,
                         context: str = None,
                         org_metadata: dict = {}) -> Instance:
        fields: Dict[str, Field] = {}
        if self._combine_input_fields:
            qa_tokens = self.transformer_features_from_qa(question, context)
            qa_field = TextField(qa_tokens, self._token_indexers)
            fields['phrase'] = qa_field
        else:
            premise = context
            if context == "":
                premise = "N/A"
            premise_tokens = self._tokenizer.tokenize(premise)
            hypothesis_tokens = self._tokenizer.tokenize(question)
            qa_tokens = [premise_tokens, hypothesis_tokens]
            fields["premise"] = TextField(premise_tokens, self._token_indexers)
            fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)

        if answer_id is not None:
            fields['label'] = LabelField(answer_id, skip_indexing=True)
        new_metadata = {
            "id": item_id,
            "question_text": question,
            "context": context,
            "correct_answer_index": answer_id
        }

        # TODO Alon get rid of this in production...
        if 'skills' in org_metadata:
            new_metadata.update({'skills': org_metadata['skills']})
        if 'tags' in org_metadata:
            new_metadata.update({'tags': org_metadata['tags']})

        if self._debug_prints >= 0:
            logger.info(f"Tokens: {qa_tokens}")
            logger.info(f"Label: {answer_id}")
        fields["metadata"] = MetadataField(new_metadata)
        return Instance(fields)

    def transformer_features_from_qa(self, question: str, context: str):
        if self._add_prefix:
            question = "Q: " + question
            if context is not None and len(context) > 0:
                context = "C: " + context
        if context is not None and len(context) > 0:
            tokens = self._tokenizer.tokenize_sentence_pair(question, context)
        else:
            tokens = self._tokenizer.tokenize(question)
        return tokens
