from typing import Dict, Optional, List, Any
import logging

from transformers.modeling_t5 import T5Model
from transformers.modeling_roberta import RobertaModel
from transformers.modeling_xlnet import XLNetModel
from transformers.modeling_bert import BertModel
from transformers.modeling_albert import AlbertModel
from transformers.modeling_utils import SequenceSummary
import re, json
import torch
from torch.nn.modules.linear import Linear

from allennlp.common.util import sanitize
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy


logger = logging.getLogger(__name__)

@Model.register("transformer_binary_qa")
class TransformerBinaryQA(Model):
    """
    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str = None,
                 requires_grad: bool = True,
                 transformer_weights_model: str = None,
                 num_labels: int = 2,
                 predictions_file=None,
                 layer_freeze_regexes: List[str] = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._predictions = []

        self._pretrained_model = pretrained_model

        if 't5' in pretrained_model:
            self._padding_value = 1  # The index of the RoBERTa padding token
            if transformer_weights_model:  # Override for RoBERTa only for now
                logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
                transformer_model_loaded = load_archive(transformer_weights_model)
                self._transformer_model = transformer_model_loaded.model._transformer_model
            else:
                self._transformer_model = T5Model.from_pretrained(pretrained_model)
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        if 'roberta' in pretrained_model:
            self._padding_value = 1  # The index of the RoBERTa padding token
            if transformer_weights_model:  # Override for RoBERTa only for now
                logging.info(f"Loading Transformer weights model from {transformer_weights_model}")
                transformer_model_loaded = load_archive(transformer_weights_model)
                self._transformer_model = transformer_model_loaded.model._transformer_model
            else:
                self._transformer_model = RobertaModel.from_pretrained(pretrained_model)
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        elif 'xlnet' in pretrained_model:
            self._padding_value = 5  # The index of the XLNet padding token
            self._transformer_model = XLNetModel.from_pretrained(pretrained_model)
            self.sequence_summary = SequenceSummary(self._transformer_model.config)
        elif 'albert' in pretrained_model:
            self._transformer_model = AlbertModel.from_pretrained(pretrained_model)
            self._padding_value = 0  # The index of the BERT padding token
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        elif 'bert' in pretrained_model:
            self._transformer_model = BertModel.from_pretrained(pretrained_model)
            self._padding_value = 0  # The index of the BERT padding token
            self._dropout = torch.nn.Dropout(self._transformer_model.config.hidden_dropout_prob)
        else:
            assert (ValueError)

        for name, param in self._transformer_model.named_parameters():
            if layer_freeze_regexes and requires_grad:
                grad = not any([bool(re.search(r, name)) for r in layer_freeze_regexes])
            else:
                grad = requires_grad
            if grad:
                param.requires_grad = True
            else:
                param.requires_grad = False

        transformer_config = self._transformer_model.config
        transformer_config.num_labels = num_labels
        self._output_dim = self._transformer_model.config.hidden_size

        # unifing all model classification layer
        self._classifier = Linear(self._output_dim, num_labels)
        self._classifier.weight.data.normal_(mean=0.0, std=0.02)
        self._classifier.bias.data.zero_()

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        self._debug = -1


    def forward(self,
                    phrase: Dict[str, torch.LongTensor],
                    label: torch.LongTensor = None,
                    metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        self._debug -= 1
        input_ids = phrase['tokens']['token_ids']
        segment_ids = phrase['tokens']['type_ids']

        question_mask = (input_ids != self._padding_value).long()

        # Segment ids are not used by RoBERTa
        if 'roberta' in self._pretrained_model or 't5' in self._pretrained_model:
            transformer_outputs, pooled_output = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                                         # token_type_ids=util.combine_initial_dims(segment_ids),
                                                                         attention_mask=util.combine_initial_dims(question_mask))
            cls_output = self._dropout(pooled_output)
        if 'albert' in self._pretrained_model:
            transformer_outputs, pooled_output = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                                         # token_type_ids=util.combine_initial_dims(segment_ids),
                                                                         attention_mask=util.combine_initial_dims(question_mask))
            cls_output = self._dropout(pooled_output)
        elif 'xlnet' in self._pretrained_model:
            transformer_outputs = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                          token_type_ids=util.combine_initial_dims(segment_ids),
                                                          attention_mask=util.combine_initial_dims(question_mask))
            cls_output = self.sequence_summary(transformer_outputs[0])

        elif 'bert' in self._pretrained_model:
            last_layer, pooled_output = self._transformer_model(input_ids=util.combine_initial_dims(input_ids),
                                                                token_type_ids=util.combine_initial_dims(segment_ids),
                                                                attention_mask=util.combine_initial_dims(question_mask))
            cls_output = self._dropout(pooled_output)
        else:
            assert (ValueError)

        label_logits = self._classifier(cls_output)

        output_dict = {}
        output_dict['label_logits'] = label_logits
        output_dict['label_probs'] = torch.nn.functional.softmax(label_logits, dim=1)
        output_dict['answer_index'] = label_logits.argmax(1)


        if label is not None:
            loss = self._loss(label_logits, label)
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss# TODO this is shortcut to get predictions fast..


            for e, example in enumerate(metadata):
                logits = sanitize(label_logits[e, :])
                label_probs = sanitize(output_dict['label_probs'][e, :])
                prediction = sanitize(output_dict['answer_index'][e])
                prediction_dict = {'id': example['id'], \
                                   'phrase': example['question_text'], \
                                   'context': example['context'], \
                                   'logits': logits,
                                   'label_probs': label_probs,
                                   'answer': example['correct_answer_index'],
                                   'prediction': prediction,
                                   'is_correct': (example['correct_answer_index'] == prediction) * 1.0}

                if 'skills' in example:
                    prediction_dict['skills'] = example['skills']
                if 'tags' in example:
                    prediction_dict['tags'] = example['tags']
                self._predictions.append(prediction_dict)

        #if self._predictions_file is not None:# and not self.training:
        #    with open(self._predictions_file, 'a') as f:
        #        for e, example in enumerate(metadata):
        #            logits = sanitize(label_logits[e, :])
        #            prediction = sanitize(output_dict['answer_index'][e])
        #            f.write(json.dumps({'id': example['id'], \
        #                                'phrase': example['question_text' ], \
        #                                'context': example['context'], \
        #                                'logits': logits,
        #                                'answer': example['correct_answer_index'],
        #                                'prediction': prediction,
        #                                'is_correct': (example['correct_answer_index'] == prediction) * 1.0}) + '\n')



        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if reset == True and not self.training:
            return {
                'EM': self._accuracy.get_metric(reset),
                'predictions': self._predictions,
            }
        else:
            return {
                'EM': self._accuracy.get_metric(reset),
            }

