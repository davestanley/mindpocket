# Simple tagger module with option to weight cross-entropy. For dealing with
# class imbalance
# Dave Stanley, Insight Data Science, 2019
from typing import Dict, Optional, List, Any

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("simple_tagger2")
class SimpleTagger2(Model):
    """
    This ``SimpleTagger`` simply encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
    predicts a tag for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 do_crossentropy_weighting: bool = False,
                 Ntags0: int = None,
                 Ntags1: int = None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SimpleTagger2, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder
        # Cross entropy weighting -davedit
        self.do_crossentropy_weighting = do_crossentropy_weighting
            # Should be integer (e.g., 0-100), but actual range doesn't matter becuase we will normalize later
        self.Ntags0 = Ntags0            # Number of tags == 0 in whole dataset
        self.Ntags1 = Ntags1            # Number of tags == 1 in whole dataset
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_classes))

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containing the original words in the sentence to be tagged under a 'words' key.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """

        #Davedit
        print_vocab = False


        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes])

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        # x = tokens['tokens']
        # y = tokens['pos_tag']
        # w = tokens['ner_tag']

        # import code
        # code.interact(local=locals())
        # import pdb
        # pdb.set_trace()


        if tags is not None:
            if self.do_crossentropy_weighting:
                # Implementing custom loss function, weight tags = 1 vs tags = 0
                # Note this only works for binary tags at present


                Nt0 = self.Ntags0          # Should correspond to non-blanks
                Nt1 = self.Ntags1             # Should correspond to blanks

                if not (Nt0 and Nt1):
                    # If either Nt0 or Nt1 are unspecified
                    Nt0 = sum(sum((tags==0).double()))
                    Nt1 = sum(sum((tags==1).double()))

                #import pdb; pdb.set_trace()

                # # Convert N blanks to weights - weight is inversely proportional to number of tags
                # t0wp = Nt1 / (Nt0 + Nt1)      # t0 weighting percent
                # t1wp = Nt0 / (Nt0 + Nt1)       # t1 weighting percent
                # mask2 = mask.clone()
                # mask2 = mask2.double()
                # mask2[tags == 1] = mask2[tags == 1] * t1wp
                # mask2[tags == 0] = mask2[tags == 0] * t0wp
                # loss = sequence_cross_entropy_with_logits(logits, tags, mask2)

                # Convert N blanks to weights - weight is inversely proportional to number of tags
                t0wp = Nt1 / (Nt0 + Nt1)*100      # t0 weighting percent
                t1wp = Nt0 / (Nt0 + Nt1)*100       # t1 weighting percent
                mask2 = mask.clone()
                mask2[tags == 1] = mask2[tags == 1] * t1wp
                mask2[tags == 0] = mask2[tags == 0] * t0wp
                loss = sequence_cross_entropy_with_logits(logits, tags, mask2)

                # # Old code, hardcoded
                # Nnonblanks = 5877.15 - 256.16     # Average number of non-blanks per article
                # Nblanks = 256.16                  # Average # blanks per article
                # blank_weight = Nnonblanks / (Nnonblanks + Nblanks)*100
                # nblank_weight = Nblanks / (Nnonblanks + Nblanks)*100
                # mask2 = mask
                # mask2[tags == 1] = mask2[tags == 1] * blank_weight
                # mask2[tags == 0] = mask2[tags == 0] * nblank_weight
                # loss = sequence_cross_entropy_with_logits(logits, tags, mask2)
            else:
                # Defualt AllenNLP loss
                loss = sequence_cross_entropy_with_logits(logits, tags, mask)

            if print_vocab:
                vocab = self.vocab
                vo = vocab.get_index_to_token_vocabulary('pos')
                out = set([v for v in vo.values()])
                print(out)
                vo = vocab.get_index_to_token_vocabulary('ner')
                out = set([v for v in vo.values()])
                print(out)
                vo = vocab.get_index_to_token_vocabulary('dependencies')
                out = set([v for v in vo.values()])
                print(out)
                # Results of vocab from 1st 20 articles. Can use these to set embedding dimensions
                # {'VBP', 'PRP$', 'SYM', 'XX', ':', 'ADD', 'NNS', 'CC', 'VBG', 'RBR', 'NNP', 'IN', 'JJ', 'TO', 'NFP', 'NNPS', 'PRP', 'LS', 'NN', 'CD', 'FW', 'MD', 'AFX', 'PDT', "''", 'RP', 'JJR', 'RB', 'VB', '``', '.', 'VBD', 'VBN', '-RRB-', 'JJS', 'RBS', '$', '@@PADDING@@', 'EX', 'HYPH', 'POS', '-LRB-', 'WP$', 'VBZ', ',', 'UH', 'WP', 'DT', 'WDT', 'WRB', '@@UNKNOWN@@'}
                # {'PERCENT', 'ORG', 'NONE', 'ORDINAL', 'MONEY', 'CARDINAL', 'NORP', 'LANGUAGE', 'DATE', 'WORK_OF_ART', 'LAW', 'LOC', 'PERSON', 'QUANTITY', 'EVENT', 'GPE', 'TIME', 'FAC', '@@PADDING@@', 'PRODUCT', '@@UNKNOWN@@'}
                # {'poss', 'attr', 'xcomp', 'npadvmod', 'agent', 'parataxis', 'mark', 'nmod', 'predet', 'compound', 'ROOT', 'intj', 'csubjpass', 'nsubjpass', 'preconj', 'amod', 'csubj', 'ccomp', 'punct', 'advcl', 'conj', 'acomp', 'oprd', 'case', 'nsubj', 'dobj', 'nummod', 'prt', 'cc', 'advmod', 'appos', 'neg', 'pcomp', 'quantmod', 'dep', 'meta', '@@PADDING@@', 'relcl', 'expl', 'acl', 'dative', 'auxpass', 'det', 'aux', 'prep', 'pobj', '@@UNKNOWN@@'}
                # Vocab size
                # Vocabulary with namespaces:  dependencies, Size: 47 || ner, Size: 21 || pos, Size: 51 || tokens, Size: 21902 || labels, Size: 2 || Non Padded Namespaces: {'*tags', '*labels'}


                # import pdb
                # pdb.set_trace()

            for metric in self.metrics.values():
                metric(logits, tags, mask.float())
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in argmax_indices]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
