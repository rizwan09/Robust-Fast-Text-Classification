from typing import Dict, Optional, Union

import numpy
from overrides import overrides
import torch
from torch import nn
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder, Maxout
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.utils import helper

@Model.register("selector")
class Selector(Model):
    """
    This class implements the Biattentive Classification Network model described
    in section 5 of `Learned in Translation: Contextualized Word Vectors (NIPS 2017)
    <https://arxiv.org/abs/1708.00107>`_ for text classification. We assume we're
    given a piece of text, and we predict some output label.

    At a high level, the model starts by embedding the tokens and running them through
    a feed-forward neural net (``pre_encode_feedforward``). Then, we encode these
    representations with a ``Seq2SeqEncoder`` (``encoder``). We run biattention
    on the encoder output represenatations (self-attention in this case, since
    the two representations that typically go into biattention are identical) and
    get out an attentive vector representation of the text. We combine this text
    representation with the encoder outputs computed earlier, and then run this through
    yet another ``Seq2SeqEncoder`` (the ``integrator``). Lastly, we take the output of the
    integrator and max, min, mean, and self-attention pool to create a final representation,
    which is passed through a maxout network or some feed-forward layers
    to output a classification (``output_layer``).

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    embedding_dropout : ``float``
        The amount of dropout to apply on the embeddings.
    pre_encode_feedforward : ``FeedForward``
        A feedforward network that is run on the embedded tokens before they
        are passed to the encoder.
    encoder : ``Seq2SeqEncoder``
        The encoder to use on the tokens.
    integrator : ``Seq2SeqEncoder``
        The encoder to use when integrating the attentive text encoding
        with the token encodings.
    integrator_dropout : ``float``
        The amount of dropout to apply on integrator output.
    output_layer : ``Union[Maxout, FeedForward]``
        The maxout or feed forward network that takes the final representations and produces
        a classification prediction.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 embedding_dropout: float,
                 cuda_device: int = -1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Selector, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._num_classes = self.vocab.get_vocab_size("labels")
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._cuda_device = cuda_device
        self._vocab = vocab
        self._linear = nn.Linear(self._text_field_embedder.get_output_dim(), 1)


        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                is_train: int = 1):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a
            distribution over the label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        sentence1 = tokens['tokens']
        embedded_text = self._text_field_embedder(tokens)
        dropped_embedded_text = self._embedding_dropout(embedded_text)

        score = self._linear(dropped_embedded_text).squeeze(2)
        pbx = F.sigmoid(score)
     
        assert pbx.size() == sentence1.size()
        selection_x = pbx.bernoulli().long()
        result_x = sentence1.mul(selection_x) #word ids that are selected; contains zeros where it's not selected (ony selected can be found by selected_x[selected_x!=0])
        selected_x = helper.get_selected_tensor2(result_x, pbx, sentence1, self._cuda_device) #sentence1_len is a numpy array
        # print(' selected_x size: ', selected_x.size())
        logpz = zsum = zdiff = -1.0
        if is_train==1:
            mask1 = (sentence1!=0).long()
            masked_selection_x =  selection_x.mul(mask1)
            logpx = -helper.binary_cross_entropy(pbx, selection_x.float().detach(), reduce = False) #as reduce is not available for this version I am doing this code myself:
            assert logpx.size()== sentence1.size()

            # batch
            logpx = logpx.mul(mask1.float()).sum(1)
            logpz = (logpx) 
            if masked_selection_x.size()[1]>1:zdiff1 = (masked_selection_x[:,1:]-masked_selection_x[:,:-1]).abs().sum(1).float()  ####T.sum(T.abs_(z[1:]-z[:-1]), axis=0, dtype=theano.config.floatX)
            else: zdiff1 = 0.0
            # assert zdiff1.size()[0] == sentence1.size()[0]
            assert logpz.size()[0] == sentence1.size()[0]

            zdiff = zdiff1

            xsum = masked_selection_x.sum(1)
            zsum = xsum

            assert zsum.size()[0] ==  sentence1.size()[0]

            assert logpz.dim() == zsum.dim()
            # if zdiff!=0.0: assert logpz.dim() == zdiff.dim()
            return selected_x, logpz, zsum.float(), zdiff
        
        # return selected_x (var), sentence1_len (numpy), selected_y (var), sentence2_len (numpy), selector_loss (var of size 1)
        return selected_x, logpz, zsum, zdiff
 

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Selector':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        embedding_dropout = params.pop("embedding_dropout")
        cuda_device = params.pop_int("cuda_device", -1)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   embedding_dropout=embedding_dropout,
                   cuda_device=cuda_device,
                   initializer=initializer,
                   regularizer=regularizer)
