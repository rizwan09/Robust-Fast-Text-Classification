from typing import Dict, Optional, Union

import numpy, math
from overrides import overrides
import torch
from torch import nn
import torch.nn.functional as F

import math, pdb, logging, time

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder, Maxout
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

logger = logging.getLogger(__name__) 

def convert_to_minutes(s):
    """Converts seconds to minutes and seconds"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


@Model.register("skim_lstm")
class SkimLSTM(Model):
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
                 pre_encode_feedforward: FeedForward,
                 big_dim: int,
                 small_dim: int,
                 # encoder: Seq2SeqEncoder,
                 # small_encoder: Seq2SeqEncoder,
                 gamma: float,
                 integrator: Seq2SeqEncoder,
                 integrator_dropout: float,
                 output_layer: Union[FeedForward, Maxout],
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SkimLSTM, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._num_classes = self.vocab.get_vocab_size("labels")

        #change line 97 and 161
        self._big_dim = big_dim
        self._small_dim = small_dim
        self._gamma = gamma
        self._global_training_step_n = 0
        # self._x = self._text_field_embedder #
        self._pre_encode_feedforward = pre_encode_feedforward
        self._x = self._pre_encode_feedforward # or text_field_embedder
        self.large_rnn = nn.LSTMCell(input_size=self._x.get_output_dim(),
                                     hidden_size=big_dim,
                                     bias=True)
        self.small_rnn = nn.LSTMCell(input_size=self._x.get_output_dim(),
                                     hidden_size=small_dim,
                                     bias=True)
        self._k = 2
        self._linear = nn.Linear(self._x.get_output_dim() + 2 * self._big_dim, self._k)
        self._softmax_layer = nn.Softmax(dim=-1)

        
        # self._encoder = encoder
        self._integrator = integrator
        # self._integrator = self.large_rnn
        
        self._self_attentive_pooling_projection = nn.Linear(
                self._integrator.get_output_dim(), 1)

        # self._self_attentive_pooling_projection = nn.Linear(
        #         self._big_dim, 1)


        self._output_layer = output_layer
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._integrator_dropout = nn.Dropout(integrator_dropout)
        self._loss_regularizer = 1e-10
        
        
        


        check_dimensions_match(text_field_embedder.get_output_dim(),
                               self._pre_encode_feedforward.get_input_dim(),
                               "text field embedder output dim",
                               "Pre-encoder feedforward input dim")
        check_dimensions_match(self._pre_encode_feedforward.get_output_dim(),
                               self._big_dim,
                               "Pre-encoder feedforward output dim",
                               "Large Encoder input dim")
        check_dimensions_match(self._big_dim,
                               self._integrator.get_input_dim(),
                               "Large Encoder output dim ",
                               "Integrator input dim")
        check_dimensions_match(self._integrator.get_output_dim() * 3,
                               self._output_layer.get_input_dim(),
                               "Integrator output dim * 3",
                               "Output layer input dim")
        # check_dimensions_match(self._big_dim * 3,
        #                        self._output_layer.get_input_dim(),
        #                        "big dim * 3",
        #                        "Output layer input dim")

        check_dimensions_match(self._output_layer.get_output_dim(),
                               self._num_classes,
                               "Output layer output dim",
                               "Number of classes.")


        ##skim-RNN
        # check_dimensions_match(self._big_dim * 3,
        #                        self._output_layer.get_input_dim(),
        #                        "Integrator output dim * 3",
        #                        "Output layer input dim")


        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def gumbel_softmax(self, x, tau = 1.0, theta = None):
        if self.training:
            u = torch.torch.rand(x.size(), dtype=x.dtype, layout=x.layout, device=x.device)
            g = -torch.log(-torch.log(u))
            tau_inverse = 1. / tau
            r_t = F.softmax(g * tau_inverse, -1)
            return r_t
        else:
            # import pdb
            # pdb.set_trace()
            if theta:
                # import pdb
                # pdb.set_trace()
                # print("theta: ", theta)
                Q_t = torch.ge(x[:,1], theta)
            else: Q_t = torch.argmax(x, dim=-1)
            return Q_t.float()

    def _initialize(self, batch_size, cell_size, is_cuda):
        init_cell =  torch.autograd.Variable(torch.Tensor(batch_size, cell_size).zero_())
        if torch.cuda.is_available() and is_cuda:
            init_cell = init_cell.cuda()
        return init_cell

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                theta: float = None) -> Dict[str, torch.Tensor]:
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
        if self.training: self._global_training_step_n += 1
        temprature_tao = max(0.5, math.exp(-1e-4*self._global_training_step_n))
        

        text_mask = util.get_text_field_mask(tokens).float()
        embedded_text = self._text_field_embedder(tokens)
        dropped_embedded_text = self._embedding_dropout(embedded_text)

        # print('dropped_embedded_text.size()', dropped_embedded_text.size())

        pre_encoded_text = self._pre_encode_feedforward(dropped_embedded_text)
        # encoded_tokens = self._encoder(pre_encoded_text, text_mask)

        #skim-RNN
        '''
        need to generate encoded_tokens rep of hidden state in batch x seq_len x dim 
        '''
        x = pre_encoded_text # or embedded_text
        # x= embedded_text
        mask = tokens["tokens"].ne(0) 
    
        batch_size = x.size()[0]

        # import pdb
        # pdb.set_trace()

        skimmed_words = 0
        total_words = 0


        p_ = []  # [batch, len, 2]
        h_ = []  # [batch, len, large_cell_size]

        h_t = self._initialize(batch_size, self._big_dim, x.is_cuda)
        h_s = self._initialize(batch_size, self._small_dim, x.is_cuda)
        h_l = self._initialize(batch_size, self._big_dim, x.is_cuda)
        c_t = self._initialize(batch_size, self._big_dim, x.is_cuda)
        c_s = self._initialize(batch_size, self._small_dim, x.is_cuda)
        c_l = self._initialize(batch_size, self._big_dim, x.is_cuda)

        last_save_time = time.time()
        time_needed = 0

        for t in range(x.size()[1]):

            x_t = x[:, t, :]
            mask_t = mask[:, t]
            mask_t[0]=1

            p_t = F.softmax(self._linear(torch.cat([x_t, h_t, c_t], dim=1)), -1)
            r_t = self.gumbel_softmax(p_t, temprature_tao, theta = theta).unsqueeze(1)


            

            if self.training:

                total_words +=1
                # skimmed_words+=1
                

                h_l, c_l = self.large_rnn(x_t, (h_t, c_t))
                
                if self._small_dim!=self._big_dim and t>0:
                    h_s, c_s = self.small_rnn(x_t, (h_t[:, :self._small_dim], c_t[:, :self._small_dim]))

                    h_tilde = torch.transpose(torch.stack(
                                    [h_l,
                                     torch.cat([h_s[:, :self._small_dim],
                                                h_t[:, self._small_dim:]],
                                               dim=1)
                                     ], dim=2), 1, 2)

                    c_tilde = torch.transpose(torch.stack(
                                    [c_l,
                                     torch.cat([c_s[:, :self._small_dim],
                                                c_t[:, self._small_dim:]],
                                               dim=1)
                                     ], dim=2), 1, 2)

                    h_t = torch.bmm(r_t, h_tilde).squeeze()
                    c_t = torch.bmm(r_t, c_tilde).squeeze()
                else:
                    h_t = h_l
                    c_t = c_l

                # import pdb
                # pdb.set_trace()

                h_.append(h_t)
                p_.append(p_t)

                


            else:
                # import pdb
                # pdb.set_trace()
                # We need a (batch x big_dim) sized rep for each t (column)
                h_t_temp = []
                c_t_temp = []

                temp = time.time()

                for btch in range(batch_size):
                    if mask_t[btch]!=0:
                        q = r_t[btch]
                        total_words +=1

                        #q==1  as paper
                        if q==0 or self._small_dim==self._big_dim:
                            h_l, c_l = self.large_rnn(x_t[btch].unsqueeze(0), (h_t[btch].unsqueeze(0), c_t[btch].unsqueeze(0)))
                            h_t_temp.append(h_l.squeeze(0))
                            c_t_temp.append(c_l.squeeze(0))
                        #q=2 as paper
                        else:
                            skimmed_words+=1
                            h_s, c_s = self.small_rnn(x_t[btch].unsqueeze(0), (h_t[btch,:self._small_dim].unsqueeze(0), c_t[btch, :self._small_dim].unsqueeze(0)))
                            h_s = h_s.squeeze(0)
                            c_s = c_s.squeeze(0)
                            h_t_temp.append(torch.cat([h_s, h_t[btch, self._small_dim:]]))
                            c_t_temp.append(torch.cat([c_s, c_t[btch, self._small_dim:]]))
                    else:
                        # import pdb
                        # pdb.set_trace()
                        h_t_temp.append(torch.zeros_like(h_t_temp[-1]))
                        c_t_temp.append(torch.zeros_like(c_t_temp[-1]))

                tm_nd_step = time.time() - temp
                time_needed = time_needed + tm_nd_step

                h_t = torch.stack(h_t_temp, dim=0)
                c_t = torch.stack(c_t_temp, dim=0)
                # import pdb
                # pdb.set_trace()

                h_.append(h_t)
                p_.append(p_t)

        
        # if not self.training: pdb.set_trace()

        nll_prob = 0
        if self.training: nll_prob=self._gamma * torch.mean(-torch.log(torch.stack(p_, dim=0)[:, :, 1])) #transpose not needed now after stack for the loss

        # import pdb
        # pdb.set_trace()

        #Classifier
        integrator_input = torch.transpose(torch.stack(h_, dim=0), 0,1)
        # import pdb
        # pdb.set_trace()
        # print('integrator_input.size(): ', integrator_input.size())
        integrated_encodings = self._integrator(integrator_input, text_mask)
        # integrated_encodings = integrator_input
        # print('integrated_encodings.size(): ', integrated_encodings.size())

        # Simple Pooling layers
        max_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, text_mask.unsqueeze(2), -1e7)
        max_pool = torch.max(max_masked_integrated_encodings, 1)[0]
        min_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, text_mask.unsqueeze(2), +1e7)
        min_pool = torch.min(min_masked_integrated_encodings, 1)[0]
        mean_pool = torch.sum(integrated_encodings, 1) / torch.sum(text_mask, 1, keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self._self_attentive_pooling_projection(
                integrated_encodings).squeeze(2)
        self_weights = util.masked_softmax(self_attentive_logits, text_mask)
        self_attentive_pool = util.weighted_sum(integrated_encodings, self_weights)
        # print("max_pool.size(): ", max_pool.size())

        pooled_representations = torch.cat([max_pool, min_pool, mean_pool], 1)
        # print('pooled_representations.size(): ', pooled_representations.size())
        pooled_representations_dropped = self._integrator_dropout(pooled_representations)
        # print('pooled_representations_dropped.size(): ', pooled_representations_dropped.size())

        logits = self._output_layer(pooled_representations_dropped)
        # print('logits.size(): ', logits.size())
        class_probabilities = F.softmax(logits, dim=-1)
        # print('class_probabilities.size(): ', class_probabilities.size())
        # exit()

        output_dict = {'logits': logits, 'class_probabilities': class_probabilities, 'rep': pooled_representations_dropped}
        if label is not None:
            loss = self.loss(logits, label.squeeze(-1)) 
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss + self._loss_regularizer*nll_prob
            output_dict["skimmed_words"] = skimmed_words
            output_dict['total_words'] = total_words
            output_dict['time_needed'] = time_needed
            
        
        # pdb.set_trace()
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["class_probabilities"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BiattentiveClassificationNetwork':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        embedding_dropout = params.pop("embedding_dropout")
        pre_encode_feedforward = FeedForward.from_params(params.pop("pre_encode_feedforward"))
        # encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))

        # small_encoder = Seq2SeqEncoder.from_params(params.pop("small_encoder"))
        big_dim = params.pop("big_dim")
        small_dim = params.pop("small_dim")
        gamma = params.pop("gamma")

        integrator = Seq2SeqEncoder.from_params(params.pop("integrator"))
        integrator_dropout = params.pop("integrator_dropout")

        output_layer_params = params.pop("output_layer")
        if "activations" in output_layer_params:
            output_layer = FeedForward.from_params(output_layer_params)
        else:
            output_layer = Maxout.from_params(output_layer_params)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   embedding_dropout=embedding_dropout,
                   pre_encode_feedforward=pre_encode_feedforward,
                   big_dim=big_dim,
                   small_dim=small_dim,
                   # encoder=encoder,
                   # small_encoder=small_encoder,
                   gamma=gamma,
                   integrator=integrator,
                   integrator_dropout=integrator_dropout,
                   output_layer=output_layer,
                   initializer=initializer,
                   regularizer=regularizer)
