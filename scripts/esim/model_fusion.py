"""
Definition of the ESIM model.
"""
# Aurelien Coet, 2018.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention, SoftmaxAttention_Decompose
from .utils import get_mask, replace_masked


class ESIM_f(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu",
                 embedding_dim_other=50):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(ESIM_f, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_dim_other = embedding_dim_other
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings,
                                            )
        self._word_embedding.weight.data.requires_grad = False

        self._word_embedding_other = nn.Embedding(self.vocab_size,
                                                  self.embedding_dim_other,
                                                  padding_idx=padding_idx)

        self.fusion_w1 = nn.Linear(self.embedding_dim + self.embedding_dim_other, self.hidden_size * 2)
        self.fusion_w2 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim + self.embedding_dim_other,
                                        self.hidden_size,
                                        bidirectional=True)

        #self._attention = SoftmaxAttention()
        self._attention_relate = SoftmaxAttention_Decompose()

        self._projection_1 = nn.Sequential(nn.Linear(1*2*self.hidden_size,
                                                   self.hidden_size),
                                                     nn.ReLU())
        self._projection_2 = nn.Sequential(nn.Linear(1*2*self.hidden_size,
                                                      self.hidden_size),
                                                      nn.ReLU())

        self._composition_1 = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)
        self._composition_2 = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self.fusion_w3 =  nn.Linear(2 * 4 * self.hidden_size, 2 * 4 * self.hidden_size)
        self.fusion_w4 =  nn.Linear(2 * 4 * self.hidden_size, 2 * 4 * self.hidden_size)

        self.fusion_w5 =  nn.Linear(2 * 1 * self.hidden_size, 2 * 1 * self.hidden_size)
        self.fusion_w6 =  nn.Linear(2 * 1 * self.hidden_size, 2 * 1 * self.hidden_size)

        self.fusion_w7 =  nn.Linear(2 * 1 * self.hidden_size, 2 * 1 * self.hidden_size)
        self.fusion_w8 =  nn.Linear(2 * 1 * self.hidden_size, 2 * 1 * self.hidden_size)


        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths)\
            .to(self.device)

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)
        embedded_premises_other = self._word_embedding_other(premises)
        embedded_hypotheses_other = self._word_embedding_other(hypotheses)
        embedded_hypotheses = torch.cat((embedded_hypotheses, embedded_hypotheses_other), -1)
        embedded_premises = torch.cat((embedded_premises, embedded_premises_other), -1)

        fusion_hypotheses_1 = self.fusion_w1(embedded_hypotheses)
        fusion_premises_1 = self.fusion_w1(embedded_premises)


        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)

        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)

        # fusion
        fusion_hypotheses_2 = self.fusion_w2(encoded_hypotheses)
        fusion_premises_2 = self.fusion_w2(encoded_premises)

        h_size = fusion_hypotheses_2.size(1)
        p_size = fusion_premises_2.size(1)

        fusion_hypotheses_1 = fusion_hypotheses_1[:,:h_size,:]
        fusion_premises_1 = fusion_premises_1[:,:p_size,:]

        fusion_h = F.sigmoid(fusion_hypotheses_1 + fusion_hypotheses_2)
        fusion_p = F.sigmoid(fusion_premises_1 + fusion_premises_2)
        encoded_hypotheses = fusion_h * fusion_hypotheses_1 + (1 - fusion_h) * fusion_hypotheses_2
        encoded_premises = fusion_p * fusion_premises_1 + (1 - fusion_p) * fusion_premises_2

        attended_premises, attended_hypotheses , attended_premises_un, attended_hypotheses_un =\
            self._attention_relate(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)
        relate = self.realte_unralte(encoded_premises, attended_premises, encoded_hypotheses, attended_hypotheses,
                                     premises_lengths, hypotheses_lengths, premises_mask, hypotheses_mask,
                                     self._projection_1, self._composition_1, self.fusion_w5, self.fusion_w6)
        unrelate = self.realte_unralte(encoded_premises, attended_premises_un, encoded_hypotheses,
                                       attended_hypotheses_un,
                                     premises_lengths, hypotheses_lengths, premises_mask, hypotheses_mask,
                                       self._projection_2, self._composition_2, self.fusion_w7, self.fusion_w8)

        relate_fu = self.fusion_w3(relate)
        unrelate_fu = self.fusion_w4(unrelate)
        fusion = F.sigmoid(relate_fu + unrelate_fu)
        relation = fusion * relate + (1-fusion) * unrelate
        #relation = torch.cat((relate, unrelate), -1)

        logits = self._classification(relation)
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities

    def realte_unralte(self, encoded_premises, attended_premises, encoded_hypotheses, attended_hypotheses,
                       premises_lengths, hypotheses_lengths, premises_mask, hypotheses_mask,
                       projection, composition, fusion_1, fusion_2):

        fusion_pre = fusion_1(encoded_premises) + fusion_2(attended_premises)
        fusion_ = F.sigmoid(fusion_pre)
        enhanced_premises = fusion_ * attended_premises + (1-fusion_) * encoded_premises

        fusion_hyp = fusion_1(encoded_hypotheses) + fusion_2(attended_hypotheses)
        fusion_ = F.sigmoid(fusion_hyp)
        enhanced_hypotheses = fusion_ * attended_hypotheses + (1-fusion_) * encoded_hypotheses

        '''
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)
        '''
        projected_premises = projection(enhanced_premises)
        projected_hypotheses = projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = composition(projected_premises, premises_lengths)
        v_bj = composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)
        return v

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
