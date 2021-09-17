"""
Implementation of "Attention is All You Need"
"""
import torch
import torch.nn as nn

import onmt
from onmt.encoders.encoder import EncoderBase
# from onmt.utils.misc import aeq
from onmt.modules.position_ffn import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        super(TransformerEncoderLayer, self).__init__()

        self.doc_len = 200
        self.his_num = 3
        self.utt_len = 50
        self.his_len = self.his_num * self.utt_len
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.avg_pool = nn.AvgPool1d(self.utt_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, his, doc=None, lengths=None, knl_lengths=None):
        # process his and doc embeddings
        emb_his = self.embeddings(his)
        emb_his = emb_his.transpose(0, 1).contiguous()
        words_his = his[:, :, 0].transpose(0, 1)
        w_batch, w_len = words_his.size()
        padding_idx_his = self.embeddings.word_padding_idx
        mask_his = words_his.data.eq(padding_idx_his).unsqueeze(1)  # [B, 1, T]
        emb_doc = self.embeddings(doc)
        emb_doc = emb_doc.transpose(0, 1).contiguous()
        words_doc = doc[:, :, 0].transpose(0, 1)
        padding_idx_doc = self.embeddings.word_padding_idx
        mask_doc = words_doc.data.eq(padding_idx_doc).unsqueeze(1)  # [B, 1, T]

        # his_word, his_utt = SA(his)
        his_norm = self.layer_norm1(emb_his)
        his_word, _ = self.self_attn(his_norm, his_norm, his_norm, mask=mask_his)
        his_word = self.dropout(his_word) + emb_his
        his_utt = self.avg_pool(his_word.transpose(1, 2)).transpose(1, 2)

        # doc_word = SA(doc)
        doc_norm = self.layer_norm2(emb_doc)
        doc_word, _ = self.self_attn(doc_norm, doc_norm, doc_norm, mask=mask_doc)
        doc_word = self.dropout(doc_word) + emb_doc

        # doc' = cross_attention(his,doc)
        # ?? use layer norm and dropout ?
        score_word = torch.matmul(doc_word, his_word.transpose(1, 2))
        score_utt = torch.matmul(doc_word, his_utt.transpose(1, 2))
        n_score_utt = score_utt.reshape(w_batch, self.doc_len, self.his_num, 1).repeat((1, 1, 1, self.utt_len)).reshape(w_batch, self.doc_len, self.his_len)
        # ?? use softmax?
        score = score_word * n_score_utt
        n_doc = torch.matmul(score, his_word)

        return his_word, n_doc


class TransformerEncoder(EncoderBase):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):

        super(TransformerEncoder, self).__init__()
        self.transformer = TransformerEncoderLayer(num_layers, d_model, heads, d_ff,
                 dropout, embeddings)

    def forward(self, his, doc=None, lengths=None, knl_lengths=None):

        # ?? only one layer ?
        his_word, n_doc = self.transformer(his, doc)

        return his_word, n_doc

