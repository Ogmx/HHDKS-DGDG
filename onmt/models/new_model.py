""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)

        return dec_out, attns


class ConvModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ConvModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, knl, src, tgt, src_lengths, knl_lengths):
        tgt = tgt[:-1]
        enc_state, memory_bank, lengths = self.encoder(src, src_lengths)
        knowledge_encoding, _, _ = self.encoder(knl, knl_lengths)
        self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      knowledge_encoding=knowledge_encoding,
                                      memory_lengths=src_lengths)
        return dec_out, attns


class KTransformerModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(KTransformerModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, knl, src, tgt, src_lengths, knl_lengths):
        tgt = tgt[:-1]
        enc_state, his_bank, src_bank, knl_bank, lengths = self.encoder(src, knl, src_lengths, knl_lengths)

        self.decoder.init_state(his_bank, knl[600:, :, :], None, None)
        # emb, decode1_bank, decode1_mask = self.encoder.histransformer(first_dec_words, None)
        dec_out, attns = self.decoder(tgt, his_bank, knl_bank, memory_lengths=None)     # his + doc k+1
        
        return dec_out, attns

