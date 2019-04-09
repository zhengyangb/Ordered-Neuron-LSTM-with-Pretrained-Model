import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
from ON_LSTM import ONLSTMStack
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

class OpenAIGPTONLSTMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model_embeddings_weights, args):
        super(OpenAIGPTONLSTMHead, self).__init__()
        self.n_embd = args.emsize
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits

class GPTRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, chunk_size, nlayers,  dropout=0.5, dropouth=0.5, dropouti=0.5,
                 dropoute=0.1, wdrop=0, tie_weights=False, ):
        super(GPTRNNModel, self).__init__()
        self.transformer = OpenAIGPTModel.from_pretrained('openai-gpt')
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Linear(768, ninp)

        assert rnn_type in ['LSTM'], 'RNN type is not supported'
        self.rnn = ONLSTMStack(
            [ninp] + [nhid] * (nlayers - 1) + [ninp],
            chunk_size=chunk_size,
            dropconnect=wdrop,
            dropout=dropouth
        )
        self.decoder = nn.Linear(ninp, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # if tie_weights:
        #     #if nhid != ninp:
        #     #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder.weight = self.encoder.weight
        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights



    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self, pre_emb):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        # if pre_emb is not None:
        #     self.encoder.weight.data[:pre_emb.size(0), :pre_emb.size(1)] = torch.FloatTensor(pre_emb)

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, gpt_ids, fl_ids, return_h=False):
        emb = self.transformer(gpt_ids)  # BS * GPT_SL * GPT_EMS
        emb = torch.cat([emb[r, fl_ids[r], :].unsqueeze(0) for r in range(len(fl_ids))], dim=0)  # BS * (2*SL) * GPT_ES
        emb = torch.nn.functional.avg_pool1d(emb.permute(0, 2, 1), 2)
        emb = emb * 2  # BS * GPT_EMS * SL
        emb = emb.permute(2, 0, 1)  # BS * SL * GPT_EMS
        emb = self.encoder(emb)  # Goal: SL * BS * ES as in ONLSTM
        # emb = embedded_dropout(
        #     self.encoder, input,
        #     dropout=self.dropoute if self.training else 0
        # )
        #
        # emb = self.lockdrop(emb, self.dropouti)
        raw_output, hidden, raw_outputs, outputs, distances = self.rnn(emb, hidden)
        self.distance = distances

        output = self.lockdrop(raw_output, self.dropout)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        else:
            return result, hidden

    def init_hidden(self, bsz):
        return self.rnn.init_hidden(bsz)
