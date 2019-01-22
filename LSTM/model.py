import torch
import torch.nn as nn
import torch.nn.functional as F


# encoding the sentences
class Encoder(nn.Module):
    def __init__(self, embeds, embedding_dim, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embeds = embeds
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder_gru = nn.GRU(self.embedding_dim,
                                  self.hidden_size,
                                  batch_first=True,
                                  bidirectional=True,
                                  num_layers=self.num_layers)

    def forward(self, x):
        """
        :param x: (batch, t_len)
        :return: h(batch, n_layer, hidden_size)
                  out(batch, t_len, hidden_size) hidden state of gru
        """
        e = self.embeds(x)

        # out (batch, time_step, hidden_size*bidirection)
        # h (batch, n_layers*bidirection, hidden_size)
        out, h = self.encoder_gru(e)

        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        h = h[:self.num_layers]

        # h = h[:self.num_layers] + h[self.num_layers:]
        return h, out


# pointer-generator
# calculate generation probability
class Pointer(nn.Module):
    def __init__(self, embeds, embedding_size, hidden_size):
        super(Pointer, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embeds = embeds

        # prob
        self.p_gen = nn.Linear(embedding_size+hidden_size*2, 1)

    def forward(self, context, h, x):
        """
        :param context: (batch, hidden_size) attention vector
        :param h: (batch, hidden_size) decoder state
        :param x: (batch, embedding_size) decoder input
        :return: generation probability
        """
        x = self.embeds(x)
        context = context.squeeze()
        prob = self.p_gen(torch.cat((context, h[-1], x), dim=1))  # (batch, 1)
        return F.sigmoid(prob)


# calculate attention weights and return context vector
# coverage mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size, t_len, is_coverage):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.t_len = t_len
        self.is_coverage = is_coverage

        # attention
        self.attn1 = nn.Linear(self.hidden_size*2, 1)
        # coverage mechanism
        if is_coverage:
            self.attn2 = nn.Linear(self.hidden_size*2+t_len, 1)

    def forward(self, hidden, encoder_out, cover_vector):
        """
        :param hidden: (n_layers, batch, hidden_size) decoder hidden state
        :param encoder_out: (batch, time_step, hidden_size) encoder gru hidden state
        :param cover_vector: (batch, 1, time_step)
        :return: attn_weight (batch, 1, time_step)
                  context (batch, 1, hidden_size) attention vector
        """

        # hidden (n_layers, batch, hidden_size) -> (1, batch, hidden_size)
        #     -> (batch, 1, hidden_size) -> (batch, time_step, hidden_size)
        hidden = hidden[-1].view(-1, 1, self.hidden_size).repeat(1, encoder_out.size(1), 1)

        # attn_weights (batch, 1, time_step)
        # (batch. time_step, 1) -> (batch, 1, time_step)
        if self.is_coverage:
            cover_vector = cover_vector.repeat(1, encoder_out.size(1), 1)
            vector = torch.cat((hidden, encoder_out, cover_vector), dim=2)
            attn_weights = self.attn2(F.tanh(vector)).squeeze(2)
            attn_weights = F.softmax(attn_weights).unsqueeze(1)
        else:
            vector = torch.cat((hidden, encoder_out), dim=2)
            attn_weights = self.attn1(F.tanh(vector)).squeeze(2)
            attn_weights = F.softmax(attn_weights).unsqueeze(1)

        context = torch.bmm(attn_weights, encoder_out)  # (batch, 1, hidden_size)
        return attn_weights, context


class AttnDecoder(nn.Module):
    def __init__(self, attention, embeds, vocab_size, embedding_dim, hidden_size, s_len, num_layers):
        super(AttnDecoder, self).__init__()
        self.attention = attention
        self.embeds = embeds
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.s_len = s_len
        self.num_layers = num_layers

        # decoder
        self.decoder_gru = nn.GRU(self.embedding_dim+self.hidden_size,
                                  self.hidden_size,
                                  batch_first=True,
                                  num_layers=num_layers)
        self.decoder_vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, h, encoder_output, cover_vector):
        """
        :param x: (batch, 1) decoder input
        :param h: (batch, n_layer, hidden_size) code
        :param encoder_output: (batch, t_len, hidden_size) encoder gru hidden state
        :param cover_vector: (batch, 1, hidden_size) coverage vector
        :return: attn_weight (batch, 1, time_step)
                  context (batch, 1, hidden_size)
                  out (batch, hidden_size) decoder output
                  h (batch, n_layer, hidden_size) decoder hidden state
        """
        e = self.embeds(x).unsqueeze(1)
        attn_weight, context = self.attention(h, encoder_output, cover_vector)
        inputs = torch.cat((e, context), dim=2)
        out, h = self.decoder_gru(inputs, h)
        return attn_weight, context, out, h


class AttnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, hidden_size, t_len, s_len, bos, pointer, is_coverage):
        super(AttnSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.t_len = t_len
        self.s_len = s_len
        self.pointer = pointer
        self.is_coverage = is_coverage

        # <bos>
        self.bos = bos

        self.output_Linear = nn.Linear(self.hidden_size, self.vocab_size)

    def output_layer(self, x):
        """
        :param x: (batch, hidden_size) decoder output
        :return: (batch, vocab_size)
        """
        return self.output_Linear(x)

    # add <bos> to sentence
    def convert(self, x):
        """
        :param x:(batch, s_len) (word_1, word_2, ... , word_n)
        :return:(batch, s_len) (<bos>, word_1, ... , word_n-1)
        """
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x[:, :-1]

    # pointer-generate
    # calculate the final distribution by generator probabilities and context
    def final_distribution(self, attn_weight, x, gen, prob):
        """
        :param attn_weight: (batch, 1, time_step)
        :param x: (batch, t_len) encoder input
        :param gen: (batch, vocab_size) vocabulary distribution
        :param prob: generate probability
        :return: output (batch, vocab_size) final distribution
        """
        if torch.cuda.is_available():
            output = torch.zeros((x.size(0), self.vocab_size)).type(torch.cuda.FloatTensor)
        else:
            output = torch.zeros((x.size(0), self.vocab_size)).type(torch.FloatTensor)

        # add generator probabilities to output
        gen = F.softmax(gen, dim=1)  # (batch, vocab_size)
        output[:, :self.vocab_size] = prob * gen

        # add pointer probabilities to output
        # ptr (batch, t_len)
        ptr = attn_weight.squeeze()
        output.scatter_add_(1, x, (1-prob) * ptr)
        return output

    # coverage loss
    def cover_loss(self, attn_weights, cover_vector):
        loss = 2*torch.sum(torch.min(attn_weights, cover_vector))/attn_weights.size(0)
        return loss

    def forward(self, x, y):
        """
        :param x: (batch, t_len) encoder input
        :param y: (batch, s_len) decoder input
        :return: loss: coverage loss
                  outputs (batch, t_len, vocabulary)
        """
        h, encoder_out = self.encoder(x)
        # add <bos>
        y = self.convert(y)
        result = []
        # initialize cover vector
        if self.is_coverage:
            if torch.cuda.is_available():
                cover_vector = torch.zeros((x.size(0), 1, self.t_len)).type(torch.cuda.FloatTensor)
                loss = torch.zeros((1,), requires_grad=True).type(torch.cuda.FloatTensor)
            else:
                cover_vector = torch.zeros((x.size(0), 1, self.t_len)).type(torch.FloatTensor)
                loss = torch.zeros((1,), requires_grad=True).type(torch.FloatTensor)
        else:
            cover_vector = None
            loss = None
        # decoder
        for i in range(y.size(1)):
            if self.is_coverage:
                attn_weights, context, out, h = self.decoder(y[:, i], h, encoder_out, cover_vector)
                gen = self.output_layer(out).squeeze()
                loss = loss + self.cover_loss(attn_weights, cover_vector)

                # the sum of attention distributions over all previous decoder time steps
                cover_vector = cover_vector + attn_weights

            else:
                attn_weights, context, out, h = self.decoder(y[:, i], h, encoder_out, cover_vector)
                gen = self.output_layer(out).squeeze()

            if self.pointer:
                # calculate generation probability
                prob = self.pointer(context, h, y[:, i])
                final = self.final_distribution(attn_weights, x, gen, prob)
                result.append(torch.log(final))
            else:
                result.append(gen)

        outputs = torch.stack(result)
        if self.is_coverage:
            loss = loss/self.s_len
        return loss, torch.transpose(outputs, 0, 1)


# embedding
class Embeds(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size):
        super(Embeds, self).__init__()
        if embeddings is None:
            self.embeds = nn.Embedding(vocab_size, embedding_size)
        else:
            self.embeds = nn.Embedding.from_pretrained(embeddings)

    def forward(self, x):
        """
        :param x: (batch, t_len)
        :return: (batch, t_len, embedding_size)
        """
        return self.embeds(x)


class Decoder(nn.Module):
    def __init__(self, embeds, vocab_size, embedding_dim, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embeds = embeds
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # decoder
        self.decoder_gru = nn.GRU(self.embedding_dim,
                                  self.hidden_size,
                                  batch_first=True,
                                  num_layers=num_layers)
        self.decoder_vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, h):
        e = self.embeds(x)
        out, h = self.decoder_gru(e, h)
        return out, h


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, hidden_size, bos):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # <bos>
        self.bos = bos

        self.output_Linear = nn.Linear(self.hidden_size, self.vocab_size)

    def convert(self, x):
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x[:, :-1]

    def output_layer(self, x):
        input = x.contiguous().view(-1, self.hidden_size)
        output = self.output_Linear(input)
        return output.view(-1, x.size(1), self.vocab_size)

    def forward(self, x, y):
        code, _ = self.encoder(x)
        y = self.convert(y)
        out, h = self.decoder(y, code)
        out = self.output_layer(out)
        return out
