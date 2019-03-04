import torch
import math


def get_prob(model, y, hidden, encoder_outputs, attention):
    if attention:
        out = (torch.ones(1) * y).type(torch.LongTensor) # batch
    else:
        out = (torch.ones(1) * y).type(torch.LongTensor).view(1, -1)
    out, h = model.decoder(out, hidden, encoder_outputs)
    out = model.output_layer(out).squeeze()
    data = torch.nn.functional.softmax(out)  # batch dim=1
    return data, h


def max_prob(candidate):
    v = 0
    p = 0 # position of max
    for i in range(len(candidate)):
        if candidate[i][-1] > v:
            v = candidate[i][-1]
            p = i
    return p


# beam search
def beam_search(model, hidden, encoder_outputs, s_len, bos, beam_size, attention):
    """
    :param model:
    :param hidden: encoder hidden state
    :param s_len: summary length
    :param bos:
    :param beam_size:
    :return:
    """
    # init
    path = [[[bos], hidden, 0.0]]
    for i in range(s_len):
        candidate = []
        for j in range(len(path)):
            data, hidden = get_prob(model, path[j][0][-1], path[j][1], encoder_outputs, attention)
            sorted, indices = torch.sort(data, descending=True)
            pre_path = path[j][0]
            pre_prob = path[j][-1]
            for k in range(beam_size):
                p = pre_path.copy()
                p.append(int(indices[k]))
                prob = math.log(sorted[k]) + pre_prob
                candidate.append([p, hidden, prob])
        path = []
        if i == s_len - 1:
            r = candidate.pop(max_prob(candidate))
            return r[0]
        else:
            for w in range(beam_size):
                path.append(candidate.pop(max_prob(candidate)))


# class beam():
#     def __init__(self, model, s_len, bos, beam_size):
#         self.model = model
#         self.s_len = s_len # summary length
#         self.bos = bos
#         self.beam_size = beam_size
#
#         hyp = [] # The path at each time step
#         scorce = [] # The score for each translation on the beam
#
#     def get_prob(self, y, hidden):
#
#
#     def beam_search(self, hidden):
#         # init
#         path = [[[self.bos], 0.0]]
#         for i in range(self.s_len):
#             candidate = []
#             for j in range(len(path)):
#                 data = self.get_prob(path[j][-1], hidden) # encoder hidden state
