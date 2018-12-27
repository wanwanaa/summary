import torch
import torch.utils.data as Data
import pickle
import numpy as np
from LSTM.autoencoder import Autoencoder
from LCSTS_char.data_utils import index2sentence
from LSTM.data import load_embeddings
import time

# embeddings
filename = 'DATA/result/glove_embeddings_300d.npy'
embeddings = load_embeddings(filename)

# model
if torch.cuda.is_available():
    model = Autoencoder(embeddings, 4000, 22, 300, 512).cuda()
else:
    model = Autoencoder(embeddings, 4000, 22, 300, 512)

optim = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.CrossEntropyLoss()


def save_model(model, epoch):
    filename = 'LSTM/models/model_' + str(epoch) + '.pkl'
    torch.save(model.state_dict(), filename)
    print('model save at ', filename)


def load_model():
    model = Autoencoder(embeddings, 4000, 22, 300, 512)
    model.load_state_dict(torch.load('LSTM/models/autoencoder/model_14.pkl', map_location='cpu'))
    return model


def train(epoch, train_data):
    for step, x in enumerate(train_data):
        if torch.cuda.is_available():
            x = x.cuda()
        y = model(x)
        if torch.cuda.is_available():
            a = list(x[0].cpu().numpy())
            b = list(torch.argmax(y[0], dim=1).cpu().numpy())
        else:
            a = list(x[0].numpy())
            b = list(torch.argmax(y[0], dim=1).numpy())
        x = x.view(-1)
        y = y.view(-1, 4000)
        loss = loss_func(y, x)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 200 == 0:
            # result = list(torch.argmax(y, dim=1).numpy())
            # print(index2sentence(result, idx2word))
            print(index2sentence(a, idx2word))
            print(index2sentence(b, idx2word))
            print('epoch:', epoch, '|step:', step, '|train_loss: %.4f' % loss.item())
            # save_model(model, step)
    save_model(model, epoch)


def test(model, test_data):
    r = []
    for step, x in enumerate(test_data):
        h = model.encoder(x.view(-1, 22))
        out = (torch.ones(x.size(0))*2)
        result = []
        for i in range(22):
            out = out.type(torch.LongTensor).view(-1, 1)
            out, h = model.decoder(out, h)
            out = torch.squeeze(model.output_layer(out))
            out = torch.nn.functional.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)
            result.append(out.numpy())
            # if out.item() == 3:
            #     break
            # out = model.embeddings(out)
        result = np.transpose(np.array(result))
        for i in range(result.shape[0]):
            # sen1 = index2sentence(list(x[i]), idx2word)
            sen = index2sentence(list(result[i]), idx2word)
            r.append(''.join(sen))
    with open('DATA/result/autoencoder_result.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(r))


if __name__ == '__main__':
    start = time.time()
    filename_train = 'DATA/result/train_summary.npy'
    data_train = np.load(filename_train)
    train_data = Data.DataLoader(torch.LongTensor(data_train), 128, shuffle=True, num_workers=2)

    filename_test = 'DATA/result/test_summary.npy'
    data_test = np.load(filename_test)
    test_data = Data.DataLoader(torch.LongTensor(data_test), 128, shuffle=False, num_workers=2)

    # idx2word
    f = open('DATA/result/index2word.pkl', 'rb')
    idx2word = pickle.load(f)

    model = load_model()
    test(model, test_data)

    # for epoch in range(20):
    #     train(epoch, train_data)
    #     # test(epoch, test_data)
    end = time.time()
    print(end-start)