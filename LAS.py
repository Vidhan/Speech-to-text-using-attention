import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn

start_char = '0'
pyramid_layers = 3
vocab_size = 33
mel_freq = 40

batch = 32 
epochs = 25
key_dimension = 128
value_dimension = 128
e_hidden_dimension = 300
hidden_dimension = 300
embedding_dimension = 400


def to_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def to_tensor(numpy_array):
    return torch.from_numpy(numpy_array).float()


class MyCustomDataset(Dataset):

    def __init__(self, features, transcripts, vocab):
        self.features = features
        self.transcripts = transcripts
        self.vocab = vocab

    def __getitem__(self, index):
        X = self.features[index]
        Y = self.transcripts[index]
        array = [self.vocab.index(character) for character in Y]
        Y_x = [self.vocab.index(start_char)] + array
        Y_y = array + [self.vocab.index(start_char)]
        return (X, Y_x, Y_y)

    def __len__(self):
        return len(self.transcripts)


class MyDataLoader():

    def __call__(self, batch):

        batch_new = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

        X = [x[0] for x in batch_new]
        X = [x if x.shape[0] % (2**pyramid_layers) == 0
             else x[: -(x.shape[0] % (2**pyramid_layers)), :] for x in X]
        Y_x = [np.asarray(x[1]) for x in batch_new]
        Y_y = [np.asarray(x[2]) for x in batch_new]

        X_sizes = [x.shape[0] for x in X]  # list length = batch_size
        Y_sizes = [x.shape[0] for x in Y_x]  # list length = batch_size

        max_Y = max(Y_sizes)
        Y_x = [np.pad(x, (0, max_Y - x.shape[0]), mode='constant')
               for x in Y_x]
        Y_y = [np.pad(x, (0, max_Y - x.shape[0]), mode='constant')
               for x in Y_y]

        Y_x = np.dstack(Y_x).squeeze()
        Y_y = np.dstack(Y_y).squeeze()
        Y_x = to_variable(to_tensor(Y_x)).long()  # max_transcript X batch_size
        Y_y = to_variable(to_tensor(Y_y)).long()  # max_transcript X batch_size

        X = [np.pad(x, [(0, X_sizes[0] - x.shape[0]), (0, 0)], mode='constant')
             for x in X]
        X = np.dstack(X)
        X = np.swapaxes(X, 1, 2)
        X = to_variable(to_tensor(X))  # max_utterance X batch_size X mel_freq

        return pack_padded_sequence(X, X_sizes), Y_x, Y_y, Y_sizes, X.shape[1]


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.h_0_1 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.c_0_1 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.h_0_2 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.c_0_2 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.h_0_3 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.c_0_3 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.h_0_4 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.c_0_4 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())

        self.rnn1 = nn.LSTM(input_size=mel_freq,
                            hidden_size=e_hidden_dimension, num_layers=1,
                            bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=4 * e_hidden_dimension,
                            hidden_size=e_hidden_dimension, num_layers=1,
                            bidirectional=True)
        self.rnn3 = nn.LSTM(input_size=4 * e_hidden_dimension,
                            hidden_size=e_hidden_dimension, num_layers=1,
                            bidirectional=True)
        self.rnn4 = nn.LSTM(input_size=4 * e_hidden_dimension,
                            hidden_size=e_hidden_dimension, num_layers=1,
                            bidirectional=True)
        self.projection1 = nn.Linear(in_features=2 * e_hidden_dimension,
                                     out_features=key_dimension)
        self.projection2 = nn.Linear(in_features=2 * e_hidden_dimension,
                                     out_features=value_dimension)

    def forward(self, h, batch):
        h_0_1 = self.h_0_1.expand(-1, batch, -1).contiguous()
        c_0_1 = self.c_0_1.expand(-1, batch, -1).contiguous()
        h_0_2 = self.h_0_2.expand(-1, batch, -1).contiguous()
        c_0_2 = self.c_0_2.expand(-1, batch, -1).contiguous()
        h_0_3 = self.h_0_3.expand(-1, batch, -1).contiguous()
        c_0_3 = self.c_0_3.expand(-1, batch, -1).contiguous()
        h_0_4 = self.h_0_4.expand(-1, batch, -1).contiguous()
        c_0_4 = self.c_0_4.expand(-1, batch, -1).contiguous()
        h, state = self.rnn1(h, (h_0_1, c_0_1))
        pad_array, seq_length = pad_packed_sequence(sequence=h,
                                                    padding_value=0,
                                                    batch_first=False)

        pad_array = torch.transpose(pad_array, 0, 1)
        sizes = list(pad_array.size())
        pad_array = pad_array.contiguous().view(sizes[0], int(sizes[1] / 2),
                                                -1)
        pad_array = torch.transpose(pad_array, 0, 1)
        seq_length = [int(x / 2) for x in seq_length]

        h = pack_padded_sequence(pad_array, seq_length)
        h, state = self.rnn2(h, (h_0_2, c_0_2))
        pad_array, seq_length = pad_packed_sequence(sequence=h,
                                                    padding_value=0,
                                                    batch_first=False)

        pad_array = torch.transpose(pad_array, 0, 1)
        sizes = list(pad_array.size())
        pad_array = pad_array.contiguous().view(sizes[0], int(sizes[1] / 2),
                                                -1)
        pad_array = torch.transpose(pad_array, 0, 1)
        seq_length = [int(x / 2) for x in seq_length]

        h = pack_padded_sequence(pad_array, seq_length)
        h, state = self.rnn3(h, (h_0_3, c_0_3))
        pad_array, seq_length = pad_packed_sequence(sequence=h,
                                                    padding_value=0,
                                                    batch_first=False)

        pad_array = torch.transpose(pad_array, 0, 1)
        sizes = list(pad_array.size())
        pad_array = pad_array.contiguous().view(sizes[0], int(sizes[1] / 2),
                                                -1)
        pad_array = torch.transpose(pad_array, 0, 1)
        seq_length = [int(x / 2) for x in seq_length]

        h = pack_padded_sequence(pad_array, seq_length)
        h, state = self.rnn4(h, (h_0_4, c_0_4))
        pad_array, seq_length = pad_packed_sequence(sequence=h,
                                                    padding_value=0,
                                                    batch_first=False)

        key = self.projection1(pad_array)  # (utterance_l, batch, key)
        value = self.projection2(pad_array)  # (utterance_l, batch, value)

        return key, value


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.phi = nn.Linear(in_features=hidden_dimension,
                             out_features=key_dimension)
        self.softmax = nn.Softmax(dim=-1)
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dimension)
        self.h0 = nn.Parameter(torch.FloatTensor(1, hidden_dimension).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(1, hidden_dimension).zero_())
        self.lstmCell = nn.LSTMCell(input_size=embedding_dimension
                                    + value_dimension,
                                    hidden_size=hidden_dimension)
        self.projection = nn.Linear(in_features=hidden_dimension
                                    + value_dimension,
                                    out_features=vocab_size)

    # input should be #timesteps X #batch_size
    # keys should be # (utterance length, batch size, key dimension)
    # values should be # (utterance length, batch size, value dimension)
    def forward(self, input, keys, values):
        h = self.h0.expand(input.shape[1], -1)
        c = self.c0.expand(input.shape[1], -1)
        keys = keys.permute(1, 2, 0)  # batch, key, utterance_length
        values = values.permute(1, 0, 2)  # bath, utterance_length, value

        query = self.phi(h)  # (batch_size, key_dimension)
        query = torch.unsqueeze(query, dim=1)  # (batch, 1,  key)
        energy = torch.bmm(query, keys)  # (batch_size, 1, utter_length)
        attention = self.softmax(energy)  # (batch_size, 1, utter_length)
        context = torch.bmm(attention, values)  # batch_size X 1 X value
        context = torch.squeeze(context)  # batch_size X value_dimension

        logit_list = []
        for step in range(input.shape[0]):  # timesteps
            character_input = input[step:step + 1, :]
            character = self.embedding(character_input)  # 1 X batch X embed
            character = torch.squeeze(character)  # batch_size X embed
            character = torch.cat((character, context), dim=-1)

            h, c = self.lstmCell(character, (h, c))

            logits = self.projection(torch.cat((h, context), dim=-1))
            logit_list.append(logits)

            query = self.phi(h)  # (batch_size, key_dimension)
            query = torch.unsqueeze(query, dim=1)  # (batch_size, 1,  key)
            energy = torch.bmm(query, keys)  # (batch_size, 1, utter_length)
            attention = self.softmax(energy)  # (batch_size, 1, utter_length)
            context = torch.bmm(attention, values)  # batch_size X 1 X value
            context = torch.squeeze(context)  # batch_size X value_dimension

        logit_list = torch.stack(logit_list)  # timesteps X batch_size X vocab
        return logit_list


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, X, batch_size, Y_x):
        keys, values = self.encoder(X, batch_size)
        logits = self.decoder(Y_x, keys, values)
        return logits


def train():

    features = np.load('train.npy')  # utterances
    transcripts = np.load('train_transcripts.npy')  # transcipts

    features_valid = np.load('dev.npy')
    transcripts_valid = np.load('dev_transcripts.npy')

    s = set(start_char)
    for transcript in transcripts:
        s = s | set(transcript)

    vocab = sorted(s)
    #np.save("vocab.npy", vocab)

    model = Model()
    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    #model.load_state_dict(torch.load('model_params19.pt'))

    dataset = MyCustomDataset(features, transcripts, vocab)
    dataset_valid = MyCustomDataset(features_valid, transcripts_valid, vocab)
    data_loader = DataLoader(
        dataset, batch_size=batch, shuffle=True, collate_fn=MyDataLoader())
    data_loader_valid = DataLoader(
        dataset_valid, batch_size=batch, shuffle=True,
        collate_fn=MyDataLoader())

    print ("Training begins")
    for epoch in range(0, epochs):
        losses = []
        model.train()

        # X_x and X_y => transcript_l X batch 
        for (X, Y_x, Y_y, Y_sizes, batch_size) in data_loader:
            optim.zero_grad()
            logits = model(X, batch_size, Y_x)  # L X N X V
            Y_sizes = torch.Tensor(Y_sizes).view(1, -1)  # 1 X N

            mask = torch.arange(1, Y_y.shape[0] + 1).view(-1, 1)  # L X 1
            mask = mask <= Y_sizes  # L X N

            logits = logits.view(-1, vocab_size)  # (L*N, vocab_size)
            Y_y = Y_y.view(-1, 1)  # (L*N, 1)
            mask = to_variable(mask.view(-1, 1))  # (L*N, 1)

            preds = torch.masked_select(logits, mask).view(-1, vocab_size)  # (No. of 1s, vocab_size)
            targets = torch.masked_select(Y_y, mask)

            loss = loss_fn(preds, targets)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            optim.step()

        torch.save(model.state_dict(), 'model3_params' + str(epoch) + '.pt')
        print("Epoch {} Training Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses))))

        losses_valid = []
        model.eval()
        for (X, Y_x, Y_y, Y_sizes, batch_size) in data_loader_valid:

            logits = model(X, batch_size, Y_x)  # L X N X V
            Y_sizes = torch.Tensor(Y_sizes).view(1, -1)  # 1 X N

            mask = torch.arange(1, Y_y.shape[0] + 1).view(-1, 1)  # L X 1
            mask = mask <= Y_sizes  # L X N

            logits = logits.view(-1, vocab_size)  # (L*N, vocab_size)
            Y_y = Y_y.view(-1, 1)  # (L*N, 1)
            mask = to_variable(mask.view(-1, 1))  # (L*N, 1)

            preds = torch.masked_select(logits, mask).view(-1, vocab_size)  # (No. of 1s, vocab_size)
            targets = torch.masked_select(Y_y, mask)

            loss = loss_fn(preds, targets)
            losses_valid.append(loss.data.cpu().numpy())

        print("Epoch {} Validation Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses_valid))))

        with open("3loss" + str(epoch), 'w') as f:
            f.write("Epoch {} Training Loss: {:.4f}\n".format(epoch, np.asscalar(np.mean(losses))))
            f.write("Epoch {} Validation Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses_valid))))

train()


# recitation6
'''
Two options in general

- Do some masking on the loss function to zero out the padding areas


Working with padded tensors and a mask should be the easiest for attention. So given utterances you can produce keys, values and a mask.

# Baseline is around 18 (on validation). {loss/utter}
# loss is summed over the length and meaned over the batch size

The standard is to always use teacher forcing during training 100 % . Reducing that to 90 % or lower can make you generate better things as test time.

    In terms of actual value to use, calculate the mean of the sum for each transcript. So sum along the character dimension then mean along the batch dimension.
'''

# Our projection is an MLP(linear, leaky relu, linear)

# if you have a stack of 3 LSTMs, use the h0 of the
# last LSTM in the stack to generate your additional query.
#(hidden+value size)->(embedding)->(vocab)
#self.projection.weight = self.embedding.weight
# self.lstmCell2 = LSTMCell(input_size=300, hidden_size=300)
# self.lstmCell3 = LSTMCell(input_size=300, hidden_size=300)
# TODO: is cropping to multiple of 8 a good idea? what if utterance become size 0
# TODO: masked softmax
