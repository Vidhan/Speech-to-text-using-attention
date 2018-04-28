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
sos = 5


batch = 1
key_dimension = 128
value_dimension = 128
e_hidden_dimension = 300
hidden_dimension = 300
embedding_dimension = 400
max_len = 500

def write_results(predictions, output_file='predictions_final.txt'):
    with open(output_file, 'w') as f:
        f.write("Id,Predicted\n")
        i = 0
        for y in predictions:
            f.write("%d,%s\n" % (i, y))
            i = i + 1


def to_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def to_tensor(numpy_array):
    return torch.from_numpy(numpy_array).float()


class MyCustomTestDataset(Dataset):

    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        X = self.features[index]
        return X

    def __len__(self):
        return len(self.features)


class MyTestDataLoader():

    def __call__(self, batch):
        X = sorted(batch, key=lambda x: x.shape[0], reverse=True)
        X = [x if x.shape[0] % (2**pyramid_layers) == 0
             else x[: -(x.shape[0] % (2**pyramid_layers)), :] for x in X]

        X_sizes = [x.shape[0] for x in X]  # list length = batch_size

        X = [np.pad(x, [(0, X_sizes[0] - x.shape[0]), (0, 0)], mode='constant')
             for x in X]
        X = np.dstack(X)
        X = np.swapaxes(X, 1, 2)
        X = to_variable(to_tensor(X))  # max_utterance X batch_size X mel_freq

        return pack_padded_sequence(X, X_sizes), X.shape[1]


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        '''
        self.h_0_1 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.c_0_1 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.h_0_2 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.c_0_2 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.h_0_3 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.c_0_3 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.h_0_4 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        self.c_0_4 = nn.Parameter(torch.FloatTensor(2, 1, e_hidden_dimension).zero_())
        '''
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
        '''
        h_0_1 = self.h_0_1.expand(-1, batch, -1).contiguous()
        c_0_1 = self.c_0_1.expand(-1, batch, -1).contiguous()
        h_0_2 = self.h_0_2.expand(-1, batch, -1).contiguous()
        c_0_2 = self.c_0_2.expand(-1, batch, -1).contiguous()
        h_0_3 = self.h_0_3.expand(-1, batch, -1).contiguous()
        c_0_3 = self.c_0_3.expand(-1, batch, -1).contiguous()
        h_0_4 = self.h_0_4.expand(-1, batch, -1).contiguous()
        c_0_4 = self.c_0_4.expand(-1, batch, -1).contiguous()
        '''
        h, state = self.rnn1(h)#, (h_0_1, c_0_1))
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
        h, state = self.rnn2(h) #, (h_0_2, c_0_2))
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
        h, state = self.rnn3(h) #, (h_0_3, c_0_3))
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
        h, state = self.rnn4(h) #, (h_0_4, c_0_4))
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
    def forward(self, keys, values):
        h = self.h0.expand(keys.shape[1], -1)
        c = self.c0.expand(keys.shape[1], -1)
        keys = keys.permute(1, 2, 0)  # batch, key, utterance_length
        values = values.permute(1, 0, 2)  # bath, utterance_length, value

        query = self.phi(h)  # (batch_size, key_dimension)
        query = torch.unsqueeze(query, dim=1)  # (batch, 1,  key)
        energy = torch.bmm(query, keys)  # (batch_size, 1, utter_length)
        attention = self.softmax(energy)  # (batch_size, 1, utter_length)
        context = torch.bmm(attention, values)  # batch_size X 1 X value
        context = torch.squeeze(context).view(keys.shape[0], -1)  # batch_size X value_dimension

        character_input = to_variable((torch.zeros(1, keys.shape[0]) + sos).long())  # (1 X batch_size)

        index_list = []
        for step in range(max_len):  # timesteps

            character = self.embedding(character_input)  # 1 X batch X embed
            character = torch.squeeze(character).view(keys.shape[0], -1)  # batch_size X embed

            character = torch.cat((character, context), dim=-1)

            h, c = self.lstmCell(character, (h, c))

            logits = self.projection(torch.cat((h, context), dim=-1))  # batch_size X vocab
            predicted_character = torch.max(logits, 1)[1].view(1, -1)
            character_input = predicted_character
            val = predicted_character.data.cpu().numpy()[0][0]
            index_list.append(val)

            query = self.phi(h)  # (batch_size, key_dimension)
            query = torch.unsqueeze(query, dim=1)  # (batch_size, 1,  key)
            energy = torch.bmm(query, keys)  # (batch_size, 1, utter_length)
            attention = self.softmax(energy)  # (batch_size, 1, utter_length)
            context = torch.bmm(attention, values)  # batch_size X 1 X value
            context = torch.squeeze(context).view(keys.shape[0], -1)  # batch_size X value_dimension

        return index_list


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, X, batch_size):
        keys, values = self.encoder(X, batch_size)
        logits = self.decoder(keys, values)
        return logits


def test():
    test_features = np.load('test.npy')  # utterances
    vocab = np.load('vocab.npy') 
    model = Model()
    
    if torch.cuda.is_available():
        model = model.cuda()

    model.load_state_dict(torch.load('model_params19.pt', map_location=lambda storage, loc: storage))

    dataset = MyCustomTestDataset(test_features)

    data_loader = DataLoader(
        dataset, batch_size=batch, shuffle=False, collate_fn=MyTestDataLoader())

    output_list = []
    model.eval()
    for (X, batch_size) in data_loader:
        index_list = model(X, batch_size)
        output_string = ""
        for index in index_list:
            if (vocab[index] != start_char): 
                output_string += str(vocab[index])
        print (output_string)
        output_list.append(output_string)

    write_results(output_list)

test()

