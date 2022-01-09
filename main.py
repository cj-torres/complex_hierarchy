# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import language_builder2 as lb
import csv
from random import shuffle
from random import sample
import torch

def random_love_note():
    love_strings = ["I love you",
                    "I miss you Nina",
                    "you're the love of my life",
                    "I can't wait to see you",
                    "<3",
                    "I'm thinking of you",
                    "I hope you're having a good day",
                    "you're a cutie",
                    ":*",
                    "XO"
                    ]
    return sample(love_strings,1)[0]



class SimpleClassifier(torch.nn.Module):
    def __init__(self, alphabet_sz, embed_dim, hidden_sz, final_layer_sz):
        super(SimpleClassifier, self).__init__()
        self.hidden_size = hidden_sz
        self.embedding = torch.nn.Embedding(alphabet_sz, embed_dim, padding_idx=0)
        self.rnn = torch.nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_sz,
            batch_first=True
        )
        self.final_layer = torch.nn.Linear(hidden_sz, final_layer_sz)
        self.final_transform = torch.nn.Tanh()
        self.out = torch.nn.Linear(final_layer_sz, 1)
        self.out_f = torch.nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.zeros_(p.data)

    def forward(self, x, lengths):
        bs, seq_sz = x.size()
        h_t = torch.stack([torch.zeros(bs).to(x.device)] * self.hidden_size, dim=1).unsqueeze(dim=0)
        embeds = self.embedding(x)

        seq, out_h = self.rnn(embeds, h_t)
        seq = seq[torch.arange(seq.size(0)),lengths,:]
        final = self.final_transform(self.final_layer(seq))
        y_hat = self.out_f(self.out(final))

        return y_hat.squeeze()


class LSTMBranchSequencer(torch.nn.Module):

    def __init__(self, alphabet_sz, embed_dim, hidden_sz, final_layer_sz, output_sz):
        super(LSTMBranchSequencer, self).__init__()
        self.hidden_size = hidden_sz
        self.embedding = torch.nn.Embedding(alphabet_sz, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_sz,
            batch_first=True
        )
        self.final_layer = torch.nn.Linear(hidden_sz, final_layer_sz)
        self.final_transform = torch.nn.Tanh()
        self.out = torch.nn.Linear(final_layer_sz, output_sz)
        self.out_f = torch.nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.zeros_(p.data)

    def forward(self, x):
        embeds = self.embedding(x)

        seq, out_h = self.lstm(embeds)
        final = self.final_transform(self.final_layer(seq))
        y_hat = self.out_f(self.out(final))

        return y_hat.squeeze()


class LSTMCLassifier(torch.nn.Module):

    def __init__(self, alphabet_sz, embed_dim, hidden_sz, final_layer_sz):
        super(LSTMCLassifier, self).__init__()
        self.hidden_size = hidden_sz
        self.embedding = torch.nn.Embedding(alphabet_sz, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_sz,
            batch_first=True
        )
        self.final_layer = torch.nn.Linear(hidden_sz, final_layer_sz)
        self.final_transform = torch.nn.Tanh()
        self.out = torch.nn.Linear(final_layer_sz, 1)
        self.out_f = torch.nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.zeros_(p.data)

    def forward(self, x, lengths):
        #bs, seq_sz = x.size()
        #h_t = torch.stack([torch.zeros(bs).to(x.device)] * self.hidden_size, dim=1).unsqueeze(dim=0)
        embeds = self.embedding(x)

        seq, out_h = self.lstm(embeds)
        seq = seq[torch.arange(seq.size(0)),lengths,:]
        final = self.final_transform(self.final_layer(seq))
        y_hat = self.out_f(self.out(final))

        return y_hat.squeeze()


def bernoulli_loss(y, y_hat):
    return -(y * y_hat.log() + (1 - y) * (1 - y_hat).log()).mean()


def bernoulli_loss_cont(y, y_hat, mask):
    return -(torch.xlogy(y[mask], y_hat[mask]) + torch.xlogy(1-y[mask], 1-y_hat[mask])).mean(dim=0).sum(dim=-1)


def correct_guesses(y, y_hat, mask):
    return (y_hat[mask].round() == y[mask]).sum() / torch.numel(y[mask])


def train(model, x_train, y_train, lengths_train, x_test, y_test, lengths_test, epochs):
    model.to("cuda")
    x = x_train.to("cuda")
    x_test = x_test.to("cuda")
    y = y_train.to("cuda")
    y_test = y_test.to("cuda")
    op = torch.optim.Adam(model.parameters())
    best_loss = torch.tensor([float('inf')]).squeeze()
    for i in range(epochs):
        op.zero_grad()
        y_hat = model(x, lengths_train)
        loss = bernoulli_loss(y, y_hat)
        loss.backward()
        op.step()

        y_test_hat = model(x_test, lengths_test)
        loss_test = bernoulli_loss(y_test, y_test_hat)

        #if (i+1) % 10 == 0:
        #    print(loss_test.item())

        if loss_test >= best_loss:
            early_stop_counter += 1
            if early_stop_counter > 4:
                model = torch.load("best_net_cache.pt")
                print("Best loss was: %s and %s" % (best_loss.item(), random_love_note()))
                return model, best_loss
        else:
            best_loss = loss_test
            early_stop_counter = 0
            torch.save(model,"best_net_cache.pt")
    print("Best loss was: %s and %s" % (best_loss.item(), random_love_note()))
    return model, best_loss


def seq_train(model, x_train, y_train, mask_train, x_test, y_test, mask_test, target_accuracy, cutoff_epochs, batch_sz):
    from math import ceil
    model.to("cuda")

    batches = len(x_train)
    batch_q = ceil(batches/batch_sz)
    batch_r = batches % batch_sz
    indxs = [i*batch_sz for i in range(batch_q)]
    indxs[-1] = indxs[-1] - batch_sz + batch_r

    x_test = x_test.to("cuda")
    y_test = y_test.to("cuda")
    op = torch.optim.Adam(model.parameters())
    best_loss = torch.tensor([float('inf')]).squeeze()
    percent_correct = 0
    early_stop_counter = 0

    for i in range(cutoff_epochs):
        for param in model.parameters():
            param.grad = None
        for indx1, indx2 in zip(indxs, indxs[1:]):
            mask = mask_train[indx1:indx2]
            x = x_train[indx1:indx2].to("cuda")
            y = y_train[indx1:indx2].to("cuda")
            y_hat = model(x)
            loss = bernoulli_loss_cont(y, y_hat, mask)
            loss.backward()

            del mask, x, y
        op.step()

        with torch.no_grad():
            y_test_hat = model(x_test)
            loss_test = bernoulli_loss_cont(y_test, y_test_hat, mask_test)

            test_percent_correct = correct_guesses(y_test, y_test_hat, mask_test)

        if loss_test >= best_loss:
            early_stop_counter += 1
            if early_stop_counter > 4:
                model = torch.load("best_net_cache.pt")
                print("Best loss was: %s, Accuracy: %s, and %s" % (best_loss.item(), percent_correct.item(), random_love_note()))
                return model, best_loss, percent_correct
        else:
            best_loss = loss_test
            early_stop_counter = 0
            percent_correct = test_percent_correct
            torch.save(model,"best_net_cache.pt")
        print("Accuracy: %s, counter: %d" % (percent_correct.item(), early_stop_counter))
        if percent_correct > target_accuracy: break
    print("Best loss was: %s, Accuracy: %s, and %s" % (best_loss.item(), percent_correct.item(), random_love_note()))
    return model, best_loss, percent_correct


def random_split(x, y, length, r=.8):
    indx = torch.BoolTensor([i<(r*len(y)) for i, _ in enumerate(y)])
    shuffle(indx)
    x_train = x[indx]
    x_test = x[~indx]
    y_train = y[indx]
    y_test = y[~indx]
    length_train = length[indx]
    length_test = length[~indx]

    return (x_train, y_train, length_train), (x_test, y_test, length_test)


def set_normalize(x, y, lengths, r=1):
    x_true = x[y.type(torch.BoolTensor)]
    x_false = x[~y.type(torch.BoolTensor)]
    y_true = y[y.type(torch.BoolTensor)]
    y_false = y[~y.type(torch.BoolTensor)]
    lengths_true = lengths[y.type(torch.BoolTensor)]
    lengths_false = lengths[~y.type(torch.BoolTensor)]
    indx = torch.BoolTensor([i<(r*len(x_true)) for i, _ in enumerate(x_false)])
    shuffle(indx)

    subset_x_f = x_false[indx]
    subset_y_f = y_false[indx]
    subset_lengths_f = lengths_false[indx]

    new_x = torch.cat([x_true, subset_x_f])
    new_y = torch.cat([y_true, subset_y_f])
    new_length = torch.cat([lengths_true, subset_lengths_f])

    return new_x, new_y, new_length


def model_to_list(m):
    weight_list = []
    for l in m.parameters():
        try:
            weight_list.extend([w.item() for w in l.view(-1)])
        except ValueError:
            weight_list.extend([w.item() for c in l for w in c])
    return weight_list


def load_as_tensor(file_name):
    with open(file_name, mode='r') as file:
        csvfile = csv.reader(file)
        t = torch.Tensor([[float(x) for x in l] for l in csvfile])
    return t


def laplace_fit(x):
    m = x.median()
    b = (x-m).abs().mean()
    return torch.distributions.Laplace(m,b)


def gauss_fit(x):
    mu = x.mean()
    sigma2 = x.std().pow(2)
    return torch.distributions.Normal(mu, sigma2)


if __name__ == '__main__':
    target_accuracies = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
    x, y, mask = lb.make_dyck1_io_cont(5000)
    accuracy =  []
    with open('dyck1_model_c_lstm_seq1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for target in target_accuracies:
            for i in range(5000):
                print("Model %d" % (i+1))
                model = LSTMBranchSequencer(4, 2, 4, 4, 3)
                (x1, y1, lengths1), (x_t, y_t, lengths_t) = random_split(x, y, mask)
                model, best_loss, percent_correct = seq_train(model, x1, y1, lengths1, x_t, y_t, lengths_t, target, 500, 1024)
                weights = model_to_list(model)
                writer.writerow(weights)
                accuracy.append([best_loss.item(), percent_correct.item()])
    with open('dyck1_model_c_lstm_loss_seq1.csv', 'w', newline='') as accuracy_file:
        writer = csv.writer(accuracy_file)
        writer.writerow(["r", "accuracy"])
        writer.writerows([[a] for a in accuracy])

    x, y, mask = lb.make_anbn_io_cont(5000)
    accuracy = []
    with open('anbn_model_c_lstm_seq1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for target in target_accuracies:
            for i in range(5000):
                print("Model %d" % (i+1))
                model = LSTMBranchSequencer(4, 2, 4, 4, 3)
                (x1, y1, lengths1), (x_t, y_t, lengths_t) = random_split(x, y, mask)
                model, best_loss, percent_correct = seq_train(model, x1, y1, lengths1, x_t, y_t, lengths_t, target, 500, 1024)
                weights = model_to_list(model)
                writer.writerow(weights)
                accuracy.append([best_loss.item(), percent_correct.item()])
    with open('anbn_model_c_lstm_loss_seq1.csv', 'w', newline='') as accuracy_file:
        writer = csv.writer(accuracy_file)
        writer.writerow(["r", "accuracy"])
        writer.writerows([[a] for a in accuracy])








# if __name__ == '__main__':
#     x, y, lengths = lb.make_dyck1_io(5000)
#     accuracy =  []
#     with open('dyck1_model_c_lstm2.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         for i in range(1000):
#             print("Model %d" % (i+1))
#             model = LSTMBranchSequencer(3, 2, 6, 4)
#             x1, y1, lengths1 = set_normalize(x, y, lengths)
#             (x1, y1, lengths1), (x_t, y_t, lengths_t) = random_split(x, y, lengths)
#             model, best_loss = train(model, x1, y1, lengths1, x_t, y_t, lengths_t, 200)
#             weights = model_to_list(model)
#             writer.writerow(weights)
#             accuracy.append(best_loss.item())
#     with open('dyck1_model_c_lstm_loss2.csv', 'w', newline='') as accuracy_file:
#         writer = csv.writer(csvfile)
#         writer.writerows([[a] for a in accuracy])
#
#     x, y, lengths = lb.make_anbn_io(5000)
#     accuracy = []
#     with open('anbn_model_c_lstm2.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         for i in range(1000):
#             print("Model %d" % (i+1))
#             model = LSTMBranchSequencer(3, 2, 6, 4)
#             x1, y1, lengths1 = set_normalize(x, y, lengths)
#             (x1, y1, lengths1), (x_t, y_t, lengths_t) = random_split(x, y, lengths)
#             model, best_loss = train(model, x1, y1, lengths1, x_t, y_t, lengths_t, 200)
#             weights = model_to_list(model)
#             writer.writerow(weights)
#             accuracy.append(best_loss.item())
#     with open('anbn_model_c_lstm_loss2.csv', 'w', newline='') as accuracy_file:
#         writer = csv.writer(csvfile)
#         writer.writerows([[a] for a in accuracy])
    # x, y, lengths = lb.make_io(lb.anbm)
    # accuracy = []
    # with open('anbm_model_r_lstm.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for i in range(1000):
    #         print("Model %d" % (i+1))
    #         model = LSTMCLassifier(3, 2, 10, 8)
    #         x1, y1, lengths1 = set_normalize(x, y, lengths)
    #         (x1, y1, lengths1), (x_t, y_t, lengths_t) = random_split(x, y, lengths)
    #         model, best_loss = train(model, x1, y1, lengths1, x_t, y_t, lengths_t, 200)
    #         weights = model_to_list(model)
    #         writer.writerow(weights)
    #         accuracy.append(best_loss.item())
    # with open('anbm_model_r_lstm_loss.csv', 'w', newline='') as accuracy_file:
    #     writer = csv.writer(csvfile)
    #     writer.writerows([[a] for a in accuracy])
    # x, y, lengths = lb.make_io(lb.abn)
    # accuracy = []
    # with open('abn_model_r_lstm.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for i in range(1000):
    #         print("Model %d" % (i + 1))
    #         model = LSTMCLassifier(3, 2, 10, 8)
    #         x1, y1, lengths1 = set_normalize(x, y, lengths)
    #         (x1, y1, lengths1), (x_t, y_t, lengths_t) = random_split(x, y, lengths)
    #         model, best_loss = train(model, x1, y1, lengths1, x_t, y_t, lengths_t, 200)
    #         weights = model_to_list(model)
    #         writer.writerow(weights)
    #         accuracy.append(best_loss.item())
    # with open('abn_model_r_lstm_loss.csv', 'w', newline='') as accuracy_file:
    #     writer = csv.writer(csvfile)
    #     writer.writerows([[a] for a in accuracy])



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
