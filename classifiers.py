# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import language_builders as lb
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


# class SimpleClassifier(torch.nn.Module):
#     def __init__(self, alphabet_sz, embed_dim, hidden_sz, final_layer_sz):
#         super(SimpleClassifier, self).__init__()
#         self.hidden_size = hidden_sz
#         self.embedding = torch.nn.Embedding(alphabet_sz, embed_dim, padding_idx=0)
#         self.rnn = torch.nn.RNN(
#             input_size=embed_dim,
#             hidden_size=hidden_sz,
#             batch_first=True
#         )
#         self.final_layer = torch.nn.Linear(hidden_sz, final_layer_sz)
#         self.final_transform = torch.nn.Tanh()
#         self.out = torch.nn.Linear(final_layer_sz, 1)
#         self.out_f = torch.nn.Sigmoid()
#         self.init_weights()
#
#     def init_weights(self):
#         for p in self.parameters():
#             if p.data.ndimension() >= 2:
#                 torch.nn.init.normal_(p.data)
#             else:
#                 torch.nn.init.zeros_(p.data)
#
#     def forward(self, x, lengths):
#         bs, seq_sz = x.size()
#         h_t = torch.stack([torch.zeros(bs).to(x.device)] * self.hidden_size, dim=1).unsqueeze(dim=0)
#         embeds = self.embedding(x)
#
#         seq, out_h = self.rnn(embeds, h_t)
#         seq = seq[torch.arange(seq.size(0)),lengths,:]
#         final = self.final_transform(self.final_layer(seq))
#         y_hat = self.out_f(self.out(final))
#
#         return y_hat.squeeze()


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
            torch.nn.init.normal_(p.data)

    def forward(self, x):
        embeds = self.embedding(x)

        seq, out_h = self.lstm(embeds)
        final = self.final_transform(self.final_layer(seq))
        y_hat = self.out_f(self.out(final))

        return y_hat.squeeze()


class LSTMClassifier(torch.nn.Module):

    def __init__(self, alphabet_sz, embed_dim, hidden_sz, final_layer_sz):
        super(LSTMClassifier, self).__init__()
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
                torch.nn.init.normal_(p.data)
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


class LSTMSequencer(torch.nn.Module):

    def __init__(self, alphabet_sz, embed_dim, hidden_sz, final_layer_sz, output_sz):
        super(LSTMSequencer, self).__init__()
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
        self.out_f = torch.nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.normal_(p.data)
            else:
                torch.nn.init.zeros_(p.data)

    def forward(self, x):
        #bs, seq_sz = x.size()
        #h_t = torch.stack([torch.zeros(bs).to(x.device)] * self.hidden_size, dim=1).unsqueeze(dim=0)
        embeds = self.embedding(x)

        seq, out_h = self.lstm(embeds)
        final = self.final_transform(self.final_layer(seq))
        y_hat = self.out_f(self.out(final))

        return y_hat.squeeze()


class TransformerSequencer(torch.nn.Module):

    def __init__(self, alphabet_sz, embed_dim, transformer_depth, n_head, feedforward_dim, output_sz):
        super(TransformerSequencer, self).__init__()
        self.depth = transformer_depth
        self.mask = alphabet_sz-1
        self.n_head = n_head
        self.feedforward_dim = feedforward_dim
        self.embedding = torch.nn.Embedding(alphabet_sz, embed_dim, padding_idx=0)
        self.t_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            dim_feedforward=feedforward_dim,
            nhead=n_head,
            batch_first=True
        )

        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer=self.t_layer,
            num_layers=transformer_depth
        )

        self.out = torch.nn.Linear(embed_dim, output_sz)
        self.out_f = torch.nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.normal_(p.data)
            else:
                torch.nn.init.zeros_(p.data)

    def forward(self, x):
        embeds = self.embedding(x)

        out = self.transformer(embeds)
        y_hat = self.out_f(self.out(out))

        return y_hat.squeeze()


def bernoulli_loss(y, y_hat):
    return -(y * y_hat.log() + (1 - y) * (1 - y_hat).log()).mean()


def bernoulli_loss_cont(y, y_hat, mask):
    return -(torch.xlogy(y[mask], y_hat[mask]) + torch.xlogy(1-y[mask], 1-y_hat[mask])).sum(dim=1).mean(dim=0)


def correct_guesses_batch_seq(y, y_hat, mask):
    return (y_hat[mask].round() == y[mask]).sum() / torch.numel(y[mask])


def correct_guesses_seq(y, y_hat, mask):
    return (torch.argmax(y_hat, dim=-1)[mask] == y[mask]).sum() / torch.numel(y[mask])


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


def branch_seq_train(model, language_set, target_loss, is_loss, batch_sz, max_epochs, increment, patience=20):
    from math import ceil
    from random import sample



    model.to("cuda")

    batches = len(language_set.train_input)
    batch_q = ceil(batches/batch_sz)
    batch_r = batches % batch_sz
    indxs = [i*batch_sz for i in range(batch_q)]
    indxs[-1] = indxs[-1] - batch_sz + batch_r

    x_train = language_set.train_input.to("cuda")
    y_train = language_set.train_output.to("cuda")
    mask_train = language_set.train_mask.to("cuda")

    x_test = language_set.test_input.to("cuda")
    y_test = language_set.test_output.to("cuda")
    mask_test = language_set.test_mask.to("cuda")

    op = torch.optim.Adam(model.parameters(), lr=.0005)
    best_loss = torch.tensor([float('inf')]).squeeze()
    loss_test = torch.tensor([float('inf')]).squeeze()
    percent_correct = 0
    early_stop_counter = 0

    indices = range(batches)
    train_percent_correct = 0
    test_percent_correct = 0
    epoch = 0
    while (is_loss and loss_test > target_loss) or ((not is_loss) and test_percent_correct < target_loss) and epoch < max_epochs:
        batch = torch.tensor(sample(indices, batch_sz)).type(torch.LongTensor)
        for param in model.parameters():
            param.grad = None
        x = x_train[batch].to("cuda")
        y = y_train[batch].to("cuda")
        mask = mask_train[batch].to("cuda")
        y_hat = model(x)
        loss = bernoulli_loss_cont(y_hat, y)
        loss.backward()
        train_percent_correct = correct_guesses_batch_seq(y, y_hat, mask)
        del mask, x, y

        op.step()

        with torch.no_grad():
            y_test_hat = model(x_test)
            loss_test = bernoulli_loss_cont(y_test, y_test_hat, mask_test)

            test_percent_correct = correct_guesses_batch_seq(y_test, y_test_hat, mask_test)

        if loss_test >= best_loss:
            early_stop_counter += 1
            if early_stop_counter > patience:
                model.load_state_dict(torch.load("best_net_cache.ptr"))
                print("Best loss was: %s, Accuracy: %s, Train Accuracy: %s" %
                      (best_loss.item(), percent_correct.item(), train_percent_correct.item()))
                return model, best_loss, percent_correct
        else:
            best_loss = loss_test
            early_stop_counter = 0
            percent_correct = test_percent_correct
            torch.save(model.state_dict(), "best_net_cache.ptr")
        print("Accuracy: %s, loss: %s, counter: %d, train accuracy: %s" %
              (percent_correct.item(), loss_test.item(), early_stop_counter, train_percent_correct.item()))

        if epoch % increment == 0:
            yield model, best_loss, percent_correct

        epoch+=1

    print("Best loss was: %s, Accuracy: %s, Train Accuracy: %s" %
          (best_loss.item(), percent_correct.item(), train_percent_correct.item()))

    return model, best_loss, percent_correct


def seq_train(model, language_set, batch_sz, increment, max_epochs, patience=20):
    from math import ceil
    from random import sample

    model.to("cuda")

    ce_loss = torch.nn.CrossEntropyLoss()

    batches = len(language_set.train_input)
    batch_q = ceil(batches/batch_sz)
    batch_r = batches % batch_sz
    indxs = [i*batch_sz for i in range(batch_q)]
    indxs[-1] = indxs[-1] - batch_sz + batch_r

    x_train = language_set.train_input.to("cuda")
    y_train = language_set.train_output.to("cuda")
    mask_train = language_set.train_mask.to("cuda")

    x_test = language_set.test_input.to("cuda")
    y_test = language_set.test_output.to("cuda")
    mask_test = language_set.test_mask.to("cuda")
    op = torch.optim.Adam(model.parameters(), lr=.0005, weight_decay=.05)
    best_loss = torch.tensor([float('inf')]).squeeze()
    #loss_test = torch.tensor([float('inf')]).squeeze()
    percent_correct = 0
    early_stop_counter = 0

    indices = range(batches)
    train_percent_correct = 0
    #test_percent_correct = 0
    epoch = 0
    while epoch < max_epochs:
        batch = torch.tensor(sample(indices, batch_sz)).type(torch.LongTensor)
        for param in model.parameters():
            param.grad = None
        x = x_train[batch].to("cuda")
        y = y_train[batch].to("cuda")
        mask = mask_train[batch].to("cuda")
        y_hat = model(x)

        loss = ce_loss(y_hat[mask], y[mask])
        loss.backward()
        train_percent_correct = correct_guesses_seq(y, y_hat, mask)
        del mask, x, y

        op.step()

        with torch.no_grad():
            y_test_hat = model(x_test)
            test_mask = language_set.test_mask
            loss_test = ce_loss(y_test_hat[test_mask], y_test[test_mask])

            test_percent_correct = correct_guesses_seq(y_test, y_test_hat, mask_test)

        if loss_test >= best_loss:
            early_stop_counter += 1
            if early_stop_counter > patience:
                model.load_state_dict(torch.load("best_net_cache.ptr"))
                print("Best loss was: %s, Accuracy: %s, Train Accuracy: %s" %
                      (best_loss.item(), percent_correct.item(), train_percent_correct.item()))
                return model, best_loss, percent_correct, best_epoch
        else:
            best_epoch = epoch + 1
            best_loss = loss_test
            early_stop_counter = 0
            percent_correct = test_percent_correct
            torch.save(model.state_dict(), "best_net_cache.ptr")

        print("Accuracy: %s, loss: %s, counter: %d, train accuracy: %s" %
              (percent_correct.item(), loss_test.item(), early_stop_counter, train_percent_correct.item()))

        epoch += 1

        if epoch % increment == 0: #and epoch != max_epochs:
            print("Saving epoch %d" % (epoch))
            yield model, loss_test, test_percent_correct, epoch

    print("Best loss was: %s, Accuracy: %s, Train Accuracy: %s" %
          (best_loss.item(), percent_correct.item(), train_percent_correct.item()))
    #return model, best_loss, percent_correct, epoch


def seq_transformers_train(model, language_set, batch_sz, mask_percent, increment, max_epochs, patience=20):
    from math import ceil
    from random import sample

    model.to("cuda")

    ce_loss = torch.nn.CrossEntropyLoss()

    batches = len(language_set.train_input)
    batch_q = ceil(batches/batch_sz)
    batch_r = batches % batch_sz
    indxs = [i*batch_sz for i in range(batch_q)]
    indxs[-1] = indxs[-1] - batch_sz + batch_r

    x_train = language_set.train_input.to("cuda")
    y_train = language_set.train_input.to("cuda")
    mask_train = language_set.train_mask.to("cuda")

    x_test = language_set.test_input.to("cuda")
    y_test = language_set.test_input.to("cuda")
    mask_test = language_set.test_mask.to("cuda")
    op = torch.optim.Adam(model.parameters(), lr=.0005, weight_decay=.05)
    best_loss = torch.tensor([float('inf')]).squeeze()
    #loss_test = torch.tensor([float('inf')]).squeeze()
    percent_correct = 0
    early_stop_counter = 0

    indices = range(batches)
    train_percent_correct = 0
    #test_percent_correct = 0
    epoch = 0
    while epoch < max_epochs:
        batch = torch.tensor(sample(indices, batch_sz)).type(torch.LongTensor)
        for param in model.parameters():
            param.grad = None
        x = x_train[batch].clone().to("cuda")
        y = y_train[batch].to("cuda")
        mask = torch.distributions.Bernoulli(mask_train[batch].to("cuda")*mask_percent).sample().type(torch.BoolTensor)
        x[mask] = model.mask
        y_hat = model(x)

        loss = ce_loss(y_hat[mask], y[mask])
        loss.backward()
        train_percent_correct = correct_guesses_seq(y, y_hat, mask)
        del mask, x, y

        op.step()

        with torch.no_grad():
            x_test_masked = x_test.clone()
            test_mask = language_set.test_mask
            mask = torch.distributions.Bernoulli(test_mask.to("cuda")*mask_percent).sample().type(torch.BoolTensor)

            x_test_masked[mask] = model.mask

            y_test_hat = model(x_test)
            loss_test = ce_loss(y_test_hat[mask], y_test[mask])

            test_percent_correct = correct_guesses_seq(y_test, y_test_hat, mask_test)

        if loss_test >= best_loss:
            early_stop_counter += 1
            if early_stop_counter > patience:
                model.load_state_dict(torch.load("best_net_cache.ptr"))
                print("Best loss was: %s, Accuracy: %s, Train Accuracy: %s" %
                      (best_loss.item(), percent_correct.item(), train_percent_correct.item()))
                return model, best_loss, percent_correct, best_epoch
        else:
            best_epoch = epoch + 1
            best_loss = loss_test
            early_stop_counter = 0
            percent_correct = test_percent_correct
            torch.save(model.state_dict(), "best_net_cache.ptr")



        print("Accuracy: %s, loss: %s, counter: %d, train accuracy: %s" %
              (percent_correct.item(), loss_test.item(), early_stop_counter, train_percent_correct.item()))

        epoch += 1

        if epoch % increment == 0: #and epoch != max_epochs:
            print("Saving epoch %d" % (epoch))
            yield model, loss_test, test_percent_correct, epoch

    print("Best loss was: %s, Accuracy: %s, Train Accuracy: %s" %
          (best_loss.item(), percent_correct.item(), train_percent_correct.item()))



def random_split_no_overfit(x, y, length, r=.85):
    # should work, as all sets should be sorted
    cut = int(len(x)*r)
    x_train = x[:cut]
    x_test = x[cut:]
    y_train = y[:cut]
    y_test = y[cut:]
    length_train = length[:cut]
    length_test = length[cut:]

    return (x_train, y_train, length_train), (x_test, y_test, length_test)


def random_split_no_overfit(x, y, length, r=.85):
    # should work, as all sets should be sorted
    cut = int(len(x)*r)
    x_train = x[:cut]
    x_test = x[cut:]
    y_train = y[:cut]
    y_test = y[cut:]
    length_train = length[:cut]
    length_test = length[cut:]

    return (x_train, y_train, length_train), (x_test, y_test, length_test)


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
