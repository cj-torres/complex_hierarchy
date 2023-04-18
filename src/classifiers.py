# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import language_builders as lb
import csv
from random import shuffle
from random import sample
from copy import deepcopy
import torch
import mdl_tools as mdl
import variational_ib as vib


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


class RNNBranchSequencer(torch.nn.Module):

    def __init__(self, alphabet_sz, embed_dim, hidden_sz, final_layer_sz, output_sz):
        super(RNNBranchSequencer, self).__init__()
        self.hidden_size = hidden_sz
        self.embedding = torch.nn.Embedding(alphabet_sz, embed_dim, padding_idx=0)
        self.rnn = torch.nn.RNN(
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

        seq, out_h = self.rnn(embeds)
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

    def __init__(self, alphabet_sz, embed_dim, hidden_sz, final_layer_sz):
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
        self.out = torch.nn.Linear(final_layer_sz, alphabet_sz)
        #self.out_f = torch.nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            torch.nn.init.normal_(p.data)

    def forward(self, x):
        #bs, seq_sz = x.size()
        #h_t = torch.stack([torch.zeros(bs).to(x.device)] * self.hidden_size, dim=1).unsqueeze(dim=0)
        embeds = self.embedding(x)

        seq, out_h = self.lstm(embeds)
        final = self.final_transform(self.final_layer(seq))
        out = self.out(final)

        return out.squeeze()


class RNNSequencer(torch.nn.Module):

    def __init__(self, alphabet_sz, embed_dim, hidden_sz, final_layer_sz):
        super(RNNSequencer, self).__init__()
        self.hidden_size = hidden_sz
        self.embedding = torch.nn.Embedding(alphabet_sz, embed_dim, padding_idx=0)
        self.rnn = torch.nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_sz,
            batch_first=True
        )
        self.final_layer = torch.nn.Linear(hidden_sz, final_layer_sz)
        self.final_transform = torch.nn.Tanh()
        self.out = torch.nn.Linear(final_layer_sz, alphabet_sz)
        #self.out_f = torch.nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            torch.nn.init.normal_(p.data)

    def forward(self, x):
        #bs, seq_sz = x.size()
        #h_t = torch.stack([torch.zeros(bs).to(x.device)] * self.hidden_size, dim=1).unsqueeze(dim=0)
        embeds = self.embedding(x)

        seq, out_h = self.rnn(embeds)
        final = self.final_transform(self.final_layer(seq))
        out = self.out(final)

        return out.squeeze()


class PositionalEncoding(torch.nn.Module):
    # stolen from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.Tensor([10000.0])) / d_model))
        self.pe = torch.zeros(max_len, 1, d_model)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = torch.nn.Parameter(self.pe, requires_grad=False)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerSequencer(torch.nn.Module):

    def __init__(self, alphabet_sz, embed_dim, transformer_depth, n_head, feedforward_dim, output_sz):
        super(TransformerSequencer, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=2048)
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

        #self.out = torch.nn.Linear(embed_dim, output_sz)
        #self.out_f = torch.nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            #if p.data.ndimension() >= 2:
            torch.nn.init.normal_(p.data)
            #else:
            #    torch.nn.init.zeros_(p.data)

    def forward(self, x, debug=False):
        embeds = self.embedding(x)
        embeds = self.pos_encoder(embeds)
        out = self.transformer(embeds)

        if debug:
            breakpoint()
        return out.squeeze()


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


def branch_seq_train(model, language_set, batch_sz, max_epochs, increment, lam, patience=float("inf"),
                     l0_regularized=False, sub_batch_sz=8):
    from math import ceil
    from random import sample

    if l0_regularized:
        model = mdl.L0_Regularizer(model, lam)

    model.to("cuda")

    batches = len(language_set.train_input)
    batch_q = ceil(batches/batch_sz)
    batch_r = batches % batch_sz
    indxs = [i*batch_sz for i in range(batch_q)]
    indxs[-1] = indxs[-1] - batch_sz + batch_r

    x_train = language_set.train_input#.to("cuda")
    y_train = language_set.train_output#.to("cuda")
    mask_train = language_set.train_mask#.to("cuda")

    op = torch.optim.Adam(model.parameters(), lr=.03)
    #lr_s = torch.optim.lr_scheduler.StepLR(op, )
    best_loss = torch.tensor([float('inf')]).squeeze()
    percent_correct = 0
    early_stop_counter = 0

    indices = range(batches)
    train_percent_correct = 0
    epoch = 1
    size = torch.tensor([0])
    while epoch <= max_epochs:
        temp_indices = list(deepcopy(indices))
        shuffle(temp_indices)
        batches = batch_split(temp_indices, batch_sz)
        for batch in batches:
            for sub_batch in batch_split(batch, sub_batch_sz):
                sub_batch_tens = torch.tensor(sub_batch).type(torch.LongTensor)
                model.train()
                x = x_train[sub_batch_tens].to("cuda")
                y = y_train[sub_batch_tens].to("cuda")
                mask = mask_train[sub_batch_tens].to("cuda")
                y_hat = model(x)
                loss = bernoulli_loss_cont(y, y_hat, mask)

                #breakpoint()

                if l0_regularized:
                    re_loss = model.regularization()
                    loss = loss + re_loss
                loss.backward()
                train_percent_correct = correct_guesses_batch_seq(y, y_hat, mask)

                del mask, x, y

                op.step()

                if l0_regularized:
                    model.constrain_parameters()

        with torch.no_grad():
            model.eval()

            x_test = language_set.test_input.to("cuda")
            y_test = language_set.test_output.to("cuda")
            mask_test = language_set.test_mask.to("cuda")

            y_test_hat = model(x_test)
            loss_test = bernoulli_loss_cont(y_test, y_test_hat, mask_test)

            test_percent_correct = correct_guesses_batch_seq(y_test, y_test_hat, mask_test)

            del x_test, y_test, mask_test

        if loss_test >= best_loss:
            early_stop_counter += 1
            if early_stop_counter > patience:
                model.load_state_dict(torch.load("best_net_cache.ptr"))
                print("Best loss was: %s, Accuracy: %s, Train Accuracy: %s" %
                      (best_loss.item(), percent_correct.item(), train_percent_correct.item()))
                return model, loss_test, test_percent_correct, epoch
        else:
            best_loss = loss_test
            early_stop_counter = 0
            percent_correct = test_percent_correct
            torch.save(model.state_dict(), "best_net_cache.ptr")
        if l0_regularized:
            size = model.count_l0()
        print(("Epoch: %d, Accuracy: %s, loss: %s, counter: %d, train accuracy: %s"+", network size: %s"*l0_regularized)
              % ((epoch, test_percent_correct.item(), loss_test.item(), early_stop_counter, train_percent_correct.item())
                 + (size.item(),)*l0_regularized))

        if epoch % increment == 0:
            print("Saving epoch %d" % (epoch))
            yield model, loss_test, test_percent_correct, epoch


        epoch+=1

    print("Best loss was: %s, Accuracy: %s, Train Accuracy: %s" %
          (best_loss.item(), percent_correct.item(), train_percent_correct.item()))

    return model, best_loss, percent_correct, epoch


def vib_seq_train(model, language_set, batch_sz, max_epochs, increment, lam, patience=float("inf")):
    from math import ceil
    from random import sample

    model.to("cuda")

    batches = len(language_set.train_input)
    batch_q = ceil(batches/batch_sz)
    batch_r = batches % batch_sz
    indxs = [i*batch_sz for i in range(batch_q)]
    indxs[-1] = indxs[-1] - batch_sz + batch_r

    x_train = language_set.train_input #.to("cuda")
    y_train = language_set.train_output #.to("cuda")
    mask_train = language_set.train_mask #.to("cuda")

    op = torch.optim.Adam(model.parameters(), lr=.03)
    best_loss = torch.tensor([float('inf')]).squeeze()
    percent_correct = 0
    early_stop_counter = 0

    indices = range(batches)
    train_percent_correct = 0

    epoch = 1
    while epoch <= max_epochs:
        temp_indices = list(deepcopy(indices))
        shuffle(temp_indices)
        batches = batch_split(temp_indices, batch_sz)
        for batch in batches:
            for param in model.parameters():
                param.grad = None
            batch_tens = torch.tensor(batch).type(torch.LongTensor)
            model.train()
            x = x_train[batch_tens].to("cuda")
            y = y_train[batch_tens].to("cuda")
            mask = mask_train[batch_tens].to("cuda")
            y_hat, h_stats, h_seq, log_probs = model(x)

            ce_loss = model.lm_loss(y, y_hat, mask)
            mi_loss = model.mi_loss(h_seq, log_probs, mask)

            loss = ce_loss + lam * mi_loss

            loss.backward()
            train_percent_correct = correct_guesses_seq(y, y_hat, mask)

            del mask, x, y

            op.step()

        with torch.no_grad():
            model.eval()

            x_test = language_set.test_input.to("cuda")
            y_test = language_set.test_output.to("cuda")
            mask_test = language_set.test_mask.to("cuda")

            y_test_hat, h_stats, h_seq, log_probs = model(x_test)
            test_mask = language_set.test_mask
            loss_test = model.lm_loss(y_test, y_test_hat, test_mask)
            mi_test = model.mi_loss(h_seq, log_probs, test_mask)

            test_percent_correct = correct_guesses_seq(y_test, y_test_hat, mask_test)

        if loss_test >= best_loss:
            early_stop_counter += 1
            if early_stop_counter > patience:
                model.load_state_dict(torch.load("best_net_cache.ptr"))
                print("Best loss was: %s, Accuracy: %s, Train Accuracy: %s" %
                      (best_loss.item(), percent_correct.item(), train_percent_correct.item()))
                return model, loss_test, mi_test, test_percent_correct, epoch
        else:
            best_loss = loss_test
            early_stop_counter = 0
            percent_correct = test_percent_correct
            torch.save(model.state_dict(), "best_net_cache.ptr")
        print(
                "Epoch: %d, Accuracy: %s, CE Loss: %s, MI Loss: %s, counter: %d, train accuracy: %s" %
                (epoch, test_percent_correct.item(), loss_test.item(), mi_test.item(), early_stop_counter,
                 train_percent_correct.item())
              )

        if epoch % increment == 0:
            print("Saving epoch %d" % epoch)
            yield model, loss_test, mi_test, test_percent_correct, epoch

        epoch += 1

    print("Best loss was: %s, Accuracy: %s, Train Accuracy: %s" %
          (best_loss.item(), percent_correct.item(), train_percent_correct.item()))

    return model, best_loss, mi_test, percent_correct, epoch


def seq_train(model, language_set, batch_sz, max_epochs, increment, lam, patience=float("inf"), l0_regularized=False, sub_batch_sz = 8):
    from math import ceil
    from random import sample

    if l0_regularized:
        model = mdl.L0_Regularizer(model, lam)

    model.to("cuda")

    ce_loss = torch.nn.CrossEntropyLoss()

    batches = len(language_set.train_input)
    batch_q = ceil(batches/batch_sz)
    batch_r = batches % batch_sz
    indxs = [i*batch_sz for i in range(batch_q)]
    indxs[-1] = indxs[-1] - batch_sz + batch_r

    x_train = language_set.train_input #.to("cuda")
    y_train = language_set.train_output #.to("cuda")
    mask_train = language_set.train_mask #.to("cuda")

    op = torch.optim.Adam(model.parameters(), lr=.03)
    best_loss = torch.tensor([float('inf')]).squeeze()
    percent_correct = 0
    early_stop_counter = 0

    indices = range(batches)
    train_percent_correct = 0
    #test_percent_correct = 0
    epoch = 1
    while epoch <= max_epochs:
        temp_indices = list(deepcopy(indices))
        shuffle(temp_indices)
        batches = batch_split(temp_indices, batch_sz)
        for batch in batches:
            for sub_batch in batch_split(batch, sub_batch_sz):
                for param in model.parameters():
                    param.grad = None
                sub_batch_tens = torch.tensor(sub_batch).type(torch.LongTensor)
                model.train()
                x = x_train[sub_batch_tens].to("cuda")
                y = y_train[sub_batch_tens].to("cuda")
                mask = mask_train[sub_batch_tens].to("cuda")
                y_hat = model(x)

                loss = ce_loss(y_hat[mask], y[mask])
                if l0_regularized:
                    re_loss = model.regularization()
                    loss = loss + re_loss
                loss.backward()
                train_percent_correct = correct_guesses_seq(y, y_hat, mask)

                del mask, x, y

                op.step()

                if l0_regularized:
                    model.constrain_parameters()

        with torch.no_grad():
            model.eval()

            x_test = language_set.test_input.to("cuda")
            y_test = language_set.test_output.to("cuda")
            mask_test = language_set.test_mask.to("cuda")

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
                return model, loss_test, test_percent_correct, epoch
        else:
            best_loss = loss_test
            early_stop_counter = 0
            percent_correct = test_percent_correct
            torch.save(model.state_dict(), "best_net_cache.ptr")
        size = torch.tensor([0])
        if l0_regularized:
            size = model.count_l0()
        print((
                          "Epoch: %d, Accuracy: %s, loss: %s, counter: %d, train accuracy: %s" + ", network size: %s" * l0_regularized)
              % ((epoch, test_percent_correct.item(), loss_test.item(), early_stop_counter,
                  train_percent_correct.item())
                 + (size.item(),) * l0_regularized))

        if epoch % increment == 0:
            print("Saving epoch %d" % (epoch))
            yield model, loss_test, test_percent_correct, epoch

        epoch += 1

    print("Best loss was: %s, Accuracy: %s, Train Accuracy: %s" %
          (best_loss.item(), percent_correct.item(), train_percent_correct.item()))

    return model, best_loss, percent_correct, epoch
    #return model, best_loss, percent_correct, epoch


def seq_transformers_train(model, language_set, batch_sz, mask_percent, increment, max_epochs, patience=20, weight_decay=False):
    from math import ceil
    from random import sample

    #model#.to("cuda")

    ce_loss = torch.nn.CrossEntropyLoss()

    batches = len(language_set.train_input)
    batch_q = ceil(batches/batch_sz)
    batch_r = batches % batch_sz
    indxs = [i*batch_sz for i in range(batch_q)]
    indxs[-1] = indxs[-1] - batch_sz + batch_r

    x_train = language_set.train_input#.to("cuda")
    y_train = language_set.train_input#.to("cuda")
    mask_train = language_set.train_mask#.to("cuda")

    #x_test = language_set.test_input#.to("cuda")
    #y_test = language_set.test_input#.to("cuda")
    #mask_test = language_set.test_mask#.to("cuda")
    if weight_decay:
        op = torch.optim.Adam(model.parameters(), lr=.0005, weight_decay=.005)
    else:
        op = torch.optim.Adam(model.parameters(), lr=.0005)
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
        y = y_train[batch].clone().to("cuda")
        mask = torch.distributions.Bernoulli(mask_train[batch]*mask_percent).sample().type(torch.BoolTensor).to("cuda")
        x[mask] = model.mask
        y_hat = model(x)

        loss = ce_loss(y_hat[mask], y[mask])
        loss.backward()
        train_percent_correct = correct_guesses_seq(y, y_hat, mask)
        del mask, x, y

        op.step()

        with torch.no_grad():
            x_test = language_set.test_input.to("cuda")
            y_test = language_set.test_input.to("cuda")
            mask_test = language_set.test_mask.to("cuda")
            x_test_masked = x_test.clone()
            test_mask = language_set.test_mask
            mask = torch.distributions.Bernoulli(test_mask*mask_percent).sample().type(torch.BoolTensor).to("cuda")

            x_test_masked[mask] = model.mask

            y_test_hat = model(x_test)
            loss_test = ce_loss(y_test_hat[mask], y_test[mask])

            test_percent_correct = correct_guesses_seq(y_test, y_test_hat, mask_test)

            del x_test, y_test, mask_test

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


def pfa_train(model, language_set, batch_sz, increment, max_epochs, lam, patience=float("inf")):
    #with torch.autograd.set_detect_anomaly(True):
    from math import ceil
    from random import sample

    #model #.to("cuda")

    batches = len(language_set.train_input)
    batch_q = ceil(batches / batch_sz)
    batch_r = batches % batch_sz
    indxs = [i * batch_sz for i in range(batch_q)]
    indxs[-1] = indxs[-1] - batch_sz + batch_r

    x_train = language_set.train_output # .to("cuda")

    op = torch.optim.Adam(model.parameters(), lr=.03)
    best_loss = torch.tensor([float('inf')]).squeeze()
    early_stop_counter = 0

    indices = range(batches)
    epoch = 1
    while epoch <= max_epochs:
        #for i in range(batch_sz // 16):
        for param in model.parameters():
            param.grad = None
        batch = torch.tensor(sample(indices, batch_sz)).type(torch.LongTensor)
        model.train()
        x = x_train[batch] #.to("cuda")

        loss = model(x)
        loss.backward()

        del x

        op.step()
        #print("Sub-epoch")

        with torch.no_grad():
            model.eval()

            x_test = language_set.test_output #.to("cuda")

            loss_test = model(x_test)

        if loss_test >= best_loss:
            early_stop_counter += 1
            if early_stop_counter > patience:
                model.load_state_dict(torch.load("best_net_cache.ptr"))
                print("Best loss was: %s, Train loss: %s" %
                      (best_loss.item(), loss_test.item()))
                return model, loss_test, epoch
        else:
            best_loss = loss_test
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_net_cache.ptr")
        print((
                      "Epoch: %d, loss: %s, counter: %d, train loss: %s")
              % (epoch, loss_test.item(), early_stop_counter,
                  loss_test.item()))

        if epoch % increment == 0:
            print("Saving epoch %d" % (epoch))
            yield model, loss_test, epoch

        epoch += 1

    print("Best loss was: %s, Train loss: %s" %
          (best_loss.item(), loss_test.item()))

    return model, best_loss, epoch


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
        weight_list.extend([w.item() for w in list(torch.flatten(l))])
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


def batch_split(indices, batch_sz):
    for i in range(0, len(indices), batch_sz):
        yield indices[i:min(i+batch_sz, len(indices))]



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
