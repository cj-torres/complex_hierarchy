import torch
import entmax

class PFA(torch.nn.Module):
    def __init__(self, vocab_size, num_states):
        super(PFA, self).__init__()
        Q = vocab_size
        X = num_states
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.alpha = torch.nn.Parameter(torch.FloatTensor(Q).unsqueeze(dim=0))
        self.omega = torch.nn.Parameter(torch.FloatTensor(Q).unsqueeze(dim=0))
        self.E_shape = (Q, X)
        self.T_shape = (X, Q, Q)
        self.E = torch.nn.Parameter(torch.FloatTensor(*self.E_shape))
        self.T = torch.nn.Parameter(torch.FloatTensor(*self.T_shape))
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.uniform_(p.data, 0, 1)

    def encode_symbols(self, symbols):
        return torch.LongTensor([self.vocab.index(x) for x in symbols])

    def encode_path(self, path):
        return torch.LongTensor([self.states.index(q) for q in path])

    def p_path(self, qs):
        q_initial, *_, q_final = qs
        result = 1.0
        result *= self.alpha[q_initial]
        for t in range(len(qs) - 1):
          result *= self.T[qs[t], qs[t + 1]]
        result *= self.omega[q_final]
        return result

    def p_words_given_path(self, qs, xs):
        """ Return p(words | path) under the given PFA. """
        result = 1.0
        for q, x in zip(qs, xs):
          result *= self.E[q, x]
        return result

    def p_joint(self, qs, xs):
        """
        Return p(path, words).
        If words is length K, path must be length K+1 (because we include the initial state).
        """
        assert len(qs) == len(xs) + 1
        q_initial, *_, q_final = qs
        result = torch.tensor([1.0])

        result *= self.alpha[q_initial]  # multiply in probability to start in the first state of the path
        for t, x in enumerate(xs):
          result *= self.E[qs[t], x]
          result *= self.T[qs[t], qs[t + 1]]
        result *= self.omega[q_final]  # multiply in probability to end at the last state of the path

        return result

    def forward(self, xs, padding=0):
        # B x L
        device = xs.device
        #bishengliu's unpadding code https://gist.github.com/binshengliu/ecb15b68e14f8c70d5da10244f08beba
        mask = (xs != padding)
        valid = (mask.sum(dim=1))
        xs = xs[mask].split(valid.tolist())


        if self.training:
            prob = torch.tensor([0.]).to(device)
            for seq in xs:
              K = len(seq)
              chart = torch.zeros(self.num_states, K + 1).to(device)
              for t in reversed(range(K + 1)):
                for q_ in range(self.num_states):
                  if t == K:
                    chart[q_, t] = entmax.sparsemax_loss(self.omega, torch.tensor([q_]))
                  else:
                    chart[q_, t] = entmax.sparsemax_loss(self.E[q_:q_+1], torch.tensor([seq[t]])) + \
                                   (entmax.sparsemax_loss(self.T[q_:q_+1, seq[t]].expand(self.num_states, self.num_states),
                                                          torch.arange(self.num_states,
                                                                       dtype=torch.long)) +
                                   chart[:, t + 1].clone()).logsumexp(dim=-1)
                #breakpoint()
              #breakpoint()
              prob += (entmax.sparsemax_loss(self.alpha.expand(self.num_states, self.num_states), torch.arange(self.num_states,
                                                                     dtype=torch.long)) + \
                      chart[:, 0].clone()).logsumexp(dim=-1)
              #if total_nll <= 0:
              #    breakpoint()
              print(prob)
            return prob

        else:
            prob = torch.tensor([0.]).to(device)
            for seq in xs:
              K = len(seq)
              chart = torch.zeros(self.num_states, K + 1).to(device)
              for t in reversed(range(K + 1)):
                for q_ in range(self.num_states):
                  if t == K:
                    chart[q_, t] = torch.log(entmax.sparsemax(self.omega, dim=-1)[:, q_])
                  else:
                    chart[q_, t] = torch.log(entmax.sparsemax(self.E, dim=-1)[q_, seq[t]]) + \
                                   (torch.log(entmax.sparsemax(self.T, dim=-1)[q_, seq[t]]) +
                                   chart[:, t + 1].clone()).logsumexp(dim=-1)
                #breakpoint()
              prob += torch.log((entmax.sparsemax(self.alpha, dim=-1) + chart[:, 0].clone()).logsumexp(dim=-1))
              #if total_nll <= 0:
              #    breakpoint()
              print(prob)
            return prob


def inverse_softplus(x, beta=torch.tensor([10.]), threshold = .5):
    return torch.log(torch.exp(beta*x)-1)/beta