import torch, math
import numpy as np


class GaussianSampler(torch.nn.Module):
    """
    Module to sample from multivariate Gaussian distributions.
    Converts input into a sampled vector of the same size.
    """

    def __init__(self, input_size: int):
        """
        :param input_size: size of vector that will be transformed to twice its length
        Variables:
        gauss_parameter_generator is a matrix that transforms input into a vector, first half of vector is the mean,
        second half the standard deviation
        """
        super(GaussianSampler, self).__init__()
        self.input_size = input_size
        self.gauss_parameter_generator = torch.nn.Linear(self.input_size, self.input_size*2)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.zeros_(p.data)

    def sample(self, stats):
        """
        Takes vector of parameters, transforms these into distributions using supported torch class
        Returns parameters and samples
        :param stats: vector of parameters
        :return: Tuple(Tensor of means, tensor of standard deviations, tensor of samples)
        """
        mu = stats[:, :, self.input_size:]
        std = torch.nn.functional.softplus(stats[:, :, :self.input_size], beta=1)
        norm = torch.distributions.normal.Normal(mu, std)
        samples = norm.rsample()
        log_probs = norm.log_prob(samples).sum(dim=-1)
        return mu, std, samples, log_probs

    def forward(self, x: torch.Tensor):
        """
        Takes input vector x, produces samples
        :param x: Input tensor
        :return: Tuple(Tensor of samples, tensor of means, tensor of standard deviations)
        """
        # x is of size BxTxH
        gauss_parameters = self.gauss_parameter_generator(x)
        mu, std, sample, log_probs = self.sample(gauss_parameters)
        return sample, mu, std, log_probs


class GaussianLSTM(torch.nn.Module):
    """
    LSTM with a sampling layer on the output
    """
    def __init__(self, input_sz: int, hidden_sz: int, num_layers: int):
        super(GaussianLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_size=self.input_sz,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.h_gauss_sampler = GaussianSampler(hidden_sz)

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                torch.nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor,
                init_states = None):

        bs, seq_sz, _ = x.size()

        if init_states is None:
            h_t, c_t = (torch.stack([torch.zeros(self.num_layers, self.hidden_size).to(x.device)] * bs, dim=-2),
                        torch.stack([torch.zeros(self.num_layers, self.hidden_size).to(x.device)] * bs, dim=-2))
        else:
            h_t, c_t = init_states
        output, (_, _) = self.lstm(x, (h_t, c_t))
        h_seq, h_mus, h_stds, log_probs = self.h_gauss_sampler(output)

        return h_seq, (h_mus, h_stds), log_probs


class SequentialVariationalIB(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_size, final_layer_sz, n_flows, n_layers):

        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.noise_size = hidden_size
        self.final_layer_sz = final_layer_sz
        self.flow = create_model(hidden_size, hidden_size, n_flows, n_layers)

        self.__build_model()

    def __build_model(self):
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0
        )

        self.lstm = GaussianLSTM(self.embedding_dim, self.hidden_size, self.num_layers)

        self.initial_state = torch.nn.Parameter(torch.stack(
            [torch.randn(self.num_layers, self.hidden_size),
             torch.randn(self.num_layers, self.hidden_size)]
        ))

        self.final_layer = torch.nn.Linear(self.noise_size, self.final_layer_sz)
        self.decoder = torch.nn.Linear(self.final_layer_sz, self.vocab_size)

    def get_initial_state(self, batch_size):
        init_a, init_b = self.initial_state
        return torch.stack([init_a] * batch_size, dim=-2), torch.stack([init_b] * batch_size, dim=-2)  # L x B x H

    def encode(self, X):
        # X is of shape B x T
        batch_size, seq_len = X.shape
        embedding = self.word_embedding(X)
        init_h, init_c = self.get_initial_state(batch_size)  # init_h is L x B x H
        h_seq, (h_mus, h_stds), log_probs = self.lstm(embedding, (init_h, init_c))  # output is B x T x H
        return h_seq, (h_mus, h_stds), log_probs

    def decode(self, h):
        pre = self.final_layer(h)
        logits = self.decoder(pre)
        return logits

    def forward(self, x):
        h_seq, h_stats, log_probs = self.encode(x)
        y_hat = self.decode(h_seq)

        return y_hat, h_stats, h_seq, log_probs

    def lm_loss(self, y, y_hat, mask):
        """
        Just the cross entropy loss
        :param y: Target tensor
        :param y_hat: Output tensor
        :param mask: Mask selecting only relevant entries for the loss
        :return: Cross entropy
        """
        ce = torch.nn.CrossEntropyLoss()
        # Y contains the target token indices. Shape B x T
        # Y_hat contains distributions
        ce_loss = ce(y_hat[mask], y[mask])
        return ce_loss

    def mi_loss(self, zx_seq, zx_log_probs, mask):
        '''
        Calculates KL divergence between
        :param zx_seq: Samples from distribution of z given x
        :param zx_log_probs: Sequence of log probabilities for z given x for zx_seq
        :param mask: Mask of relevant entries
        :return: Mean KL divergence between distribution specified by stats and Z (a standard multivariate normal)
        '''
        #std = stats[1]
        #mu = stats[0]

        #std_flat = std[mask]  # B*TxH remove padded entries
        #mu_flat = mu[mask]  # B*TxH

        zx_seq_flat = zx_seq[mask]
        norm = torch.distributions.normal.Normal(torch.zeros_like(zx_seq_flat[0]), torch.ones_like(zx_seq_flat[0]))

        y_samples, log_dets = self.flow(zx_seq_flat)
        y_log_likelihood = norm.log_prob(y_samples).sum(dim=-1) + log_dets

        kld = zx_log_probs[mask] - y_log_likelihood
        mi = kld.mean()

        # z given x is a sequence of distributions
        #zx_normals = torch.distributions.multivariate_normal.MultivariateNormal(mu_flat, std_flat)

        # z is a standard normal multivariate with the same dimensions as z given x
        #z = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros_like(mu_flat),
        #                                                               torch.diag_embed(torch.ones_like(mu_flat)))
        #kld = torch.distributions.kl_divergence(zx_normals, z)

        return mi

###
# de Cao et al. 2019 code for BNAF below
#
#
###


class Sequential(torch.nn.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for computing the output of
    the function alongside with the log-det-Jacobian of such transformation.
    """

    def forward(self, inputs: torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """

        log_det_jacobian = 0.0
        for i, module in enumerate(self._modules.values()):
            inputs, log_det_jacobian_ = module(inputs)
            log_det_jacobian = log_det_jacobian + log_det_jacobian_
        return inputs, log_det_jacobian


class BNAF(torch.nn.Sequential):
    """
    Class that extends ``torch.nn.Sequential`` for constructing a Block Neural
    Normalizing Flow.
    """

    def __init__(self, *args, res: str = None):
        """
        Parameters
        ----------
        *args : ``Iterable[torch.nn.Module]``, required.
            The modules to use.
        res : ``str``, optional (default = None).
            Which kind of residual connection to use. ``res = None`` is no residual
            connection, ``res = 'normal'`` is ``x + f(x)`` and ``res = 'gated'`` is
            ``a * x + (1 - a) * f(x)`` where ``a`` is a learnable parameter.
        """

        super(BNAF, self).__init__(*args)

        self.res = res

        if res == "gated":
            self.gate = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(1)))

    def forward(self, inputs: torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The output tensor and the log-det-Jacobian of this transformation.
        """

        outputs = inputs
        grad = None

        for module in self._modules.values():
            outputs, grad = module(outputs, grad)

            grad = grad if len(grad.shape) == 4 else grad.view(grad.shape + [1, 1])

        assert inputs.shape[-1] == outputs.shape[-1]

        if self.res == "normal":
            return inputs + outputs, torch.nn.functional.softplus(grad.squeeze()).sum(
                -1
            )
        elif self.res == "gated":
            return self.gate.sigmoid() * outputs + (1 - self.gate.sigmoid()) * inputs, (
                torch.nn.functional.softplus(grad.squeeze() + self.gate)
                - torch.nn.functional.softplus(self.gate)
            ).sum(-1)
        else:
            return outputs, grad.squeeze().sum(-1)

    def _get_name(self):
        return "BNAF(res={})".format(self.res)


class Permutation(torch.nn.Module):
    """
    Module that outputs a permutation of its input.
    """

    def __init__(self, in_features: int, p: list = None):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features.
        p : ``list`` or ``str``, optional (default = None)
            The list of indeces that indicate the permutation. When ``p`` is not a
            list, if ``p = 'flip'``the tensor is reversed, if ``p = None`` a random
            permutation is applied.
        """

        super(Permutation, self).__init__()

        self.in_features = in_features

        if p is None:
            self.p = np.random.permutation(in_features)
        elif p == "flip":
            self.p = list(reversed(range(in_features)))
        else:
            self.p = p

    def forward(self, inputs: torch.Tensor):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        Returns
        -------
        The permuted tensor and the log-det-Jacobian of this permutation.
        """

        return inputs[:, self.p], 0

    def __repr__(self):
        return "Permutation(in_features={}, p={})".format(self.in_features, self.p)


class MaskedWeight(torch.nn.Module):
    """
    Module that implements a linear layer with block matrices with positive diagonal blocks.
    Moreover, it uses Weight Normalization (https://arxiv.org/abs/1602.07868) for stability.
    """

    def __init__(
        self, in_features: int, out_features: int, dim: int, bias: bool = True
    ):
        """
        Parameters
        ----------
        in_features : ``int``, required.
            The number of input features per each dimension ``dim``.
        out_features : ``int``, required.
            The number of output features per each dimension ``dim``.
        dim : ``int``, required.
            The number of dimensions of the input of the flow.
        bias : ``bool``, optional (default = True).
            Whether to add a parametrizable bias.
        """

        super(MaskedWeight, self).__init__()
        self.in_features, self.out_features, self.dim = in_features, out_features, dim

        weight = torch.zeros(out_features, in_features)
        for i in range(dim):
            weight[
                i * out_features // dim : (i + 1) * out_features // dim,
                0 : (i + 1) * in_features // dim,
            ] = torch.nn.init.xavier_uniform_(
                torch.Tensor(out_features // dim, (i + 1) * in_features // dim)
            )

        self._weight = torch.nn.Parameter(weight)
        self._diag_weight = torch.nn.Parameter(
            torch.nn.init.uniform_(torch.Tensor(out_features, 1)).log()
        )

        self.bias = (
            torch.nn.Parameter(
                torch.nn.init.uniform_(
                    torch.Tensor(out_features),
                    -1 / math.sqrt(out_features),
                    1 / math.sqrt(out_features),
                )
            )
            if bias
            else 0
        )

        mask_d = torch.zeros_like(weight)
        for i in range(dim):
            mask_d[
                i * (out_features // dim) : (i + 1) * (out_features // dim),
                i * (in_features // dim) : (i + 1) * (in_features // dim),
            ] = 1

        self.register_buffer("mask_d", mask_d)

        mask_o = torch.ones_like(weight)
        for i in range(dim):
            mask_o[
                i * (out_features // dim) : (i + 1) * (out_features // dim),
                i * (in_features // dim) :,
            ] = 0

        self.register_buffer("mask_o", mask_o)

    def get_weights(self):
        """
        Computes the weight matrix using masks and weight normalization.
        It also compute the log diagonal blocks of it.
        """

        w = torch.exp(self._weight) * self.mask_d + self._weight * self.mask_o

        w_squared_norm = (w ** 2).sum(-1, keepdim=True)

        w = self._diag_weight.exp() * w / w_squared_norm.sqrt()

        wpl = self._diag_weight + self._weight - 0.5 * torch.log(w_squared_norm)

        return w.t(), wpl.t()[self.mask_d.bool().t()].view(
            self.dim, self.in_features // self.dim, self.out_features // self.dim
        )

    def forward(self, inputs, grad: torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal block of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
        transformations combined with this transformation.
        """

        w, wpl = self.get_weights()

        g = wpl.transpose(-2, -1).unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)

        return (
            inputs.matmul(w) + self.bias,
            torch.logsumexp(g.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3), -1)
            if grad is not None
            else g,
        )

    def __repr__(self):
        return "MaskedWeight(in_features={}, out_features={}, dim={}, bias={})".format(
            self.in_features,
            self.out_features,
            self.dim,
            not isinstance(self.bias, int),
        )


class Tanh(torch.nn.Tanh):
    """
    Class that extends ``torch.nn.Tanh`` additionally computing the log diagonal
    blocks of the Jacobian.
    """

    def forward(self, inputs, grad: torch.Tensor = None):
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            The input tensor.
        grad : ``torch.Tensor``, optional (default = None).
            The log diagonal blocks of the partial Jacobian of previous transformations.
        Returns
        -------
        The output tensor and the log diagonal blocks of the partial log-Jacobian of previous
        transformations combined with this transformation.
        """

        g = -2 * (inputs - math.log(2) + torch.nn.functional.softplus(-2 * inputs))
        return (
            torch.tanh(inputs),
            (g.view(grad.shape) + grad) if grad is not None else g,
        )


def create_model(n_dims, hidden_dim, n_flows, n_layers, residual=None):
    # batch first, then data of size n_dims

    flows = []
    for f in range(n_flows):
        layers = []
        for _ in range(n_layers - 1):
            layers.append(
                MaskedWeight(
                    n_dims * hidden_dim,
                    n_dims * hidden_dim,
                    dim=n_dims,
                )
            )
            layers.append(Tanh())

        flows.append(
            BNAF(
                *(
                    [
                        MaskedWeight(
                            n_dims, n_dims * hidden_dim, dim=n_dims
                        ),
                        Tanh(),
                    ]
                    + layers
                    + [
                        MaskedWeight(
                            n_dims * hidden_dim, n_dims, dim=n_dims
                        )
                    ]
                ),
                res=residual if f < n_flows - 1 else None
            )
        )

        if f < n_flows - 1:
            flows.append(Permutation(n_dims, "flip"))

    model = Sequential(*flows)
    return model