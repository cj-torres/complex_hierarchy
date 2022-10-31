import torch, math, types


class L0_Regularizer(torch.nn.Module):

    def __init__(self, original_module: torch.nn.Module, lam: float,
                      weight_decay: float = 0, is_neural: bool = False,
                    temperature: float = 2/3, droprate_init=0.5,
                 limit_a = -.1, limit_b = 1.1, epsilon = 1e-6
                 ):
        # is_neural support only for models consisting of 1-d and 2-d tensors
        # other support could be chaotic
        super(L0_Regularizer, self).__init__()
        self.module = original_module

        self.pre_parameters = torch.nn.ParameterDict(
            {name + "_p": param for name, param in self.module.named_parameters()}
        )

        self.param_names = [name for name, param in self.module.named_parameters()]
        self.is_neural = is_neural
        self.mask_parameters = torch.nn.ParameterDict()
        self.module.original_forward = self.module.forward
        self.lam = lam
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.droprate_init = droprate_init
        self.limit_a = limit_a
        self.limit_b = limit_b
        self.epsilon = epsilon

        if is_neural:
            for name, param in self.pre_parameters.items():
                if len(param.size()) == 2:
                    mask = torch.nn.Parameter(torch.Tensor(param.size()[0]))
                    self.mask_parameters.update({name + "_m": mask})

        else:
            for name, param in self.pre_parameters.items():
                mask = torch.nn.Parameter(torch.Tensor(param.size()))
                self.mask_parameters.update({name + "_m": mask})

    ''' 
    Below code direct copy with adaptations from codebase for: 
    
    Louizos, C., Welling, M., & Kingma, D. P. (2017). 
    Learning sparse neural networks through L_0 regularization. 
    arXiv preprint arXiv:1712.01312.
    '''

    def reset_parameters(self):
        for name, weight in self.module.pre_parameters():
            if "bias" in name:
                torch.nn.init.constant(weight, 0.0)
            else:
                torch.nn.init.xavier_uniform(weight)

        for name, weight in self.module.mask_parameters():
            torch.nn.init.normal(weight, math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)


    def constrain_parameters(self):
        for name, weight in self.module.mask_parameters():
            weight.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x, param):
        """Implements the CDF of the 'stretched' concrete distribution"""
        '''references parameters'''
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.nn.functional.sigmoid(
            logits * self.temperature - self.mask_parameters[param+"_m"]).clamp(min=self.epsilon, max=1 - self.epsilon)

    def quantile_concrete(self, x, param):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        '''references parameters'''
        y = torch.nn.functional.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.mask_parameters[param+"_m"]) / self.temperature)
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def _reg_w(self, param):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        '''is_neural is old method, calculates wrt columns first multiplied by expected values of gates
           second method calculates wrt all parameters
        '''

        # why is this negative? will investigate behavior at testing
        if self.is_neural:
            logpw_col = torch.sum(- (.5 * self.weight_decay * self.pre_parameters[param+"_p"].pow(2)) - self.lam, 1)
            logpw = torch.sum((1 - self.cdf_qz(0, param)) * logpw_col)
        else:
            logpw_l2 = - (.5 * self.weight_decay * param.pow(2)) - self.lam
            logpw = torch.sum((1 - self.cdf_qz(0, param)) * logpw_l2)

        return logpw

    def regularization(self):
        r_total = torch.Tensor([])
        for param in self.param_names:
            r_total = torch.cat([r_total, self._reg_w(param)])
        return r_total.sum()

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        # Variable deprecated and removed
        eps = self.floatTensor(size).uniform_(self.epsilon, 1-self.epsilon)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.in_features))
            z = self.quantile_concrete(eps)
            return torch.nn.functional.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.nn.functional.sigmoid(self.qz_loga).view(1, self.in_features).expand(batch_size, self.in_features)
            return torch.nn.functional.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.in_features)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask.view(self.in_features, 1) * self.weights

    def forward(self, input):
        if self.local_rep or not self.training:
            z = self.sample_z(input.size(0), sample=self.training)
            xin = input.mul(z)
            output = xin.mm(self.weights)
        else:
            weights = self.sample_weights()
            output = input.mm(weights)
        if self.use_bias:
            output.add_(self.bias)
        return output

    def new_forward():


    def forward():
