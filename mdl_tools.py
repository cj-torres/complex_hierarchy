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
            {name.replace(".", "#") + "_p": param for name, param in self.module.named_parameters()}
        )

        self.param_names = [name.replace(".", "#") for name, param in self.module.named_parameters()]
        self.is_neural = is_neural
        self.mask_parameters = torch.nn.ParameterDict()
        self.lam = lam
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.droprate_init = droprate_init
        self.limit_a = limit_a
        self.limit_b = limit_b
        self.epsilon = epsilon

        if is_neural:
            for name, param in self.module.named_parameters():
                if len(param.size()) == 2:
                    mask = torch.nn.Parameter(torch.Tensor(param.size()[0]))
                    self.mask_parameters.update({name.replace(".", "#") + "_m": mask})

        else:
            for name, param in self.module.named_parameters():
                mask = torch.nn.Parameter(torch.Tensor(param.size()))
                self.mask_parameters.update({name.replace(".", "#") + "_m": mask})

        # below code guts the module of its previous parameters,
        # allowing them to be replaced by non-leaf tensors

        for name in self.param_names:
            L0_Regularizer.recursive_del(self.module, name)

        self.reset_parameters()
    ''' 
    Below code direct copy with adaptations from codebase for: 
    
    Louizos, C., Welling, M., & Kingma, D. P. (2017). 
    Learning sparse neural networks through L_0 regularization. 
    arXiv preprint arXiv:1712.01312.
    '''

    def reset_parameters(self):
        for name, weight in self.pre_parameters.items():
            if "bias" in name:
                torch.nn.init.constant_(weight, 0.0)
            else:
                torch.nn.init.xavier_uniform_(weight)

        for name, weight in self.mask_parameters.items():
            torch.nn.init.normal_(weight, math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

    def constrain_parameters(self):
        for name, weight in self.mask_parameters.items():
            weight.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x, param):
        """Implements the CDF of the 'stretched' concrete distribution"""
        # references parameters
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(
            logits * self.temperature - self.mask_parameters[param+"_m"]).clamp(min=self.epsilon, max=1 - self.epsilon)

    def quantile_concrete(self, x, param):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        # references parameters
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.mask_parameters[param+"_m"]) / self.temperature)
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def _reg_w(self, param):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        """is_neural is old method, calculates wrt columns first multiplied by expected values of gates
           second method calculates wrt all parameters
        """

        # why is this negative? will investigate behavior at testing
        # reversed negative value, value should increase with description length
        if self.is_neural:
            logpw_col = torch.sum(- (.5 * self.weight_decay * self.pre_parameters[param+"_p"].pow(2)) - self.lam, 1)
            logpw = torch.sum((1 - self.cdf_qz(0, param)) * logpw_col)
        else:
            logpw_l2 = - (.5 * self.weight_decay * self.pre_parameters[param+"_p"].pow(2)) - self.lam
            logpw = torch.sum((1 - self.cdf_qz(0, param)) * logpw_l2)

        return -logpw

    def regularization(self):
        r_total = torch.Tensor([])
        for param in self.param_names:
            r_total = torch.cat([r_total, self._reg_w(param).unsqueeze(dim=0)])
        return r_total.sum()

    def count_l0(self):
        total = []
        for param in self.param_names:
            total.append(torch.sum(1 - self.cdf_qz(0, param)).unsqueeze(dim=0))
        return torch.cat(total).sum()

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        # Variable deprecated and removed
        eps = torch.FloatTensor(size).uniform_(self.epsilon, 1-self.epsilon)
        return eps

    def sample_z(self, batch_size, param, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        new_size = torch.Size([batch_size]) + self.mask_parameters[param+"_m"].size()
        if sample:
            eps = self.get_eps(new_size)
            z = self.quantile_concrete(eps)
            return torch.nn.functional.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(self.mask_parameters[param+"_m"]).unsqueeze(dim=0).expand(new_size)
            return torch.nn.functional.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)

    def sample_weights(self, param):
        z = self.quantile_concrete(self.get_eps(self.mask_parameters[param+"_m"].size()), param)
        mask = torch.nn.functional.hardtanh(z, min_val=0, max_val=1)
        return mask * self.pre_parameters[param+"_p"]

    def forward(self, input):
        """rewrite parameters (tensors) of core module and feedforward"""
        for param in self.param_names:
            L0_Regularizer.recursive_set(self.module, param, self.sample_weights(param))

        return self.module(input)

    @staticmethod
    def recursive_get(obj, att_name):
        if "#" in att_name:
            first, last = att_name.split("#", 1)
            L0_Regularizer.recursive_get(getattr(obj, first), last)
        else:
            return getattr(obj, att_name)

    @staticmethod
    def recursive_set(obj, att_name, val):
        if "#" in att_name:
            first, last = att_name.split("#", 1)
            L0_Regularizer.recursive_set(getattr(obj, first), last, val)
        else:
            setattr(obj, att_name, val)

    @staticmethod
    def recursive_del(obj, att_name):
        if "#" in att_name:
            first, last = att_name.split("#", 1)
            L0_Regularizer.recursive_del(getattr(obj, first), last)
        else:
            delattr(obj, att_name)


""" UNIT TESTS """


# 1 - element test
def test_1():

    in_tensors = torch.randn(500, 5)
    target = torch.zeros_like(in_tensors)
    target[:, 0] = in_tensors[:, 0]

    model = L0_Regularizer(torch.nn.Linear(5, 5), .01)
    opt = torch.optim.Adam(model.parameters())

    for i in range(10000):
        opt.zero_grad()
        loss = ((model(in_tensors) - target).pow(2)).mean()
        re_loss = model.regularization()

        total = loss - re_loss

        total.backward()
        opt.step()
        if i%100 == 0:
            print(loss.item())
            print(model.count_l0().item())


# all-element test
def test_2():

    in_tensors = torch.randn(500, 5)
    target = in_tensors.sum(dim=1).unsqueeze(dim=1).expand(-1, 5)

    model = L0_Regularizer(torch.nn.Linear(5, 5), .01)
    opt = torch.optim.Adam(model.parameters())

    for i in range(10000):
        opt.zero_grad()
        loss = ((model(in_tensors) - target).pow(2)).mean()
        re_loss = model.regularization()

        total = loss - re_loss

        total.backward()
        opt.step()
        if i%100 == 0:
            print(loss.item())
            print(model.count_l0().item())