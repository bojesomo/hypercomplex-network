from .ops import *
import math
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_ntuple
from .parametrize import register_parametrization
from scipy.stats import chi
from torch import _VF
from functools import partial
from numpy.random import RandomState
from einops import rearrange


class Hamilton(nn.Module):
    def __init__(self, n_divs=4, comp_mat=None):
        super().__init__()
        self.register_buffer("n_divs", torch.tensor(n_divs))
        if comp_mat is None:
            comp_mat = get_comp_mat(n_divs)  # TODO - move this to the caller to reduce timing
        self.register_buffer('comp_mat', torch.from_numpy(comp_mat))

    def forward(self, weights):
        cat_kernel_hypercomplex_i = []
        for comp_i in self.comp_mat:
            kernel_hypercomplex_i = []
            for idx, ii in enumerate(comp_i):
                itr = np.abs(ii)
                sign = np.sign(ii) if np.sign(ii) != 0 else 1
                kernel_hypercomplex_i.append(sign * weights[itr])
            cat_kernel_hypercomplex_i.append(torch.cat(kernel_hypercomplex_i, dim=0))

        return torch.cat(cat_kernel_hypercomplex_i, dim=1)


class Concatenate(nn.Module):
    def __init__(self, dim=-1, n_divs=1):
        super(Concatenate, self).__init__()
        self.n_divs = n_divs
        self.dim = dim

    def forward(self, x):
        x_splits = [torch.chunk(x_i, chunks=self.n_divs, dim=self.dim) for x_i in x]
        components = [torch.cat([x_split[component] for x_split in x_splits], dim=self.dim) for component
                      in range(self.n_divs)]
        return torch.cat(components, dim=self.dim)


# Dropout using same mask for different components
class HyperDropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False, n_divs=1, dim=-1, ignore_print=None, name=None):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.n_divs = n_divs
        self.dim = dim
        self.inplace = inplace

        self.my_list = ['p', 'inplace', 'n_divs']
        self.ignore_print = [] if ignore_print is None else ignore_print
        self.name = name

    def forward(self, x):
        if self.training:
            shape = torch.tensor(x.shape)
            shape[self.dim] //= self.n_divs
            repeat_dim = [1] * x.dim()
            repeat_dim[self.dim] = self.n_divs
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            mask = binomial.sample(shape).repeat(*repeat_dim) * (1.0/(1-self.p))
            return x.mul_(mask) if self.inplace else x.mul(mask)
        return x
    
    # def extra_repr(self) -> str:
    #     return f"p= {self.p}, inplace={self.inplace}, n_divs={self.n_divs}"
    def extra_repr(self) -> str:
        extra = ', '.join(
            [f"{key}={self.__getattribute__(key)}" for key in
             self.my_list if key not in self.ignore_print])
        return extra

    def __repr__(self):
        name = self.name or 'HyperDropout'
        return f"{name}({self.extra_repr()})"
    

# ############################
# '''Hypercomplex Linear!'''
# ###########################
class HyperLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, n_divs=4, ignore_print=None, name=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_divs = n_divs
        self.weight = nn.Parameter(torch.rand(size=(n_divs, self.out_features // n_divs, self.in_features // n_divs)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.my_list = ['in_features', 'out_features', 'bias', 'n_divs']
        self.ignore_print = [] if ignore_print is None else ignore_print
        self.name = name

        self.comp_mat = get_comp_mat(n_divs)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(1) + self.weight.size(2) * self.n_divs))
        self.w_shape = self.weight.shape
        phase = np.random.uniform(low=-np.pi, high=np.pi, size=self.w_shape[1:]).astype(np.float32)
        phi = np.random.uniform(low=-1, high=1, size=(self.n_divs - 1, *self.w_shape[1:])).astype(np.float32)
        phi /= np.sqrt((phi ** 2 + 1e-4).sum(axis=0))
        modulus = chi.rvs(df=self.n_divs, loc=0, scale=stdv, size=self.w_shape[1:]).astype(np.float32)

        weight = [modulus * np.cos(phase)]
        weight.extend([modulus * phi_ * np.sin(phase) for phi_ in phi])

        self.weight.data = torch.from_numpy(np.stack(weight))

        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        hamilton = fast_hypercomplex(self.weight, self.n_divs, comp_mat=self.comp_mat)
        return F.linear(x, hamilton, self.bias)

    def extra_repr(self) -> str:
        extra = ', '.join(
            [f"{key}={self.__getattribute__(key) if key is not 'bias' else str(self.bias is not None)}"
             for key in self.my_list if key not in self.ignore_print])
        return extra

    # def extra_repr(self) -> str:
    #     return f"in_features={self.in_features}, out_features={self.out_features}, " + \
    #            f"bias={self.bias is not None}, n_divs={self.n_divs}"
    def __repr__(self):
        name = self.name or 'HyperLinear'
        return f"{name}({self.extra_repr()})"


class HyperLinear_(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, n_divs=4):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.n_divs = n_divs
        # self.weight = nn.Parameter(torch.FloatTensor(self.in_features // n_divs, self.out_features))
        self.weight = nn.Parameter(torch.rand(size=(n_divs, self.out_features // n_divs, self.in_features // n_divs)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.comp_mat = get_comp_mat(n_divs)
        self._reset_parameters()
        register_parametrization(self, 'weight', Hamilton(n_divs=self.n_divs, comp_mat=self.comp_mat))

    def _reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(1) + self.weight.size(2) * self.n_divs))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, " + \
               f"bias={self.bias is not None}, n_divs={self.n_divs}"


# ############################
# '''Hypercomplex Conv'''
# ############################
class HyperConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, n_divs=4,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', operation='convolution2d', ignore_print=None, name=None):
        # hypercomplex_format=True):

        super(HyperConv, self).__init__()
        n = int(''.join(c for c in operation if c.isdigit()))
        self.n_divs = n_divs
        self.in_channels = in_channels  # // self.n_divs
        self.out_channels = out_channels  # // self.n_divs
        self.stride = to_ntuple(n)(stride)
        self.padding = to_ntuple(n)(padding)
        self.groups = groups
        self.dilation = to_ntuple(n)(dilation)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.operation = operation

        assert (self.in_channels % self.groups == 0)
        assert (self.out_channels % self.groups == 0)

        self.kernel_size, self.w_shape = self.get_kernel_and_weight_shape(kernel_size)

        self.weight = nn.Parameter(torch.rand(*self.w_shape))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.my_list = ['in_channels', 'out_channels', 'bias', 'kernel_size', 'stride', 'padding', 'n_divs',
                        'operation']
        self.ignore_print = [] if ignore_print is None else ignore_print
        self.name = name

        self.comp_mat = get_comp_mat(n_divs)
        self.reset_parameters()

    def get_kernel_and_weight_shape(self, kernel_size):
        ks = None
        if self.operation == 'convolution1d':
            if type(kernel_size) is not int:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 1d convolution. The kernel size
                    must be integer in the case. Found kernel_size = """ + str(kernel_size)
                )
            else:
                ks = tuple((kernel_size,))
        else:  # in case it is 2d or 3d.
            if type(kernel_size) is int:
                if self.operation == 'convolution2d':
                    ks = to_ntuple(2)(kernel_size)  # (kernel_size, kernel_size)
                elif self.operation == 'convolution3d':
                    ks = to_ntuple(3)(kernel_size)  # (kernel_size, kernel_size, kernel_size)
            else:  # elif type(kernel_size) is not int:
                if self.operation == 'convolution2d' and len(kernel_size) != 2:
                    raise ValueError(
                        """An invalid kernel_size was supplied for a 2d convolution. The kernel size
                        must be either an integer or a tuple of 2. Found kernel_size = """ + str(kernel_size)
                    )
                elif self.operation == 'convolution3d' and len(kernel_size) != 3:
                    raise ValueError(
                        """An invalid kernel_size was supplied for a 3d convolution. The kernel size
                        must be either an integer or a tuple of 3. Found kernel_size = """ + str(kernel_size)
                    )
                else:
                    ks = kernel_size
        w_shape = (self.n_divs, self.out_channels // self.n_divs, self.in_channels // self.groups // self.n_divs) + \
                  (*ks,)
        return ks, w_shape

    def reset_parameters(self):
        # print(self.w_shape)
        # receptive_field = np.prod(self.w_shape[3:])
        receptive_field = np.prod(self.kernel_size)
        # fan_in = self.n_divs * self.w_shape[2] * receptive_field
        # fan_out = self.w_shape[1] * receptive_field
        n_kernel = len(self.kernel_size)
        in_features = self.w_shape[-n_kernel]
        out_features = self.w_shape[-(n_kernel+1)]
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field

        if self.init_criterion == 'glorot':
            stdv = np.sqrt(2 / (fan_in + fan_out))
        elif self.init_criterion == 'he':
            stdv = np.sqrt(2 / fan_in)
        else:
            raise ValueError('Invalid criterion: ' + self.init_criterion)
        # print(stdv)
        if self.weight_init == 'hypercomplex':
            stdv /= np.sqrt(self.n_divs)
        # print(stdv)
        phase = np.random.uniform(low=-np.pi, high=np.pi, size=self.w_shape[1:]).astype(np.float32)
        phi = np.random.uniform(low=-1, high=1, size=(self.n_divs - 1, *self.w_shape[1:])).astype(np.float32)
        phi /= np.sqrt((phi ** 2 + 1e-4).sum(axis=0))
        modulus = chi.rvs(df=self.n_divs, loc=0, scale=stdv, size=self.w_shape[1:]).astype(np.float32)

        weight = [modulus * np.cos(phase)]
        weight.extend([modulus * phi_ * np.sin(phase) for phi_ in phi])

        self.weight.data = torch.from_numpy(np.stack(weight))

        # self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        assert 3 <= x.dim() <= 5, "The convolutional input x is either 3, 4 or 5 dimensional. x.dim = " + str(
            x.dim())
        kernels = fast_hypercomplex(self.weight, n_divs=self.n_divs, comp_mat=self.comp_mat)
        convfunc = {3: F.conv1d, 4: F.conv2d, 5: F.conv3d}[x.dim()]
        return convfunc(x, kernels, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)

    def extra_repr(self) -> str:
        extra = ', '.join(
            [f"{key}={self.__getattribute__(key) if key is not 'bias' else str(self.bias is not None)}" for key in
             self.my_list if key not in self.ignore_print])
        return extra

    def __repr__(self):
        name = self.name or 'HyperConv'
        return f"{name}({self.extra_repr()})"


# ############################
# '''Hypercomplex ConvTranspose'''
# ############################
class HyperConvTranspose(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, n_divs=4,
                 dilation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='hypercomplex', operation='convolution2d', ignore_print=None, name=None):
        # hypercomplex_format=True):

        super().__init__()
        n = int(''.join(c for c in operation if c.isdigit()))
        self.n_divs = n_divs
        self.in_channels = in_channels  # // self.n_divs
        self.out_channels = out_channels  # // self.n_divs
        self.stride = to_ntuple(n)(stride)
        self.padding = to_ntuple(n)(padding)
        self.output_padding = to_ntuple(n)(output_padding)
        self.groups = groups
        self.dilation = to_ntuple(n)(dilation)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.operation = operation

        assert (self.in_channels % self.groups == 0)
        assert (self.out_channels % self.groups == 0)

        self.kernel_size, self.w_shape = self.get_kernel_and_weight_shape(kernel_size)

        self.weight = nn.Parameter(torch.rand(*self.w_shape))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.my_list = ['in_channels', 'out_channels', 'bias', 'kernel_size', 'stride', 'padding', 'output_padding',
                        'n_divs', 'operation']
        self.ignore_print = [] if ignore_print is None else ignore_print
        self.name = name

        self.comp_mat = get_comp_mat(n_divs)
        self.reset_parameters()

    def get_kernel_and_weight_shape(self, kernel_size):
        ks = None
        if self.operation == 'convolution1d':
            if type(kernel_size) is not int:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 1d convolution. The kernel size
                    must be integer in the case. Found kernel_size = """ + str(kernel_size)
                )
            else:
                ks = tuple((kernel_size,))
        else:  # in case it is 2d or 3d.
            if type(kernel_size) is int:
                if self.operation == 'convolution2d':
                    ks = to_ntuple(2)(kernel_size)  # (kernel_size, kernel_size)
                elif self.operation == 'convolution3d':
                    ks = to_ntuple(3)(kernel_size)  # (kernel_size, kernel_size, kernel_size)
            else:  # elif type(kernel_size) is not int:
                if self.operation == 'convolution2d' and len(kernel_size) != 2:
                    raise ValueError(
                        """An invalid kernel_size was supplied for a 2d convolution. The kernel size
                        must be either an integer or a tuple of 2. Found kernel_size = """ + str(kernel_size)
                    )
                elif self.operation == 'convolution3d' and len(kernel_size) != 3:
                    raise ValueError(
                        """An invalid kernel_size was supplied for a 3d convolution. The kernel size
                        must be either an integer or a tuple of 3. Found kernel_size = """ + str(kernel_size)
                    )
                else:
                    ks = kernel_size
        w_shape = (self.n_divs, self.out_channels // self.n_divs, self.in_channels // self.groups // self.n_divs) + \
                  (*ks,)
        return ks, w_shape

    def reset_parameters(self):
        # print(self.w_shape)
        # receptive_field = np.prod(self.w_shape[3:])
        receptive_field = np.prod(self.kernel_size)
        # fan_in = self.n_divs * self.w_shape[2] * receptive_field
        # fan_out = self.w_shape[1] * receptive_field
        n_kernel = len(self.kernel_size)
        in_features = self.w_shape[-n_kernel]
        out_features = self.w_shape[-(n_kernel + 1)]
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field

        if self.init_criterion == 'glorot':
            stdv = np.sqrt(2 / (fan_in + fan_out))
        elif self.init_criterion == 'he':
            stdv = np.sqrt(2 / fan_in)
        else:
            raise ValueError('Invalid criterion: ' + self.init_criterion)
        # print(stdv)
        if self.weight_init == 'hypercomplex':
            stdv /= np.sqrt(self.n_divs)
        # print(stdv)
        phase = np.random.uniform(low=-np.pi, high=np.pi, size=self.w_shape[1:]).astype(np.float32)
        phi = np.random.uniform(low=-1, high=1, size=(self.n_divs - 1, *self.w_shape[1:])).astype(np.float32)
        phi /= np.sqrt((phi ** 2 + 1e-4).sum(axis=0))
        modulus = chi.rvs(df=self.n_divs, loc=0, scale=stdv, size=self.w_shape[1:]).astype(np.float32)

        weight = [modulus * np.cos(phase)]
        weight.extend([modulus * phi_ * np.sin(phase) for phi_ in phi])

        self.weight.data = torch.from_numpy(np.stack(weight))

        # self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        assert 3 <= x.dim() <= 5, "The convolutional input x is either 3, 4 or 5 dimensional. x.dim = " + str(
            x.dim())
        kernels = fast_hypercomplex(self.weight, n_divs=self.n_divs, comp_mat=self.comp_mat)
        convfunc = {3: F.conv_transpose1d, 4: F.conv_transpose2d, 5: F.conv_transpose3d}[x.dim()]
        return convfunc(x, kernels, self.bias, self.stride, self.padding, self.output_padding, self.groups,
                        self.dilation)

    def extra_repr(self) -> str:
        extra = ', '.join(
            [f"{key}={self.__getattribute__(key) if key is not 'bias' else str(self.bias is not None)}" for key in
             self.my_list if key not in self.ignore_print])
        return extra

    def __repr__(self):
        name = self.name or 'HyperConvTranspose'
        return f"{name}({self.extra_repr()})"


# ################################\
# # Hypercomplex BatchNorm
# ##########################
class HBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_components=4, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        # super().__init__(num_features, eps=1e-5, momentum=0.1,
        #                  affine=True, track_running_stats=True)
        super().__init__()
        self.num_components = num_components
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.num_learnable_components = self.num_components * (self.num_components + 1) // 2
        if self.affine:
            self.weight = nn.Parameter(
                torch.diag(torch.Tensor([1. / torch.sqrt(torch.Tensor([self.num_components]))] * self.num_components)))
            # self.weight = nn.Parameter(torch.Tensor(self.num_learnable_components))  # TODO - gamma
            self.bias = nn.Parameter(torch.zeros(self.num_components))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_components))
            # self.register_buffer('running_var', torch.ones(*(self.num_components, self.num_components)))
            self.register_buffer('running_var', torch.diag(torch.Tensor([1. / torch.sqrt(torch.Tensor([self.num_components]))] * self.num_components)))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        # self._reset_parameters()
        # self.ab = [(t * self.num_components - sum(range(t)), self.num_components - t) for t in
        #            range(self.num_components)]
        # self.weight_mat = torch.zeros(num_components, num_components)

        idx_ui = torch.triu_indices(self.num_components, self.num_components)
        idx = []
        for n in range(self.num_components):
            idx.extend(
                [sum(list(range(self.num_components - 1, self.num_components - 1 - i, -1))) + n for i in range(n + 1)]
                [1:])
        self.idx = idx
        self.idx_upper = tuple(idx_ui)
        self.idx_extra = tuple(idx_ui.flipud()[:, idx])

    def _reset_parameters(self) -> None:
        # self.reset_parameters()
        if self.affine:
            self.bias.data.zero_()
            value = 1 / np.sqrt(self.num_components)
            diag_points = [t * self.num_components - sum(range(t)) for t in range(self.num_components)]
            for comp_idx in range(self.num_learnable_components):
                if comp_idx in diag_points:
                    self.weight.data[comp_idx].fill_(value)
                else:
                    self.weight.data[comp_idx].fill_(0)

    def _reshape_mat(self):
        self.weight_mat = torch.zeros((self.num_components, self.num_components), device=self.weight.device)
        self.weight_mat[self.idx_upper] = self.weight
        self.weight_mat[self.idx_extra] = self.weight[self.idx]

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError(f'expected 4D input (got {x.dim()}D input)')

    def forward(self, x):
        self._check_input_dim(x)
        batch, channel, height, width = x.shape
        # y = x.transpose(0, 1)
        # return_shape = y.shape
        # total_planes = return_shape[0]
        # dim_planes = total_planes // self.num_components
        # y = torch.cat([*[y[dim_planes * dim: dim_planes * (dim + 1), :].contiguous().view(1, -1) for dim in
        #                  range(self.num_components)]], dim=0)
        y = rearrange(x, 'b (n c) h w -> n (c b h w)', n=self.num_components)
        # y = rearrange(x, 'b (n c) h w -> n (b c h w)', n=self.num_components)
        # mu = reduce(x, 'b (n c) h w -> n', 'mean', n=self.num_components)

        mu = y.mean(dim=1)
        sigma2 = cov(y, rowvar=True)
        sigma2 = sigma2 + self.eps * torch.eye(self.num_components, self.num_components, device=sigma2.device)
        W = torch.linalg.cholesky_ex(torch.inverse(sigma2)).L
        if self.training is not True:
            y = y - self.running_mean.view(-1, 1)
            y = torch.mm(self.running_var, y)
        else:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * W
            y = y - mu.view(-1, 1)
            y = torch.mm(W, y)
        # self._reshape_mat()
        # y = torch.mm(self.weight_mat, y) + self.bias[:, None]
        y = torch.mm(self.weight, y) + self.bias[:, None]

        # y = torch.cat([*[row.view(dim_planes, row.size(0) // dim_planes) for row in y]], dim=0)
        # return y.view(return_shape).transpose(0, 1)
        y = rearrange(y, 'n (c b h w) -> b (n c) h w', n=self.num_components, b=batch, h=height, w=width).contiguous()
        return y


def partial_(func, name, **keywords):
    # name = str(HyperConv).split('.')[-1].replace("'>",'')
    def newfunc(*fargs, **fkeywords):
        newkeywords = {**keywords, **fkeywords}
        return func(*fargs, name=name, ignore_print=list(keywords.keys()), **newkeywords)
    newfunc.func = func
    newfunc.keywords = keywords
    # newfunc.ignore_print = list(keywords.keys())
    return newfunc


HyperConv1d = partial_(HyperConv, 'HyperConv1d', operation='convolution1d')
HyperConv2d = partial_(HyperConv, 'HyperConv2d', operation='convolution2d')
HyperConv3d = partial_(HyperConv, 'HyperConv3d', operation='convolution3d')

ComplexConv1d = partial_(HyperConv, 'ComplexConv1d', operation='convolution1d', n_divs=2)
ComplexConv2d = partial_(HyperConv, 'ComplexConv2d', operation='convolution2d', n_divs=2)
ComplexConv3d = partial_(HyperConv, 'ComplexConv3d', operation='convolution3d', n_divs=2)

QuaternionConv1d = partial_(HyperConv, 'QuaternionConv1d', operation='convolution1d', n_divs=4)
QuaternionConv2d = partial_(HyperConv, 'QauternionConv2d', operation='convolution2d', n_divs=4)
QuaternionConv3d = partial_(HyperConv, 'QuaternionConv3d', operation='convolution3d', n_divs=4)

OctonionConv1d = partial_(HyperConv, 'OctonionConv1d', operation='convolution1d', n_divs=8)
OctonionConv2d = partial_(HyperConv, 'OctonionConv2d', operation='convolution2d', n_divs=8)
OctonionConv3d = partial_(HyperConv, 'OctonionConv3d', operation='convolution3d', n_divs=8)

SedenionConv1d = partial_(HyperConv, 'SedenionConv1d', operation='convolution1d', n_divs=16)
SedenionConv2d = partial_(HyperConv, 'SedenionConv2d', operation='convolution2d', n_divs=16)
SedenionConv3d = partial_(HyperConv, 'SedenionConv3d', operation='convolution3d', n_divs=16)

HyperConvTranspose1d = partial_(HyperConvTranspose, 'HyperConvTranspose1d', operation='convolution1d')
HyperConvTranspose2d = partial_(HyperConvTranspose, 'HyperConvTranspose2d', operation='convolution2d')
HyperConvTranspose3d = partial_(HyperConvTranspose, 'HyperConvTranspose3d', operation='convolution3d')

ComplexConvTranspose1d = partial_(HyperConvTranspose, 'ComplexConvTranspose1d', operation='convolution1d', n_divs=2)
ComplexConvTranspose2d = partial_(HyperConvTranspose, 'ComplexConvTranspose2d', operation='convolution2d', n_divs=2)
ComplexConvTranspose3d = partial_(HyperConvTranspose, 'ComplexConvTranspose3d', operation='convolution3d', n_divs=2)

QuaternionConvTranspose1d = partial_(HyperConvTranspose, 'QuaternionConvTranspose1d', operation='convolution1d',
                                     n_divs=4)
QuaternionConvTranspose2d = partial_(HyperConvTranspose, 'QauternionConvTranspose2d', operation='convolution2d',
                                     n_divs=4)
QuaternionConvTranspose3d = partial_(HyperConvTranspose, 'QuaternionConvTranspose3d', operation='convolution3d',
                                     n_divs=4)

OctonionConvTranspose1d = partial_(HyperConvTranspose, 'OctonionConvTranspose1d', operation='convolution1d', n_divs=8)
OctonionConvTranspose2d = partial_(HyperConvTranspose, 'OctonionConvTranspose2d', operation='convolution2d', n_divs=8)
OctonionConvTranspose3d = partial_(HyperConvTranspose, 'OctonionConvTranspose3d', operation='convolution3d', n_divs=8)

SedenionConvTranspose1d = partial_(HyperConvTranspose, 'SedenionConvTranspose1d', operation='convolution1d', n_divs=16)
SedenionConvTranspose2d = partial_(HyperConvTranspose, 'SedenionConvTranspose2d', operation='convolution2d', n_divs=16)
SedenionConvTranspose3d = partial_(HyperConvTranspose, 'SedenionConvTranspose3d', operation='convolution3d', n_divs=16)

ComplexLinear = partial_(HyperLinear, 'ComplexLinear', n_divs=2)

QuaternionLinear = partial_(HyperLinear, 'QuaternionLinear', n_divs=4)

OctonionLinear = partial_(HyperLinear, 'OctonionLinear', n_divs=8)

SedenionLinear = partial_(HyperLinear, 'SedenionLinear', n_divs=16)

DropoutLinear = partial_(HyperDropout, 'DropoutLinear', dim=-1)

HyperDropout1d = partial_(HyperDropout, 'HyperDropout1d', dim=1)
HyperDropout2d = partial_(HyperDropout, 'HyperDropout2d', dim=1)
HyperDropout3d = partial_(HyperDropout, 'HyperDropout3d', dim=1)

ComplexDropout1d = partial_(HyperDropout, 'ComplexDropout1d', dim=1, n_divs=2)
ComplexDropout2d = partial_(HyperDropout, 'ComplexDropout2d', dim=1, n_divs=2)
ComplexDropout3d = partial_(HyperDropout, 'ComplexDropout3d', dim=1, n_divs=2)

QuaternionDropout1d = partial_(HyperDropout, 'QuaternionDropout1d', dim=1, n_divs=4)
QuaternionDropout2d = partial_(HyperDropout, 'QauternionDropout2d', dim=1, n_divs=4)
QuaternionDropout3d = partial_(HyperDropout, 'QuaternionDropout3d', dim=1, n_divs=4)

OctonionDropout1d = partial_(HyperDropout, 'OctonionDropout1d', dim=1, n_divs=8)
OctonionDropout2d = partial_(HyperDropout, 'OctonionDropout2d', dim=1, n_divs=8)
OctonionDropout3d = partial_(HyperDropout, 'OctonionDropout3d', dim=1, n_divs=8)

SedenionDropout1d = partial_(HyperDropout, 'SedenionDropout1d', dim=1, n_divs=16)
SedenionDropout2d = partial_(HyperDropout, 'SedenionDropout2d', dim=1, n_divs=16)
SedenionDropout3d = partial_(HyperDropout, 'SedenionDropout3d', dim=1, n_divs=16)
