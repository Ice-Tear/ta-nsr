import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.misc import config_to_primitive
from .network_utils import get_activation
import tinycudann as tcnn

def get_mlp(n_input_dims, n_output_dims, config):
    if config.otype == 'VanillaMLP':
        network = VanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == 'SDFMLP':
        network = SDFMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == 'VanillaMLPCutlass':
        network = VanillaMLPCutlass(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == 'LipshitzMLP':
        network = LipshitzMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    else:
        with torch.cuda.device(0):
            network = tcnn.Network(n_input_dims, n_output_dims, config_to_primitive(config))
            if config.get('sphere_init', False):
                sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network)
    return network

class VanillaMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = config['n_neurons'], config['n_hidden_layers']
        self.sphere_init, self.weight_norm = config.get('sphere_init', False), config.get('weight_norm', False)
        self.sphere_init_radius = config.get('sphere_init_radius', 0.5)
        self.layers = [self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False), self.make_activation()]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False), self.make_activation()]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = getattr(F, config['output_activation']) if config['output_activation'] != 'none' else lambda x : x
        
    @torch.cuda.amp.autocast(False)
    def forward(self, x):
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x
    
    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True) # network without bias will degrade quality
        
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(dim_in), std=0.0001)
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dim_out))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer   

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)
        
class SDFMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = config['n_neurons'], config['n_hidden_layers']
        self.sphere_init, self.weight_norm = config.get('sphere_init', False), config.get('weight_norm', False)
        self.sphere_init_radius = config.get('sphere_init_radius', 0.5)
        layer_dims = [dim_in] + [self.n_neurons] * self.n_hidden_layers + [dim_out]

        self.layers = torch.nn.ModuleList()

        # Hidden layers
        layer_dim_pairs = list(zip(layer_dims[:-1], layer_dims[1:]))
        for li, (k_in, k_out) in enumerate(layer_dim_pairs):
            linear = torch.nn.Linear(k_in, k_out)
            if self.sphere_init:
                self._geometric_init(linear, k_in, k_out, first=(li == 0))
            if self.weight_norm:
                linear = nn.utils.weight_norm(linear)
            self.layers.append(linear)
            if li == len(layer_dim_pairs) - 1:
                self.layers[-1].bias.data.fill_(0.0)
        # SDF prediction layer
        self.linear_sdf = torch.nn.Linear(k_in, 1)
        if self.sphere_init:
            self._geometric_init_sdf(self.linear_sdf, k_in, out_bias=self.sphere_init_radius)
        
        self.output_activation = get_activation(config['output_activation'])
        self.activation = self.make_activation()
    
    @torch.cuda.amp.autocast(False)
    def forward(self, x, with_feature=True):
        feat = x.float()
        for li, linear in enumerate(self.layers):
            if li != len(self.layers) - 1 or with_feature:
                feat_pre = linear(feat)
                feat_activ = self.activation(feat_pre)
            if li == len(self.layers) - 1:
                out = [self.linear_sdf(feat).float(),
                       feat_activ.float() if with_feature else None]
            feat = feat_activ
        return out
        
    def _geometric_init(self, linear, k_in, k_out, first=False):
        torch.nn.init.constant_(linear.bias, 0.0)
        torch.nn.init.normal_(linear.weight, 0.0, np.sqrt(2 / k_out))
        if first:
            torch.nn.init.constant_(linear.weight[:, 3:], 0.0)  # positional encodings

    def _geometric_init_sdf(self, linear, k_in, out_bias=0.):
        torch.nn.init.normal_(linear.weight, mean=np.sqrt(np.pi / k_in), std=0.0001)
        torch.nn.init.constant_(linear.bias, -out_bias)
    
    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)

# FIXME: CutlassLayer doesn't work well. No clue...
class VanillaMLPCutlass(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = config['n_neurons'], config['n_hidden_layers']
        self.sphere_init, self.weight_norm = config.get('sphere_init', False), config.get('weight_norm', False)
        self.sphere_init_radius = config.get('sphere_init_radius', 0.5)
        self.layers = [self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False), self.make_activation()]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False), self.make_activation()]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = getattr(F, config['output_activation']) if config['output_activation'] != 'none' else lambda x : x
        
    @torch.cuda.amp.autocast(False)
    def forward(self, x):
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x
    
    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True) # network without bias will degrade quality
        
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(dim_in), std=0.0001)
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dim_out))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        
        layer = init_cutlass_with_torch_linear(layer)
        return layer   

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)

class CutlassLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, bias=False, dtype=torch.float16):
        super().__init__()
        config = {
            "otype": "CutlassMLP",
            "activation": "None",
            "output_activation": "None",
            "n_neurons": 0,
            "n_hidden_layers": 0,
        }
        self.weight = tcnn.Network(inputs_dim, outputs_dim, config)
        self.bias = torch.nn.Parameter(torch.Tensor(outputs_dim)).cuda() if bias else 0.0
        self.dtype = dtype

    def forward(self, inputs):
        return self.weight(inputs).to(self.dtype) + self.bias.to(self.dtype)



def leaky_relu_init(m, negative_slope=0.2):
    gain = np.sqrt(2.0 / (1.0 + negative_slope ** 2))

    if isinstance(m, torch.nn.Conv1d):
        
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Conv2d):
        print('1')
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose1d):
        print('1')
        ksize = m.kernel_size[0] // 2
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose2d):
        print('1')
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose3d):
        print('1')
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features
        # print(n1)
        # print(n2)
        # print('======')
        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        
        return

  
    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()

    if isinstance(m, torch.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]


def apply_weight_init_fn(m, fn, negative_slope=1.0):

    should_initialize_weight=True
    if not hasattr(m, "weights_initialized"): #if we don't have this then we need to intiialzie
        # fn(m, is_linear, scale)
        should_initialize_weight=True
        
    elif m.weights_initialized==False: #if we have it but it's set to false
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    else:
        print("skipping weight init on ", m)
        should_initialize_weight=False

    if should_initialize_weight:
        # fn(m, is_linear, scale)
        fn(m,negative_slope)
        # m.weights_initialized=True
        for module in m.children():
            apply_weight_init_fn(module, fn, negative_slope)

class LipshitzModule(torch.nn.Module):
    def __init__(self, in_channels, nr_out_channels_per_layer, last_layer_linear):
        super(LipshitzModule, self).__init__()
        self.last_layer_linear=last_layer_linear
        self.layers=torch.nn.ModuleList()
        # self.layers=[]
        for i in range(len(nr_out_channels_per_layer)):
            cur_out_channels=nr_out_channels_per_layer[i]
            self.layers.append(  torch.nn.Linear(in_channels, cur_out_channels)   )
            in_channels=cur_out_channels
        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        if last_layer_linear:
            leaky_relu_init(self.layers[-1], negative_slope=1.0)

        #we make each weight separately because we want to add the normalize to it
        self.weights_per_layer=torch.nn.ParameterList()
        self.biases_per_layer=torch.nn.ParameterList()
        for i in range(len(self.layers)):
            self.weights_per_layer.append(self.layers[i].weight)
            self.biases_per_layer.append(self.layers[i].bias)

        self.lipshitz_bound_per_layer=torch.nn.ParameterList()
        for i in range(len(self.layers)):
            max_w= torch.max(torch.sum(torch.abs(self.weights_per_layer[i]), dim=1))
            #we actually make the initial value quite large because we don't want at the beggining to hinder the rgb model in any way. A large c means that the scale will be 1
            c = torch.nn.Parameter(torch.ones((1)) * max_w * 2) 
            self.lipshitz_bound_per_layer.append(c)

        self.weights_initialized=True #so that apply_weight_init_fn doesnt initialize anything

    def normalization(self, w, softplus_ci):
        absrowsum = torch.sum(torch.abs(w), dim=1)
        # scale = torch.minimum(torch.tensor(1.0), softplus_ci/absrowsum)
        # this is faster than the previous line since we don't constantly recreate a torch.tensor(1.0)
        scale = softplus_ci/absrowsum
        scale = torch.clamp(scale, max=1.0)
        return w * scale[:,None]

    def lipshitz_bound_full(self):
        lipshitz_full=1.0
        for i in range(len(self.layers)):
            lipshitz_full=lipshitz_full * torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])

        return lipshitz_full

    def forward(self, x):
        for i in range(len(self.layers)):
            weight=self.weights_per_layer[i]
            bias=self.biases_per_layer[i]

            weight=self.normalization(weight, torch.nn.functional.softplus(self.lipshitz_bound_per_layer[i])  )

            x=torch.nn.functional.linear(x, weight, bias)

            is_last_layer=i==(len(self.layers) - 1)

            if is_last_layer and self.last_layer_linear:
                pass
            else:
                x=torch.nn.functional.gelu(x)
        return x

# copy from permutoSDF https://github/RaduAlexandru/permuto_sdf
class LipshitzMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = config['n_neurons'], config['n_hidden_layers']
        out_dim = []
        for i in range(self.n_hidden_layers):
            out_dim.append(self.n_neurons)
        out_dim.append(dim_out)
        self.layers = LipshitzModule(dim_in, out_dim, last_layer_linear=True)
        self.output_activation = getattr(F, config['output_activation']) if config['output_activation'] != 'none' else lambda x : x

    @torch.cuda.amp.autocast(False)
    def forward(self, x):
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x

def sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network):
    print('Initialize tcnn MLP to approximately represent a sphere.')
    """
    from https://github.com/NVlabs/tiny-cuda-nn/issues/96
    It's the weight matrices of each layer laid out in row-major order and then concatenated.
    Notably: inputs and output dimensions are padded to multiples of 8 (CutlassMLP) or 16 (FullyFusedMLP).
    The padded input dimensions get a constant value of 1.0,
    whereas the padded output dimensions are simply ignored,
    so the weights pertaining to those can have any value.
    """
    # padto = 16 if config.otype == 'FullyFusedMLP' else 8
    padto = 16 # in tinycudann 1.7, maybe all mlp are padded to multiples of 16
    n_input_dims = n_input_dims + (padto - n_input_dims % padto) % padto
    n_output_dims = n_output_dims + (padto - n_output_dims % padto) % padto
    data = list(network.parameters())[0].data
    assert data.shape[0] == (n_input_dims + n_output_dims) * config.n_neurons + (config.n_hidden_layers - 1) * config.n_neurons**2
    new_data = []
    # first layer
    weight = torch.zeros((config.n_neurons, n_input_dims)).to(data)
    torch.nn.init.constant_(weight[:, 3:], 0.0)
    torch.nn.init.normal_(weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(config.n_neurons))
    new_data.append(weight.flatten())
    # hidden layers
    for i in range(config.n_hidden_layers - 1):
        weight = torch.zeros((config.n_neurons, config.n_neurons)).to(data)
        torch.nn.init.normal_(weight, 0.0, np.sqrt(2) / np.sqrt(config.n_neurons))
        new_data.append(weight.flatten())
    # last layer
    weight = torch.zeros((n_output_dims, config.n_neurons)).to(data)
    torch.nn.init.normal_(weight, mean=np.sqrt(np.pi) / np.sqrt(config.n_neurons), std=0.0001)
    new_data.append(weight.flatten())
    new_data = torch.cat(new_data)
    data.copy_(new_data)

def init_cutlass_with_torch_linear(linear):
    k_out, k_in = linear.weight.data.shape
    # turn it to CutlassLayer
    batch_size_granularity = 16
    padded_dim_in_size = (k_in + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity
    padded_dim_out_size = (k_out + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity
    weight = linear.weight.data.cuda()
    weight = F.pad(weight, pad=(0,padded_dim_in_size-k_in,0,padded_dim_out_size-k_out)).half() # 
    bias = linear.bias.cuda()
    linear = CutlassLayer(k_in, k_out, bias=True, dtype=torch.float32)
    linear.weight.params.data = weight.flatten()
    linear.bias = bias
    return linear