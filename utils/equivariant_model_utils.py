import numpy as np
import torch
from torch import nn

def glorot_init(shape):
    # initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    initialization_range = np.sqrt(2.0 / (shape[-2] + shape[-1]))
    return torch.tensor(np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32))

class Scalar_Linear(nn.Module):
    def __init__(self, in_dim, out_dim, activation = torch.nn.LeakyReLU(negative_slope=1.0e-2)): #activation=torch.nn.SiLU()):
        super(Scalar_Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation=activation
        self.linear = nn.Linear(in_dim, out_dim)
        # self.reset_parameters()
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain = 1.)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        output_shape = list(x.size())
        output_shape[-1] = self.out_dim
        x = x.reshape(-1, x.shape[-1])
        y = self.linear(x)
        y = self.activation(y)
        return y.reshape(output_shape)
    
    # def scalar_neuron(input, weight, bias, activation=torch.nn.SiLU()):
    #     output_shape = list(input.size())
    #     output_shape[-1] = weight.size(1)
    #     input = input.reshape([-1, input.size(-1)])
    #     output = torch.matmul(input, weight) + bias
    #     output = activation(output)
    #     return output.reshape(output_shape)

class Scalar_MLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, activation = torch.nn.LeakyReLU(negative_slope=1.0e-2), use_batchnorm = True): #activation=torch.nn.SiLU()):
        super(Scalar_MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation=activation
        self.linear = nn.Linear(in_dim, mid_dim)
        self.linear2 = nn.Linear(mid_dim, out_dim)
        # self.reset_parameters()
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(mid_dim)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain = 1.)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        output_shape = list(x.size())
        output_shape[-1] = self.out_dim
        x = x.reshape(-1, x.shape[-1])
        y = self.linear(x)
        if self.use_batchnorm:
            y = self.bn(y)
        y = self.activation(y)
        y = self.linear2(y)
        return y.reshape(output_shape)

class Vector_Relu(nn.Module):
    def __init__(self, W_in, W_out, leaky = True, alpha = 0.2):
        super(Vector_Relu, self).__init__()
        # self.Q_weight = nn.Parameter(glorot_init([input_vec_size + v_dim, input_vec_size + v_dim]))
        # self.K_weight = nn.Parameter(glorot_init([input_vec_size + v_dim, input_vec_size + v_dim]))
        self.Q_weight = nn.Parameter(glorot_init([W_in, W_out]))
        self.K_weight = nn.Parameter(glorot_init([W_in, W_out]))
        self.leaky = leaky
        self.eps = 1e-7
        self.alpha = alpha

    def forward(self, input):
        output_shape = list(input.size())
        output_shape[-2] = self.Q_weight.size(1)
        input = input.reshape([-1, input.size(-2), input.size(-1)])
        input = torch.transpose(input, -1, -2)
        Q = torch.matmul(input, self.Q_weight)
        K = torch.matmul(input, self.K_weight)
        inner_product = torch.einsum('nic,  nic->nc', Q, K)
        inner_product = torch.unsqueeze(inner_product * (inner_product < 0), dim=1)
        k_norm = torch.linalg.norm(K, dim=1)
        k_norm = torch.unsqueeze(k_norm, dim=1) # + self.eps # NaN issue here
        output = Q - inner_product * K / (torch.square(k_norm) + self.eps)
        output = torch.transpose(output, -1, -2)
        if self.leaky:
            input = torch.transpose(input, -1, -2)
            return self.alpha * input.reshape(output_shape) + (1 - self.alpha) * output.reshape(output_shape)
        return output.reshape(output_shape)

    # def Vector_Relu(input, Q_weight, K_weight):
    #     output_shape = list(input.size())
    #     output_shape[-2] = Q_weight.size(1)
    #     input = input.reshape([-1, input.size(-2), input.size(-1)])
    #     input = torch.transpose(input, -1, -2)
    #     # output = torch.matmul(input, weight)
    #     Q = torch.matmul(input, Q_weight)
    #     K = torch.matmul(input, K_weight)
    #     inner_product = torch.einsum('nic,  nic->nc', Q, K)
    #     inner_product = torch.unsqueeze(inner_product * (inner_product < 0), dim=1)
    #     k_norm = torch.linalg.norm(K, dim=1)
    #     k_norm = torch.unsqueeze(k_norm, dim=1) + SMALL_NUMBER
    #     output = Q - inner_product * K / torch.square(k_norm)
    #     output = torch.transpose(output, -1, -2)
    #     return output.reshape(output_shape)

    # def Vector_Relu_leaky(self, input, Q_weight, K_weight, alpha=0.3):
    #     output_shape = list(input.size())
    #     output_shape[-2] = Q_weight.size(1)
    #     input = input.reshape([-1, input.size(-2), input.size(-1)])
    #     input = torch.transpose(input, -1, -2)
    #     # output = torch.matmul(input, weight)
    #     Q = torch.matmul(input, Q_weight)
    #     K = torch.matmul(input, K_weight)
    #     inner_product = torch.einsum('nic,  nic->nc', Q, K)
    #     inner_product = torch.unsqueeze(inner_product * (inner_product < 0), dim=1)
    #     k_norm = torch.linalg.norm(K, dim=1)
    #     k_norm = torch.unsqueeze(k_norm, dim=1) + SMALL_NUMBER
    #     output = Q - inner_product * K / torch.square(k_norm)
    #     output = torch.transpose(output, -1, -2)
    #     input = torch.transpose(input, -1, -2)
    #     return alpha * input.reshape(output_shape) + (1 - alpha) * output.reshape(output_shape)

class Vector_Linear(nn.Module):
    def __init__(self, in_dim, out_dim, activation = torch.nn.LeakyReLU(negative_slope=1.0e-2)): #activation=torch.nn.SiLU()):
        super(Vector_Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation=activation
        self.linear = nn.Linear(in_dim, out_dim, bias = False)
        # self.reset_parameters()
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain = 1.)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, input):
        output_shape = list(input.size())
        output_shape[-2] = self.out_dim
        input = input.reshape([-1, input.size(-2), input.size(-1)])
        input = torch.transpose(input, -1, -2)
        output = self.linear(input)
        output = torch.transpose(output, -1, -2)
        return output.reshape(output_shape)

    # def vector_linear(self, input, weight):
    #     output_shape = list(input.size())
    #     output_shape[-2] = weight.size(1)
    #     input = input.reshape([-1, input.size(-2), input.size(-1)])
    #     input = torch.transpose(input, -1, -2)
    #     output = torch.matmul(input, weight)
    #     output = torch.transpose(output, -1, -2)
    #     return output.reshape(output_shape)

class Vector_MLP(nn.Module):
    def __init__(self, W_in, W_out, in_dim, out_dim, leaky = True, alpha = 0.2, use_batchnorm = True):
        super(Vector_MLP, self).__init__()
        assert(W_out == in_dim)
        self.vector_relu = Vector_Relu(W_in, W_out, leaky = leaky, alpha = alpha)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.batchnorm = VNBatchNorm(in_dim)
        self.vlinear = Vector_Linear(in_dim, out_dim)
    #     self.reset_parameters()
        
    # def reset_parameters(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             torch.nn.init.xavier_normal_(p, gain=1.)
    #         else:
    #             torch.nn.init.zeros_(p)

    def forward(self, input):
        # print("a", input.shape)
        hidden = self.vector_relu(input)
        # print("b", hidden.shape)
        if self.use_batchnorm == True:
            hidden = self.batchnorm(hidden)
        # print("c", hidden.shape)
        output = self.vlinear(hidden)
        # print("d", output.shape)
        return output

    # def fully_connected_vec(self, vec, non_linear_Q, non_linear_K, output_weight, activation='leaky_relu'):
    #     # if activation == 'leaky_relu':
    #     hidden = self.Vector_Relu_leaky(vec, non_linear_Q, non_linear_K)
    #     # else:
    #         # hidden = Vector_Relu(vec, non_linear_Q, non_linear_K)
    #     output = self.vector_linear(hidden, output_weight)
    #     return output

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain = 1.)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out

class VNLeakyReLU(nn.Module):
    def __init__(self, W_in, W_out, alpha=0.2, leaky = True):
        super(VNLeakyReLU, self).__init__()
        self.map_to_dir = nn.Linear(W_in, W_out, bias = False)
        self.alpha = alpha
        self.leaky = leaky
        self.eps = 1e-7
        self.alpha = alpha
        self.W_out = W_out
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain = 1.)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # import ipdb; ipdb.set_trace()
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        xout = (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+self.eps))*d))
        if self.leaky:
            x_out = self.alpha * x + (1-self.alpha) * xout
        else:
            x_out = xout
        return x_out

# class VNLinearAndLeakyReLU(nn.Module):
class VN_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, W_in, W_out, leaky=True, use_batchnorm=True, alpha=0.2):
        # super(VNLinearAndLeakyReLU, self).__init__()
        super(VN_MLP, self).__init__()
        assert(out_channels == W_in)
        self.use_batchnorm = use_batchnorm
        self.alpha = alpha
        
        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(W_in, W_out, alpha, leaky)
        
        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = VNBatchNorm(out_channels)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        # import ipdb; ipdb.set_trace()
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm:
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out

class VNBatchNorm(nn.Module):
    def __init__(self, num_features):
        super(VNBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        # print(num_features)
        # self.num_features = num_features
        self.eps = 1e-7
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        # print(self.num_features, "Dog", x.shape) # B, N_feat, 3]
        norm = torch.norm(x, dim=2) + self.eps
        # print("Dog2", norm.shape) # B, N_feat
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        return x


class VNLayerNorm(nn.Module):
    def __init__(self, num_features):
        super(VNLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(num_features)
        # print(num_features)
        # self.num_features = num_features
        self.eps = 1e-7
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        # print(self.num_features, "Dog", x.shape) # B, N_feat, 3]
        norm = torch.norm(x, dim=2) + self.eps # N x C
        # print("Dog2", norm.shape) # B, N_feat
        norm_bn = self.ln(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        return x
