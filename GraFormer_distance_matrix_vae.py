from __future__ import absolute_import

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
import Vae as V
from torch.nn.parameter import Parameter
from ChebConv_distance_matrix_vae import ChebConv, _ResChebGC
import smile_coordinate as sc
import molecule_optimizer as mo
 
sample_batch = 10
device = torch.device("cuda")
model_vae = V.VAE().to(device)

batch_size = 2000
max_atom_num = 9
dimension = 128
z_dim = 20
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


gan_edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4],
                          [0, 5], [5, 6], [6, 7], [7, 8],
                          ], dtype=torch.long)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features=layer.size=512
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True) #标准差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 #eps为系数因子,b_2为偏置


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class GraAttenLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GraAttenLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        ##print("GraAttenLayerx的形状为:",x.shape)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(Q, K, V, mask=None, dropout=None):
    # Query=Key=Value: [batch_size, 8, max_len, 64]
    ##print("Q的形状为:",Q.shape)
    ##print("K的形状为:",K.shape)
    d_k = Q.size(-1)
    # Q * K.T = [batch_size, 8, max_len, 64] * [batch_size, 8, 64, max_len]
    # scores: [batch_size, 8, max_len, max_len]
    ##print("d_k",d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    ##print("score的shape",scores.shape)
    # padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    ##print("p_attn的shape",p_attn.shape)
    ##print("v的形状为:",V.shape)
    ##print()
    return torch.matmul(p_attn, V), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(1)
        ##print("query的形状为:",query.shape)
        nbatches = query.size(0)
        ##print("nbatches的值为:",nbatches)
        Q, K, V = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]
        ##print("Q的值为:",)
        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)

        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True]]])


class LAM_Gconv(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(LAM_Gconv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        ##print("A的形状为:",A.shape)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        ##print("x的形状为:",X.shape)
        ##print("laplacian_batch(A_hat):",A_hat.shape)
        ##print(batch)
        ##print(A_hat)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X)) #矩阵相乘
        if self.activation is not None:
            X = self.activation(X)
        return X


class GraphNet(nn.Module):

    def __init__(self, in_features=44, out_features=44, n_pts=9):
        super(GraphNet, self).__init__()

        self.A_hat = Parameter(torch.eye(n_pts).float(), requires_grad=True)
        self.gconv1 = LAM_Gconv(in_features, in_features * 2)
        self.gconv2 = LAM_Gconv(in_features * 2, out_features, activation=None)

    def forward(self, X):
        ##print("X的形状为:",X.shape)
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1


class GraFormer(nn.Module):
    def __init__(self, hid_dim=128, coords_dim=(44, 3), num_layers=4,
                 n_head=4,  dropout=0.1, n_pts=9):
        super(GraFormer, self).__init__()
        self.n_layers = num_layers
        #self.adj = adj

        _gconv_input = ChebConv(in_c=coords_dim[0], out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        
        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(input_dim=hid_dim, output_dim=hid_dim,
                                                hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_input = _gconv_input
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=3, K=2)

    def forward(self,adj_list,mul_data,x, mask,flag):
        self.adj_list = adj_list
        self.mul_data = mul_data
        #record_result_list = []
        #print("x的形状为:",x.shape)
        #print("adj_list的形状为:",self.adj_list.shape)
        #for i in range(0,len(self.adj_list)):
        #print(adj_list[100])
        #print(mul_data[100])
        out = self.gconv_input(x, self.adj_list,self.mul_data)
            ##print("x[i]为，ajd_list[i]为:",x[i],adj_list[i])
        # for i in range(0,len(out)):
        #     if torch.isnan(out[i].any()):
        #print(out[100])
        for j in range(self.n_layers):
            out = self.atten_layers[j](out, mask)
            out = self.gconv_layers[j](adj = self.adj_list,x = out,mul_data=mul_data)
            ##print("output的形状:",out.shape)
        graph_embedding = out
        node_embedding = x
        #print("graph_embedding的shape",graph_embedding.shape)
        #print("node_embedding的shape",node_embedding.shape)
        if flag == True:  #训练
            x_reconst,mu,log_var = model_vae(graph_embedding,node_embedding) 
            #np.save('./layer_data.npy',layer_data)
            x_reconst = x_reconst.cuda()
            mu = mu.cuda()
            log_var = log_var.cuda()
            #加入vae模型
            #out = VAE(out)
            kl_div = -0.5 * torch.sum(1+log_var - mu.pow(2)-log_var.exp())
            kl_div = kl_div.cuda()
            #out_concat = torch.cat([graph_embedding,node_embedding])
            
        if flag == False:  #验证
            model_vae.eval()
            x_reconst,mu,log_var = model_vae(graph_embedding,node_embedding) 
            #np.save('./layer_data.npy',layer_data)
            x_reconst = x_reconst.cuda()
            mu = mu.cuda()
            log_var = log_var.cuda()
            #加入vae模型
            #out = VAE(out)
            kl_div = -0.5 * torch.sum(1+log_var - mu.pow(2)-log_var.exp())
            kl_div = kl_div.cuda()
            with torch.no_grad():
                z = torch.randn(x.size(0),max_atom_num,z_dim).to(device)
                #print(z.shape)
                out = model_vae.decode(z)
                #print("out_vae的形状为：",out_vae.shape)
                out_concat = torch.cat([graph_embedding,out])
                #print("out_concat的形状为：",out_concat.shape)
        #print(out.shape)
        #mse_torch = torch_distance_matrix(out)
        #print(mse_torch.shape)
        #record_result_list.append(out)
        #list_to_tensor = []
        # for i in range(0,len(record_result_list)):
        #     list_to_tensor.append(record_result_list[i])
        # record_result_list = torch.stack(list_to_tensor)
        # record_result_list = record_result_list.squeeze(1)
        ##print("record_result_list的形状为:",record_result_list.shape)
        out = self.gconv_output(out,self.adj_list,self.mul_data)
        #print("out4:",out[100])
        if flag == False:
            test_sdf_path = './gemo_sample_test_smile.sdf'
            predict_path = './gemo_sample_test_3d.sdf'
            true_path = './gemo_sample_test_true.sdf'
            sc.modify_sample_3d_coordinate(out,test_sdf_path,predict_path)
            out = mo.write_while_inferring(true_path,predict_path)
        if flag == 'test':
            test_smile_path = './rdkit_folder/sample_test_smile.sdf'
            predict_path = './rdkit_folder/sample_predict_3d.sdf'
            true_path = './rdkit_folder/sample_test_true.sdf'
            sc.modify_sample_3d_coordinate(out,test_smile_path,predict_path)
            out = mo.write_while_inferring(true_path,predict_path)
        return out,kl_div

def torch_distance_matrix(data):
    mse_list = []
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            for k in range(0,len(data[i])):
                mse = torch.sqrt((data[i][j]-data[i][k])**2)
                mse_list.append(mse)
    
    #mse_array = np.array(mse_list)
    mse_sqrt = []
    for mse_error in mse_list:
        mse = torch.sqrt(mse_error[0]**2 + mse_error[1]**2 + mse_error[2]**2)
        mse_sqrt.append(mse)
    mse_sqrt = torch.stack(mse_sqrt,0)
    mse_sqrt = mse_sqrt.view(len(data),len(data[0]),len(data[0]))
    return mse_sqrt




# if __name__ == '__main__':
#     adj = adj_mx_from_edges(num_pts=9, edges=gan_edges, sparse=False)
#     ##print(adj)
#     model = GraFormer(hid_dim=128, coords_dim=(44, 3))
#     # model_test = GraFormer(adj=adj, hid_dim=64, coords_dim=(2, 3), n_pts=21,
#     #                           num_layers=5, n_head=4, dropout=0)
#     x = torch.zeros((2, 9, 44))
#     ##print(model(adj,x, src_mask))

