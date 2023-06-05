import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import scipy.sparse as sp


# body_edges = torch.tensor([[0, 1], [1, 2], [2, 3],
#                          [0, 4], [4, 5], [5, 6],
#                          [0, 7], [7, 8], [8, 9],
#                          [8, 10], [10, 11], [11, 12],
#                          [8, 13], [13, 14], [14, 15]], dtype=torch.long)




def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) #求和
    r_inv = np.power(rowsum, -1).flatten() #将数组展开成一维数组
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv) #对角矩阵，稀疏矩阵生成
    mx = r_mat_inv.dot(mx)      #点积
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):  #根据边创建邻接矩阵
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx) #有向图转无向图，创建对称的邻接矩阵，推导过程略，总之能把一个矩阵变换成对称的
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0])) #对加入自连接的邻接矩阵进行对称归一化处理，因为邻接矩阵的对角都是0，和特征矩阵内积相当于将邻接矩阵做了加权和，节点特征的值成为了邻接矩阵的权，自身的特征被忽略。为避免这种情况，可以先给A加上一个单位矩阵I,单位矩阵。它是个方阵，从左上角到右下角的对角线（称为主对角线）上的元素均为1。除此以外全都为0。这样，使得邻接矩阵的对角元素变成1。

    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx) 
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1,1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph,mul_data):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [B, N, N]
        :return: convolution result, [B, N, D]
        """
        #inputs = inputs.unsqueeze(1)
        #print("inputs的长度为",inputs.shape)
        #print("graph的形状为:",graph.shape)
        ###print("graph的长度为",len(graph))
        ####print("inputs的形状为:",inputs.size) 
        ###print("graphs为:",graph.shape)  
        #L = ChebConv.get_laplacian(graph, self.normalize)  # [B,N, N] 下三角矩阵
        #print("L的形状为:",L.shape)
        mul_L = self.cheb_polynomial(mul_data)       # [K, B, N, N]
        #print("mul_L的形状为:",mul_L.shape)
        #print("mul_data的形状为:",mul_L.shape)
        mul_L = mul_L.view(3,mul_L.size(0),9,9)
        #print("mul_L的形状为:",mul_L.shape)
        result = torch.matmul(mul_L, inputs)           # [K, B, N, C]
        
        #result = result.unsqueeze(2)
        #print("result的形状为:",result.shape)
        #print("weight的形状为:",self.weight.shape)
        result = torch.matmul(result, self.weight)     # [K, B, N, D] 输入与权重之间相乘 X*W
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]
        #print("result的形状为:",result.shape)
        return result

    def cheb_polynomial(self, batch_laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian. 根据拉普拉斯图计算切比雪夫多项式

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        record_multi_order_laplacian = []
        for laplacian in batch_laplacian:
            N = laplacian.size(0)  # [N, N]
            multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
            multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

            if self.K == 1:
                return multi_order_laplacian
            else:
                multi_order_laplacian[1] = laplacian
                if self.K == 2:
                    return multi_order_laplacian
                else:
                    for k in range(2, self.K):
                        multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                                multi_order_laplacian[k-2]
            record_multi_order_laplacian.append(multi_order_laplacian)
        record_multi_order_laplacian = torch.stack(record_multi_order_laplacian,0)
        return record_multi_order_laplacian

    @staticmethod
    def get_laplacian(batch_graph, normalize): #拉普拉斯矩阵
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize: #对称归一化的拉普拉斯矩阵
            batch_L = []
            for graph in batch_graph:
                graph = np.array(graph)
                graph = torch.from_numpy(graph)
                ###print("graph的size为:",graph.size(0))
                D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2)) #度数阵
                
                ###print(graph)
                ###print("graph的数据类型为:",graph.dtype)
                graph = graph.to(torch.float32)
                ###print("graph的数据类型为:",graph.dtype)
                ###print(D)
                L = torch.eye(graph.size(0)) - torch.mm(torch.mm(D, graph), D) #L = D -1/2 * L * D -1/2
                batch_L.append(L)
        else:
            for graph in batch_graph:
                D = torch.diag(torch.sum(graph, dim=-1))
                L = D - graph    #graph 是图表示的邻接矩阵
                batch_L.append(L)
        #batch_L = np.array(batch_L)
        #batch_L = batch_L.astype('float32')
        batch_L = torch.stack(batch_L,0)
        ###print(L)
        return batch_L


class _GraphConv(nn.Module): #图卷积
    def __init__(self, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = ChebConv(input_dim, output_dim, K=2)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, adj,mul_data):
        x = self.gconv(x, adj,mul_data)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResChebGC(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResChebGC, self).__init__()
        #self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)
        ####print("adj的形状为:",adj.shape)
        ####print("adi",adj)
        ###print("hello")
    def forward(self,x,adj,mul_data):
        self.adj = adj
        ###print("_ResChebGC中x的形状为:",x.shape)
        residual = x
        out = self.gconv1(x, self.adj,mul_data)
        out = self.gconv2(out, self.adj,mul_data)
        ####print("adj的形状为:",adj.shape)
        
        ###print("self.adj的形状为:",self.adj.shape)
        ###print("self.adj",self.adj)
        return residual + out
    # def test(self,adj,x):
    #     residual 


class ChebNet(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(ChebNet, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)

    def forward(self, x,mul_data): 
        out = self.gconv1(x, self.adj,mul_data)
        out = self.gconv2(out, self.adj,mul_data)
        return out



