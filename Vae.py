import os
#from ot import toc
#from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import torch.utils.data as data
#torch.cuda.set_device(1) # 这句用来设置pytorch在哪块GPU上运行
device = torch.device('cuda')

# layer_data = np.load('./layer_data.npy')
# print(layer_data.shape)
# size = 23040
# h_dim = 400
# z_dim = 20
# num_epochs = 100
# batch_size = 20
# learning_rate = 1e-3


# sample_dir = 'samples'
# if not os.path.exists(sample_dir):
#     os.makedirs(sample_dir)

class Mydataset(data.Dataset):

    def __init__(self,graph_embedding,node_embedding):
        self.graph_embedding = graph_embedding
        self.node_embedding = node_embedding
        self.idx = list()
        for item in graph_embedding:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        data = self.graph_embedding[index]
        node_embedding_data = self.node_embedding[index]
        return data,node_embedding_data

    def __len__(self):
        return len(self.idx)


# 数据加载器
# dataset = Mydataset(layer_data)
# data_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                           batch_size=batch_size, 
#                                           shuffle=False)

class VAE(nn.Module):
    def __init__(self, size=128,redisual_size=4, eva_redisual_size=100,h_dim=40, z_dim=20):
        super(VAE, self).__init__()
        
        self.size = size
        self.h_dim = h_dim
        self.red = nn.Linear(redisual_size, h_dim)
        self.eva_red = nn.Linear(eva_redisual_size, h_dim)
        #print("size.device",self.size.device)
        self.fc1 = nn.Linear(self.size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim) # 均值 向量
        self.fc3 = nn.Linear(h_dim, z_dim) # 保准方差 向量
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, self.size)
        
    # 编码过程
    def encode(self, x):
        #print(x.shape)
        # if(x.size(0)==4):
        #     #self.szie = x.size(1)
        #     h = F.relu(self.fc1(x.cuda())).cuda()
        #     #print(x.size(1).device)
        #     #print(x.device)
        # #print()

        # if(x.size(0)==200):
        #     #print(x.shape)
        #     h = F.relu(self.fc1(x.cuda())).cuda()
        # if(x.size(0)==100):
        #     h = F.relu(self.fc1(x.cuda())).cuda()
        h = F.relu(self.fc1(x.cuda())).cuda()
        return self.fc2(h).cuda(), self.fc3(h).cuda()
    
    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
 
    # 解码过程
    def decode(self, z):
        h = F.relu(self.fc4(z))
        #print(h.shape)
        return F.sigmoid(self.fc5(h))
    
    # 整个前向传播过程：编码-》解码
    def forward(self, x,node_embedding):
        mu, log_var = self.encode(x)
        mu = mu.cuda()
        log_var = log_var.cuda()
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

    

# model = VAE().to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     for i, (x) in enumerate(data_loader):
#         # 获取样本，并前向传播
#         #print(x.shape)
#         #print(x)
#         x = x.to(device).view(-1, size)
#         x_reconst, mu, log_var = model(x)
#         # print(x[0])
#         # print(x_reconst[0])
#         # print(x_reconst.shape)
#         # 计算重构损失和KL散度（KL散度用于衡量两种分布的相似程度）
#         # KL散度的计算可以参考论文或者文章开头的链接
#         # reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
#         kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
#         # 反向传播和优化
#         loss = kl_div
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1) % 2 == 0:
#             print ("Epoch[{}/{}], Step [{}/{}], KL Div: {:.4f}" 
#                    .format(epoch+1, num_epochs, i+1, len(data_loader),  kl_div.item()))

#     # 利用训练的模型进行测试
#     with torch.no_grad():
#         # 随机生成的图像
#         z = torch.randn(batch_size, z_dim).to(device)
#         print(z.shape)
#         out = model.decode(z).view(-1, batch_size, 9, 128)
        #save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))
 
    #     # 重构的图像
    #     out, _, _ = model(x)
    #     x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        #save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))

# reconsPath = './samples/reconst-1.png'
# Image = mpimg.imread(reconsPath)
# plt.imshow(Image) # 显示图片
# plt.axis('off') # 不显示坐标轴
# plt.show()