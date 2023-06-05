import os 
import numpy as np
import torch
import pandas as pd
import GraFormer_distance_matrix_vae
#import mpnn_qm9
#from mpnn_qm9 import MPNNModel
import tensorflow as tf
from rdkit.Chem import rdmolops
import rdkit
from rdkit import Chem
from torch import nn
import tqdm
from torch_geometric.data import  Data
import GCN_featurization
import smile_coordinate as sc
import molecule_optimizer as mo
import random as rd 
import Vae as V


src_mask = torch.tensor([[[True]]]).cuda()
device = torch.device("cuda")
model_vae = V.VAE().to(device)
kl_weight = 0.01

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] =str(seed)
    rd.seed(seed)

def read_csv():
    
    #tf.compat.v1.disable_eager_execution()
    csv_file = './qm9.csv'
    df = pd.read_csv(csv_file,usecols=[0,1])
    smile_list = list(df.iloc[:,1])
    #df = np.array(df)
    ##print(df.iloc[:,1])
    id_list = list(df.iloc[:,0])
    return df,smile_list,id_list

def dataset(df):
    permuted_indices = np.random.permutation(np.arange(df.shape[0]))

    # Train set: 80 % of data
    train_index = permuted_indices[: int(df.shape[0] * 0.8)]
    x_train = mpnn_qm9.graphs_from_smiles(df.iloc[train_index].smiles)
    #y_train = df.iloc[train_index].p_np
    ##print(x_train)
    # Valid set: 19 % of data
    #valid_index = permuted_indices[int(df.shape[0] * 0.8) : int(df AverageMeter.shape[0] * 0.99)]
    #x_valid = mpnn_qm9.graphs_from_smiles(df.iloc[valid_index].smiles)
    #y_valid = df.iloc[valid_index].p_np

    # Test set: 1 % of data
    test_index = permuted_indices[int(df.shape[0] * 0.99) :]
    x_test = mpnn_qm9.graphs_from_smiles(df.iloc[test_index].smiles)
    return #x_train,x_valid,x_test

def graph_matrix(smile_list):
    adj_matrix_list = []
    for smile in smile_list:
        mol = rdkit.Chem.MolFromSmiles(smile)
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        ##print(adj_matrix)
        adj_matrix_list.append(adj_matrix)
    adj_matrix_list = torch.tensor(adj_matrix_list)
    return adj_matrix_list

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_ckpt(state, ckpt_path, suffix=None):
    if suffix is None:
        suffix = 'epoch_{:04d}'.format(state['epoch'])

    file_path = os.path.join(ckpt_path, 'ckpt_{}.pth.tar'.format(suffix))
    torch.save(state, file_path)


def wrap(func, unsqueeze, *args):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

import torch.utils.data as data
import numpy as np

x = np.array(range(80)).reshape(8, 10) # 模拟输入， 8个样本，每个样本长度为10
y = np.array(range(8))  # 模拟对应样本的标签， 8个标签

def adjs2ctable(adjs):
    xline=torch.eye(adjs.shape[2]).to(adjs)
    xline=torch.tile(torch.tile(xline.unsqueeze(0),(adjs.shape[1],1,1)).unsqueeze(0),(adjs.shape[0],1,1,1))
    adjs_=adjs-xline
    ctable=torch.sum(adjs_,dim=1,dtype=torch.double)
    ctable[:,0,0]=1
    tmp=ctable[:,0,2]
    tmp=torch.where(tmp==0.0,0.5,tmp)
    ctable[:,0,2]=tmp 
    ctable[:,2,0]=tmp 
    tmp=ctable[:,1,2]
    tmp=torch.where(tmp==0.0,0.5,tmp)
    ctable[:,1,2]=tmp
    ctable[:,2,1]=tmp    
    return ctable

def Adjs2zmat(adjs,mask=None): 
    ctable=adjs2ctable(adjs)
    max_size=ctable.shape[1]
    Bctable=ctable
    bt,it,jt=torch.where(Bctable>0)
    rbij=Bctable[bt,it,jt]
    id=torch.where((it>jt)|(it<3))
    bt=bt[id];it=it[id];jt=jt[id];rbij=rbij[id]
    zb=torch.stack((bt,it,jt),dim=1)
    newindex=bt*max_size+jt
    Actable=torch.index_select(ctable.view(-1,max_size),dim=0,index=newindex)
    indext,kt=torch.where(Actable>0)
    rbjk=Actable[indext,kt]
    bt=torch.index_select(bt,dim=0,index=indext)
    it=torch.index_select(it,dim=0,index=indext)
    jt=torch.index_select(jt,dim=0,index=indext)
    rbij=torch.index_select(rbij,dim=0,index=indext)
    ids=torch.where((it>kt)|(it<3))
    bt=bt[ids];it=it[ids];jt=jt[ids];kt=kt[ids];rbij=rbij[ids];rbjk=rbjk[ids]
    za=torch.stack((bt,it,jt,kt),dim=1)
    newindex=bt*max_size+kt
    Dctable=torch.index_select(ctable.view(-1,max_size),dim=0,index=newindex)
    indext,lt=torch.where(Dctable>0)
    rbkl=Dctable[indext,lt]
    bt=torch.index_select(bt,dim=0,index=indext)
    it=torch.index_select(it,dim=0,index=indext)
    jt=torch.index_select(jt,dim=0,index=indext)
    kt=torch.index_select(kt,dim=0,index=indext)
    rbij=torch.index_select(rbij,dim=0,index=indext)
    rbjk=torch.index_select(rbjk,dim=0,index=indext)
    ids=torch.where((it>lt)|(it<3))
    bt=bt[ids];it=it[ids];jt=jt[ids];kt=kt[ids];lt=lt[ids];rbij=rbij[ids];rbjk=rbjk[ids];rbkl=rbkl[ids]
    ids_all=torch.where(((it!=jt)&(it!=kt)&(it!=lt)&(jt!=kt)&(kt!=lt)&(jt!=lt)&(it>4)&(rbij+rbjk+rbkl==3.0)))[0]
    ids_3=torch.where((it==3)&(rbij+rbjk+rbkl==3.0)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=lt)&(jt!=kt)&(kt!=lt)))[0]
    if ids_3.shape[0]==0:
        ids_3=torch.where((it==3)&(rbij+rbjk+rbkl==2.5)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=lt)&(jt!=kt)&(kt!=lt)))[0]
    ids_4=torch.where((it==4)&(rbij+rbjk+rbkl==3.0)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=lt)&(jt!=kt)&(kt!=lt)))[0]
    if ids_4.shape[0]==0:
        ids_4=torch.where((it==4)&(rbij+rbjk+rbkl==2.5)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=lt)&(jt!=kt)&(kt!=lt)))[0] 
    ids_2=torch.where((it==2)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=kt))&(rbij+rbjk+rbkl==3))[0]
    if ids_2.shape[0]==0:
        ids_2=torch.where((it==2)&((it!=jt)&(it!=kt)&(it!=lt)&(jt!=kt))&(rbij+rbjk+rbkl==2.5))[0]
    ids_1=torch.where((it<=1))[0]
    ids,_=torch.sort(torch.cat([ids_1,ids_2,ids_3,ids_4,ids_all]))
    bt=bt[ids];it=it[ids];jt=jt[ids];kt=kt[ids];lt=lt[ids];rbij=rbij[ids];rbjk=rbjk[ids];rbkl=rbkl[ids]
    Zmat=torch.stack((bt,it,jt,kt,lt,rbij+rbjk+rbkl),dim=1)
    uni=bt*max_size+it
    uni,ids=np.unique(uni.detach().cpu().numpy(),return_index=True)
    bt=bt[ids];it=it[ids];jt=jt[ids];kt=kt[ids];lt=lt[ids];rbij=rbij[ids];rbjk=rbjk[ids];rbkl=rbkl[ids]
    uni,counts=torch.unique(bt,return_counts=True)
    Zmat=torch.stack((bt,it,jt,kt,lt,rbij+rbjk+rbkl),dim=1)
    pt=0
    Zlist=[]
    for cid,count in enumerate(counts):
        mat=torch.zeros(max_size,6)
        mat[:,0]=cid 
        mat[:count]=Zmat.narrow(0,pt,count) 
        Zlist.append(mat)    
        pt+=count
    Zmat=torch.stack(Zlist,dim=0).to(torch.int32)
    return Zmat

def get_laplacian(batch_graph, normalize): #拉普拉斯矩阵
    """
    return the laplacian of the graph.

    :param graph: the graph structure without self loop, [N, N].
    :param normalize: whether to used the normalized laplacian.
    :return: graph laplacian.
    """
    #对称归一化的拉普拉斯矩阵
    if normalize:
        batch_L = []
        for graph in batch_graph:
            graph = np.array(graph)
            graph = torch.from_numpy(graph)
            ##print("graph的size为:",graph.size(0))
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2)) #度数阵
            
            ##print(graph)
            ##print("graph的数据类型为:",graph.dtype)
            graph = graph.to(torch.float32)
            ##print("graph的数据类型为:",graph.dtype)
            ##print(D)
            L = torch.eye(graph.size(0)) - torch.mm(torch.mm(D, graph), D) #L = D -1/2 * L * D -1/2
            batch_L.append(L)
    else:
        for graph in batch_graph:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph    #graph 是图表示的邻接矩阵
            batch_L.append(L)
    #batch_L = np.array(batch_L)
    #batch_L = batch_L.astype('float32')
    #batch_L = torch.stack(batch_L,0)
    ##print(L)
    return batch_L


class Mydataset(data.Dataset):

    def __init__(self,id_list,smile_list,adj_list,mul_L_list, x, distance_matrix,coordinate):
        self.x = x
        self.y = distance_matrix
        self.id_list = id_list
        self.mul_L = mul_L_list
        self.smile_list = smile_list
        self.adj = adj_list
        self.coordinate = coordinate
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        id_data = self.id_list[index]
        smile_data = self.smile_list[index]
        input_data = self.idx[index]
        adj_data = self.adj[index]
        target = self.y[index]
        mul_data = self.mul_L[index]
        coordinate_data = self.coordinate[index]
        return id_data,smile_data,adj_data, mul_data,input_data, target,coordinate_data

    def __len__(self):
        return len(self.idx)


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

def train(dataloader,model_pos,device,criterion, optimizer, lr_now):
    #batch_time = AverageMeter()
    #data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_dis_loss = AverageMeter()
    epoch_kl_loss = AverageMeter()
    #model_pos.train()
    # Switch to train mode
    torch.set_grad_enabled(True)
    #data_input = np.load('./data/qm9_data_2d.npz')
    model_pos.train()
    #end = time.time()
    #dl = tqdm(dataloader)
    result_3d = []
    smile_list = []
    epoch = 0
    #layer_data_list = []
    for index,(smile_data,adj_data,mul_data,input_data,target,coordinate_data) in enumerate(dataloader):
        #print("training:",index)
        #epoch += 0
        num_poses = target.size(0)
        adj_data,input_data,target,mul_data = adj_data.to(device),input_data.to(device),target.to(device),mul_data.to(device)
        outputs_3d,kl_div = model_pos(adj_data,mul_data,input_data,src_mask,True)
        # print(outputs_3d[0].shape)
        # print(outputs_3d[1].shape)
        # print(outputs_3d[2].shape)
        # print(type(outputs_3d[2]))
        # print(outputs_3d.size(0))
        # print(outputs_3d.size(1))
        #outputs_3d_vae_input = outputs_3d.view(-1,outputs_3d.size(0)*outputs_3d.size(1)*outputs_3d.size(2))
        #print("epoch:%s" %index,outputs_3d_vae_input.shape)
        #print("output",outputs_3d_vae_input.device)
        outputs_3d = outputs_3d.cuda()
        # layer_data = layer_data.cuda()
        # print(layer_data.shape)
        dist_torch = torch.cdist(outputs_3d,outputs_3d,p=2)
        #kl_div = -0.5 * torch.sum(1+log_var - mu.pow(2)-log_var.exp())
        dist_torch = dist_torch.cuda()
        #angel_torch = get_torsions(outputs_3d)
        kl_div = kl_div.cuda()
        optimizer.zero_grad()
        #outputs_3d = torch.tensor(outputs_3d)
        dis_loss = criterion(dist_torch,target)
        #angel_loss = criterion(angel_torch,angel_target)
        loss_3d_pos = dis_loss + kl_weight * kl_div
        loss_3d_pos.backward()
        optimizer.step()
        epoch_dis_loss.update(dis_loss.item(),num_poses)
        epoch_kl_loss.update(kl_div.item(),num_poses)
        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
        #result_3d.append(record)
        smile_list.append(smile_data)
        #print("loss",loss_3d_pos)
        #layer_data_list.append(layer_data)
    return epoch_loss_3d_pos.avg, lr_now,result_3d,smile_list,epoch_dis_loss.avg,epoch_kl_loss.avg #layer_data_list

class KabschRMSD(nn.Module):
    def __init__(self) -> None:
        super(KabschRMSD, self).__init__()

    def forward(self, ligs_coords_pred, ligs_coords):
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
            lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

            A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)

            U, S, Vt = torch.linalg.svd(A)

            corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            rotation = (U @ corr_mat) @ Vt
            translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)

            lig_coords = (rotation @ lig_coords.t()).t() + translation
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        return torch.tensor(rmsds).mean()

def kab_rmsd_train():
    epoch_loss_3d_pos = AverageMeter()
    #model_pos.train()
    # Switch to train mode
    torch.set_grad_enabled(True)
    #data_input = np.load('./data/qm9_data_2d.npz')
    model_pos.train()
    #end = time.time()
    #dl = tqdm(dataloader)
    result_3d = []
    smile_list = []
    
    for index,(smile_data,adj_data,mul_data,input_data,target,coordinate_data) in enumerate(dataloader):
        #print("training:",index)
        num_poses = target.size(0)
        adj_data,input_data,target,mul_data,coordinate_data = adj_data.to(device),input_data.to(device),target.to(device),mul_data.to(device)
        outputs_3d = model_pos(adj_data,mul_data,input_data,src_mask).cuda()
        #mse_torch = torch.cdist(outputs_3d,outputs_3d,p=2)
        #mse_torch = mse_torch.cuda()
        optimizer.zero_grad()
        #outputs_3d = torch.tensor(outputs_3d)
        loss_3d_pos = criterion(mse_torch,target)
        loss_3d_pos.backward()
        optimizer.step()
        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
        #result_3d.append(record)
        smile_list.append(smile_data)
    return epoch_loss_3d_pos.avg, lr_now,result_3d,smile_list


def evaluate(dataloader,model_pos, device):
    epoch_loss_3d_pos = AverageMeter()
    #model_pos.train()
    # Switch to train mode
    epoch_dis_loss = AverageMeter()
    epoch_kl_loss = AverageMeter()
    torch.set_grad_enabled(False)
    #data_input = np.load('./data/qm9_data_2d.npz')
    model_pos.eval()
    #end = time.time()
    #dl = tqdm(dataloader)
    result_3d = []
    smile_list = []
    true_3d = []
    #layer_data_list = []
    for index,(smile_data,adj_data,mul_data,input_data,target,coordinate_data) in enumerate(dataloader):

        num_poses = target.size(0)
        adj_data,input_data,target,mul_data = adj_data.to(device),input_data.to(device),target.to(device),mul_data.to(device)
        #print("evaluate中input_data的形状为:",input_data.shape)
        outputs_3d,kl_div = model_pos(adj_data,mul_data,input_data,src_mask,False)
        #print(type(outputs_3d))
        #outputs_3d_vae_input = outputs_3d.view(-1,outputs_3d.size(0)*outputs_3d.size(1)*outputs_3d.size(2))
        record = outputs_3d.detach().cpu()
        # x_reconst,mu,log_var = model_vae(outputs_3d_vae_input) 
        # x_reconst = x_reconst.cuda()
        # mu = mu.cuda()
        # log_var = log_var.cuda()
        outputs_3d = outputs_3d.cuda()
        kl_div = kl_div.cuda()
        mse_torch = torch.cdist(outputs_3d,outputs_3d,p=2)
        #kl_div = -0.5 * torch.sum(1+log_var - mu.pow(2)-log_var.exp())
        #kl_div = kl_div.cuda()
        #mse_torch = torch.cdist(outputs_3d,outputs_3d,p=2)
        mse_torch = mse_torch.cuda()
        #optimizer.zero_grad()
        #outputs_3d = torch.tensor(outputs_3d)
        #layer_data_list.append(layer_data)
        dis_loss = criterion(mse_torch,target)
        loss_3d_pos = criterion(mse_torch,target)+ kl_weight * kl_div
        #loss_3d_pos.backward()
        #optimizer.step()
        epoch_dis_loss.update(dis_loss.item(),num_poses)
        epoch_kl_loss.update(kl_div.item(),num_poses)
        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
        result_3d.append(record)
        smile_list.append(smile_data)
        true_3d.append(coordinate_data)
    return epoch_loss_3d_pos.avg,result_3d,smile_list,true_3d,epoch_dis_loss.avg,epoch_kl_loss.avg

def test(dataloader,model_pos, device):
    epoch_loss_3d_pos = AverageMeter()
    #model_pos.train()
    # Switch to train mode
    torch.set_grad_enabled(False)
    #data_input = np.load('./data/qm9_data_2d.npz')
    model_pos.eval()
    #end = time.time()
    #dl = tqdm(dataloader)
    result_3d = []
    smile_list = []
    for index,(smile_data,adj_data,mul_data,input_data,target) in enumerate(dataloader):

        num_poses = target.size(0)
        adj_data,input_data,target,mul_data = adj_data.to(device),input_data.to(device),target.to(device),mul_data.to(device)
        outputs_3d = model_pos(adj_data,mul_data,input_data,src_mask).cuda()
        #print(type(outputs_3d))
        record = outputs_3d.detach().cpu()
        mse_torch = torch.cdist(outputs_3d,outputs_3d,p=2)
        mse_torch = mse_torch.cuda()
        #optimizer.zero_grad()
        #outputs_3d = torch.tensor(outputs_3d)
        
        loss_3d_pos = criterion(mse_torch,target)
        #loss_3d_pos.backward()
        #optimizer.step()
        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
        result_3d.append(record)
        smile_list.append(smile_data)
    
    return epoch_loss_3d_pos.avg,result_3d,smile_list
#def calc

def get_torsions(outputs_3d):
    atom_counter = 0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    #                     if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                    #                         or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                    #                         continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList

def get_3d_coordinate(sdf_path):
    coordinate_3d = []
    atom_num_list = []
    sum = 0
    atom_num = 0
    sdf = Chem.SDMolSupplier(sdf_path)
    for mol in sdf:
        sum += 1
        num_atoms = mol.GetNumAtoms()
        for i in range(0,num_atoms):
            atom_num += 1
            x = mol.GetConformer().GetAtomPosition(i).x
            y = mol.GetConformer().GetAtomPosition(i).y
            z = mol.GetConformer().GetAtomPosition(i).z
            coordinate_3d.append(x)
            coordinate_3d.append(y)
            coordinate_3d.append(z)
        atom_num_list.append(num_atoms)
    coordinate_3d = np.array(coordinate_3d)
    coordinate_3d = coordinate_3d.reshape(sum,9,3)
    coordinate_3d = torch.from_numpy(coordinate_3d)
    #coordinate_3d = torch.from_numpy(coordinate_3d)
    #print(coordinate_3d.shape)
    return coordinate_3d,sum

class KabschRMSD(nn.Module):
    def __init__(self) -> None:
        super(KabschRMSD, self).__init__()

    def forward(self, ligs_coords_pred, ligs_coords):
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
            lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

            A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)

            U, S, Vt = torch.linalg.svd(A)

            corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            rotation = (U @ corr_mat) @ Vt
            translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)

            lig_coords = (rotation @ lig_coords.t()).t() + translation
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        return torch.tensor(rmsds).mean()

def calc_rmsd():
    predict_optimizer_path = './optimizer_mol.sdf'
    true_sdf_path = './test_true.sdf'
    predict_coordinate, predict_sum = get_3d_coordinate(predict_optimizer_path)
    target_coordinate, target_sum = get_3d_coordinate(true_sdf_path)
    kab = KabschRMSD()
    kab_rmsd = kab(predict_coordinate,target_coordinate)
    print("kab_rmsd",kab_rmsd)

# def sample_from_test():
#     if os.path.exists(log_dir):


if __name__ == "__main__":
    setup_seed(50)
    df,smile_list,id_list = read_csv()
   
    # #x_train,__,__ = dataset(df) 
    # ##print(smile_list)
    # adj_matrix_list = graph_matrix(smile_list)
    # #print("adj_list的长度:",len(adj_matrix_list))
    # ##print(adj_matrix_list[0])
    # # atom_features,bond_features,pair_indices,molecule_indicator,mpnn = MPNNModel(
    # #     atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0],
    # # )
    # ##print(atom_features)
    model = GraFormer_distance_matrix_vae.GraFormer(hid_dim=128,n_pts=9).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-5)
    criterion = nn.MSELoss(reduction='mean').to(device)
    ##print(model)
    #print("smile_list的长度:",len(smile_list))
    # data_list = []
    # for smile in smile_list:
    #     tg_data = GCN_featurization.featurize_mol_from_smiles(smile)
    #     data_list.append(tg_data)
    # #print("data_list的长度:",len(data_list))
    # ##print(data_list)
    # # start_epoch = 0
    # # epoch = 100
    # #x = atom_features
    # x_list = []
    # for i in range(0,len(data_list)):
    #     x = data_list[i].x
    #     x_list.append(x)
    
    #x = torch.stack(x_list,0)
    #x = data_list[0].x
    #print("x的形状为:",x.shape)
    data_target = np.load('./qm9_distance_matrix.npy',allow_pickle=True)
    data_target = torch.from_numpy(data_target)
    data_target = data_target.float()
    distance_matrix = data_target
    adj_matrix_list = np.load('./adj_array.npy',allow_pickle=True)
    x_list = np.load('./x_array.npy',allow_pickle=True)
    mul_L_list = np.load('./mul_array.npy',allow_pickle=True)
    coordinate = np.load('./qm9_3d.npy',allow_pickle=True)
    #print(mul_L_list.shape)
    #adj_matrix_list = adj_matrix_list.tolist()
    #print("data_target的形状为:",data_target.shape)
    # x_array = np.array(x_list)
    # adj_matrix_array = np.array(adj_matrix_list)
    # np.save('./x_array.npy',x_array)
    # np.save('./adj_array.npy',adj_matrix_array)
    # mul_L_list = get_laplacian(adj_matrix_list,True)
    # np.save('./mul_array.npy',mul_L_list)
    # print(len(mul_L_list))
    datasets = Mydataset(id_list,smile_list,adj_matrix_list,mul_L_list,x_list, distance_matrix,coordinate)  # 初始化
    # #print(datasets.__len__())
    train_size = int(0.80 * len(datasets))
    validate_size = int (0.10 * len(datasets))
    test_size = len(datasets) - train_size - validate_size
    train_dataset , validate_dataset , test_dataset = torch.utils.data.random_split(datasets,[train_size,validate_size,test_size],generator=torch.Generator().manual_seed(30))
    train_loader = data.DataLoader(train_dataset,batch_size=200)
    validate_loader = data.DataLoader(validate_dataset,batch_size=200)
    test_loader = data.DataLoader(test_dataset,batch_size=200)
    #dataloader = data.DataLoader(datasets, batch_size=200) 

    # for index,(adj_data,mul_data,input_data,target) in enumerate(dataloader):
    #     print('adj_data的形状为：' ,adj_data.shape)
    #     print('input_data的形状为' , input_data.shape)
    #     print('target的形状为' , target.shape)
    # ##print(x_list)
    # #result = model(adj_matrix_list,x_list,src_mask)
    # ##print(len(result))
    log_dir = './qm9_3d.pth'
    
    if os.path.exists(log_dir):
        for index,(id_data,smile_data,adj_data,mul_data,input_data,target,coordinate) in enumerate(test_loader):
            sum = 0
            test_id_data = id_data
            test_smile_data = smile_data
            test_adj_data = adj_data.cuda()
            test_mul_data = mul_data.cuda()
            test_input_data = input_data.cuda()
            test_true_coordinate = coordinate
            # sum += 1
            # if (sum>1):
            #     break
        checkpoint = torch.load(log_dir)
        print("exist")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(test_smile_data[0])
        np.save('./sample_id_data.npy',id_data)
        #test_smile_data = np.array(test_smile_data)
        sample_test_mol_list = sc.get_sample_mol_list(test_smile_data)
        print(len(sample_test_mol_list))
        #print(sample_test_mol_list)
        test_sdf_path = './sample_test_smile.sdf'
        sc.save_mol_to_sdf(sample_test_mol_list,test_sdf_path)

        for i in range(1,101):
            outputs_3d,kl_div = model(test_adj_data,test_mul_data,test_input_data,src_mask,False)
            predict_path = './sample_test_3d.sdf'
            true_path = './sample_test_true.sdf'
            print(outputs_3d[0][0])
            print(len(outputs_3d))
            #print()
            sc.modify_sample_3d_coordinate(outputs_3d,test_sdf_path,predict_path)

            sc.modify_sample_3d_coordinate(test_true_coordinate,test_sdf_path,true_path)
            mo.write_while_inferring(i)
        # mo.write_while_inferring()
        # print(outputs_3d.shape)
        
    # log_dir = './dataloader/qm9_3d_dist.pth'
    # if os.path.exists(log_dir):
    #     checkpoint = torch.load(log_dir)
    #     print("exist")
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_epoch = 0
        end_epoch = 400
        # for epoch in range(start_epoch, epoch):
        #     epoch_loss, lr_now, glob_step = train(train_loader, model_pos, criterion, optimizer, device, args.lr, lr_now,
        #                                           glob_step, args.lr_decay, args.lr_gamma, max_norm=True)
        #adj = GraFormer.adj_mx_from_edges(num_pts=9, edges=gan_edges, sparse=False)
        min_loss = 100
        record_epoch_loss = []
        record_validate_loss = []
        record_test_loss = []
        record_train_dis_loss = []
        record_train_kl_div_loss = []
        record_validate_dis_loss = []
        record_validate_kl_div_loss = []
        lr_list = []
        true_3d_list = []
        #test_smile_list
        lr_now = 1e-5
        
        for epoch in range(start_epoch,end_epoch):
            
            train_epoch_loss, lr_now,train_result_3d,train_smile_list,dis_loss,kl_div_loss  = train(dataloader=train_loader,model_pos=model, device=device,criterion=criterion, optimizer=optimizer, lr_now=lr_now)
            evaluate_epoch_loss,result_3d,smile_list,true_3d,eva_dis_loss,eva_kl_div_loss = evaluate(dataloader=validate_loader,model_pos=model,device=device)
            #test_epoch_loss,result_3d,smile_list = evaluate(dataloader=test_loader,model_pos=model,device=device)
            record_epoch_loss.append(train_epoch_loss)
            record_validate_loss.append(evaluate_epoch_loss)
            record_train_dis_loss.append(dis_loss)
            record_train_kl_div_loss.append(kl_div_loss)
            record_validate_dis_loss.append(eva_dis_loss)
            record_validate_kl_div_loss.append(eva_kl_div_loss)
            #record_test_loss.append(test_epoch_loss)
            #np.save('./train_layer_data.npy',train_layer_data)
            lr_list.append(lr_now)
            if (epoch%10==0 and epoch>0):
                print("第%s次train_total_loss为%s,DA_loss为%s,kl_div_loss为%s:" %(epoch,train_epoch_loss,dis_loss,kl_div_loss))
                print("第%s次evaluate_total_loss为%s,DA_loss为%s,kl_div_loss为%s:" %(epoch,evaluate_epoch_loss,eva_dis_loss,eva_kl_div_loss))
                #print("第%s次test_distance_matrix_loss为:" %epoch,test_epoch_loss)
                with open('./distance_matrix_loss_save.txt','a+') as f:
                    
                    f.write("第"+str(epoch)+"次的train_distance_matrix_loss为:"+str(train_epoch_loss)+'\n')
                    f.write("第"+str(epoch)+"次的evaluate_distance_matrix_loss为:"+str(evaluate_epoch_loss)+'\n')
                    #f.write("第"+str(epoch)+"次的test_distance_matrix_loss为:"+str(test_epoch_loss)+'\n')
                f.close()
            # if (evaluate_epoch_loss < min_loss):
            #     min_loss = evaluate_epoch_loss
            #     torch.save({
            #                     'epoch':epoch,
            #                     'model_state_dict':model.state_dict(),
            #                     'optimizer_state_dict':optimizer.state_dict(),
            #                     'loss':train_epoch_loss
            #                 },'./qm9_3d.pth')
            if (epoch%60==0 and epoch!=0):
                for p in optimizer.param_groups:
                    p['lr'] *= 0.8
                torch.save({
                                'epoch':epoch,
                                'model_state_dict':model.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                'loss':train_epoch_loss
                            },'./qm9_3d.pth')
            
                #np.save('./epoch_loss_test.npy',record_epoch_loss)
            
        # torch.save({
        #     'epoch':epoch,
        #     'model_state_dict':model.state_dict(),
        #     'optimizer_state_dict':optimizer.state_dict(),
        #     'loss':epoch_loss
        # },'./qm9_3d.pth')
        record_epoch_loss = np.array(record_epoch_loss)
        record_validate_loss = np.array(record_validate_loss)
        #record_test_loss = np.array(record_test_loss)
        record_train_dis_loss = np.array(record_train_dis_loss)
        record_train_kl_div_loss =  np.array(record_train_kl_div_loss)
        record_validate_dis_loss =  np.array(record_validate_dis_loss)
        record_validate_kl_div_loss =  np.array(record_validate_kl_div_loss)
        lr_list = np.array(lr_list)
        result_3d = np.array(result_3d)
        smile_list = np.array(smile_list)
        train_smile_list = np.array(train_smile_list)
        test_true_coordinate = np.array(true_3d)
        
        #print("test_true:",test_true_coordinate[0][0])
        np.save('./epoch_train_loss.npy',record_epoch_loss)
        np.save('./epoch_evaluate_loss.npy',record_validate_loss)
        np.save('./train_dis_loss.npy',record_train_dis_loss)
        np.save('./train_kl_loss.npy',record_train_kl_div_loss)
        np.save('./validate_dis_loss.npy',record_validate_dis_loss)
        np.save('./validate_kl_div_loss.npy',record_validate_kl_div_loss)
        np.save('./result_3d.npy',result_3d)
        np.save('./smile_data.npy',smile_list)
        np.save('./true_coordinate.npy',test_true_coordinate)
        #np.save('./epoch_test_loss.npy',record_test_loss)
        np.save('./lr_rate.npy',lr_list)
        np.save('./train_smile.npy',train_smile_list)
        predict_path = './result_3d.npy'
        true_path = './true_coordinate.npy'
        test_smile_list = sc.get_mol_list(smile_list)
        #print("smile长度：",len(test_smile_list))
        save_path = './test_smile.sdf'
        sc.save_mol_to_sdf(test_smile_list,save_path)
        coordinate_array = sc.cat_tensor(predict_path)
        true_coordinate = sc.cat_tensor(true_path)
        #print("coordinate_array的shape:",coordinate_array.shape)
        predict_path = './test_3d.sdf'
        true_path = './test_true.sdf'
        sc.modify_3d_coordinate(coordinate_array,predict_path)
        sc.modify_3d_coordinate(true_coordinate,true_path)
        #mo.write_while_inferring()

        #calc_rmsd()
    # for epoch in range(start_epoch, epoch):
    #     epoch_loss, lr_now, glob_step = train(train_loader, model_pos, criterion, optimizer, device, args.lr, lr_now,
    #                                           glob_step, args.lr_decay, args.lr_gamma, max_norm=True)
    #adj = GraFormer.adj_mx_from_edges(num_pts=9, edges=gan_edges, sparse=False)
    #predict_result = model(x_list,src_mask)
    # result = []
    # epoch_loss_3d_pos = AverageMeter()
    # torch.set_grad_enabled(False)
    # model.eval()
    # epoch_loss_3d_pos = AverageMeter()
    # #model_pos.train()
    # # Switch to train mode
    # torch.set_grad_enabled(False)
    # #data_input = np.load('./data/qm9_data_2d.npz')
    # model_pos.eval()
    # #end = time.time()
    # #dl = tqdm(dataloader)
    # for index,(adj_data,mul_data,input_data,target) in enumerate(dataloader):

    #     num_poses = data_target.size(0)
    #     adj_data,input_data,target,mul_data = adj_data.to(device),input_data.to(device),target.to(device),mul_data.to(device)
    #     outputs_3d = model_pos(adj_data,mul_data,input_data,src_mask).cuda()
    #     #print(type(outputs_3d))
    #     mse_torch = torch.cdist(outputs_3d,outputs_3d,p=2)
    #     mse_torch = mse_torch.cuda()
    #     #optimizer.zero_grad()
    #     #outputs_3d = torch.tensor(outputs_3d)
        
    #     loss_3d_pos = criterion(mse_torch,target)
    #     #loss_3d_pos.backward()
    #     #optimizer.step()
    #     epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
    # for index,(adj_data,mul_data,input_data,target) in enumerate(test_loader):

    #     num_poses = data_target.size(0)
    #     adj_data,input_data,target,mul_data = adj_data.to(device),input_data.to(device),target.to(device),mul_data.to(device)
    #     outputs_3d = model(adj_data,mul_data,input_data,src_mask).cuda()
    #     #print(type(outputs_3d))
    #     result.append(outputs_3d)
    #     mse_torch = torch.cdist(outputs_3d,outputs_3d,p=2)
    #     mse_torch = mse_torch.cuda()
    #     #optimizer.zero_grad()
    #     #outputs_3d = torch.tensor(outputs_3d)
        
    #     loss_3d_pos = criterion(mse_torch,target)
    #     #loss_3d_pos.backward()
    #     #optimizer.step()
    #     epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
    # with open('./record_loss.txt','a+') as g:
    #     string = "testdataset中的distance_matrix的loss为:"
    #     g.write(string+str(epoch_loss_3d_pos.avg))
    # g.close()
    # print("testdataset中的distance_matrix的loss为:",epoch_loss_3d_pos.avg)
    #result = model(adj_data,mul_data,input_data,src_mask)
    #print(result.shape)
    ##print("input",data_input)
    ##print(predict_result)
    # for i in result:
    #     with open('./predict_coordinate_test.txt','a+') as f:
    #         f.write(str(i))
    #     f.close()
    # predict_numpy = result.detach().numpy()
    # np.save('./predict_coordinate_9_dataloader.npy',predict_numpy)
