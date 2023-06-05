import os 
import numpy as np
import torch
import pandas as pd
import GraFormer_distance_matrix
#import mpnn_qm9
#from mpnn_qm9 import MPNNModel
import tensorflow as tf
from rdkit.Chem import rdmolops
import rdkit
from torch import nn
import tqdm
from torch_geometric.data import  Data
import GCN_featurization
import warnings
import torch.nn.utils as torch_utils
np.random.seed(42)
tf.random.set_seed(42)

src_mask = torch.tensor([[[True]]]).cuda()
device = torch.device("cuda")
def read_csv():
    
    #tf.compat.v1.disable_eager_execution()
    csv_file = './qm9.csv'
    df = pd.read_csv(csv_file,usecols=[0,1])
    smile_list = list(df.iloc[:,1])
    id_list = list(df.iloc[:,0])
    #df = np.array(df)
    ##print(df.iloc[:,1])
    return df,smile_list,id_list

def dataset(df):
    permuted_indices = np.random.permutation(np.arange(df.shape[0]))

    # Train set: 80 % of data
    train_index = permuted_indices[: int(df.shape[0] * 0.8)]
    x_train = mpnn_qm9.graphs_from_smiles(df.iloc[train_index].smiles)
    #y_train = df.iloc[train_index].p_np
    ##print(x_train)
    # Valid set: 19 % of data
    valid_index = permuted_indices[int(df.shape[0] * 0.8) : int(df.shape[0] * 0.99)]
    x_valid = mpnn_qm9.graphs_from_smiles(df.iloc[valid_index].smiles)
    #y_valid = df.iloc[valid_index].p_np

    # Test set: 1 % of data
    test_index = permuted_indices[int(df.shape[0] * 0.99) :]
    x_test = mpnn_qm9.graphs_from_smiles(df.iloc[test_index].smiles)
    return x_train,x_valid,x_test

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

# x = np.array(range(80)).reshape(8, 10) # 模拟输入， 8个样本，每个样本长度为10
# y = np.array(range(8))  # 模拟对应样本的标签， 8个标签

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

    def __init__(self,id_list,smile_list,adj_list,mul_L_list, x, y):
        self.x = x
        self.y = y
        self.id_list = id_list
        self.mul_L = mul_L_list
        self.smile_list = smile_list
        self.adj = adj_list
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
        return id_data,smile_data,adj_data, mul_data,input_data, target

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

def xyz2ic(x,zmat,mask=None,eps=1e-7):
    batch_num=x.shape[0]
    max_atom_num=x.shape[1]
    zmat=zmat.view(-1,6)
    it=zmat[:,0]*max_atom_num+zmat[:,1]
    jt=zmat[:,0]*max_atom_num+zmat[:,2]
    kt=zmat[:,0]*max_atom_num+zmat[:,3]
    lt=zmat[:,0]*max_atom_num+zmat[:,4]
    x=x.cuda().view(-1,3)
    #print(x.is_cuda)
    it = it.cuda()
    jt = jt.cuda()
    kt = kt.cuda()
    lt = lt.cuda()
    #print(it.is_cuda)
    x0=torch.index_select(x,dim=0,index=it).view(-1,max_atom_num,3)
    x1=torch.index_select(x,dim=0,index=jt).view(-1,max_atom_num,3 )
    x2=torch.index_select(x,dim=0,index=kt).view(-1,max_atom_num,3)
    x3=torch.index_select(x,dim=0,index=lt).view(-1,max_atom_num,3)
    dist,J_dist = dist_deriv(x0,x1)
    angle,J_angle = angle_deriv(x0,x1,x2)
    dihedral,J_dihedral = torsion_deriv(x0,x1,x2,x3)
    return dist,angle,dihedral,J_dist,J_angle,J_dihedral

def angle_deriv(x1, x2, x3, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """
        computes angle between input points together with
        the Jacobian wrt to `x1`
    """
    r12 = x1 - x2
    r12_norm = torch.norm(r12, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(r12_norm < eps):
            warnings.warn("singular division in angle computation")
    if enforce_boundaries:
        r12_norm = r12_norm.clamp_min(eps)

    rn12 = r12 / r12_norm

    J = (torch.eye(3).to(x1) - outer(rn12, rn12)) / r12_norm[..., None]

    r32 = x3 - x2
    r32_norm = torch.norm(r32, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(r32_norm < eps):
            warnings.warn("singular division in angle computation")
    if enforce_boundaries:
        r32_norm = r32_norm.clamp_min(eps)

    rn32 = r32 / r32_norm

    cos_angle = torch.sum(rn12 * rn32, dim=-1)
    J = rn32[..., None, :] @ J

    if raise_warnings:
        if torch.any((cos_angle < -1. + eps) & (cos_angle > 1. - eps)):
            warnings.warn("singular radians in angle computation")
    if enforce_boundaries:
        cos_angle = cos_angle.clamp(-1. + eps, 1. - eps)

    a = torch.acos(cos_angle)

    J = -J / torch.sqrt(1.0 - cos_angle.pow(2)[..., None, None])

    return a, J[..., 0, :]

def train(dataloader,model_pos,device,criterion, optimizer, lr_now):
    #batch_time = AverageMeter()
    #data_time = AverageMeter()
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
    id_list = []
    count = 0
    error_count = 0
    for index,(id_data,smile_data,adj_data,mul_data,input_data,target) in enumerate(dataloader):
        #print("training:",index)
        num_poses = target.size(0)
        count += 1
        adj_data,input_data,target,mul_data = adj_data.to(device),input_data.to(device),target.to(device),mul_data.to(device)
        outputs_3d = model_pos(adj_data,mul_data,input_data,src_mask).cuda()
        adj = adj_data.unsqueeze(1)
        zmat = Adjs2zmat(adj)
        #mse_torch = torch.cdist(outputs_3d,outputs_3d,p=2)
        #mse_torch = mse_torch.cuda()
        error_smile_data = []
        error_target_data = []
        dist_predict,angle_predict,dihedral_predict,__,__,___ = xyz2ic(outputs_3d,zmat,mask=None,eps=1e-7)
        dist_reference,angle_reference,dihedral_reference,__,__,___ = xyz2ic(target,zmat,mask=None,eps=1e-7)
        #dihedral_predict = torch.where(dihedral_predict == 0, torch.ones_like(torch.tensor(0.1).cuda()), dihedral_predict)
        #dihedral_reference = torch.where(dihedral_reference == 0, torch.ones_like(torch.tensor(0.1).cuda()), dihedral_reference)
        print(dihedral_predict)
        print(dihedral_reference)
        if (torch.isnan(dihedral_reference).any()==True or torch.isnan(angle_reference).any()==True or torch.isnan(dihedral_predict).any()==True or torch.isnan(angle_predict).any()==True):
            print(torch.isnan(dihedral_reference).any(), torch.isnan(angle_reference).any(),torch.isnan(dihedral_predict).any(),torch.isnan(angle_predict).any())
            error_count += 1
            error_smile_data.append(smile_data)
            error_target_data.append(target)
            np.save('./error_dihedral_smile.npy',error_smile_data)
            np.save('./error_target_data.npy',target.detach().cpu().numpy())
        if (torch.isnan(dihedral_reference).any()==False and torch.isnan(angle_reference).any()==False and torch.isnan(dihedral_predict).any()==False and torch.isnan(angle_predict).any()==False):
            print(torch.isnan(dihedral_reference).any(), torch.isnan(angle_reference).any(),torch.isnan(dihedral_predict).any(),torch.isnan(angle_predict).any())
            print("no problem")

        dist_predict = dist_predict.cuda()
        dist_reference = dist_reference.cuda()
        angle_predict = angle_predict.cuda()
        angle_reference = angle_reference.cuda()
        dihedral_predict = dihedral_predict.cuda()
        dihedral_reference = dihedral_reference.cuda()
        grad_clip = 1.0
        optimizer.zero_grad()

        #outputs_3d = torch.tensor(outputs_3d)
        #predict = dist_predict + angle_predict + dihedral_predict
        #reference = dist_reference + angle_reference + dihedral_reference
        loss_3d_pos = criterion(dist_predict,dist_reference) + 0.1 * criterion(angle_predict,angle_reference) + 0.1 * criterion(dihedral_predict,dihedral_reference).detach()
        
        loss_3d_pos.backward()
        torch_utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
        #result_3d.append(record)
        smile_list.append(smile_data)
        id_list.append(id_data)
        print("count:",count)
        print("error_count:",error_count)
    return epoch_loss_3d_pos.avg, lr_now,result_3d,smile_list,id_list

def torsion_deriv(x1, x2, x3, x4, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """
        computes torsion angle between input points together with
        the Jacobian wrt to `x1`.
    """
    b0 = -1.0 * (x2 - x1)

    # TODO not used can be removed in next refactor
    # db0_dx1 = torch.eye(3).to(x1)

    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1norm = torch.norm(b1, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(b1norm < eps):
            warnings.warn("singular division in distance computation")
    if enforce_boundaries:
        b1norm = b1norm.clamp_min(eps)

    b1_normalized = b1 / b1norm

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    #
    # dv_db0 = jacobian of v wrt b0

    v = b0 - torch.sum(b0 * b1_normalized, dim=-1, keepdim=True) * b1_normalized
    dv_db0 = torch.eye(3)[None, None, :, :].to(x1) - outer(b1_normalized, b1_normalized)

    w = b2 - torch.sum(b2 * b1_normalized, dim=-1, keepdim=True) * b1_normalized

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    #
    # dx_dv = jacobian of x wrt v
    x = torch.sum(v * w, dim=-1, keepdim=True)
    dx_dv = w[..., None, :]

    # b1xv = fast cross product between b1_normalized and v
    # given by multiplying v with the skew of b1_normalized
    # (see https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Cross_product)
    #
    # db1xv_dv = Jacobian of b1xv wrt v
    A = skew(b1_normalized)
    b1xv = (A @ (v[..., None]))[..., 0]
    db1xv_dv = A

    # y = dot product of b1xv and w
    # dy_db1xv = Jacobian of v wrt b1xv
    y = torch.sum(b1xv * w, dim=-1, keepdim=True)
    dy_db1xv = w[..., None, :]

    x = x[..., None]
    y = y[..., None]

    # a = torsion angle spanned by unit vector (x, y)
    # xysq = squared norm of (x, y)
    # da_dx = Jacobian of a wrt xysq
    a = torch.atan2(y, x)
    xysq = x.pow(2) + y.pow(2)

    if raise_warnings:
        if torch.any(xysq < eps):
            warnings.warn("singular division in torsion computation")
    if enforce_boundaries:
        xysq = xysq.clamp_min(eps)

    da_dx = -y / xysq
    da_dy = x / xysq

    # compute derivative with chain rule
    J = da_dx @ dx_dv @ dv_db0 + da_dy @ dy_db1xv @ db1xv_dv @ dv_db0
    return a[..., 0, 0], J[..., 0, :]

def outer(x, y):
    """ outer product between input tensors """
    return x[..., None] @ y[..., None, :]


def dist_deriv(x1, x2, eps=1e-7, enforce_boundaries=True, raise_warnings=True):
    """
        computes distance between input points together with
        the Jacobian wrt to `x1`
    """
    r = x2 - x1
    rnorm = torch.norm(r, dim=-1, keepdim=True)

    if raise_warnings:
        if torch.any(rnorm < eps):
            warnings.warn("singular division in distance computation")
    if enforce_boundaries:
        rnorm = rnorm.clamp_min(eps)

    dist = rnorm[..., 0]
    J = -r / rnorm
    # J = _safe_div(-r, rnorm)
    return dist, J

def skew(x):
    """
        returns skew symmetric 3x3 form of a 3 dim vector
    """
    assert len(x.shape) > 1, "`x` requires at least 2 dimensions"
    zero = torch.zeros(*x.shape[:-1]).to(x)
    a, b, c = x[..., 0], x[..., 1], x[..., 2]
    s = torch.stack(
        [
            torch.stack([zero, c, -b], dim=-1),
            torch.stack([-c, zero, a], dim=-1),
            torch.stack([b, -a, zero], dim=-1),
        ],
        dim=-1,
    )
    return s


def evaluate(dataloader,model_pos, device):
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
    id_list = []
    for index,(id_data,smile_data,adj_data,mul_data,input_data,target) in enumerate(dataloader):

        num_poses = target.size(0)
        adj_data,input_data,target,mul_data = adj_data.to(device),input_data.to(device),target.to(device),mul_data.to(device)
        outputs_3d = model_pos(adj_data,mul_data,input_data,src_mask).cuda()
        #print(type(outputs_3d))
        record = outputs_3d.detach().cpu()
        adj = adj_data.unsqueeze(1)
        zmat = Adjs2zmat(adj)
        #mse_torch = torch.cdist(outputs_3d,outputs_3d,p=2)
        #mse_torch = mse_torch.cuda()
        dist_predict,angle_predict,dihedral_predict,__,__,___ = xyz2ic(outputs_3d,zmat,mask=None,eps=1e-7)
        dist_reference,angle_reference,dihedral_reference,__,__,___ = xyz2ic(target,zmat,mask=None,eps=1e-7)
        dist_predict = dist_predict.cuda()
        dist_reference = dist_reference.cuda()
        angle_predict = angle_predict.cuda()
        angle_reference = angle_reference.cuda()
        dihedral_predict = dihedral_predict.cuda()
        dihedral_reference = dihedral_reference.cuda()
        predict = dist_predict + angle_predict + dihedral_predict
        reference = dist_reference + angle_reference + dihedral_reference
        loss_3d_pos = criterion(dist_predict,dist_reference) + 0.1 * criterion(angle_predict,angle_reference) +  0.1 * criterion(dihedral_predict,dihedral_reference)
        #loss_3d_pos.backward()
        #optimizer.step()
        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
        result_3d.append(record)
        smile_list.append(smile_data)
        id_list.append(id_data)
    return epoch_loss_3d_pos.avg,result_3d,smile_list,id_list


def get_torsions(mol_list):
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

def test(dataloader,model_pos, device):
    epoch_loss_3d_pos = AverageMeter()
    #model_pos.train()
    # Switch to train mode
    torch.set_grad_enabled(False)
    #data_input = np.load('./data/qm9_data_2d.npz')
    model_pos.eval()
    #end = time.time()
    #dl = tqdm(dataloader)
    for index,(adj_data,mul_data,input_data,target) in enumerate(dataloader):

        num_poses = data_target.size(0)
        adj_data,input_data,target,mul_data = adj_data.to(device),input_data.to(device),target.to(device),mul_data.to(device)
        outputs_3d = model_pos(adj_data,mul_data,input_data,src_mask).cuda()
        #print(type(outputs_3d))
        mse_torch = torch.cdist(outputs_3d,outputs_3d,p=2)
        mse_torch = mse_torch.cuda()
        #optimizer.zero_grad()
        #outputs_3d = torch.tensor(outputs_3d)
        
        loss_3d_pos = criterion(mse_torch,target)
        #loss_3d_pos.backward()
        #optimizer.step()
        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
    
    return epoch_loss_3d_pos.avg, lr_now
#def calc

if __name__ == "__main__":

    df,smile_list,id_list = read_csv()
    #print(id_list[0])
    # id_list = np.array(id_list)
    # np.save('./id_list.npy',id_list)
    # print("success")
    # #x_train,__,__ = dataset(df) 
    # ##print(smile_list)
    # adj_matrix_list = graph_matrix(smile_list)
    # #print("adj_list的长度:",len(adj_matrix_list))
    # ##print(adj_matrix_list[0])
    # # atom_features,bond_features,pair_indices,molecule_indicator,mpnn = MPNNModel(
    # #     atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0],
    # # )
    # ##print(atom_features)
    weight_decay = 0.01
    model = GraFormer_distance_matrix.GraFormer(hid_dim=128,n_pts=9).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-5,weight_decay=weight_decay)
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
    data_target = np.load('./qm9_3d.npy',allow_pickle=True)
    data_target = torch.from_numpy(data_target)
    data_target = data_target.float()
    y = data_target
    adj_matrix_list = np.load('./adj_array.npy',allow_pickle=True)
    x_list = np.load('./x_array.npy',allow_pickle=True)
    mul_L_list = np.load('./mul_array.npy',allow_pickle=True)
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
    datasets = Mydataset(id_list,smile_list,adj_matrix_list,mul_L_list,x_list, y)  # 初始化
    # #print(datasets.__len__())
    train_size = int(0.8 * len(datasets))
    validate_size = int (0.10 * len(datasets))
    test_size = len(datasets) - train_size - validate_size
    train_dataset , validate_dataset , test_dataset = torch.utils.data.random_split(datasets,[train_size,validate_size,test_size],generator=torch.Generator().manual_seed(30))
    train_loader = data.DataLoader(train_dataset,batch_size=200)
    validate_loader = data.DataLoader(validate_dataset,batch_size=200)
    test_loader = data.DataLoader(test_dataset,batch_size=200)
    #for index,(id_data,smile_data,adj_data,mul_data,input_data,target) in enumerate(test_loader):
    #dataloader = data.DataLoader(datasets, batch_size=200) 

    # for index,(adj_data,mul_data,input_data,target) in enumerate(dataloader):
    #     print('adj_data的形状为：' ,adj_data.shape)
    #     print('input_data的形状为' , input_data.shape)
    #     print('target的形状为' , target.shape)
    # ##print(x_list)
    # #result = model(adj_matrix_list,x_list,src_mask)
    # ##print(len(result))
    # log_dir = './qm9_3d.pth'
    # if os.path.exists(log_dir):
    #     checkpoint = torch.load(log_dir)
    #     print("exist")
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # log_dir = './qm9_3d_angel.pth'
    # if os.path.exists(log_dir):
    #     checkpoint = torch.load(log_dir)
    #     print("exist")
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = 0
    end_epoch = 1
    # for epoch in range(start_epoch, epoch):
    #     epoch_loss, lr_now, glob_step = train(train_loader, model_pos, criterion, optimizer, device, args.lr, lr_now,
    #                                           glob_step, args.lr_decay, args.lr_gamma, max_norm=True)
    #adj = GraFormer.adj_mx_from_edges(num_pts=9, edges=gan_edges, sparse=False)
    min_loss = 100
    record_epoch_loss = []
    record_validate_loss = []
    #record_test_loss = []
    lr_list = []
    lr_now = 1e-7
    
    for epoch in range(start_epoch,end_epoch):
        
        train_epoch_loss, lr_now,train_result_3d,train_smile_list,train_id_list  = train(dataloader=train_loader,model_pos=model, device=device,criterion=criterion, optimizer=optimizer, lr_now=lr_now)
        evaluate_epoch_loss,result_3d,smile_list,id_list = evaluate(dataloader=validate_loader,model_pos=model,device=device)
        test_epoch_loss,test_result_3d,test_smile_list,test_id_list = evaluate(dataloader=test_loader,model_pos=model,device=device)
        #np.save('./')
        #test_epoch_loss,lr_now = evaluate(dataloader=test_loader,model_pos=model,device=device)
        record_epoch_loss.append(train_epoch_loss)
        record_validate_loss.append(evaluate_epoch_loss)
        #record_test_loss.append(test_epoch_loss)
        lr_list.append(lr_now)
        if (epoch%10==0):
            print("第%s次train_distance_matrix_loss为:" %epoch,train_epoch_loss)
            print("第%s次evaluate_distance_matrix_loss为:" %epoch,evaluate_epoch_loss)
            #print("第%s次test_distance_matrix_loss为:" %epoch,test_epoch_loss)
            with open('./distance_matrix_loss_save_angel.txt','a+') as f:
                
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
        if (epoch%40==0 and epoch!=0):
            for p in optimizer.param_groups:
                p['lr'] *= 0.8
            torch.save({
                            'epoch':epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'loss':train_epoch_loss
                        },'./qm9_3d_angel.pth')
        
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
    lr_list = np.array(lr_list)
    result_3d = np.array(result_3d)
    smile_list = np.array(smile_list)
    train_smile_list = np.array(train_smile_list)
    test_smile_list = np.array(test_smile_list)
    id_list = np.array(id_list)
    np.save('./id_data.npy',id_list)
    np.save('./epoch_loss_test_angel.npy',record_epoch_loss)
    np.save('./epoch_evaluate_loss_angel.npy',record_validate_loss)
    np.save('./result_3d_angel.npy',result_3d)
    np.save('./smile_data_angel.npy',smile_list)
    #np.save('./epoch_test_loss.npy',record_test_loss)
    np.save('./lr_rate_angel.npy',lr_list)
    np.save('./train_smile_angel.npy',train_smile_list)
    np.save('./test_smile.npy',test_smile_list)
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
