from copy import deepcopy
from rdkit.Geometry import Point3D
from rdkit import Chem
import numpy as np
import torch
from datetime import datetime
from rdkit.Chem import rdMolTransforms
import math
import copy
import os 
import faulthandler
from torch.utils.data import Dataset
#from commons.process_mols import get_geometry_graph, get_lig_graph_revised, get_rdkit_coords
from dgl import batch
from rdkit.Chem import SDMolSupplier, SanitizeMol, SanitizeFlags, PropertyMol, SmilesMolSupplier, AddHs
import dgl
from rdkit.Chem import AllChem
import scipy.spatial as spa
from scipy.special import softmax
from torch.utils.data import DataLoader
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

def log(*args):
    print(f'[{datetime.now()}]', *args)

def get_geometry_graph(lig):
    coords = lig.GetConformer().GetPositions()
    edges_src = []
    edges_dst = []
    for i, atom in enumerate(lig.GetAtoms()):
        src_idx = atom.GetIdx()
        assert src_idx == i
        one_hop_dsts = [neighbor for neighbor in list(atom.GetNeighbors())]
        two_and_one_hop_idx = [neighbor.GetIdx() for neighbor in one_hop_dsts]
        for one_hop_dst in one_hop_dsts:
            for two_hop_dst in one_hop_dst.GetNeighbors():
                two_and_one_hop_idx.append(two_hop_dst.GetIdx())
        all_dst_idx = list(set(two_and_one_hop_idx))
        if len(all_dst_idx) ==0: continue
        all_dst_idx.remove(src_idx)
        all_src_idx = [src_idx] *len(all_dst_idx)
        edges_src.extend(all_src_idx)
        edges_dst.extend(all_dst_idx)
    graph = dgl.graph((torch.tensor(edges_src), torch.tensor(edges_dst)), num_nodes=lig.GetNumAtoms(), idtype=torch.long)
    graph.edata['feat'] = torch.from_numpy(np.linalg.norm(coords[edges_src] - coords[edges_dst], axis=1).astype(np.float32))
    return graph

def get_dihedral_vonMises(mol, conf, atom_idx, Z):
    Z = np.array(Z)
    v = np.zeros((2,1))
    iAtom = mol.GetAtomWithIdx(atom_idx[1])
    jAtom = mol.GetAtomWithIdx(atom_idx[2])
    k_0 = atom_idx[0]
    i = atom_idx[1]
    j = atom_idx[2]
    l_0 = atom_idx[3]
    for b1 in iAtom.GetBonds():
        k = b1.GetOtherAtomIdx(i)
        if k == j:
            continue
        for b2 in jAtom.GetBonds():
            l = b2.GetOtherAtomIdx(j)
            if l == i:
                continue
            assert k != l
            s_star = S_vec(GetDihedralFromPointCloud(Z, (k, i, j, l)))
            a_mat = A_transpose_matrix(GetDihedral(conf, (k, i, j, k_0)) + GetDihedral(conf, (l_0, i, j, l)))
            v = v + np.matmul(a_mat, s_star)
    v = v / np.linalg.norm(v)
    v = v.reshape(-1)
    return np.degrees(np.arctan2(v[1], v[0]))

def A_transpose_matrix(alpha):
    return np.array([[np.cos(np.radians(alpha)), np.sin(np.radians(alpha))],
                     [-np.sin(np.radians(alpha)), np.cos(np.radians(alpha))]], dtype=np.double)



def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.deepcopy(mol)
    #     opt_mol = add_rdkit_conformer(opt_mol)

    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]

    #     # apply transformation matrix
    #     rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))

    return opt_mol

def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t

def GetDihedralFromPointCloud(Z, atom_idx):
    p = Z[list(atom_idx)]
    b = p[:-1] - p[1:]
    b[0] *= -1 #########################
    v = np.array( [ v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]] ] )
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.degrees(np.arctan2( y, x ))

def S_vec(alpha):
    return np.array([[np.cos(np.radians(alpha))],
                     [np.sin(np.radians(alpha))]], dtype=np.double)


def get_rdkit_coords(mol, seed = None):
    ps = AllChem.ETKDGv2()
    if seed is not None:
        ps.randomSeed = seed
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    else:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    conf = mol.GetConformer()
    lig_coords = conf.GetPositions()
    return torch.tensor(lig_coords, dtype=torch.float32)

def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])

def run_corrections(lig, lig_coord, ligs_coords_pred_untuned):
    input_coords = lig_coord.detach().cpu()
    prediction = ligs_coords_pred_untuned.detach().cpu()
    lig_input = deepcopy(lig)
    conf = lig_input.GetConformer()
    for i in range(lig_input.GetNumAtoms()):
        x, y, z = input_coords.numpy()[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    lig_equibind = deepcopy(lig)
    conf = lig_equibind.GetConformer()
    for i in range(lig_equibind.GetNumAtoms()):
        x, y, z = prediction.numpy()[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    coords_pred = lig_equibind.GetConformer().GetPositions()

    Z_pt_cloud = coords_pred
    rotable_bonds = get_torsions([lig_input])
    new_dihedrals = np.zeros(len(rotable_bonds))
    for idx, r in enumerate(rotable_bonds):
        new_dihedrals[idx] = get_dihedral_vonMises(lig_input, lig_input.GetConformer(), r, Z_pt_cloud)
    optimized_mol = apply_changes(lig_input, new_dihedrals, rotable_bonds)
    optimized_conf = optimized_mol.GetConformer()
    coords_pred_optimized = optimized_conf.GetPositions()
    R, t = rigid_transform_Kabsch_3D(coords_pred_optimized.T, coords_pred.T)
    coords_pred_optimized = (R @ (coords_pred_optimized).T).T + t.squeeze()
    for i in range(optimized_mol.GetNumAtoms()):
        x, y, z = coords_pred_optimized[i]
        optimized_conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return optimized_mol

class Ligands(Dataset):
    def __init__(self, ligpath, rec_graph, args, lazy = None, slice = None, skips = None, ext = None, addH = None, rdkit_seed = None):
        self.ligpath = ligpath
        self.rec_graph = rec_graph
        self.args = args
        self.dp = args.dataset_params
        self.use_rdkit_coords = args.use_rdkit_coords
        self.device = args.device
        self.rdkit_seed = rdkit_seed
        
        ##Default argument handling
        self.skips = skips if skips is not None else set()

        extensions_requiring_conformer_generation = ["smi"]
        extensions_defaulting_to_lazy = ["smi"]

        if ext is None:
            try:
                ext = ligpath.split(".")[-1]
            except (AttributeError, KeyError):
                ext = "sdf"
        

        if lazy is None:
            if ext in extensions_defaulting_to_lazy:
                self.lazy = True
            else:
                self.lazy = False
        else:
            self.lazy = lazy

        if addH is None:
            if ext == "smi":
                addH = True
            else:
                addH = False
        self.addH = addH
        
        self.generate_conformer = ext in extensions_requiring_conformer_generation

        suppliers = {"sdf": SDMolSupplier, "smi": SmilesMolSupplier}
        supp_kwargs = {"sdf": dict(sanitize = False, removeHs =  False),
                        "smi": dict(sanitize = False)}
        self.supplier = suppliers[ext](ligpath, **supp_kwargs[ext])

        if slice is None:
            self.slice = 0, len(self.supplier)
        else:
            slice = (slice[0] if slice[0] >= 0 else len(self.supplier)+slice[0], slice[1] if slice[1] >= 0 else len(self.supplier)+slice[1])
            self.slice = tuple(slice)

        self.failed_ligs = []
        self.true_idx = []

        if not self.lazy:
            self.ligs = []
            for i in range(*self.slice):
                if i in self.skips:
                    continue
                lig = self.supplier[i]
                lig, name = self._process(lig)
                if lig is not None:
                    self.ligs.append(PropertyMol.PropertyMol(lig))
                    self.true_idx.append(i)
                else:
                    self.failed_ligs.append((i, name))

        if self.lazy:
            self._len = self.slice[1]-self.slice[0]
        else:
            self._len = len(self.ligs)

    def _process(self, lig):
        if lig is None:
            return None, None
        if self.addH:
            lig = AddHs(lig)
        if self.generate_conformer:
            get_rdkit_coords(lig, self.rdkit_seed)
        sanitize_succeded = (SanitizeMol(lig, catchErrors = True) is SanitizeFlags.SANITIZE_NONE)
        if sanitize_succeded:
            return lig, lig.GetProp("_Name")
        else:
            return None, lig.GetProp("_Name")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.lazy:
            if idx < 0:
                nonneg_idx = self._len + idx
            else:
                nonneg_idx = idx

            if nonneg_idx >= self._len or nonneg_idx < 0:
                raise IndexError(f"Index {idx} out of range for Ligands dataset with length {len(self)}")
            
            
            true_index = nonneg_idx + self.slice[0]
            if true_index in self.skips:
                return true_index, "Skipped"
            lig = self.supplier[true_index]
            lig, name = self._process(lig)
            if lig is not None:
                lig = PropertyMol.PropertyMol(lig)
            else:
                self.failed_ligs.append((true_index, name))
                return true_index, name
        elif not self.lazy:
            lig = self.ligs[idx]
            true_index = self.true_idx[idx]

        
        try:
            lig_graph = get_lig_graph_revised(lig, lig.GetProp('_Name'), max_neighbors=self.dp['lig_max_neighbors'],
                                            use_rdkit_coords=self.use_rdkit_coords, radius=self.dp['lig_graph_radius'])
        except AssertionError:
            self.failed_ligs.append((true_index, lig.GetProp("_Name")))
            return true_index, lig.GetProp("_Name")
        
        geometry_graph = get_geometry_graph(lig) if self.dp['geometry_regularization'] else None

        lig_graph.ndata["new_x"] = lig_graph.ndata["x"]
        return lig, lig_graph.ndata["new_x"], lig_graph, self.rec_graph, geometry_graph, true_index
    
    @staticmethod
    def collate(_batch):
        sample_succeeded = lambda sample: not isinstance(sample[0], int)
        sample_failed = lambda sample: isinstance(sample[0], int)
        clean_batch = tuple(filter(sample_succeeded, _batch))
        failed_in_batch = tuple(filter(sample_failed, _batch))
        if len(clean_batch) == 0:
            return None, None, None, None, None, None, failed_in_batch
        ligs, lig_coords, lig_graphs, rec_graphs, geometry_graphs, true_indices = map(list, zip(*clean_batch))
        output = (
            ligs,
            lig_coords,
            batch(lig_graphs),
            batch(rec_graphs),
            batch(geometry_graphs) if geometry_graphs[0] is not None else None,
            true_indices,
            failed_in_batch
        )
        return output

def get_mol_information(sdf_path,predict_sdf_path):
    coordinate_3d = []
    predict_coordinate_3d = []
    atom_num_list = []
    predict_atom_num_list = []
    sum = 0
    predict_sum = 0
    atom_num = 0
    predict_atom_num = 0
    mol_list = []
    sdf = Chem.SDMolSupplier(sdf_path)
    predict_sdf = Chem.SDMolSupplier(predict_sdf_path)
    for mol in sdf:
        sum += 1
        mol_list.append(mol)
    for mol in sdf:
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
    
    for mol in predict_sdf:
        num_atoms = mol.GetNumAtoms()
        for i in range(0,num_atoms):
            predict_atom_num += 1
            x = mol.GetConformer().GetAtomPosition(i).x
            y = mol.GetConformer().GetAtomPosition(i).y
            z = mol.GetConformer().GetAtomPosition(i).z
            predict_coordinate_3d.append(x)
            predict_coordinate_3d.append(y)
            predict_coordinate_3d.append(z)
        predict_atom_num_list.append(num_atoms)
    predict_coordinate_3d = np.array(predict_coordinate_3d)
    predict_coordinate_3d = predict_coordinate_3d.reshape(sum,9,3)
    coordinate_3d = torch.from_numpy(coordinate_3d)
    predict_coordinate_3d = torch.from_numpy(predict_coordinate_3d)
    return mol_list,coordinate_3d,predict_coordinate_3d

def get_result_coordinate(sdf_path):
    coordinate_3d = []
    
    atom_num_list = []
    
    sum = 0
    
    atom_num = 0
    
    mol_list = []
    sdf = Chem.SDMolSupplier(sdf_path)
    
    for mol in sdf:
        sum += 1
        mol_list.append(mol)
    for mol in sdf:
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
    
    return mol_list,coordinate_3d


def get_lig_graph_revised(mol, name, radius=20, max_neighbors=None, use_rdkit_coords=False):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    if use_rdkit_coords:
        try:
            rdkit_coords = get_rdkit_coords(mol).numpy()
            R, t = rigid_transform_Kabsch_3D(rdkit_coords.T, true_lig_coords.T)
            lig_coords = ((R @ (rdkit_coords).T).T + t.squeeze())
            log('kabsch RMSD between rdkit ligand and true ligand is ', np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())
        except Exception as e:
            lig_coords = true_lig_coords
            with open('temp_create_dataset_rdkit_timesplit_no_lig_or_rec_overlap_train.log', 'a') as f:
                f.write('Generating RDKit conformer failed for  \n')
                f.write(name)
                f.write('\n')
                f.write(str(e))
                f.write('\n')
                f.flush()
            print('Generating RDKit conformer failed for  ')
            print(name)
            print(str(e))
    else:
        lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)

    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            # log(
            #     f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
        assert i not in dst
        assert dst != []
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distance[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distance[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = lig_coords[src, :] - lig_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)

        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['feat'] = lig_atom_featurizer(mol)
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata['x'] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    if use_rdkit_coords:
        graph.ndata['new_x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    return graph

def distance_featurizer(dist_list, divisor) -> torch.Tensor:
    # you want to use a divisor that is close to 4/7 times the average distance that you want to encode
    length_scale_list = [1.5 ** x for x in range(15)]
    center_list = [0. for _ in range(15)]

    num_edge = len(dist_list)
    dist_list = np.array(dist_list)

    transformed_dist = [np.exp(- ((dist_list / divisor) ** 2) / float(length_scale))
                        for length_scale, center in zip(length_scale_list, center_list)]

    transformed_dist = np.array(transformed_dist).T
    transformed_dist = transformed_dist.reshape((num_edge, -1))
    return torch.from_numpy(transformed_dist.astype(np.float32))

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}

def lig_atom_featurizer(mol):
    #ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        g_charge = atom.GetDoubleProp('_GasteigerCharge')
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_fo.rmal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
            g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
        ])

    return torch.tensor(atom_features_list)

def write_while_inferring(sdf_path,predict_sdf_path):
    
    # full_output_path = os.path.join(args.output_directory, "output.sdf")
    # full_failed_path = os.path.join(args.output_directory, "failed.txt")
    # full_success_path = os.path.join(args.output_directory, "success.txt")

    # w_or_a = "a" if args.skip_in_output else "w"
    # with torch.no_grad(), open(full_output_path, w_or_a) as file, open(
    #     full_failed_path, "a") as failed_file, open(full_success_path, w_or_a) as success_file:
    #     with Chem.SDWriter(file) as writer:
    #         i = 0
    #         total_ligs = len(dataloader.dataset)
    #         for batch in dataloader:
    #             i += args.batch_size
    #             print(f"Entering batch ending in index {min(i, total_ligs)}/{len(dataloader.dataset)}")
    #             ligs, lig_coords, lig_graphs, rec_graphs, geometry_graphs, true_indices, failed_in_batch = batch
    #             for failure in failed_in_batch:
    #                 if failure[1] == "Skipped":
    #                     continue
    #                 failed_file.write(f"{failure[0]} {failure[1]}")
    #                 failed_file.write("\n")
    #             if ligs is None:
    #                 continue
    #             lig_graphs = lig_graphs.to(args.device)
    #             rec_graphs = rec_graphs.to(args.device)
    #             geometry_graphs = geometry_graphs.to(args.device)
                
                
    #             out_ligs, out_lig_coords, predictions, successes, failures = run_batch(model, ligs, lig_coords,
    #                                                                                    lig_graphs, rec_graphs,
    #  
    #print("hello")
    #sdf_path = './gemo_sample_test_true.sdf'
    #predict_sdf_path = './gemo_sample_test_3d.sdf'                                                                                   
    out_ligs,out_lig_coords,predictions = get_mol_information(sdf_path,predict_sdf_path)
    #print(out_lig_coords[0][0][0])
    #print(predictions[0][0][0])
    opt_mols = [run_corrections(lig, lig_coord, prediction) for lig, lig_coord, prediction in zip(out_ligs, out_lig_coords, predictions)]
    result_sdf = './sample_100/result.sdf'
    writer = Chem.SDWriter(result_sdf)
    for mol in opt_mols:
        writer.write(mol)
    mol_list , output_3d =  get_result_coordinate(result_sdf)
    return output_3d
        #success_file.write(f"{success[0]} {success[1]}")
        #success_file.write("\n")
        # print(f"written {mol.GetProp('_Name')} to output")
    # for failure in failures:
    #     failed_file.write(f"{failure[0]} {failure[1]}")
    #     failed_file.write("\n")


#if __name__ == '__main__':
    # args, cmdline_args = parse_arguments(arglist)
    
    # args = get_default_args(args, cmdline_args)
    # assert args.output_directory, "An output directory should be specified"
    # assert args.ligands_sdf, "No ligand sdf specified"
    # assert args.rec_pdb, "No protein specified"
    # seed_all(args.seed)
    
    # os.makedirs(args.output_directory, exist_ok = True)

    # success_path = os.path.join(args.output_directory, "success.txt")
    # failed_path = os.path.join(args.output_directory, "failed.txt")
    # if os.path.exists(success_path) and os.path.exists(failed_path) and args.skip_in_output:
    #     with open(success_path) as successes, open(failed_path) as failures:
    #         previous_work = successes.readlines()


    #     previous_work = set(map(lambda tup: int(tup.split(" ")[0]), previous_work))
    #     print(f"Found {len(previous_work)} previously calculated ligands")
    # else:
    #     previous_work = None
    
        
    # rec_graph, model = load_rec_and_model(args)
    # if args.lig_slice is not None:
    #     lig_slice = tuple(map(int, args.lig_slice.split(",")))
    # else:
    #     lig_slice = None
    
    #lig_data = Ligands(args.ligands_sdf, rec_graph, args, slice = lig_slice, skips = previous_work, lazy = args.lazy_dataload)
    #lig_loader = DataLoader(lig_data, batch_size = args.batch_size, collate_fn = lig_data.collate, num_workers = args.n_workers_data_load)

    # full_failed_path = os.path.join(args.output_directory, "failed.txt")
    # with open(full_failed_path, "a" if args.skip_in_output else "w") as failed_file:
    #     for failure in lig_data.failed_ligs:
    #         failed_file.write(f"{failure[0]} {failure[1]}")
    #         failed_file.write("\n")
    
    #write_while_inferring()