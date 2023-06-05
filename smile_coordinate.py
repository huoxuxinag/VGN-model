import numpy as np
#import feature_generator 
#import modify_coordinate
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
def get_mol_list(smiles_array):
    mol_list = []
    for i in range(0,len(smiles_array)):
        for j in range(0,len(smiles_array[i])):
            smiles = str(smiles_array[i][j])
            #print(smiles)
            molecule = Chem.MolFromSmiles(smiles)
            mol_list.append(molecule)
    return mol_list

def get_sample_mol_list(smiles_array):
    mol_list = []
    for i in range(0,len(smiles_array)):
        #for j in range(0,len(smiles_array[i])):
        smiles = str(smiles_array[i])
        #print(smiles)
        molecule = Chem.MolFromSmiles(smiles)
        mol_list.append(molecule)
    return mol_list

def get_Geom_test_mol_list(Geom_smiles_list):
    mol_list = []
    for batch_smiles in Geom_smiles_list:
        for smiles in batch_smiles:
            smiles = str(smiles)
            molecule = Chem.MolFromSmiles(smiles)
            mol_list.append(molecule)
    return mol_list

def save_mol_to_sdf(mol_list,path):
    writer = Chem.SDWriter(path)
    for mol in mol_list:
        writer.write(mol)

    writer.close()



# def save_geom_mol_to_sdf(geom_list,path):
#     writer = Chem.SDWriter(path)
#     for batch_mol in gemo

from rdkit import Chem
#mol_suppl = Chem.SDMolSupplier('./data/qm9_3d_test.sdf')
import numpy as np
#import torch
#from features.feature_generator import mol_information

def modify_3d_coordinate(predict_coordinate,path):
    coordinate_3d = []
    atom_num_list = []
    mol_num = -1
    atom_num = 0
    sdf_writer = Chem.SDWriter(path)
    sdf = Chem.SDMolSupplier('./test_smile.sdf')
    for mol in sdf:
        #print(mol)
        mol_num += 1
        num_atoms = mol.GetNumAtoms()
        for i in range(0,num_atoms):
            atom_num += 1
            mol.GetConformer().SetAtomPosition(i,(float(predict_coordinate[mol_num][i][0]),float(predict_coordinate[mol_num][i][1]),float(predict_coordinate[mol_num][i][2]))) 
        sdf_writer.write(mol) 
            #mol.GetConformer().SetAtomPosition(i) 
            #mol.GetConformer().SetAtomPosition(i)  
            #coordinate_3d.append(x)
            #coordinate_3d.append(y)
            #coordinate_3d.append(z)
        #atom_num_list.append(coordinate_3d)
    #coordinate_3d = np.array(coordinate_3d)
    #coordinate_3d = coordinate_3d.reshape(atom_num,3)
    #coordinate_3d = torch.from_numpy(coordinate_3d)
    #return coordinate_3d,atom_num_list,mol_num

def modify_3d_Gemo_coordinate(predict_coordinate,path):
    coordinate_3d = []
    atom_num_list = []
    mol_num = -1
    atom_num = 0
    sdf_writer = Chem.SDWriter(path)
    sdf = Chem.SDMolSupplier('./test_smile.sdf')
    for mol in sdf:
        #print(mol)
        mol_num += 1
        num_atoms = mol.GetNumAtoms()
        for i in range(0,num_atoms):
            atom_num += 1
            mol.GetConformer().SetAtomPosition(i,(float(predict_coordinate[mol_num][i][0]),float(predict_coordinate[mol_num][i][1]),float(predict_coordinate[mol_num][i][2]))) 
        sdf_writer.write(mol) 

def compute_3d_coordinate():
    coordinate_3d = []
    atom_num_list = []
    sum = 0
    sdf_writer = Chem.SDWriter('./model_save/test_3d_true.sdf')
    sdf = Chem.SDMolSupplier('./model_save/test_true.sdf')
    for mol in sdf:
        sum += 1
        m2 = AllChem.AddHs(mol)
        AllChem.EmbedMolecule(m2,randomSeed=1)
        #AllChem.MMFFOptimizeMolecule(m2)
        m3 = Chem.RemoveHs(m2)
        sdf_writer.write(m3)
    sdf_writer.close()
#predict_200 = np.load('./data/predict_coordinate_9_200.npy')
def cat_tensor(path):
    data = np.load(path,allow_pickle=True)
    #print(len(data))
    result_3d = data[0]
    for i in range(1,len(data)):
        #print(data[i].shape)
        result_3d = torch.cat((result_3d,data[i]),0)

    #print(result_3d.shape)
    result_3d_array = np.array(result_3d)
    #print(result_3d_array.shape)
    #np.save('./model_save/result_3d_array.npy',result_3d_array)
    #print(result_3d_array[0][0])
    return result_3d_array


def modify_sample_3d_coordinate(predict_coordinate,suppiler_path,path):
    coordinate_3d = []
    atom_num_list = []
    mol_num = -1
    atom_num = 0
    sdf_writer = Chem.SDWriter(path)
    sdf = Chem.SDMolSupplier(suppiler_path)
    print(len(sdf))
    for mol in sdf:
        #print(mol)
        mol_num += 1
        num_atoms = mol.GetNumAtoms()
        for i in range(0,num_atoms):
            atom_num += 1
            #print(float(predict_coordinate[mol_num][i][0]))
            mol.GetConformer().SetAtomPosition(i,(float(predict_coordinate[mol_num][i][0]),float(predict_coordinate[mol_num][i][1]),float(predict_coordinate[mol_num][i][2]))) 
            
        sdf_writer.write(mol) 

def modify_sample_Gemo_3d_coordinate(predict_coordinate,suppiler_path,path):
    coordinate_3d = []
    atom_num_list = []
    mol_num = -1
    atom_num = 0
    sdf_writer = Chem.SDWriter(path)
    sdf = Chem.SDMolSupplier(suppiler_path)
    for mol in sdf:
        #print(mol)
        mol_num += 1
        num_atoms = mol.GetNumAtoms()
        for i in range(0,num_atoms):
            atom_num += 1
            mol.GetConformer().SetAtomPosition(i,(float(predict_coordinate[mol_num][i][0]),float(predict_coordinate[mol_num][i][1]),float(predict_coordinate[mol_num][i][2]))) 
        sdf_writer.write(mol) 


def test():
    result_3d_array = cat_tensor()
    smile_data = np.load('./model_save/smile_data.npy',allow_pickle=True)
    result_3d = np.load('./model_save/result_3d.npy',allow_pickle=True)



    #print(result_3d.shape)
    #result_3d_array = np.array(result_3d)
    #print(result_3d_array.shape)
    #np.save('./model_save/result_3d_array.npy',result_3d_array)
    #print(result_3d_array[0][0])

    #print(smile_data[0][0])
    #print(result_3d_array[0][0][0])
    #result_3d_array = result_3d_array.reshape(len(result_3d_array)*9,3)
    smile_list = get_mol_list(smile_data)
    save_mol_to_sdf(smile_list)
    compute_3d_coordinate()
    modify_3d_coordinate(result_3d_array)


