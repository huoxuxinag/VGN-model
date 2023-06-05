import msgpack
import os
import numpy as np
import rdkit
from rdkit import Chem
def get_smiles_data():
    direc = "./"
    qm9_file = os.path.join(direc,"qm9_featurized.msgpack")
    unpacker = msgpack.Unpacker(open(qm9_file,"rb"))
    #qm9_lk = next(iter(unpacker))
    overall_qm9_dic = {}
    #print(len(qm9_lk))
    for qm9_dic in unpacker:
        overall_qm9_dic.update(qm9_dic)
    smiles = list(overall_qm9_dic.keys())[:]
    qm9_dic_list = []
    for i in range(0,len(smiles)):
        qm9_dic_list.append(overall_qm9_dic[smiles[i]]) 
    print(len(qm9_dic_list))
    qm9_smile_list = []
    qm9_confor = []
    qm9_xyz_smiles = []
    for j in range(0,len(smiles)):
        for k in range(0,len(overall_qm9_dic[smiles[j]])):
            qm9_smile_list.append(smiles[j])
            qm9_xyz_smiles.append(overall_qm9_dic[smiles[j]][k]['xyz2mol_smiles'])
    # for j in range(0,len(smiles)):
    #     for k in range(0,len(overall_qm9_dic[smiles[j]]["conformers"])):
    #         #qm9_confor.update({smiles[j]:overall_qm9_dic[smiles[j]]["conformers"][k]['xyz']})
    #         qm9_confor.append(overall_qm9_dic[smiles[j]]["conformers"][k]['xyz'])
    #         qm9_smile_list.append(smiles[j])
    #         qm9_xyz_smiles.append(overall_qm9_dic[smiles[j]]["conformers"][k]['xyz'])
    #qm9_dict = []
    # for smile in smiles:
    #     qm9_dict.append(overall_qm9_dic[smile])
    # qm9_conformation = {}
    # for i in range(0,len(smiles)):
    #     for j in range(0,len(overall_qm9_dic[i]["conformers"])):
    #         qm9_conformation.update(smiles[i],overall_qm9_dic[i]["conformers"][j])

    #qm9_dic = [smile for smile in smiles qm9_lk[smiles]]
    # sample_smiles = list(qm9_lk.keys())[:]
    #print(sample_smiles)
    #sample_sub_dic = qm9_lk[sample_smiles]
    #print(len(sample_sub_dic))
    #tutle = {key:val for key,val in sample_sub_dic.items() if key!='conformers'}
    #print(tutle)
    #print(sample_sub_dic['conformers'][0])
    return qm9_smile_list,qm9_xyz_smiles

def load_qm9_data():
    qm9_smile_data = np.load('./geom_qm9_smile.npy',allow_pickle=True)
    qm9_confor_data = np.load('./geom_qm9_confor_xyz.npy',allow_pickle=True)
    

def smile_to_sdf(smile_list):
    mol_list = []
    sdf_writer = Chem.SDWriter('./geo_qm9_sdf.sdf')
    error_id =[]
    count = 0
    for smile in smile_list:
        count += 1
        mol = Chem.MolFromSmiles(smile)
        #print(mol)
        mol_list.append(mol)
        try :
            sdf_writer.write(mol)
        except :
            error_id.append(count)
    # for mol in mol
    sdf_writer.close()
    return mol_list,error_id

def xyz_remove_h(qm9_xyz_list):
    qm9_xyz_remove_h_list = []
    qm9_xyz_remove_h_total = []
    for i in range(0,len(qm9_xyz_list)):
        for j in range(0,len(qm9_xyz_list[i])):
            if qm9_xyz_list[i][j][0] != float(1):
                qm9_xyz_remove_h_list.append(qm9_xyz_list[i][j])
            qm9_xyz_remove_h_array = np.array(qm9_xyz_remove_h_list)
        qm9_xyz_remove_h_total.append(qm9_xyz_remove_h_array)
    return qm9_xyz_remove_h_total

def test_atom_id(suppiler_sdf_path):
    sdf = Chem.SDMolSupplier(suppiler_sdf_path)
    mol_num = 0
    for mol in sdf:
        mol_num += 1
        print(Chem.MolToSmiles(mol))
        if mol_num>1:
            break

if __name__ == '__main__':

    qm9_smile_list , qm9_xyz_smiles = get_smiles_data()
    print(qm9_smile_list[0])
    print(qm9_xyz_smiles[0])
    # error_id_array = np.load('./error_id_array.npy',allow_pickle=True)
    # test = Chem.MolFromSmiles(qm9_smile_list[0])
    # test_smiles = Chem.MolToSmiles(test)
    # print(test_smiles)
    # print(qm9_smile_list[0])
    # test_sdf_writer = Chem.SDWriter('./test_one.sdf')
    # test_sdf_writer.write(test)
    # test_sdf_writer.close()
    # suppiler_path = './test_one.sdf'
    # #geom_sdf_suppiler_path 
    # sdf = Chem.SDMolSupplier(suppiler_path)
    # for mol in sdf:
    #     print(Chem.MolToSmiles(mol))
    # #mol_list , error_id_array =  
    # #print(error_id_array[0])
    # suppiler_sdf_path = './geo_qm9_sdf.sdf'
    # geom_sdf = Chem.SDMolSupplier(suppiler_sdf_path)
    # print(Chem.MolToSmiles(geom_sdf[0]))
    # #print(qm9_confor[0][:][0])
    # for i in range(0,len(qm9_confor[0])):
    #     print(qm9_confor[0][i][0])
    # for i in range(0,geom_sdf[0].GetNumAtoms()):
    #     print(i,geom_sdf[0].GetAtomWithIdx(i).GetAtomicNum())
    #test_atom_id(suppiler_path)
    # # print(len(qm9_confor))
    # # print(len(qm9_smile_list))
    # qm9_confor_array = np.array(qm9_confor)
    # qm9_smile_array = np.array(qm9_smile_list)
    # #np.save('./geom_qm9_smile.npy',qm9_smile_array)
    # #np.save('./geom_qm9_confor_xyz.npy',qm9_confor_array)
    
    # print(len(qm9_confor))
    # print(len(qm9_smile_list))
    # print(qm9_confor[0])
    # qm9_remove_h_total = xyz_remove_h(qm9_confor)
    # qm9_remove_h_array = np.array(qm9_remove_h_total)
    # np.save('./qm9_remove_h_conformer.npy',qm9_remove_h_array)
    # print(len(qm9_remove_h_total))
    # print(qm9_remove_h_total[0])
    #mol_list , error_id_list = smile_to_sdf(qm9_smile_list)
    #error_id_array = np.array(error_id_list)
    #np.save('./error_id_array.npy',error_id_array)
    #print(len(mol_list))
    #print(len(error_id_list))
    #print(qm9_smile_list[0:10])
    # smiles_data,qm9_confor = get_smiles_data()
    # print(smiles_data[0])
    # print(len(smiles_data))
    # #print(qm9_confor)
    # smile_list = list(qm9_confor)
    # confor_list = list(qm9_confor.values())
    # smiles_array = np.array(smile_list)
    # confor_array = np.array(confor_list)
    # np.save('./geom_qm9_smile.npy',smiles_array)
    # np.save('./geom_qm9_confor_xyz.npy',confor_array)
    # # for key,values in qm9_confor:
    #     print("%s,%s"%key %values)
    #print(len(qm9_all_smiles))
    #print(smiles[0])
    #qm9_smile_data,qm9_confor_data = load_qm9_data()
    # print(len(qm9_smile_data))
    # print((qm9_confor_data[9][0]))
    #print(qm9_smile_data[0:2])
    #print(qm9_confor_data[0:2])
    # qm9_smile_data,qm9_confor_data = load_qm9_data()
    # print(len(qm9_smile_data))
    # print(len(qm9_confor_data))
    # #print(qm9_smile_data[0:3])
    # #print(qm9_confor_data[0:3])
    # mol_list,error_id = smile_to_sdf(qm9_smile_data)
    # print(len(mol_list))
    # print(len(error_id))
    # print(len(qm9_smile_data))
    # print(len(qm9_confor_data))