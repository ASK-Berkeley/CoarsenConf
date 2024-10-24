import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
import networkx as nx

import torch#, tqdm
import tqdm as tqdm_outer
from tqdm import tqdm
import torch.nn.functional as F
# from multiprocessing import Pool
import torch.multiprocessing as mp
# import dgl.multiprocessing as mp
# import multiprocessing as mp
#mp.set_start_method('spawn') # use 'spawn' method instead of 'fork'
#mp.set_sharing_strategy('file_system')
import glob, pickle, random
import os
import os.path as osp
import copy
# from torch_geometric.data import Dataset, DataLoader
# from torch_geometric.transforms import BaseTransform
from collections import defaultdict
from molecule_utils import *
# from torch_scatter import scatter
# from torch_geometric.data import Dataset, Data, DataLoader
from dgl.data import DGLDataset
from dgl.dataloading import DataLoader
# import psutil
import concurrent.futures

dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
qm9_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
drugs_types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
               'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
               'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
               'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}

def check_distances(molecule, geometry_graph, B = False):
    src, dst = geometry_graph.edges()
    src = src.long()
    dst = dst.long()
    generated_coords = molecule.ndata['x'] #!!! THIS WAS THE BUG X_TRUE DOES NOT MAKE SENSE FOR RDKIT
    # print("Generated Coords for dist check", generated_coords)
    d_squared = torch.sum((generated_coords[src] - generated_coords[dst]) ** 2, dim=1)
    error = torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2)
    # if B:
    #     print("[B] Distance Loss Check", error)
    # else:
    #     print("[A] Distance Loss Check", error)
    #print(geometry_graph.edata['feat'])
    #print(np.linalg.norm(((generated_coords[src] - generated_coords[dst]).numpy()), axis = 1))
    #print(np.linalg.norm(((molecule.ndata['x'][src] - molecule.ndata['x'][dst]).numpy()), axis = 1))
    return error

def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def get_bond_idx(m, i, j):
    bond = m.GetBondBetweenAtoms(i,j)
#     info = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    if bond == None:
        return 4 #-1e9 # not inf for stab
    else:
#         typ = bond.GetBondType()
        return bonds[bond.GetBondType()]

QM9_DIMS = ([5, 4, 2, 8, 6, 8, 4, 6, 5], 0)
DRUG_DIMS = ([35, 4, 2, 8, 6, 8, 4, 6, 5], 0)

def featurize_mol_new(mol, types=drugs_types, conf_id = 0, use_rdkit_coords = False, seed = 0, radius = 4, max_neighbors=None, old_rdkit = False, use_mmff = True):
    if type(types) is str:
        if types == 'qm9':
            types = qm9_types
        elif types == 'drugs':
            types = drugs_types
    # print(conf_id)
    N = mol.GetNumAtoms()
    atom_type_idx = []
    atomic_number = []
    atom_features = []
    chiral_tag = []
    ring = mol.GetRingInfo()
    for i, atom in enumerate(mol.GetAtoms()):
        atom_type_idx.append(types[atom.GetSymbol()])
        chiral_tag.append(chirality[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())
        
        atom_features.extend([atom.GetAtomicNum(), 1 if atom.GetIsAromatic() else 0])
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2]))#6
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))#8
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))#4
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                              int(ring.IsAtomInRingOfSize(i, 4)),
                              int(ring.IsAtomInRingOfSize(i, 5)),
                              int(ring.IsAtomInRingOfSize(i, 6)),
                              int(ring.IsAtomInRingOfSize(i, 7)),
                              int(ring.IsAtomInRingOfSize(i, 8))])#6
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3])) # 5

    x1 = F.one_hot(torch.tensor(atom_type_idx), num_classes=len(types)) # 5
    x2 = torch.tensor(atom_features).view(N, -1) # 39
    x3 = torch.tensor(chiral_tag).view(N, -1).to(torch.float) # 1
    node_features = torch.cat([x1.to(torch.float), x3, x2], dim=-1)
    
    conf = mol.GetConformer(conf_id)
    true_lig_coords = conf.GetPositions()
    lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    remove_centroid = True
    # print("Lower Level Featurize", use_rdkit_coords, lig_coords)
    if remove_centroid:
        lig_coords -= np.mean(lig_coords, axis = 0)
        true_lig_coords -= np.mean(true_lig_coords, axis = 0)
        
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)
    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    bond_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            #print( f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
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
        bond_list.extend([get_bond_idx(mol, int(i), int(d)) for d in dst])
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)
    graph.ndata['feat'] = node_features #lig_atom_featurizer(mol)
    graph.ndata['ref_feat'] = node_features
    edge_type = torch.from_numpy(np.asarray(bond_list).astype(np.float32)).type(torch.long) #torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)+1).to(torch.float)
    graph.edata['feat'] = torch.cat((distance_featurizer(dist_list, 0.75), edge_attr), dim = -1)
    graph.ndata['x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    graph.ndata['x_ref'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    graph.ndata['x_true'] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    return graph

def featurize_mol(mol, types=drugs_types, use_rdkit_coords = False, seed = 0, radius = 4, max_neighbors=None, old_rdkit = False, use_mmff = True):
    if type(types) is str:
        if types == 'qm9':
            types = qm9_types
        elif types == 'drugs':
            types = drugs_types
    
    N = mol.GetNumAtoms()
    atom_type_idx = []
    atomic_number = []
    atom_features = []
    chiral_tag = []
    ring = mol.GetRingInfo()
    for i, atom in enumerate(mol.GetAtoms()):
        atom_type_idx.append(types[atom.GetSymbol()])
        chiral_tag.append(chirality[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())
        
        atom_features.extend([atom.GetAtomicNum(), 1 if atom.GetIsAromatic() else 0])
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2]))#6
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))#8
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))#4
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                              int(ring.IsAtomInRingOfSize(i, 4)),
                              int(ring.IsAtomInRingOfSize(i, 5)),
                              int(ring.IsAtomInRingOfSize(i, 6)),
                              int(ring.IsAtomInRingOfSize(i, 7)),
                              int(ring.IsAtomInRingOfSize(i, 8))])#6
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3])) # 5

#     z = torch.tensor(atomic_number, dtype=torch.long)

#     row, col, edge_type = [], [], []
#     for bond in mol.GetBonds():
#         start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
#         row += [start, end]
#         col += [end, start]
#         edge_type += 2 * [bonds[bond.GetBondType()]]

#     edge_index = torch.tensor([row, col], dtype=torch.long)
#     edge_type = torch.tensor(edge_type, dtype=torch.long)
#     edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    x1 = F.one_hot(torch.tensor(atom_type_idx), num_classes=len(types)) # 5
    x2 = torch.tensor(atom_features).view(N, -1) # 39
    x3 = torch.tensor(chiral_tag).view(N, -1).to(torch.float) # 1
    node_features = torch.cat([x1.to(torch.float), x3, x2], dim=-1)
    
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    # import ipdb; ipdb.set_trace()
#     use_rdkit_coords = True
    if use_rdkit_coords:
        if old_rdkit:
            rdkit_coords = get_rdkit_coords_old(mol, seed)
        else:
            rdkit_coords = get_rdkit_coords(mol, seed, use_mmff) #.numpy()
        if rdkit_coords is None:
            return None
        R, t = rigid_transform_Kabsch_3D(rdkit_coords.T, true_lig_coords.T)
        lig_coords = ((R @ (rdkit_coords).T).T + t.squeeze())
        #print('kabsch RMSD between rdkit ligand and true ligand is ', np.sqrt(np.sum((rdkit_coords - true_lig_coords) ** 2, axis=1).mean()).item())
        #print('kabsch RMSD between aligned rdkit ligand and true ligand is ', np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())
#         lig_coords = align(torch.from_numpy(rdkit_coords), torch.from_numpy(true_lig_coords)).numpy()
#         # print('LOSS kabsch RMSD between rdkit ligand and true ligand is ', np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())
        loss = torch.nn.MSELoss()
        loss_error = loss(torch.from_numpy(true_lig_coords), torch.from_numpy(lig_coords)).cpu().detach().numpy().item()
        # print('LOSS kabsch MSE between rdkit ligand and true ligand is ', loss_error )
    else:
        lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    remove_centroid = True
    # print("Lower Level Featurize", use_rdkit_coords, lig_coords)
    if remove_centroid:
        lig_coords -= np.mean(lig_coords, axis = 0)
        true_lig_coords -= np.mean(true_lig_coords, axis = 0)
    if use_rdkit_coords:
        loss = torch.nn.MSELoss()
        loss_error = loss(torch.from_numpy(true_lig_coords), torch.from_numpy(lig_coords)).cpu().detach().numpy().item()
        # print('[No MEAN] LOSS kabsch MSE between rdkit ligand and true ligand is ', loss_error )
        
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)
    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    bond_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            #print( f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
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
#         bonds = []
#         for d in dst:
#             bonds.append(get_bond(mol, int(i), int(d)))
#         bond_list.extend(bonds)
        bond_list.extend([get_bond_idx(mol, int(i), int(d)) for d in dst])
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['feat'] = node_features #lig_atom_featurizer(mol)
    # graph.ndata['mol_feat'] = node_features
    # if use_rdkit_coords:
    graph.ndata['ref_feat'] = node_features
#     graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
#     graph.edata['bond_type'] = torch.from_numpy(np.asarray(bond_list).astype(np.float32))
    edge_type = torch.from_numpy(np.asarray(bond_list).astype(np.float32)).type(torch.long) #torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)+1).to(torch.float)
    graph.edata['feat'] = torch.cat((distance_featurizer(dist_list, 0.75), edge_attr), dim = -1)
#     graph.edata['feat'] = torch.cat((distance_featurizer(dist_list, 0.75),torch.from_numpy(np.asarray(bond_list).astype(np.float32)).view(-1,1)), dim = -1)
    graph.ndata['x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    graph.ndata['x_ref'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    graph.ndata['x_true'] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    # if use_rdkit_coords:
    #     graph.ndata['rdkit_loss'] = torch.from_numpy(np.array([loss_error]*node_features.shape[0]).astype(np.float32)).reshape(-1,1)
    return graph

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())
        
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
        
    return G

def get_transformation_mask(mol, pyg_data = None):
#     G = to_networkx(pyg_data, to_undirected=False)
    G = mol_to_nx(mol)
    to_rotate = []
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edges = edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate

class ConformerDataset(DGLDataset):
    def __init__(self, root, split_path, mode, types, dataset, num_workers=1, limit_molecules=None,
                 cache_path=None, pickle_dir=None, boltzmann_resampler=None, raw_dir='/data/QM9/dgl', save_dir='/data/QM9/dgl',
                 force_reload=False, verbose=False, transform=None, name = "qm9",
                 invariant_latent_dim = 64, equivariant_latent_dim = 32, use_diffusion_angle_def = False, old_rdkit = False):
        # part of the featurisation and filtering code taken from GeoMol https://github.com/PattanaikL/GeoMol
#         super(ConformerDataset, self).__init__(name, transform) /data
        self.D = invariant_latent_dim
        self.F = equivariant_latent_dim
        self.root = root
        self.types = types
        self.failures = defaultdict(int)
        self.dataset = dataset
        self.boltzmann_resampler = boltzmann_resampler
        self.cache_path = cache_path
        self.use_diffusion_angle_def = use_diffusion_angle_def
        # print("Cache", cache_path)
        # if cache_path: cache_path += "." + mode
        self.use_name = name
        self.split_path = split_path
        self.mode = mode
        self.pickle_dir = pickle_dir
        self.num_workers = num_workers
        self.limit_molecules = limit_molecules
        self.old_rdkit = old_rdkit
        super(ConformerDataset, self).__init__(name, raw_dir = raw_dir, save_dir = save_dir, transform = transform)
        
    def process(self):
        if self.cache_path and os.path.exists(self.cache_path):
            print('Reusing preprocessing from cache', self.cache_path)
            with open(self.cache_path, "rb") as f:
                self.datapoints = pickle.load(f)
        else:
            print("Preprocessing")
            if self.dataset == 'qm9':
                 self.datapoints = self.preprocess_datapoints(self.root, self.split_path, self.pickle_dir, self.mode, self.num_workers, self.limit_molecules)
            else:
                self.datapoints = self.preprocess_datapoints_chunk(self.root, self.split_path, self.pickle_dir, self.mode, self.num_workers, self.limit_molecules)
            if self.cache_path:
                print("Caching at", self.cache_path)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.datapoints, f)
            
    def preprocess_datapoints_chunk(self, root, split_path, pickle_dir, mode, num_workers, limit_molecules):
        mols_per_pickle = 1000
        split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
        if limit_molecules:
            split = split[:limit_molecules]
        smiles = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))
        smiles = smiles[split]
        self.open_pickles = {}
        smiles = [smi[len(root):-7] for smi in smiles]
        print('Preparing to process', len(smiles), 'smiles')
        chunk_size = len(smiles)//5
        all_smiles = smiles
        smiles = []
        old_name = self.use_name
        total_count = 0
        for i in range(6):
            smiles = all_smiles[i*chunk_size : (i+1)*chunk_size]
            datapoints = []
            if num_workers > 1:
                results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    with tqdm(total=len(smiles)) as pbar:
                        futures = [executor.submit(self.filter_smiles_mp, entry) for entry in smiles]
                        for future in concurrent.futures.as_completed(futures):
                            pbar.update(1)
                    molecules = [future.result() for future in concurrent.futures.as_completed(futures)]
                    datapoints = [item for sublist in molecules for item in sublist if sublist is not None and sublist[0] is not None]
                    
            print('Fetched', len(datapoints), 'mols successfully')
            total_count += len(datapoints)
            print('Fetched Total', total_count, 'mols successfully')
            print(self.failures)
            if pickle_dir: del self.current_pickle
            self.datapoints = datapoints
            self.use_name = old_name + f"_{i}"
            self.save()
        return datapoints
    
    def preprocess_datapoints(self, root, split_path, pickle_dir, mode, num_workers, limit_molecules):
        mols_per_pickle = 1000
        split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
        if limit_molecules:
            split = split[:limit_molecules]
        smiles = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))
        smiles = smiles[split]

        self.open_pickles = {}
        if False and pickle_dir:
            smiles = [(i // mols_per_pickle, smi[len(root):-7]) for i, smi in zip(split, smiles)]
            if limit_molecules:
                smiles = smiles[:limit_molecules]
            self.current_pickle = (None, None)
            self.pickle_dir = pickle_dir
        else:
            smiles = [smi[len(root):-7] for smi in smiles]

        print('Preparing to process', len(smiles), 'smiles')
        datapoints = []
        if num_workers > 1:
            results = []
            # with mp.Pool(num_workers) as pool:
            #     # results = list(tqdm(pool.imap_unordered(self.filter_smiles_mp, smiles), total=len(smiles)))
            #     results = pool.map_async(self.filter_smiles_mp, smiles)
            #     pool.close()
            #     # wait for all tasks to complete and processes to close
            #     pool.join()
            # datapoints = [item for sublist in results for item in sublist if sublist[0] is not None] 
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Create a tqdm progress bar for the list of SMILES strings
                with tqdm(total=len(smiles)) as pbar:
                    # Submit the jobs to the thread pool
                    futures = [executor.submit(self.filter_smiles_mp, entry) for entry in smiles]

                    # Iterate over the futures and update the progress bar
                    for future in concurrent.futures.as_completed(futures):
                        pbar.update(1)

                # Wait for the jobs to complete and extract the resulting molecules
                molecules = [future.result() for future in concurrent.futures.as_completed(futures)]
                datapoints = [item for sublist in molecules for item in sublist if sublist is not None and sublist[0] is not None]
                

        else:
            # if num_workers > 1:
            #     p = Pool(num_workers)
            #     p.__enter__()
            count = 0
            with tqdm_outer.tqdm(total=len(smiles)) as pbar:
                map_fn = p.imap if num_workers > 1 else map
                for t in map_fn(self.filter_smiles_mp, smiles):
                    if t and t[0] is not None:
    #                     datapoints.append(t)
                        datapoints.extend(t)
                        count += 1
                        if count > 0 and count % 10000 == 0:
                            print("Saving...", count)
                            self.datapoints = datapoints
                            self.save()
                    pbar.update()
            if num_workers > 1: p.__exit__(None, None, None)
        print('Fetched', len(datapoints), 'mols successfully')
        print(self.failures)
        if pickle_dir: del self.current_pickle
        return datapoints
        
    def filter_smiles_mp(self, smile):
        if type(smile) is tuple:
            pickle_id, smile = smile
            current_id, current_pickle = self.current_pickle
            if current_id != pickle_id:
                path = osp.join(self.pickle_dir, str(pickle_id).zfill(3) + '.pickle')
                if not osp.exists(path):
                    self.failures[f'std_pickle{pickle_id}_not_found'] += 1
                    # print("A")
                    return [None]
                with open(path, 'rb') as f:
                    self.current_pickle = current_id, current_pickle = pickle_id, pickle.load(f)
            if smile not in current_pickle:
                self.failures['smile_not_in_std_pickle'] += 1
                # print("B")
                return [None]
            mol_dic = current_pickle[smile]

        else:
            if not os.path.exists(os.path.join(self.root, smile + '.pickle')):
                self.failures['raw_pickle_not_found'] += 1
                # print("C")
                return [None]
            pickle_file = osp.join(self.root, smile + '.pickle')
            mol_dic = self.open_pickle(pickle_file)

        smile = mol_dic['smiles']

        if '.' in smile:
            self.failures['dot_in_smile'] += 1
            # print("D")
            return [None]

        # filter mols rdkit can't intrinsically handle
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            self.failures['mol_from_smiles_failed'] += 1
            # print("E")
            return [None]

        mol = mol_dic['conformers'][0]['rd_mol']
        
        # xc = mol.GetConformer().GetPositions()
        # print("filter mol POS", mol.GetConformer().GetPositions())
        N = mol.GetNumAtoms()
        if not mol.HasSubstructMatch(dihedral_pattern):
            self.failures['no_substruct_match'] += 1
            # print("F")
            return [None]

        if N < 4:
            self.failures['mol_too_small'] += 1
            # print("G")
            return [None]
        # print("A filter")
        datas = self.featurize_mol(mol_dic, old_rdkit = self.old_rdkit)
        if not datas or len(datas) == 0:
            self.failures['featurize_mol_failed'] += 1
            # print("H")
            return [None]
        results_A = []
        results_B = []
        bad_idx_A, bad_idx_B = [], []
        for idx, data in enumerate(datas):
            if not data:
                self.failures['featurize_mol_failed_A'] += 1
                bad_idx_A.append(idx)
                results_A.append(None)
                continue
            mol = mol_dic['conformers'][idx]['rd_mol']
            edge_mask, mask_rotate = get_transformation_mask(mol, data)
            if np.sum(edge_mask) < 0.5: #TODO: Do we need this since we are using GEOMOL
                self.failures['no_rotable_bonds'] += 1
                bad_idx_A.append(idx)
                results_A.append(None)
                continue
            # print("filter SMILE", smile)
            try:
                A_frags, A_frag_ids, A_adj, A_out, A_bond_break, A_cg_bonds, A_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = self.use_diffusion_angle_def)
                A_cg = conditional_coarsen_3d(data, A_frag_ids, A_cg_map, A_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
            except:
                self.failures['coarsening error'] += 1
                bad_idx_A.append(idx)
                results_A.append(None)
                continue
            # xcc = mol.GetConformer().GetPositions()
            # print("filter mol POS2", xcc == xc)
            geometry_graph_A = get_geometry_graph(mol)
            err = check_distances(data, geometry_graph_A)
            assert(err < 1e-3)
            # if err.item() > 1e-3:
            #     import ipdb; ipdb.set_trace()
            #     data = self.featurize_mol(mol_dic)
            Ap = create_pooling_graph(data, A_frag_ids)
            geometry_graph_A_cg = get_coarse_geometry_graph(A_cg, A_cg_map)
            # print(A_frag_ids, A_cg_bonds, A_cg_map)
            results_A.append((data, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
            # import ipdb; ipdb.set_trace()
            # print("\nB featurizing")
            # rdmol_dic = copy.deepcopy(mol_dic)
        data_Bs = self.featurize_mol(mol_dic, use_rdkit_coords = True, old_rdkit = self.old_rdkit)
        for idx, data_B in enumerate(data_Bs):
            if idx in set(bad_idx_A):
                bad_idx_B.append(idx)
                results_B.append(None)
                continue
              
            if not data_B:
                self.failures['featurize_mol_failed_B'] += 1
                bad_idx_B.append(idx)
                # print("BAD B", idx, len(data_Bs))
                results_B.append(None)
                continue
                # return [None]
            mol = mol_dic['conformers'][idx]['rd_mol']
            B_frags, B_frag_ids, B_adj, B_out, B_bond_break, B_cg_bonds, B_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = self.use_diffusion_angle_def)
            B_cg = conditional_coarsen_3d(data_B, B_frag_ids, B_cg_map, B_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
#             geometry_graph_B = copy.deepcopy(geometry_graph_A) #get_geometry_graph(mol)
            geometry_graph_B = get_geometry_graph(mol)
            Bp = create_pooling_graph(data_B, B_frag_ids)
            geometry_graph_B_cg = get_coarse_geometry_graph(B_cg, B_cg_map)
            err = check_distances(data_B, geometry_graph_B, True)
            assert(err < 1e-3)
            # if err.item() > 1e-3:
            #     import ipdb; ipdb.set_trace()
            #     data_B = self.featurize_mol(mol_dic, use_rdkit_coords = True)
    #         data.edge_mask = torch.tensor(edge_mask)
    #         data.mask_rotate = mask_rotate
#             return ((data, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids), (data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg))
            results_B.append((data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
        assert(len(results_A) == len(results_B))
        bad_idx = set(bad_idx_A) | set(bad_idx_B)
        results_A = [x for idx, x in enumerate(results_A) if idx not in bad_idx]
        results_B = [x for idx, x in enumerate(results_B) if idx not in bad_idx]
        assert(len(results_A) == len(results_B))
        if len(results_A) == 0 or len(results_B) == 0:
            # print("Bad Input")
            # print("I")
            return [None]
        # print(smile, len(results_A))
        return [(a,b) for a,b in zip(results_A, results_B)]

    def len(self):
        return len(self.datapoints)

    def get(self, idx):
        data = self.datapoints[idx]
        return copy.deepcopy(data)
    
    def __getitem__(self, idx):
        r"""Gets the data object at index.
        """
        data = self.datapoints[idx]
        return copy.deepcopy(data)

    def __len__(self):
        r"""The number of examples in the dataset."""
        return len(self.datapoints)
    
    def save(self):
        # return True
        graphs, infos = [], []
        for A, B in self.datapoints:
            data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids = A
            data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids = B
            infos.append((A_frag_ids, B_frag_ids))
            graphs.extend([data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg])
        dgl.data.utils.save_info(self.save_dir + f'/{self.use_name}_infos.bin', infos)
        dgl.data.utils.save_graphs(self.save_dir + f'/{self.use_name}_graphs.bin', graphs)
        print("Saved Successfully", self.save_dir, self.use_name, len(self.datapoints))
    
    def load(self):
        if self.dataset == "qm9" or self.dataset == 'xl':
            graphs, _ = dgl.data.utils.load_graphs(self.save_dir + f'/{self.use_name}_graphs.bin')
            info = dgl.data.utils.load_info(self.save_dir + f'/{self.use_name}_infos.bin')
            count = 0
            results_A, results_B = [], []
            for i in range(0, len(graphs), 10):
                AB = graphs[i: i+10]
                A_frag_ids, B_frag_ids = info[count]
                count += 1
                data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg = AB[:5]
                data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg = AB[5:]
                results_A.append((data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
                results_B.append((data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
            self.datapoints = [(a,b) for a,b in zip(results_A, results_B)]
            print("Loaded Successfully",  self.save_dir, self.use_name, len(self.datapoints))
        else:
            if False:
                try:
                    count = 0
                    results_A, results_B = [], []
                    self.datapoints = []
                    for chunk in range(6): #! operating on the small chunk for multi gpu debugging
                        if chunk > 7: 
                            print("Skipping large chunks for debugging")
                            break
                        print(f"Loading Chunk {chunk} ...")
                        graphs, _ = dgl.data.utils.load_graphs(self.save_dir + f'/{self.use_name}_{chunk}_graphs.bin')
                        info = dgl.data.utils.load_info(self.save_dir + f'/{self.use_name}_{chunk}_infos.bin')
                        count = 0
                        print(f"Loading Chunk {chunk} = {len(graphs)//10}")
                        results_A, results_B = [], []
                        cur, mark = None, 0
                        
                        book = defaultdict(int)
                        for i in range(0, len(graphs), 10):
                            AB = graphs[i: i+10]
                            A_frag_ids, B_frag_ids = info[count]
                            count += 1
                            data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg = AB[:5]
                            data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg = AB[5:]
                            guess = (data_A.ndata['x'].shape[0], A_cg.ndata['x'].shape[0], tuple(tuple(s) for s in A_frag_ids))
                            if book[guess] < 2:
                                book[guess] += 1
                            else:
                                continue
                            # if cur == None:
                            #     cur = guess
                            #     mark =+ 1
                            # elif mark >= 5:
                            #     continue
                            # elif mark >= 2 and cur == guess:
                            #     mark += 1
                            #     continue
                            # elif mark < 2 and cur == guess:
                            #     mark += 1
                            # elif cur !=  guess:
                            #     mark = 1
                            #     cur = guess
                            results_A.append((data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
                            results_B.append((data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
                        # import ipdb; ipdb.set_trace()
                        self.datapoints.extend([(a,b) for a,b in zip(results_A, results_B)])
                    print("Loaded Successfully",  self.save_dir, self.use_name, len(self.datapoints))
                except:
                    import ipdb; ipdb.set_trace()
            else:
                try:
                    count = 0
                    results_A, results_B = [], []
                    self.datapoints = []
                    for chunk in range(6): #! operating on the small chunk for multi gpu debugging
                        if chunk > 7: 
                            print("Skipping large chunks for debugging")
                            break
                        print(f"Loading Chunk {chunk} ...")
                        graphs, _ = dgl.data.utils.load_graphs(self.save_dir + f'/{self.use_name}_{chunk}_graphs.bin')
                        info = dgl.data.utils.load_info(self.save_dir + f'/{self.use_name}_{chunk}_infos.bin')
                        count = 0
                        print(f"Loading Chunk {chunk} = {len(graphs)//10}")
                        results_A, results_B = [], []
                        for i in range(0, len(graphs), 10):
                            AB = graphs[i: i+10]
                            A_frag_ids, B_frag_ids = info[count]
                            count += 1
                            data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg = AB[:5]
                            data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg = AB[5:]
                            results_A.append((data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
                            results_B.append((data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
                        self.datapoints.extend([(a,b) for a,b in zip(results_A, results_B)])
                    print("Loaded Successfully",  self.save_dir, self.use_name, len(self.datapoints))
                except:
                    import ipdb; ipdb.set_trace()
                
    
    def has_cache(self):
        if self.dataset == "qm9":
            return os.path.exists(os.path.join(self.save_dir, f'{self.use_name}_graphs.bin'))
        else:
            return os.path.exists(os.path.join(self.save_dir, f'{self.use_name}_0_graphs.bin'))

    def __repr__(self):
        return f'Dataset("{self.name}", num_graphs={len(self)},' + \
               f' save_path={self.save_path})'

    def open_pickle(self, mol_path):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    def featurize_mol(self, mol_dic, use_rdkit_coords = False, limit = 5, old_rdkit = False):
        confs = mol_dic['conformers']
        name = mol_dic["smiles"]

        mol_ = Chem.MolFromSmiles(name)
        canonical_smi = Chem.MolToSmiles(mol_, isomericSmiles=False)

        pos = []
        # weights = []
        datas = []
        for idx, conf in enumerate(confs):
            if limit > 0 and idx >= limit:
                break
            mol = conf['rd_mol']

            # filter for conformers that may have reacted
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
            except Exception as e:
                print(e)
                continue

            if conf_canonical_smi != canonical_smi:
                datas.append(None)
                continue

            pos.append(torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float))
            # weights.append(conf['boltzmannweight'])
            # # ! Is this the issue I am seeing in the qm9 data?
            # # TODO: is this it doggo?
            # if pos[-1].shape[0] != Chem.AddHs(mol).GetNumAtoms():
            #     import ipdb; idpb.set_trace
            #     datas.append(None)
            #     continue
            
            correct_mol = mol
            mol_features = featurize_mol(correct_mol, self.types, use_rdkit_coords = use_rdkit_coords, old_rdkit=old_rdkit)
            datas.append(mol_features)
            # if self.boltzmann_resampler is not None:
            #     # torsional Boltzmann generator uses only the local structure of the first conformer
            #     break
#             if True:
#                 break #! only look at first of the dataset for now since we only need 1. Causes distance issues otherwise

        # return None if no non-reactive conformers were found
        if len(pos) == 0:
            return None
        # import ipdb; ipdb.set_trace()
        # print("\n Mid Level POS \n", pos)
        # print()
        
#         THIS IS GETTING PASSED IN -1 INDEX OF CONFS POSITION
#         data = featurize_mol(correct_mol, self.types, use_rdkit_coords = use_rdkit_coords)
        # dos = Draw.MolDrawOptions()
        # dos.addAtomIndices=True
        # img = Draw.MolToImage(correct_mol, options=dos)
        # display(img)
        #! TODO figure out how to handle boltzman weights
#         normalized_weights = list(np.array(weights) / np.sum(weights))
#         if np.isnan(normalized_weights).sum() != 0:
#             print(name, len(confs), len(pos), weights)
#             normalized_weights = [1 / len(weights)] * len(weights)
#         data.canonical_smi, data.mol, data.pos, data.weights = canonical_smi, correct_mol, pos, normalized_weights
        return datas

    def resample_all(self, resampler, temperature=None):
        ess = []
        for data in tqdm.tqdm(self.datapoints):
            ess.append(resampler.resample(data, temperature=temperature))
        return ess

def collate(samples):
    A, B = map(list, zip(*samples))
#     data, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg
#  .to('cuda:0') causes errors
    A_graph = dgl.batch([x[0] for x in A])
    geo_A = dgl.batch([x[1] for x in A])
    Ap = dgl.batch([x[2] for x in A])
    A_cg = dgl.batch([x[3] for x in A])
    geo_A_cg = dgl.batch([x[4] for x in A])
    frag_ids = [x[5] for x in A]
    
    B_graph = dgl.batch([x[0] for x in B])
    geo_B = dgl.batch([x[1] for x in B])
    Bp = dgl.batch([x[2] for x in B])
    B_cg = dgl.batch([x[3] for x in B])
    geo_B_cg = dgl.batch([x[4] for x in B])
    B_frag_ids = [x[5] for x in B]
    return (A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids), (B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids)

def load_torsional_data(batch_size = 32, mode = 'train', data_dir='/data/QM9/qm9/',
                dataset='qm9', limit_mols=0, log_dir='./test_run', num_workers=1, restart_dir=None, seed=0,
                 split_path='/data/QM9/split.npy',
                 std_pickles=None):
# def load_torsional_data(batch_size = 32, mode = 'train', data_dir='/data/QM9/qm9/',
#                 dataset='qm9', limit_mols=0, log_dir='./test_run', num_workers=1, restart_dir=None, seed=0,
#                  split_path='/data/QM9/split.npy',
#                  std_pickles=None): #   std_pickles='/data/QM9/standardized_pickles'):
    types = qm9_types if dataset == 'qm9' else drugs_types
    use_diffusion_angle_def = False
    data = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
                                   types=types, transform=None,
                                   num_workers=num_workers,
                                   limit_molecules=limit_mols, #args.limit_train_mols,
                                   cache_path=None, #args.cache,
                                   name=f'{dataset}_{mode}_{limit_mols}_final',
                                   pickle_dir=std_pickles,
                                   use_diffusion_angle_def=use_diffusion_angle_def,
                                   boltzmann_resampler=None)
    
    
    # data2 = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
    #                                types=types, transform=None,
    #                                num_workers=num_workers,
    #                                limit_molecules=limit_mols, #args.limit_train_mols,
    #                                cache_path=None, #args.cache,
    #                                name=f'{dataset}_{mode}_{limit_mols}_good',
    #                                pickle_dir=std_pickles,
    #                                use_diffusion_angle_def=use_diffusion_angle_def,
    #                                boltzmann_resampler=None,
    #                                old_rdkit = True)
    # import ipdb; ipdb.set_trace()
    dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers,
                                            collate_fn = collate)
    return dataloader, data

def load_torsional_data_local(batch_size = 32, mode = 'train', data_dir='/data/QM9/qm9/',
                dataset='qm9', limit_mols=0, log_dir='./test_run', num_workers=1, restart_dir=None, seed=0,
                 split_path='/data/QM9/split.npy',
                 std_pickles=None): #   std_pickles='/data/QM9/standardized_pickles'):
    types = qm9_types if dataset == 'qm9' else drugs_types
    use_diffusion_angle_def = False
    data = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
                                   types=types, transform=None,
                                   num_workers=num_workers,
                                   limit_molecules=limit_mols, #args.limit_train_mols,
                                   cache_path=None, #args.cache,
                                   name=f'{dataset}_{mode}_{limit_mols}_final',
                                   pickle_dir=std_pickles,
                                   use_diffusion_angle_def=use_diffusion_angle_def,
                                   raw_dir='/data/QM9/dgl', 
                                   save_dir='/data/QM9/dgl',
                                   boltzmann_resampler=None)
    
    dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers,
                                            collate_fn = collate)
    return dataloader, data



def load_drugs_conformers(batch_size = 32, mode = 'train', data_dir='/data/DRUGS/drugs/',
                dataset='drugs', limit_mols=0, log_dir='./test_run', num_workers=1, restart_dir=None, seed=0,
                 split_path='/data/DRUGS/split.npy',
                 std_pickles=None):
# def load_torsional_data(batch_size = 32, mode = 'train', data_dir='/data/QM9/qm9/',
#                 dataset='qm9', limit_mols=0, log_dir='./test_run', num_workers=1, restart_dir=None, seed=0,
#                  split_path='/data/QM9/split.npy',
#                  std_pickles=None): #   std_pickles='/data/QM9/standardized_pickles'):
    types = drugs_types
    use_diffusion_angle_def = False
    data = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
                                   types=types, transform=None,
                                   num_workers=num_workers,
                                   limit_molecules=limit_mols, #args.limit_train_mols,
                                   cache_path=None, #args.cache,
                                   name=f'{dataset}_{mode}_{limit_mols}_final',
                                   pickle_dir=std_pickles,
                                   raw_dir='/data/DRUGS/dgl', 
                                   save_dir='/data/DRUGS/dgl',
                                   use_diffusion_angle_def=use_diffusion_angle_def,
                                   boltzmann_resampler=None)
    dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers,
                                            collate_fn = collate)
    return dataloader, data


def preprocess_drugs_local(batch_size = 32, mode = 'train', data_dir='/data/DRUGS/drugs/',
                dataset='drugs', limit_mols=0, log_dir='./test_run', num_workers=1, restart_dir=None, seed=0,
                 split_path='/data/DRUGS/split.npy',
                 std_pickles=None): #   std_pickles='/data/QM9/standardized_pickles'):
    types = qm9_types if dataset == 'qm9' else drugs_types
    use_diffusion_angle_def = False
    data = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
                                   types=types, transform=None,
                                   num_workers=num_workers,
                                   limit_molecules=limit_mols, #args.limit_train_mols,
                                   cache_path=None, #args.cache,
                                   name=f'{dataset}_{mode}_{limit_mols}_final',
                                   pickle_dir=std_pickles,
                                   use_diffusion_angle_def=use_diffusion_angle_def,
                                   raw_dir='/data/DRUGS/dgl', 
                                   save_dir='/data/DRUGS/dgl',
                                   boltzmann_resampler=None)
    
    dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers,
                                            collate_fn = collate)
    return dataloader, data


def preprocess_drugs_local_fast(batch_size = 32, mode = 'train', data_dir='/data/DRUGS/drugs/',
                dataset='drugs', limit_mols=0, log_dir='./test_run', num_workers=10, restart_dir=None, seed=0,
                 split_path='/data/DRUGS/split.npy',
                 std_pickles=None): #   std_pickles='/data/QM9/standardized_pickles'):
    types = qm9_types if dataset == 'qm9' else drugs_types
    use_diffusion_angle_def = False
    data = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
                                   types=types, transform=None,
                                   num_workers=num_workers,
                                   limit_molecules=limit_mols, #args.limit_train_mols,
                                   cache_path=None, #args.cache,
                                   name=f'{dataset}_{mode}_{limit_mols}_final_fast',
                                   pickle_dir=std_pickles,
                                   use_diffusion_angle_def=use_diffusion_angle_def,
                                   raw_dir='/data/DRUGS/dgl', 
                                   save_dir='/data/DRUGS/dgl',
                                   boltzmann_resampler=None)
    
    dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers,
                                            collate_fn = collate)
    return dataloader, data


def load_big_drugs(batch_size = 32, mode = 'train', data_dir='/data/DRUGS/drugs/',
                dataset='drugs', limit_mols=0, log_dir='./test_run', num_workers=1, restart_dir=None, seed=0,
                 split_path='/data/DRUGS/split.npy',
                 std_pickles=None):
    types = qm9_types if dataset == 'qm9' else drugs_types
    use_diffusion_angle_def = False
    data = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
                                   types=types, transform=None,
                                   num_workers=num_workers,
                                   limit_molecules=limit_mols, #args.limit_train_mols,
                                   cache_path=None, #args.cache,
                                   name=f'drugs_xl_{mode}_0_final',
                                   pickle_dir=std_pickles,
                                   raw_dir='/data/DRUGS/dgl', 
                                   save_dir='/data/DRUGS/dgl',
                                   use_diffusion_angle_def=use_diffusion_angle_def,
                                   boltzmann_resampler=None)
    
    dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers,
                                            collate_fn = collate)
    return dataloader, data
