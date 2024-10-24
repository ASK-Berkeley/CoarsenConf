import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
import networkx as nx

import torch, tqdm
import torch.nn.functional as F
from multiprocessing import Pool

import glob, pickle, random
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

def check_distances(molecule, geometry_graph):
    src, dst = geometry_graph.edges()
    src = src.long()
    dst = dst.long()
    generated_coords = molecule.ndata['x_true']
    d_squared = torch.sum((generated_coords[src] - generated_coords[dst]) ** 2, dim=1)
    error = torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2)
    print("Distance Loss Check", error)
    print(geometry_graph.edata['feat'])
    print(np.linalg.norm(((generated_coords[src] - generated_coords[dst]).numpy()), axis = 1))
    print(np.linalg.norm(((molecule.ndata['x'][src] - molecule.ndata['x'][dst]).numpy()), axis = 1))
    # ! This is not returning 0 why?
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
def featurize_mol(mol, types=drugs_types, use_rdkit_coords = False, seed = 0, radius = 4, max_neighbors=None):
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
#     use_rdkit_coords = True
    if use_rdkit_coords:
        rdkit_coords = get_rdkit_coords(mol, seed) #.numpy()
        if rdkit_coords is None:
            return None
        R, t = rigid_transform_Kabsch_3D(rdkit_coords.T, true_lig_coords.T)
        lig_coords = ((R @ (rdkit_coords).T).T + t.squeeze())
        print('kabsch RMSD between rdkit ligand and true ligand is ', np.sqrt(np.sum((rdkit_coords - true_lig_coords) ** 2, axis=1).mean()).item())
        print('kabsch RMSD between aligned rdkit ligand and true ligand is ', np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())
#         lig_coords = align(torch.from_numpy(rdkit_coords), torch.from_numpy(true_lig_coords)).numpy()
#         # print('LOSS kabsch RMSD between rdkit ligand and true ligand is ', np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())
        loss = torch.nn.MSELoss()
        loss_error = loss(torch.from_numpy(true_lig_coords), torch.from_numpy(lig_coords)).cpu().detach().numpy().item()
        print('LOSS kabsch MSE between rdkit ligand and true ligand is ', loss_error )
    else:
        lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    remove_centroid = True
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
            print(
                f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
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

    # class ConformerDataset(Dataset):
#     def __init__(self, root, split_path, mode, types, dataset, transform=None, num_workers=1, limit_molecules=None,
#                  cache=None, pickle_dir=None, boltzmann_resampler=None):
#         # part of the featurisation and filtering code taken from GeoMol https://github.com/PattanaikL/GeoMol

class ConformerDataset(DGLDataset):
    def __init__(self, root, split_path, mode, types, dataset, num_workers=1, limit_molecules=None,
                 cache_path=None, pickle_dir=None, boltzmann_resampler=None, raw_dir=None, save_dir=None,
                 force_reload=False, verbose=False, transform=None, name = "TorsionalDiffusion",
                 invariant_latent_dim = 64, equivariant_latent_dim = 32, use_diffusion = False):
        # part of the featurisation and filtering code taken from GeoMol https://github.com/PattanaikL/GeoMol
#         super(ConformerDataset, self).__init__(name, transform)
        self.D = invariant_latent_dim
        self.F = equivariant_latent_dim
        self.root = root
        self.types = types
        self.failures = defaultdict(int)
        self.dataset = dataset
        self.boltzmann_resampler = boltzmann_resampler
        self.cache_path = cache_path
        self.use_diffusion = use_diffusion
        print("Cache", cache_path)
        if cache_path: cache_path += "." + mode
        
#         if cache and os.path.exists(cache):
#             print('Reusing preprocessing from cache', cache)
#             with open(cache, "rb") as f:
#                 self.datapoints = pickle.load(f)
#         else:
#             print("Preprocessing")
#             self.datapoints = self.preprocess_datapoints(root, split_path, pickle_dir, mode, num_workers, limit_molecules)
#             if cache:
#                 print("Caching at", cache)
#                 with open(cache, "wb") as f:
#                     pickle.dump(self.datapoints, f)

#         if limit_molecules:
#             self.datapoints = self.datapoints[:limit_molecules]
        self.split_path = split_path
        self.mode = mode
        self.pickle_dir = pickle_dir
        self.num_workers = num_workers
        self.limit_molecules = limit_molecules
        super(ConformerDataset, self).__init__(name, transform)

    def process(self):
        if self.cache_path and os.path.exists(self.cache_path):
            print('Reusing preprocessing from cache', self.cache_path)
            with open(self.cache_path, "rb") as f:
                self.datapoints = pickle.load(f)
        else:
            print("Preprocessing")
            self.datapoints = self.preprocess_datapoints(self.root, self.split_path, self.pickle_dir, self.mode, self.num_workers, self.limit_molecules)
            if self.cache_path:
                print("Caching at", self.cache_path)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.datapoints, f)

        if self.limit_molecules:
            self.datapoints = self.datapoints[:self.limit_molecules]
            
    def preprocess_datapoints(self, root, split_path, pickle_dir, mode, num_workers, limit_molecules):
        mols_per_pickle = 1000
        split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
        if limit_molecules:
            split = split[:limit_molecules]
        smiles = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))
        smiles = smiles[split]

        self.open_pickles = {}
        if pickle_dir:
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
            p = Pool(num_workers)
            p.__enter__()
        with tqdm.tqdm(total=len(smiles)) as pbar:
            map_fn = p.imap if num_workers > 1 else map
            for t in map_fn(self.filter_smiles, smiles):
                if t:
                    datapoints.append(t)
                pbar.update()
        if num_workers > 1: p.__exit__(None, None, None)
        print('Fetched', len(datapoints), 'mols successfully')
        print(self.failures)
        if pickle_dir: del self.current_pickle
        return datapoints

    def filter_smiles(self, smile):
        if type(smile) is tuple:
            pickle_id, smile = smile
            current_id, current_pickle = self.current_pickle
            if current_id != pickle_id:
                path = osp.join(self.pickle_dir, str(pickle_id).zfill(3) + '.pickle')
                if not osp.exists(path):
                    self.failures[f'std_pickle{pickle_id}_not_found'] += 1
                    return False
                with open(path, 'rb') as f:
                    self.current_pickle = current_id, current_pickle = pickle_id, pickle.load(f)
            if smile not in current_pickle:
                self.failures['smile_not_in_std_pickle'] += 1
                return False
            mol_dic = current_pickle[smile]

        else:
            if not os.path.exists(os.path.join(self.root, smile + '.pickle')):
                self.failures['raw_pickle_not_found'] += 1
                return False
            pickle_file = osp.join(self.root, smile + '.pickle')
            mol_dic = self.open_pickle(pickle_file)

        smile = mol_dic['smiles']

        if '.' in smile:
            self.failures['dot_in_smile'] += 1
            return False

        # filter mols rdkit can't intrinsically handle
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            self.failures['mol_from_smiles_failed'] += 1
            return False

        mol = mol_dic['conformers'][0]['rd_mol']
        N = mol.GetNumAtoms()
        if not mol.HasSubstructMatch(dihedral_pattern):
            self.failures['no_substruct_match'] += 1
            return False

        if N < 4:
            self.failures['mol_too_small'] += 1
            return False
        print("A")
        data = self.featurize_mol(mol_dic)
        if not data:
            self.failures['featurize_mol_failed'] += 1
            return False

        edge_mask, mask_rotate = get_transformation_mask(mol, data)
        if np.sum(edge_mask) < 0.5:
            self.failures['no_rotable_bonds'] += 1
            return False
        print("SMILE", smile)
        A_frags, A_frag_ids, A_adj, A_out, A_bond_break, A_cg_bonds, A_cg_map = coarsen_molecule(mol, use_diffusion = self.use_diffusion)
        A_cg = conditional_coarsen_3d(data, A_frag_ids, A_cg_map, A_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
        geometry_graph_A = get_geometry_graph(mol)
        err = check_distances(data, geometry_graph_A)
        if err.item() > 1e-3:
            import ipdb; ipdb.set_trace()
            data = self.featurize_mol(mol_dic)
        Ap = create_pooling_graph(data, A_frag_ids)
        geometry_graph_A_cg = get_coarse_geometry_graph(A_cg, A_cg_map)
        print(A_frag_ids, A_cg_bonds, A_cg_map)
        # import ipdb; ipdb.set_trace()
        print("\nB")
        # rdmol_dic = copy.deepcopy(mol_dic)
        data_B = self.featurize_mol(mol_dic, use_rdkit_coords = True)
        if not data_B:
            self.failures['featurize_mol_failed_B'] += 1
            return False
        
        B_frags, B_frag_ids, B_adj, B_out, B_bond_break, B_cg_bonds, B_cg_map = coarsen_molecule(mol, use_diffusion = self.use_diffusion)
        B_cg = conditional_coarsen_3d(data_B, B_frag_ids, B_cg_map, B_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
        geometry_graph_B = copy.deepcopy(geometry_graph_A) #get_geometry_graph(mol)
        Bp = create_pooling_graph(data_B, B_frag_ids)
        geometry_graph_B_cg = get_coarse_geometry_graph(B_cg, B_cg_map)
        err = check_distances(data_B, geometry_graph_B)
        if err.item() > 1e-3:
            import ipdb; ipdb.set_trace()
            data_B = self.featurize_mol(mol_dic, use_rdkit_coords = True)
#         data.edge_mask = torch.tensor(edge_mask)
#         data.mask_rotate = mask_rotate
        return ((data, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids), (data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg))

    def len(self):
        return len(self.datapoints)

    def get(self, idx):
        data = self.datapoints[idx]
#         if self.boltzmann_resampler:
#             self.boltzmann_resampler.try_resample(data)
        return copy.deepcopy(data)
    
    def __getitem__(self, idx):
        r"""Gets the data object at index.
        """
        data = self.datapoints[idx]
        return copy.deepcopy(data)

    def __len__(self):
        r"""The number of examples in the dataset."""
        return len(self.datapoints)


    def __repr__(self):
        return f'Dataset("{self.name}", num_graphs={len(self)},' + \
               f' save_path={self.save_path})'

    def open_pickle(self, mol_path):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    def featurize_mol(self, mol_dic, use_rdkit_coords = False):
        confs = mol_dic['conformers']
        name = mol_dic["smiles"]

        mol_ = Chem.MolFromSmiles(name)
        canonical_smi = Chem.MolToSmiles(mol_, isomericSmiles=False)

        pos = []
        weights = []
        for conf in confs:
            mol = conf['rd_mol']

            # filter for conformers that may have reacted
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
            except Exception as e:
                print(e)
                continue

            if conf_canonical_smi != canonical_smi:
                continue

            pos.append(torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float))
            weights.append(conf['boltzmannweight'])
            correct_mol = mol

            if self.boltzmann_resampler is not None:
                # torsional Boltzmann generator uses only the local structure of the first conformer
                break
            if True:
                break #! only look at first of the dataset for now since we only need 1. Causes distance issues otherwise

        # return None if no non-reactive conformers were found
        if len(pos) == 0:
            return None
        # import ipdb; ipdb.set_trace()
        data = featurize_mol(correct_mol, self.types, use_rdkit_coords = use_rdkit_coords)
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

        return data

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
    return (A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids), (B_graph, geo_B, Bp, B_cg, geo_B_cg)

def load_torsional_data(batch_size = 32, mode = 'train', data_dir='/data/QM9/qm9/',
                dataset='qm9', limit_mols=0, log_dir='./test_run', num_workers=1, restart_dir=None, seed=0,
                 split_path='/data/QM9/split.npy',
                  std_pickles='/data/QM9/standardized_pickles'):
    types = qm9_types if dataset == 'qm9' else drugs_types
    use_diffusion = False
    data = ConformerDataset(data_dir, split_path, mode, dataset=dataset,
                                   types=types, transform=None,
                                   num_workers=num_workers,
                                   limit_molecules=limit_mols, #args.limit_train_mols,
                                   cache_path=None, #args.cache,
                                   pickle_dir=std_pickles,
                                   use_diffusion=use_diffusion,
                                   boltzmann_resampler=None)
    dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers,
                                            collate_fn = collate)
    return dataloader, data
