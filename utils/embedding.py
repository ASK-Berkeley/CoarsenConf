import math
import warnings

# import pandas as pd
import dgl
import numpy as np
import scipy.spatial as spa
import torch
# from Bio.PDB import get_surface, PDBParser, ShrakeRupley
# from Bio.PDB.PDBExceptions import PDBConstructionWarning
# from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import MolFromPDBFile, AllChem, GetPeriodicTable, rdDistGeom
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from scipy import spatial
from scipy.special import softmax
from torch import nn

from geometry_utils import rigid_transform_Kabsch_3D, rigid_transform_Kabsch_3D_torch

# biopython_parser = PDBParser()
# periodic_table = GetPeriodicTable()
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

A_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 1)  # number of scalar features

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims, use_scalar_feat=True, n_feats_to_use=None):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.use_scalar_feat = use_scalar_feat
        self.n_feats_to_use = n_feats_to_use
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1]
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
            if i + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())
            if i + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_features > 0 and self.use_scalar_feat:
            x_embedding += self.linear(x[:, self.num_categorical_features:])
        if torch.isnan(x_embedding).any():
            print('nan')
        return x_embedding

class AtomEncoderTorsionalDiffusion(torch.nn.Module):
    def __init__(self, emb_dim, feature_dim):
        super(AtomEncoderTorsionalDiffusion, self).__init__()
        self.node_embedding = nn.Sequential(
            nn.Linear(feature_dim, emb_dim),
            nn.LeakyReLU(negative_slope=1e-2),# nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain = 1.)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x_embedding = self.node_embedding(x)
        return x_embedding

def lig_atom_featurizer(mol):
    ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        g_charge = atom.GetDoubleProp('_GasteigerCharge')
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
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


# sr = ShrakeRupley(probe_radius=1.4,  # in A. Default is 1.40 roughly the radius of a water molecule.
#                   n_points=100)  # resolution of the surface of each atom. Default is 100. A higher number of points results in more precise measurements, but slows down the calculation.

def get_lig_graph(mol, lig_coords, radius=20, max_neighbor=None):
    ################### Build the k-NN graph ##############################
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
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbor + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            #print(                f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
        assert i not in dst
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
    graph.ndata['x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    return graph

def get_lig_structure_graph(lig):
    coords = lig.GetConformer().GetPositions()
    weights = []
    for idx, atom in enumerate(lig.GetAtoms()):
        weights.append(atom.GetAtomicNum())
    weights = np.array(weights)
    mask = []
    angles = []
    edges = []
    distances = []
    for bond in lig.GetBonds():
        type = bond.GetBondType()
        src_idx = bond.GetBeginAtomIdx()
        dst_idx = bond.GetEndAtomIdx()
        src = lig.GetAtomWithIdx(src_idx)
        dst = lig.GetAtomWithIdx(dst_idx)
        src_neighbors = [atom.GetIdx() for atom in list(src.GetNeighbors())]
        src_neighbors.remove(dst_idx)
        src_weights = weights[src_neighbors]
        dst_neighbors = [atom.GetIdx() for atom in list(dst.GetNeighbors())]
        dst_neighbors.remove(src_idx)
        dst_weights = weights[dst_neighbors]
        src_to_dst = (coords[dst_idx] - coords[src_idx])
        if not (len(src_neighbors) > 0 and len(
                dst_neighbors) > 0) or type != Chem.rdchem.BondType.SINGLE or bond.IsInRing():
            edges.append([src_idx, dst_idx])
            distances.append(np.linalg.norm(src_to_dst))
            mask.append(0)
            angles.append(-1)
            edges.append([dst_idx, src_idx])
            distances.append(np.linalg.norm(src_to_dst))
            mask.append(0)
            angles.append(-1)
            continue
        src_neighbor_coords = coords[src_neighbors]
        dst_neighbor_coords = coords[dst_neighbors]
        src_mean_vec = np.mean(src_neighbor_coords * np.array(src_weights)[:, None] - coords[src_idx], axis=0)
        dst_mean_vec = np.mean(dst_neighbor_coords * np.array(dst_weights)[:, None] - coords[dst_idx], axis=0)
        normal = src_to_dst / np.linalg.norm(src_to_dst)
        src_mean_projection = src_mean_vec - src_mean_vec.dot(normal) * normal
        dst_mean_projection = dst_mean_vec - dst_mean_vec.dot(normal) * normal
        cos_dihedral = src_mean_projection.dot(dst_mean_projection) / (
                    np.linalg.norm(src_mean_projection) * np.linalg.norm(dst_mean_projection))
        dihedral_angle = np.arccos(cos_dihedral)
        edges.append([src_idx, dst_idx])
        mask.append(1)
        distances.append(np.linalg.norm(src_to_dst))
        angles.append(dihedral_angle)
        edges.append([dst_idx, src_idx])
        distances.append(np.linalg.norm(src_to_dst))
        mask.append(1)
        angles.append(dihedral_angle)
    edges = torch.tensor(edges)
    graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(coords), idtype=torch.int32)

    graph.ndata['feat'] = lig_atom_featurizer(lig)
    graph.ndata['weights'] = torch.from_numpy(np.array(weights).astype(np.float32))
    graph.edata['feat'] = distance_featurizer(distances, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata['x'] = torch.from_numpy(np.array(coords).astype(np.float32))

    return graph, torch.tensor(mask, dtype=bool), torch.tensor(angles, dtype=torch.float32)

def get_geometry_graph(lig, coords = None): # 2 Hop Distances
    if coords is None:
        coords = lig.GetConformer().GetPositions()
    #print("Make Geo", coords)
    coords -= np.mean(coords, axis = 0)
    #print("Make Geo remove mean", coords)
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


def get_coarse_geometry_graph(graph, cg_map): # 2 Hop Distances
    # coords = lig.GetConformer().GetPositions()
    coords = graph.ndata['x'].cpu().numpy()
    edges_src = []
    edges_dst = []
    N = coords.shape[0]
    if N == 1:
        edges_src.extend([0])
        edges_dst.extend([0])
    else:
        for src_idx in range(N):
            one_hop_dsts = [neighbor for neighbor in  cg_map[src_idx]] #list(atom.GetNeighbors())]
            two_and_one_hop_idx = [neighbor for neighbor in one_hop_dsts]
            for one_hop_dst in one_hop_dsts:
                for two_hop_dst in cg_map[one_hop_dst]:#one_hop_dst.GetNeighbors():
                    two_and_one_hop_idx.append(two_hop_dst)
            all_dst_idx = list(set(two_and_one_hop_idx))
            if len(all_dst_idx) ==0: continue
            all_dst_idx.remove(src_idx)
            all_src_idx = [src_idx] *len(all_dst_idx)
            edges_src.extend(all_src_idx)
            edges_dst.extend(all_dst_idx)
    graph = dgl.graph((torch.tensor(edges_src), torch.tensor(edges_dst)), num_nodes=N, idtype=torch.long)
    graph.edata['feat'] = torch.from_numpy(np.linalg.norm(coords[edges_src] - coords[edges_dst], axis=1).astype(np.float32))
    return graph

def isRingAromatic(mol, bondRing):
    for id in bondRing:
        if not mol.GetBondWithIdx(id).GetIsAromatic():
            return False
    return True

def get_geometry_graph_ring(lig):
    coords = lig.GetConformer().GetPositions()
    rings = lig.GetRingInfo().AtomRings()
    bond_rings = lig.GetRingInfo().BondRings()
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
        for ring_idx, ring in enumerate(rings):
            if src_idx in ring and isRingAromatic(lig,bond_rings[ring_idx]):
                all_dst_idx.extend(list(ring))
        all_dst_idx = list(set(all_dst_idx))
        if len(all_dst_idx) == 0: continue
        all_dst_idx.remove(src_idx)
        all_src_idx = [src_idx] *len(all_dst_idx)
        edges_src.extend(all_src_idx)
        edges_dst.extend(all_dst_idx)
    graph = dgl.graph((torch.tensor(edges_src), torch.tensor(edges_dst)), num_nodes=lig.GetNumAtoms(), idtype=torch.long)
    graph.edata['feat'] = torch.from_numpy(np.linalg.norm(coords[edges_src] - coords[edges_dst], axis=1).astype(np.float32))
    return graph

def get_lig_graph_multiple_conformer(mol, name, radius=20, max_neighbors=None, use_rdkit_coords=False, num_confs=10):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    try:
        count = 0
        success = False
        while not success:
            try:
                all_lig_coords = get_multiple_rdkit_coords_individual(mol,num_conf=num_confs)
                success = True
            except Exception as e:
                print(f'failed RDKit coordinate generation. Trying the {count}th time.')
                if count > 5:
                    raise Exception(e)
                count +=1

    except Exception as e:
        all_lig_coords = [true_lig_coords] * num_confs
        with open('temp_create_dataset_rdkit.log', 'a') as f:
            f.write('Generating RDKit conformer failed for  \n')
            f.write(name)
            f.write('\n')
            f.write(str(e))
            f.write('\n')
            f.flush()
        print('Generating RDKit conformer failed for  ')
        print(name)
        print(str(e))
    lig_graphs = []
    for i in range(num_confs):
        R, t = rigid_transform_Kabsch_3D(all_lig_coords[i].T, true_lig_coords.T)
        lig_coords = ((R @ (all_lig_coords[i]).T).T + t.squeeze())
        print('kabsch RMSD between rdkit ligand and true ligand is ',
            np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())

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
                print(
                    f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
            assert i not in dst
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
        lig_graphs.append(graph)
    return lig_graphs

def get_lig_graph_revised(mol, name, radius=20, max_neighbors=None, use_rdkit_coords=False):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    if use_rdkit_coords:
        try:
            rdkit_coords = get_rdkit_coords(mol).numpy()
            R, t = rigid_transform_Kabsch_3D(rdkit_coords.T, true_lig_coords.T)
            lig_coords = ((R @ (rdkit_coords).T).T + t.squeeze())
            print('kabsch RMSD between rdkit ligand and true ligand is ', np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())
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
