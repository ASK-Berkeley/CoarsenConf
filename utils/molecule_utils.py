import os
import sys
import pickle
import numpy as np
import dgl
from collections import defaultdict, Counter

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
import torch
from geometry_utils import *
import scipy.spatial as spa
from scipy import spatial
from scipy.special import softmax
from embedding import *
import networkx as nx

def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])

def align_coordinates(rd_mol, kit_mol):
    prior_coords = kit_mol.GetConformer().GetPositions()
    true_coords = rd_mol.GetConformer().GetPositions()
    R, t = rigid_transform_Kabsch_3D(prior_coords.T, true_coords.T)
    rotated_rdkit_coords = ((R @ (prior_coords).T).T + t.squeeze())
    return rotated_rdkit_coords

def align(source, target):
        # Rot, trans = rigid_transform_Kabsch_3D_torch(input.T, target.T)
        # lig_coords = ((Rot @ (input).T).T + trans.squeeze())
        # Kabsch RMSD implementation below taken from EquiBind
        # source = kit_mol.GetConformer().GetPositions()
        # target = rd_mol.GetConformer().GetPositions()
        lig_coords_pred = target
        lig_coords = source
        lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
        lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

        A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)

        U, S, Vt = torch.linalg.svd(A)

        corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
        rotation = (U @ corr_mat) @ Vt
        translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
        return (rotation @ lig_coords.t()).t() + translation

def get_torsions_geo(mol_list):
    atom_counter = 0
    torsionList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
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

def autoregressive_bfs(ids, bmap, a=1.0, b=1.1):
    if len(ids) == 1:
        return [0]
    scores = {}
    start = -1
    max_score = 0
    for k, val in bmap.items():
        sc = len(ids[k])*a + len(val)*b
        penalty = 0 if len(ids[k]) > 1 else a
        sc -= penalty
        scores[k] = sc
        max_score = max(max_score, sc)
        if max_score == sc:
            start = k
    order = []
    queue = [start]
    seen = set(queue)
    while len(queue) > 0:
        cur = queue.pop(0)
        order.append(cur)
        kids = [x for x in bmap[cur] if x not in seen]
        if len(kids) == 0:
            continue
        for k in kids:
            seen.add(k)
        kids = [x for _, x in sorted(zip([scores[y] for y in kids], kids), key=lambda pair: pair[0], reverse = True)]
        queue.extend(kids)
    return order

def autoregressive_bfs_with_reference(ids, bmap, bond_break, a=1.0, b=1.1):
    #print(ids)
    #print(bmap)
    #print(bond_break)
    if len(ids) == 1:
        return [0], [-1]
    scores = {}
    start = -1
    max_score = 0
    for k, val in bmap.items():
        sc = len(ids[k])*a + len(val)*b
        penalty = 0 if len(ids[k]) > 1 else a
        sc -= penalty
        scores[k] = sc
        max_score = max(max_score, sc)
        if max_score == sc:
            start = k
    order = []
    queue = [(start, -1)]
    reference = []
    seen = set([start])
    while len(queue) > 0:
        cur = queue.pop(0)
        reference.append(cur[1])
        cur = cur[0]
        order.append(cur)
        kids = [x for x in bmap[cur] if x not in seen]
        if len(kids) == 0:
            continue
        for k in kids:
            seen.add(k)
        kids_connect = []
#     Here we check the forward and reverse direction to ensure the reference comes from the 
        for _, x in sorted(zip([scores[y] for y in kids], kids), key=lambda pair: pair[0], reverse = True):
            for kid in ids[x]:
                if kid in bond_break.keys():
                    atom_connect = list(set(bond_break[kid]).intersection(ids[cur]))
                    if len(atom_connect) == 0:
                        continue
                    #print("ac", atom_connect, "kid -->", bond_break[kid], kid,ids[x], "cur -->", ids[cur])
                    assert(len(atom_connect) == 1)
                    atom_connect = atom_connect[0]
                    break
                atom_connect = -1
            kids_connect.append((x, atom_connect))
        kids = kids_connect
        queue.extend(kids)
    assert(all([ v > -1 for v in reference[1:] ]))
    return order, reference

def mol2graph(mol, name = "test", radius=4, max_neighbors=None, use_rdkit_coords = False, seed = 0):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    if use_rdkit_coords:
        # import ipdb; ipdb.set_trace()
        rdkit_coords = get_rdkit_coords(mol, seed) #.numpy()
        R, t = rigid_transform_Kabsch_3D(rdkit_coords.T, true_lig_coords.T)
        lig_coords = ((R @ (rdkit_coords).T).T + t.squeeze())
        #print('kabsch RMSD between rdkit ligand and true ligand is ', np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())

        lig_coords = align(torch.from_numpy(rdkit_coords), torch.from_numpy(true_lig_coords)).numpy()
        # print('LOSS kabsch RMSD between rdkit ligand and true ligand is ', np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())
        loss = torch.nn.MSELoss()
        # print('LOSS kabsch MSE between rdkit ligand and true ligand is ', loss(torch.from_numpy(true_lig_coords), torch.from_numpy(lig_coords)).cpu().detach().numpy().item())
    else:
        lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]

    remove_centroid = True
    if remove_centroid:
        lig_coords -= np.mean(lig_coords, axis = 0)
        true_lig_coords -= np.mean(true_lig_coords, axis = 0)
        #print("Remove Centroid", np.mean(lig_coords, axis = 0))

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
            #print(f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
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
    graph.ndata['mol_feat'] = lig_atom_featurizer(mol)
    if use_rdkit_coords:
        graph.ndata['rd_feat'] = graph.ndata['feat']
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata['x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32)) #! this should be the current coords not always ground truth
    graph.ndata['x_ref'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    graph.ndata['x_true'] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    return graph

def get_torsion_angles(mol):
    torsions_list = []
    G = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i)
    nodes = set(G.nodes())
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        l = list(sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append(
            (n0[0], e[0], e[1], n1[0])
        )
    return torsions_list

def coarsen_molecule_new(m):
    m = Chem.AddHs(m) #the GEOM dataset molecules have H's
    # if not use_diffusion_angle_def:
    torsions = get_torsions_geo([m])
    #  angles = [(a,b,c,d, rdMolTransforms.GetDihedralRad(mol.GetConformer()), a, b, c, d) for a,b,c,d in torsions] # GetDihedralRad GetDihedralDeg
    # else:
    #     assert(1 == 0)
    #     torsions = get_torsion_angles(m)
    #print("Torsion Angles", torsions)
    if len(torsions) > 0:
        bond_break = [(b,c) for a,b,c,d in torsions]
        adj = Chem.rdmolops.GetAdjacencyMatrix(m)
        for r,c in bond_break:
            adj[r][c] = 0
            adj[c][r] = 0
        out = Chem.rdmolops.FragmentOnBonds(m,
                                        [m.GetBondBetweenAtoms(b[0], b[1]).GetIdx() for b in bond_break],
                                        addDummies=False) # determines fake bonds which adds fake atoms
        frags = Chem.GetMolFrags(out, asMols=True)
        frag_ids = Chem.GetMolFrags(out, asMols=False) #fragsMolAtomMapping = []
        frag_ids = [set(x) for x in frag_ids]
        cg_bonds = []
        cg_map = defaultdict(list)
        for start, end in bond_break:
            a = min(start, end)
            b = max(start, end)
            A, B = -1, -1
            for i, bead in enumerate(frag_ids):
                if a in bead:
                    A = i
                elif b in bead:
                    B = i
                if A > 0 and B > 0:
                    break
            cg_map[A].append(B)
            cg_map[B].append(A)
            cg_bonds.append((min(A,B), max(A,B)))
        
        bond_break_map = defaultdict(list) # TODO can expand to torsion info as well
        for b,c in bond_break:
            bond_break_map[b].append(c)
            bond_break_map[c].append(b)

        return torsions, list(frags), frag_ids, adj, out, bond_break_map, cg_bonds, cg_map
    else:
        return torsions, [m], [list(range(m.GetNumAtoms()))], Chem.rdmolops.GetAdjacencyMatrix(m), m, [], None, None
    
def coarsen_molecule(m, use_diffusion_angle_def = False):
    m = Chem.AddHs(m) #the GEOM dataset molecules have H's
    # if not use_diffusion_angle_def:
    torsions = get_torsions_geo([m])
    #  angles = [(a,b,c,d, rdMolTransforms.GetDihedralRad(mol.GetConformer()), a, b, c, d) for a,b,c,d in torsions] # GetDihedralRad GetDihedralDeg
    # else:
    #     assert(1 == 0)
    #     torsions = get_torsion_angles(m)
    #print("Torsion Angles", torsions)
    if len(torsions) > 0:
        bond_break = [(b,c) for a,b,c,d in torsions]
        adj = Chem.rdmolops.GetAdjacencyMatrix(m)
        for r,c in bond_break:
            adj[r][c] = 0
            adj[c][r] = 0
        out = Chem.rdmolops.FragmentOnBonds(m,
                                        [m.GetBondBetweenAtoms(b[0], b[1]).GetIdx() for b in bond_break],
                                        addDummies=False) # determines fake bonds which adds fake atoms
        frags = Chem.GetMolFrags(out, asMols=True)
        frag_ids = Chem.GetMolFrags(out, asMols=False) #fragsMolAtomMapping = []
        frag_ids = [set(x) for x in frag_ids]
        cg_bonds = []
        cg_map = defaultdict(list)
        for start, end in bond_break:
            a = min(start, end)
            b = max(start, end)
            A, B = -1, -1
            for i, bead in enumerate(frag_ids):
                if a in bead:
                    A = i
                elif b in bead:
                    B = i
                if A > 0 and B > 0:
                    break
            cg_map[A].append(B)
            cg_map[B].append(A)
            cg_bonds.append((min(A,B), max(A,B)))
        
        bond_break_map = defaultdict(list) # TODO can expand to torsion info as well
        for b,c in bond_break:
            bond_break_map[b].append(c)
            bond_break_map[c].append(b)

        return list(frags), frag_ids, adj, out, bond_break_map, cg_bonds, cg_map
    else:
        return [m], [list(range(m.GetNumAtoms()))], Chem.rdmolops.GetAdjacencyMatrix(m), m, [], None, None

def create_pooling_graph(dgl_graph, frag_ids, latent_dim = 64, use_mean_node_features=True):
    N = len(frag_ids)
    n = dgl_graph.ndata['x'].shape[0]
    fine_coords = []
    coarse_coords = []
    # if use_mean_node_features:
    #     latent_dim += 5
#     M = np.zeros((N, n))
    chunks = []
    for bead, atom_ids in enumerate(frag_ids):
        subg = list(atom_ids)
        subg.sort()
        cc = dgl_graph.ndata['x'][subg,:].mean(dim=0).cpu().numpy().reshape(-1,3)
        fc = dgl_graph.ndata['x'][subg,:].cpu().numpy()
#         qprint(cc.shape, fc.shape)
        chunks.append(fc.shape[0])
        coarse_coords.append(cc)
        fine_coords.append(fc)
#         M[bead, list(atom_ids)] = 1
    fine_coords = np.concatenate(fine_coords, axis = 0)
    coarse_coords = np.concatenate(coarse_coords, axis = 0)
#     print(coarse_coords.shape, fine_coords.shape)
    coords = np.concatenate((fine_coords, coarse_coords), axis = 0)
    
    distance = spa.distance.cdist(coords, coords)
    src_list = []
    dst_list = []
    dist_list = []
    # mean_norm_list = [np.zeros((5,))]*n
    
    # prev = 0
    # for cg_bead in range(n, n+N):
    #     src = [prev + i for i in range(0, chunks[cg_bead-n])]
    #     dst = [cg_bead]*len(src)
    #     prev += len(src)
    #     src_list.extend(src)
    #     dst_list.extend(dst)
    #     valid_dist = list(distance[src, cg_bead])
    #     dist_list.extend(valid_dist)
    for idx, cg_bead in enumerate(list(range(n, n+N))):
        src = list(frag_ids[idx])
        dst = [cg_bead]*len(src)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distance[src, cg_bead])
        dist_list.extend(valid_dist)

        
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=n+N, idtype=torch.int32)

    graph.ndata['feat'] = torch.zeros((n+N, latent_dim))
    # graph.ndata['feat_pool'] = torch.zeros((n+N, latent_dim)) # for ECN updates
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata['x'] = torch.from_numpy(np.array(coords).astype(np.float32))
    # graph.ndata['x_pool'] = torch.from_numpy(np.array(coords).astype(np.float32))
    # graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    return graph
        

def conditional_coarsen_3d(dgl_graph, frag_ids, cg_map, bond_break = None, radius=4, max_neighbors=None, latent_dim_D = 64, latent_dim_F = 32,  use_mean_node_features = True):
    num_nodes = len(frag_ids)
    coords = []
    # if use_mean_node_features:
    #     latent_dim += 5
    M = np.zeros((num_nodes, dgl_graph.ndata['x'].shape[0]))
    Mmap = np.zeros((num_nodes,1))
    for bead, atom_ids in enumerate(frag_ids):
        subg = list(atom_ids)
        subg.sort()
        coords.append(dgl_graph.ndata['x'][subg,:].mean(dim=0).cpu().numpy())
        M[bead, list(atom_ids)] = 1
        Mmap[bead, 0] = len(list(atom_ids))
        # TODO can scale by MW weighted_average = (A@W)/W.sum()
#         print(subg, coords[0].shape, coords)
    # bfs_order = autoregressive_bfs(frag_ids, cg_map)
    bfs_order, ref_order = autoregressive_bfs_with_reference(frag_ids, cg_map, bond_break)
    #print(bfs_order)
    #print(ref_order)
    bfs = np.zeros((num_nodes,1))
    ref = np.zeros((num_nodes,1))
    for order_idx, bead in enumerate(bfs_order): # step 0 --> N
        bfs[bead, 0] = order_idx
    for order_idx, atom_reference in enumerate(ref_order): # step 0 --> N
        ref[order_idx, 0] = atom_reference
    # Use the below if we need the reference to be in bead order not BFS order
    # for order_idx, info in enumerate(zip(bfs_order,ref_order)):
    #     bead, atom_reference = info # step 0 --> N
    #     bfs[bead, 0] = order_idx
    #     ref[bead, 0] = atom_reference

    coords = np.asarray(coords)
    distance = spa.distance.cdist(coords, coords)
    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    if num_nodes == 1:
        src = [0]
        dst = [0]
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distance[0, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distance[0, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        # weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        # assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        # diff_vecs = coords[src, :] - coords[dst, :]  # (neigh_num, 3)
        # mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        # denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        # mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        # mean_norm_list.append(mean_vec_ratio_norm)
        mean_norm_list.append(np.zeros((5,)))
    else:
        for i in range(num_nodes):
            dst = list(np.where(distance[i, :] < radius)[0])
            dst.remove(i)
            if max_neighbors != None and len(dst) > max_neighbors:
                dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
            if len(dst) == 0:
                dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
                #print(f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}\n')
                # print(dgl_graph.ndata['x'].shape, frag_ids,  cg_map)
            assert i not in dst
            assert dst != []
            
            required_dst = cg_map[i]
            for d in required_dst:
                if d not in dst:
                    #print("[Required] adding CG edge", i, d, distance[i, d])
                    dst.append(d)
            
            src = [i] * len(dst)
            src_list.extend(src)
            dst_list.extend(dst)
            valid_dist = list(distance[i, dst])
            dist_list.extend(valid_dist)
            valid_dist_np = distance[i, dst]
            sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
            weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
            assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
            diff_vecs = coords[src, :] - coords[dst, :]  # (neigh_num, 3)
            mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
            denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
            mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
            mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['feat'] = torch.zeros((num_nodes, latent_dim_D))
    # graph.ndata['feat_pool'] = torch.zeros((num_nodes, latent_dim_D)) # for ECN updates
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata['x'] = torch.from_numpy(np.array(coords).astype(np.float32))
    # graph.ndata['x_pool'] = torch.from_numpy(np.array(coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    graph.ndata['v'] = torch.zeros((num_nodes, latent_dim_F, 3))
    # graph.ndata['M'] = torch.from_numpy(np.array(M).astype(np.float32))
    graph.ndata['cg_to_fg'] = torch.from_numpy(np.array(Mmap).astype(np.float32))
    graph.ndata['bfs'] = torch.from_numpy(np.array(bfs).astype(np.float32))
    graph.ndata['bfs_reference_point'] = torch.from_numpy(np.array(ref).astype(np.float32))
    return graph

def get_coords(rd_mol):
    rd_conf = rd_mol.GetConformers()[0]
    positions = rd_conf.GetPositions()
    Z = []
    for position in positions:
        Z.append([*position])
    return np.asarray(Z)

def get_rdkit_coords(mol, seed = None, use_mmff = True):
    id = AllChem.EmbedMultipleConfs(mol, numConfs=1)#, randomSeed=10)
    try:
        id = list(id)
    except:
        print(id, len(list(id)))
        import ipdb; ipdb.set_trace()
    # if id == -1:
    if len(id) == 0:
        #print('rdkit coords could not be generated without using random coords. using random coords now.')
        AllChem.EmbedMultipleConfs(mol, numConfs=1, useRandomCoords = True)
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        except:
            print("get_rdkit_coords: RDKit cannot generate conformer for: ", Chem.MolToSmiles(mol))
            return None
    elif use_mmff:
        AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
    conf = mol.GetConformer()
    lig_coords = conf.GetPositions()
    return np.asarray(lig_coords) #, dtype=torch.float32)

def get_rdkit_coords_old(mol, seed = None):
    ps = AllChem.ETKDGv2()
    if seed is not None:
        ps.randomSeed = seed
    id = AllChem.EmbedMolecule(mol, ps)
    # import ipdb; ipdb.set_trace()
    if id == -1:
        #print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=0)
        except:
            # print("RDKit cannot generate conformer for: ", Chem.MolToSmiles(mol))
            return None
    else:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    conf = mol.GetConformer()
    lig_coords = conf.GetPositions()
    return np.asarray(lig_coords) #, dtype=torch.float32)

from rdkit.Chem.rdchem import BondType as BT
def get_bond(m, i, j):
    bond = m.GetBondBetweenAtoms(i,j)
    info = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    if bond == None:
        return -1e9 # not inf for stab
    else:
        typ = bond.GetBondType()
        return info[typ]
        
    
def mol2graphV2(mol, name = "test", radius=4, max_neighbors=None, use_rdkit_coords=False):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)
    src_list = []
    dst_list = []
    dist_list = []
    bond_list = []
    mean_norm_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            #print(f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
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
        
        bonds = []
        for d in dst:
            bonds.append(get_bond(mol, int(i), int(d)))
        bond_list.extend(bonds)

    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['feat'] = lig_atom_featurizer(mol)
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.edata['bond_type'] = torch.from_numpy(np.asarray(bond_list).astype(np.float32))
    graph.edata['featV2'] = torch.cat((graph.edata['feat'],graph.edata['bond_type'].view(-1,1)), dim = -1)
    graph.ndata['x'] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    if use_rdkit_coords:
        graph.ndata['new_x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    return graph

def mol2graphV3(mol, name = "test", radius=4, max_neighbors=None, use_rdkit_coords=False):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    
    torsions = get_torsions_geo([mol])
    torsion_map = {}
    for b in torsions:
        torsion_map[(b[1], b[2])] = ([b[0], b[3]])
        
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)
    src_list = []
    dst_list = []
    dist_list = []
    bond_list = []
    tort_list = []
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
        
        bonds = []
        torts = []
        for d in dst:
            bonds.append(get_bond(mol, int(i), int(d)))
            ii, dd = int(i), int(d)
            key = (min(ii, dd), max(ii, dd))
            if key in torsion_map:
                val = torsion_map[key]
                torts.append(GetDihedral(conf, [val[0], key[0], key[1], val[1]]))
            else:
                torts.append(0)
        bond_list.extend(bonds)
        tort_list.extend(torts)

    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['feat'] = lig_atom_featurizer(mol)
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.edata['bond_type'] = torch.from_numpy(np.asarray(bond_list).astype(np.float32))
    graph.edata['torsion_angle'] = torch.from_numpy(np.asarray(tort_list).astype(np.float32))
    graph.edata['featV2'] = torch.cat((graph.edata['feat'],graph.edata['bond_type'].view(-1,1)), dim = -1)
    graph.edata['featV3'] = torch.cat((graph.edata['feat'],graph.edata['bond_type'].view(-1,1), graph.edata['torsion_angle'].view(-1,1)), dim = -1)
    graph.ndata['x'] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
#     if use_rdkit_coords:
#         graph.ndata['new_x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    return graph
