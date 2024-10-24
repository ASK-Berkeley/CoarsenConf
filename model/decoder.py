from utils.equivariant_model_utils import *
from utils.geometry_utils import *
from decoder_layers import IEGMN_Bidirectional
from decoder_delta_layers import IEGMN_Bidirectional_Delta
from decoder_double_delta_layers import IEGMN_Bidirectional_Double_Delta
from collections import defaultdict
import numpy as np
import copy
from torch import nn
import torch.nn.functional as F
import dgl
from tqdm import tqdm
from utils.neko_fixed_attention import neko_MultiheadAttention

import ipdb

class Decoder(nn.Module):
    def __init__(self, atom_embedder, coordinate_type, n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, latent_dim, coord_F_dim,
                 dropout, nonlin, leakyrelu_neg_slope, random_vec_dim=0, random_vec_std=1, use_scalar_features=True,
                 num_A_feats=None, save_trajectories=False, weight_sharing = True, conditional_mask=False, verbose = False, **kwargs):
        super(Decoder, self).__init__()
        # self.mha = torch.nn.MultiheadAttention(embed_dim = 3, num_heads = 1, batch_first = True)
        self.verbose = verbose
        self.mha = neko_MultiheadAttention(embed_dim = 3, num_heads = 1, batch_first = True)
        self.double = False
        self.device = device
        if coordinate_type == "delta":
            self.iegmn = IEGMN_Bidirectional_Delta(n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, latent_dim, coord_F_dim,
                 dropout, nonlin, leakyrelu_neg_slope, random_vec_dim, random_vec_std, use_scalar_features,
                 save_trajectories, weight_sharing, conditional_mask, **kwargs).cuda() #iegmn
        elif coordinate_type == "double":
            self.double = True #True
            self.iegmn = IEGMN_Bidirectional_Double_Delta(n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, latent_dim, coord_F_dim,
                 dropout, nonlin, leakyrelu_neg_slope, random_vec_dim, random_vec_std, use_scalar_features,
                 save_trajectories, weight_sharing, conditional_mask, **kwargs).cuda() #iegmn
        else:
            self.iegmn = IEGMN_Bidirectional(n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                    use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, latent_dim, coord_F_dim,
                    dropout, nonlin, leakyrelu_neg_slope, random_vec_dim, random_vec_std, use_scalar_features,
                    save_trajectories, weight_sharing, conditional_mask, **kwargs).cuda() #iegmn
        self.atom_embedder = atom_embedder # taken from the FG encoder
        D, F = latent_dim, coord_F_dim
        
        # self.h_channel_selection = Scalar_MLP(D, 2*D, D)
        # self.mse = nn.MSELoss()
        # norm = "ln"
        if kwargs['cc_norm'] == "bn":
            self.eq_norm = VNBatchNorm(F)
            self.inv_norm = nn.BatchNorm1d(D)
            self.eq_norm_2 = VNBatchNorm(3)
            self.inv_norm_2 = nn.BatchNorm1d(D)
        elif kwargs['cc_norm'] == "ln":
            self.eq_norm = VNLayerNorm(F)
            self.inv_norm = nn.LayerNorm(D)
            self.eq_norm_2 = VNLayerNorm(3)
            self.inv_norm_2 = nn.LayerNorm(D)
        else:
            assert(1 == 0)
        
        # self.feed_forward_V = Vector_MLP(F, 2*F, 2*F, F, leaky = False, use_batchnorm = False)
        self.feed_forward_V = nn.Sequential(VNLinear(F, 2*F), VN_MLP(2*F, F, F, F, leaky = False, use_batchnorm = False))
        # self.feed_forward_h = Scalar_MLP(D, 2*D, D)

        # self.feed_forward_V_3 = Vector_MLP(3, F, F, 3, leaky = False, use_batchnorm = False)
        self.feed_forward_V_3 = nn.Sequential(VNLinear(3, F), VN_MLP(F, 3, 3, 3, leaky = False, use_batchnorm = False))
        # self.feed_forward_h_3 = Scalar_MLP(D, 2*D, D)
        self.teacher_forcing = kwargs['teacher_forcing']
        self.mse_none = nn.MSELoss(reduction ='none')
    
    def get_node_mask(self, ligand_batch_num_nodes, receptor_batch_num_nodes, device):
        rows = ligand_batch_num_nodes.sum()
        cols = receptor_batch_num_nodes.sum()
        mask = torch.zeros(rows, cols, device=device)
        partial_l = 0
        partial_r = 0
        for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
            mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
            partial_l = partial_l + l_n
            partial_r = partial_r + r_n
        return mask

    def get_queries_and_mask(self, Acg, B, N, F, splits):
        # Each batch isa CG bead
        queries = []
        prev = 0
        info = list(Acg.ndata['cg_to_fg'].flatten().cpu().numpy().astype(int)) # This is used to help with the padding
        # for x in info:
        #     queries.append(B_batch.ndata['x'][prev:prev+ x, :])
        #     prev += x
        # import ipdb; ipdb.set_trace()
        splits = sum(splits, []) # flatten
        for x in splits:
            queries.append(B.ndata['x'][list(x), :])

        Q = torch.nn.utils.rnn.pad_sequence(queries, batch_first=True)
        n_max = Q.shape[1]
        attn_mask = torch.ones((N, n_max, F), dtype=torch.bool).to(self.device)
        for b in range(attn_mask.shape[0]):
            attn_mask[b, :info[b], :] = False
            attn_mask[b, info[b]:, :] = True
        return Q, attn_mask, info

    def channel_selection(self, A_cg, B, cg_frag_ids):
        # ! FF + Add Norm
        Z_V = A_cg.ndata["Z_V"]# N x F x 3
        # Z_h = A_cg.ndata["Z_h"] # N x D
        N, F, _ = Z_V.shape
        # _, D = Z_h.shape
        if self.verbose: print("[CC] V input", torch.min(Z_V).item(), torch.max(Z_V).item())
        # if self.verbose: print("[CC] h input", torch.min(Z_h).item(), torch.max(Z_h).item())
        Z_V_ff = self.feed_forward_V(Z_V)
        # Z_h_ff = self.feed_forward_h(Z_h)
        if self.verbose: print("[CC] V input FF", torch.min(Z_V_ff).item(), torch.max(Z_V_ff).item())
        # if self.verbose: print("[CC] h input FF", torch.min(Z_h_ff).item(), torch.max(Z_h_ff).item())
        Z_V = Z_V_ff + Z_V
        # Z_h = Z_h_ff + Z_h
        if self.verbose: print("[CC] V input FF add", torch.min(Z_V).item(), torch.max(Z_V).item())
        # if self.verbose: print("[CC] h input FF add", torch.min(Z_h).item(), torch.max(Z_h).item())
        Z_V = self.eq_norm(Z_V)
        # Z_h = self.inv_norm(Z_h)
        if self.verbose: print("[CC] V add norm", torch.min(Z_V).item(), torch.max(Z_V).item())
        # if self.verbose: print("[CC] h add norm", torch.min(Z_h).item(), torch.max(Z_h).item())
        # Equivariant Channel Selection
        Q, attn_mask, cg_to_fg_info = self.get_queries_and_mask(A_cg, B, N, F, cg_frag_ids)
        K = Z_V
        V = Z_V
        attn_out, attn_weights = self.mha(Q, K, V, attn_mask = attn_mask)
        res = []
        for idx, k in enumerate(cg_to_fg_info):
            res.append( attn_out[idx, :k, :]) # Pull from the parts that were not padding
        res = torch.cat(res, dim = 0)
        x_cc = res # n x 3
        # Invariant Channel Selection
        # h_og = B.ndata['rd_feat'] # n x d = 17
        # h_og_lifted = self.h_channel_selection(self.atom_embedder(h_og)) # n x D
        # h_cc = h_og_lifted + torch.repeat_interleave(Z_h, torch.tensor(cg_to_fg_info).to(self.device), dim = 0) # n x D
        # Second Add Norm
        if self.verbose: print("[CC] V cc attn update", torch.min(x_cc).item(), torch.max(x_cc).item())
        # if self.verbose: print("[CC] h cc mpnn update", torch.min(h_cc).item(), torch.max(h_cc).item())
        x_cc_ff = self.feed_forward_V_3(x_cc.unsqueeze(2)).squeeze(2)
        # h_cc_ff = self.feed_forward_h_3(h_cc)
        if self.verbose: print("[CC] V input FF 3", torch.min(x_cc_ff).item(), torch.max(x_cc_ff).item())
        # if self.verbose: print("[CC] h input FF 3", torch.min(h_cc_ff).item(), torch.max(h_cc_ff).item())
        x_cc = x_cc_ff + x_cc
        # h_cc = h_cc_ff + h_cc
        if self.verbose: print("[CC] V cc add 2", torch.min(x_cc).item(), torch.max(x_cc).item())
        # if self.verbose: print("[CC] h cc add 2", torch.min(h_cc).item(), torch.max(h_cc).item())
        x_cc = self.eq_norm_2(x_cc.unsqueeze(2)).squeeze(2)
        # h_cc = self.inv_norm_2(h_cc)
        # if self.verbose: print("same") #! the above makes it worse for some reason for bn
        h_cc = self.atom_embedder(B.ndata['ref_feat']) #! testing
        if self.verbose: print("[CC] V cc add norm --> final", torch.min(x_cc).item(), torch.max(x_cc).item())
        if self.verbose: print("[CC] h cc add norm --> final", torch.min(h_cc).item(), torch.max(h_cc).item())
        if self.verbose: print()
        return x_cc, h_cc # we have selected the features for all coarse grain beads in parallel


    def autoregressive_step(self, latent, prev = None, t = 0, geo_latent = None, geo_current = None):
        coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory = self.iegmn(latent, prev, mpnn_only = t==0,
                                                                                            geometry_graph_A = geo_latent,
                                                                                            geometry_graph_B = geo_current,
                                                                                            teacher_forcing=self.teacher_forcing,
                                                                                            atom_embedder=self.atom_embedder)
        return  coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory
    
    def sort_ids(self, all_ids, all_order):
        result = []
        for ids, order in zip(all_ids, all_order):
            sorted_frags = list(zip(ids, order))
            sorted_frags.sort(key = lambda x: x[1])
            frag_ids = [x for x, y in sorted_frags]
            result.append(frag_ids)
        return result

    def isolate_next_subgraph(self, final_molecule, id_batch, true_geo_batch):
        molecules = dgl.unbatch(final_molecule)
        geos = dgl.unbatch(true_geo_batch)
        result = []
        geo_result = []
        # valid = []
        # check = True
        for idx, ids in enumerate(id_batch):
            if ids is None:
                # valid.append((idx, False))
                # check = False
                continue
            # else:
                # valid.append((idx, True))
            fine = molecules[idx]
            subg = dgl.node_subgraph(fine, ids)
            result.append(subg)

            subgeo = dgl.node_subgraph(geos[idx], ids)
            geo_result.append(subgeo)
        return dgl.batch(result).to(self.device), dgl.batch(geo_result).to(self.device) #valid, check

    
    def gather_current_molecule(self, final_molecule, current_molecule_ids, progress, true_geo_batch):
        if current_molecule_ids is None or len(current_molecule_ids) == 0: #or torch.sum(progress) == 0
            return None, None
        if torch.sum(progress) == 0:
            return final_molecule, true_geo_batch
        molecules = dgl.unbatch(final_molecule)
        geos = dgl.unbatch(true_geo_batch)
        result = []
        geo_result = []
        for idx, ids in enumerate(current_molecule_ids):
            if progress[idx] == 0:
                continue
            fine = molecules[idx]
            subg = dgl.node_subgraph(fine, ids)
            result.append(subg)

            subgeo = dgl.node_subgraph(geos[idx], ids)
            geo_result.append(subgeo)
        return dgl.batch(result).to(self.device), dgl.batch(geo_result).to(self.device)

    def update_molecule(self, final_molecule, id_batch, coords_A, h_feats_A, latent):
        num_nodes = final_molecule.batch_num_nodes()
        start = 0
        prev = 0
        for idx, ids in enumerate(id_batch):
            if ids is None:
                start += num_nodes[idx]
                continue
            updated_ids = [start + x for x in ids]
            start += num_nodes[idx]
            cids = [prev + i for i in range(len(ids))]
            prev += len(ids)
            
            final_molecule.ndata['x_cc'][updated_ids, :] = coords_A[cids, :].to('cpu')
            final_molecule.ndata['feat_cc'][updated_ids, :] = h_feats_A[cids, :].to('cpu')

    def add_reference(self, ids, refs, progress):
        batch_idx = 0
        ids = copy.deepcopy(ids)
        for atom_ids, ref_list in zip(ids, refs):
            for idx, bead in enumerate(atom_ids):
                if self.verbose: print(idx, bead)
                partner = int(ref_list[idx].item())
                if len(bead) == 1 and partner != -1:
                    if self.verbose: print("\n start", atom_ids)
                    bead.add(partner)
                    if self.verbose: print("update with reference", atom_ids)
                    progress[batch_idx] += 1
            batch_idx += 1
        lens = [sum([len(y) for y in x]) for x in ids]
        check = all([a-b.item() == 0 for a, b in zip(lens,progress)])
        # ipdb.set_trace()
        # try:
        assert(check)
        # except:
        #     ipdb.set_trace()
        return ids, progress

    def distance_loss(self, generated_coords_all, geometry_graphs, true_coords = None):
        geom_loss = []
        for geometry_graph, generated_coords in zip(dgl.unbatch(geometry_graphs), generated_coords_all):
            src, dst = geometry_graph.edges()
            if len(src) == 0 or len(dst) == 0:
                geom_loss.append(torch.tensor([0]).to(self.device))
            src = src.long()
            dst = dst.long()
            d_squared = torch.sum((generated_coords[src] - generated_coords[dst]) ** 2, dim=1)
            geom_loss.append(1/len(src) * torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2).unsqueeze(0))
            # geom_loss.append(torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2))
        # if self.verbose: print("          [AR Distance Loss Step]", geom_loss)
        # if true_coords is not None:
        #     for a, b, c, d in zip (generated_coords_all, geom_loss, true_coords, dgl.unbatch(geometry_graphs)):
        #         if self.verbose: print("          Aligned MSE", a.shape, self.rmsd(a, c, align = True))
        #         if self.verbose: print("          Gen", a)
        #         if self.verbose: print("          True", c)
        #         if self.verbose: print("          distance", b)
        #         if self.verbose: print("          edges", d.edges())
        return torch.mean(torch.cat(geom_loss))

    def align(self, source, target):
        # Rot, trans = rigid_transform_Kabsch_3D_torch(input.T, target.T)
        # lig_coords = ((Rot @ (input).T).T + trans.squeeze())
        # Kabsch RMSD implementation below taken from EquiBind
        # if source.shape[0] == 2:
        #     return align_sets_of_two_points(target, source) #! Kabsch seems to work better and it is ok
        with torch.no_grad():
            lig_coords_pred = target
            lig_coords = source
            if source.shape[0] == 1:
                return source
            lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
            lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

            A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)+1e-7 #added noise to help with gradients
            if torch.isnan(A).any() or torch.isinf(A).any():
                print(torch.max(A))
                import ipdb; ipdb.set_trace()
                
            U, S, Vt = torch.linalg.svd(A)

            corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            rotation = (U @ corr_mat) @ Vt
            translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
        return (rotation @ lig_coords.t()).t() + translation
        # return lig_coords
    
    def mse(self, generated, true, align = False):
        # if align:
        #     true = self.align(true, generated)
        # loss = self.mse_none(true, generated)
        # ! No longer calculating an AR MSE Loss
        loss = torch.zeros_like(generated)
        return loss
    
    def get_reference(self, subgraph, molecule, ref_ids):
        # references = []
        result = []
        info = [(m, x) for m, x in zip(dgl.unbatch(molecule), ref_ids) if x is not None]
        for g, mid in zip(dgl.unbatch(subgraph), info):
            m, id = mid
            if id is None:
                continue
            r = id[0]
            if r == -1:
                # references.append(torch.zeros_like(g.ndata['x_cc']))
                g.ndata['reference_point'] = torch.zeros_like(g.ndata['x_cc']).to(g.device)
            else:
                point = molecule.ndata['x_cc'][r].reshape(1,-1)
                # references.append(point.repeat(g.ndata['x_cc'].shape[0]))
                g.ndata['reference_point'] = point.repeat(g.ndata['x_cc'].shape[0], 1).to(g.device)
            result.append(g)
        return dgl.batch(result).to(self.device) #, references

    def ar_loss_step(self, coords, coords_ref, chunks, condition_coords, condition_coords_ref, chunk_condition, align = False):#, step = 1, first_step = 1):
        loss = []
        start = 0
        bottom_up = True #self.loss_params['ar_loss_bottom_up']
        if condition_coords is not None and bottom_up:
            start_A, start_B = 0, 0
            for chunk_A, chunk_B in zip(chunks, chunk_condition):
                A, A_true = coords[start_A: start_A + chunk_A, :], coords_ref[start_A:start_A+chunk_A, :]
                B, B_true = condition_coords[start_B: start_B + chunk_B, :], condition_coords_ref[start_B:start_B+chunk_B, :]
                if A.shape[0] == 2: # when we force reference we can remove the reference form B since its in A
                    b_rows = B.shape[0]
                    common_rows = torch.all(torch.eq(B_true[:, None, :], A_true[None, :, :]), dim=-1).any(dim=-1)
                    B, B_true = B[~common_rows], B_true[~common_rows]
                    assert(B.shape[0] == B_true.shape[0] and (B.shape[0] == b_rows - 1 or B.shape[0] == b_rows))
                AB = torch.cat([A, B], dim = 0)
                AB_true = torch.cat([A_true, B_true], dim = 0)
                unmasked_loss = self.mse(AB, AB_true, align)
                mask = torch.cat([torch.ones_like(A), torch.zeros_like(B)], dim=0)
                masked_loss = torch.masked_select(unmasked_loss, mask.bool()).sum() 
                loss.append(masked_loss)
                start_A += chunk_A
                start_B += chunk_B
                if self.verbose: print("       unnormalized AR loss and A shape, B shape", masked_loss.cpu().item(), A.shape, B.shape)
        else:
            for chunk in chunks:
                sub_loss = self.mse(coords[start: start + chunk, :], coords_ref[start:start+chunk, :], align).sum()
                if self.verbose: print("      unnormalized AR first step loss ", sub_loss.cpu().item(), coords[start: start + chunk, :].shape)
                if coords[start: start + chunk, :].shape[0] == 1 or sub_loss.cpu().item()>3:
                    if self.verbose: print("       ", coords[start: start + chunk, :], coords_ref[start: start + chunk, :])
                loss.append(sub_loss)
                start += chunk
        return loss
    
    def forward(self, cg_mol_graph, rdkit_mol_graph, cg_frag_ids, true_geo_batch):
        rdkit_reference = copy.deepcopy(rdkit_mol_graph.to('cpu'))
        X_cc, H_cc = self.channel_selection(cg_mol_graph, rdkit_mol_graph, cg_frag_ids)
        rdkit_mol_graph.ndata['x_cc'] = X_cc
        rdkit_mol_graph.ndata['feat_cc'] = H_cc
        bfs = cg_mol_graph.ndata['bfs'].flatten()
        ref = cg_mol_graph.ndata['bfs_reference_point'].flatten()
        X_cc = copy.deepcopy(X_cc.detach())
        H_cc = copy.deepcopy(H_cc.detach())

        bfs_order = []
        ref_order = []
        start = 0
        for i in cg_mol_graph.batch_num_nodes():
            bfs_order.append(bfs[start : start + i])
            ref_order.append(ref[start: start + i])
            start += i

        progress = rdkit_mol_graph.batch_num_nodes().cpu()
        total_num_atoms = rdkit_mol_graph.batch_num_nodes().cpu()
        # if self.verbose: print("progress", progress)
        final_molecule = rdkit_mol_graph
        frag_ids = self.sort_ids(cg_frag_ids, bfs_order)
        # if self.verbose: print(frag_ids)
        frag_ids, progress = self.add_reference(frag_ids, ref_order, progress) #done

        frag_batch = defaultdict(list) # keys will be time steps
        max_nodes = max(cg_mol_graph.batch_num_nodes()).item()
        for t in range(max_nodes):
            for idx, frag in enumerate(frag_ids): # iterate over moelcules
                ids = None
                if t < len(frag):
                    ids = list(frag[t])
                frag_batch[t].append(ids)
        if self.double:
            ref_batch = defaultdict(list)
            for t in range(max_nodes):
                for idx, refs in enumerate(ref_order): # iterate over moelcules
                    ids = None
                    if t < len(refs):
                        ids = [int(refs[t].item())]
                    ref_batch[t].append(ids)

        # ipdb.set_trace()
        current_molecule_ids = None
        current_molecule = None
        geo_current = None
        returns = []
        
        losses = [[] for _ in range(len(progress))]#torch.zeros_like(progress)
        loss_idx = list(range(len(progress)))
        # for t in range(max_nodes):
        final_molecule = final_molecule.to('cpu')
        true_geo_batch = true_geo_batch.to('cpu')
        for t in tqdm(range(max_nodes), desc='autoregressive time steps'):
        # print("")
        # for t in range(max_nodes):
            # ipdb.set_trace()
            # print("[Auto Regressive Step]", t)
            id_batch = frag_batch[t]
            # if self.verbose: print("ID", id_batch)
            # try:
            latent, geo_latent = self.isolate_next_subgraph(final_molecule, id_batch, true_geo_batch)
            # except Exception as e:
            #     print(e)
            #     print(id_batch)
            #     assert(1==0)
            if self.double:
                r_batch = ref_batch[t]
                latent = self.get_reference(latent, final_molecule, r_batch)
            # if not check:
            #     current_molecule = self.adaptive_batching(current_molecule)
            # ipdb.set_trace()
            coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory = self.autoregressive_step(latent, current_molecule, t, geo_latent, geo_current)
            ref_coords_A = latent.ndata['x_true']
            ref_coords_B = current_molecule.ndata['x_true'] if current_molecule is not None else None
            ref_coords_B_split = [x.ndata['x_true'] for x in dgl.unbatch(current_molecule)] if current_molecule is not None else None
            model_predicted_B = current_molecule.ndata['x_cc'] if current_molecule is not None else None
            model_predicted_B_split = [x.ndata['x_cc'] for x in dgl.unbatch(current_molecule)] if current_molecule is not None else None
            gen_input = latent.ndata['x_cc']
            # if self.verbose: print("[AR step end] geom losses total from decoder", t, geom_losses)
            # dist_loss = self.distance_loss([x.ndata['x_cc'] for x in dgl.unbatch(latent)], geo_latent, [x.ndata['x_true'] for x in dgl.unbatch(latent)])
            dist_loss = torch.tensor([0])
            if self.verbose: print()
            returns.append((coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory, id_batch, ref_coords_A, ref_coords_B, ref_coords_B_split, gen_input, model_predicted_B, model_predicted_B_split, dist_loss))
            # if self.verbose: print("ID Check", returns[0][6])
                # X_cc, H_cc = self.channel_selection(cg_mol_graph, rdkit_mol_graph, cg_frag_ids)
                # coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory = self.autoregressive_step(latent, current_molecule, t, geo_latent, geo_current)

                # print(f'{t} B MSE = {torch.mean(torch.sum(ref_coords_B - coords_B, dim = 1)**2)}')
            num_molecule_chunks = [len(x) for x in id_batch if x is not None]
            loss_idx_t = [x for i, x in enumerate(loss_idx) if id_batch[i] is not None]
            num_molecule_chunks_condition = [x.shape[0] for x in ref_coords_B_split] if current_molecule is not None else None
            # ar_losses = self.ar_loss_step(coords_A, ref_coords_A, num_molecule_chunks, model_predicted_B, ref_coords_B, num_molecule_chunks_condition, True)
            # for idx, ar_loss in zip(loss_idx_t, ar_losses):
            #     losses[idx].append(ar_loss) #+= ar_loss
                
            self.update_molecule(final_molecule, id_batch, coords_A, h_feats_A, latent)
            progress -= torch.tensor([len(x) if x is not None else 0 for x in id_batch])
            if t == 0:
                current_molecule_ids = copy.deepcopy(id_batch)
            else:
                for idx, ids in enumerate(id_batch):
                    if ids is None:
                        continue
                    ids = [x for x in ids if x not in set(current_molecule_ids[idx])] #! added to prevent overlap of reference
                    current_molecule_ids[idx].extend(ids)
            # ipdb.set_trace() # erroring on dgl.unbatch(final_molecule) for some odd reason --> fixed by adding an else above
            del current_molecule, geo_current
            current_molecule, geo_current = self.gather_current_molecule(final_molecule, current_molecule_ids, progress, true_geo_batch)
            del latent, geo_latent
        # ipdb.set_trace()
        del current_molecule, geo_current
        # for idx, natoms in enumerate(total_num_atoms):
        #     losses[idx] = (sum(losses[idx])/natoms).unsqueeze(0)
        #     if self.verbose: print("[Final AR Loss]", natoms, losses[idx])
        return final_molecule.to(self.device), rdkit_reference, returns, (X_cc, H_cc), None #torch.cat(losses).mean()

