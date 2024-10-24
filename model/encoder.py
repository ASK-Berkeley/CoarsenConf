import logging
import math
import os
from datetime import datetime

import dgl
import torch
from torch import nn
from dgl import function as fn

from utils.embedding import AtomEncoder, A_feature_dims, AtomEncoderTorsionalDiffusion
from utils.model_utils import *
# from ecn_layers import *
# from ecn_3d_layers import *
from encoder_layers import *

class Encoder(nn.Module):#ECN3D(nn.Module):
    def __init__(self, n_lays, debug, device, shared_layers, noise_decay_rate, cross_msgs, noise_initial,
                 use_edge_features_in_gmn, use_mean_node_features, atom_emb_dim, latent_dim,coord_F_dim,
                 dropout, nonlin, leakyrelu_neg_slope, feature_dims = 45, random_vec_dim=0, random_vec_std=1, use_scalar_features=True,
                 save_trajectories=False, weight_sharing = True, conditional_mask=False, 
                 edge_feats_dim = 15, use_bond_in_edge_feats = True, **kwargs):
        # super(ECN3D, self).__init__()
        super(Encoder, self).__init__()
        self.debug = debug
        self.cross_msgs = cross_msgs
        self.device = device
        self.save_trajectories = save_trajectories
        self.noise_decay_rate = noise_decay_rate
        self.noise_initial = noise_initial
        self.use_edge_features_in_gmn = use_edge_features_in_gmn
        self.use_mean_node_features = use_mean_node_features
        self.random_vec_dim = random_vec_dim
        self.random_vec_std = random_vec_std
        self.weight_sharing = weight_sharing
        self.conditional_mask = conditional_mask
        # self.atom_embedder = AtomEncoder(emb_dim=atom_emb_dim - self.random_vec_dim,
                                            #  feature_dims=feature_dims, use_scalar_feat=use_scalar_features,
                                            #  n_feats_to_use=num_A_feats)
        # ! Switched from EquiBind Encoding to MLP of Torsional Diffusion
        self.atom_embedder = AtomEncoderTorsionalDiffusion(emb_dim=atom_emb_dim - self.random_vec_dim, feature_dim = feature_dims) #qm9 45
        # self.atom_embedder = AtomEncoderTorsionalDiffusion(emb_dim=atom_emb_dim - self.random_vec_dim, feature_dim = 75) #drugs

        input_node_feats_dim = atom_emb_dim #64
        if self.use_mean_node_features:
            input_node_feats_dim += 5  ### Additional features from mu_r_norm
        
        # Create First Layer
        fg_edge_feats_dim = edge_feats_dim
        if use_bond_in_edge_feats:
            fg_edge_feats_dim += 5 #this is what brings us to 20
        cg_edge_feats_dim = edge_feats_dim
        self.fine_grain_layers = nn.ModuleList()
        self.fine_grain_layers.append(
            Fine_Grain_Layer(orig_invar_feats_dim_h=input_node_feats_dim,
                        invar_feats_dim_h=input_node_feats_dim,
                        out_feats_dim_h=latent_dim,
                        edge_feats_dim = fg_edge_feats_dim,
                        nonlin=nonlin,
                        cross_msgs=self.cross_msgs,
                        leakyrelu_neg_slope=leakyrelu_neg_slope,
                        debug=debug,
                        device=device,
                        dropout=dropout,
                        save_trajectories=save_trajectories,
                        weight_sharing = self.weight_sharing, **kwargs))
        # Create N - 1 layers
        if shared_layers:
            interm_lay = Fine_Grain_Layer(orig_invar_feats_dim_h=input_node_feats_dim,
                                     invar_feats_dim_h=latent_dim,
                                     out_feats_dim_h=latent_dim,
                                     edge_feats_dim = fg_edge_feats_dim,
                                     cross_msgs=self.cross_msgs,
                                     nonlin=nonlin,
                                     leakyrelu_neg_slope=leakyrelu_neg_slope,
                                     debug=debug,
                                     device=device,
                                     dropout=dropout,
                                     save_trajectories=save_trajectories,
                                     weight_sharing = self.weight_sharing, **kwargs)
            for layer_idx in range(1, n_lays):
                self.fine_grain_layers.append(interm_lay)
        else:
            for layer_idx in range(1, n_lays):
                debug_this_layer = debug if n_lays - 1 == layer_idx else False
                self.fine_grain_layers.append(
                    Fine_Grain_Layer(orig_invar_feats_dim_h=input_node_feats_dim,
                                invar_feats_dim_h=latent_dim,
                                out_feats_dim_h=latent_dim,
                                edge_feats_dim = fg_edge_feats_dim,
                                cross_msgs=self.cross_msgs,
                                nonlin=nonlin,
                                leakyrelu_neg_slope=leakyrelu_neg_slope,
                                debug=debug_this_layer,
                                device=device,
                                dropout=dropout,
                                save_trajectories=save_trajectories,
                                weight_sharing = self.weight_sharing, **kwargs))

        # Pooling Layers
        self.pooling_layers = nn.ModuleList()
        self.pooling_layers.append(
            Pooling_3D_Layer(invar_feats_dim_h=latent_dim,
                        out_feats_dim_h=latent_dim,
                        edge_feats_dim = cg_edge_feats_dim,
                        nonlin=nonlin,
                        cross_msgs=self.cross_msgs,
                        leakyrelu_neg_slope=leakyrelu_neg_slope,
                        debug=debug,
                        device=device,
                        dropout=dropout,
                        save_trajectories=save_trajectories,
                        weight_sharing = self.weight_sharing, **kwargs))
        if shared_layers:
            interm_lay = Pooling_3D_Layer(invar_feats_dim_h=latent_dim,
                                     out_feats_dim_h=latent_dim,
                                     edge_feats_dim = cg_edge_feats_dim,
                                     cross_msgs=self.cross_msgs,
                                     nonlin=nonlin,
                                     leakyrelu_neg_slope=leakyrelu_neg_slope,
                                     debug=debug,
                                     device=device,
                                     dropout=dropout,
                                     save_trajectories=save_trajectories,
                                     weight_sharing = self.weight_sharing, **kwargs)
            for layer_idx in range(1, n_lays):
                self.pooling_layers.append(interm_lay)
        else:
            for layer_idx in range(1, n_lays):
                debug_this_layer = debug if n_lays - 1 == layer_idx else False
                self.pooling_layers.append(
                    Pooling_3D_Layer(invar_feats_dim_h=latent_dim,
                                out_feats_dim_h=latent_dim,
                                edge_feats_dim = cg_edge_feats_dim,
                                cross_msgs=self.cross_msgs,
                                nonlin=nonlin,
                                leakyrelu_neg_slope=leakyrelu_neg_slope,
                                debug=debug_this_layer,
                                device=device,
                                dropout=dropout,
                                save_trajectories=save_trajectories,
                                weight_sharing = self.weight_sharing, **kwargs))

        # Coarse Grain Layers
        self.cg_layers = nn.ModuleList()
        self.cg_layers.append(
            Coarse_Grain_3DLayer(invar_feats_dim_h=latent_dim,
                        coord_F_dim=coord_F_dim,
                        nonlin=nonlin,
                        cross_msgs=self.cross_msgs,
                        leakyrelu_neg_slope=leakyrelu_neg_slope,
                        debug=debug,
                        device=device,
                        dropout=dropout,
                        save_trajectories=save_trajectories,
                        weight_sharing = self.weight_sharing, **kwargs))
        # Create N - 1 layers
        if shared_layers:
            interm_lay = Coarse_Grain_3DLayer(invar_feats_dim_h=latent_dim,
                                     coord_F_dim=coord_F_dim,
                                     cross_msgs=self.cross_msgs,
                                     nonlin=nonlin,
                                     leakyrelu_neg_slope=leakyrelu_neg_slope,
                                     debug=debug,
                                     device=device,
                                     dropout=dropout,
                                     save_trajectories=save_trajectories,
                                     weight_sharing = self.weight_sharing, **kwargs)
            for layer_idx in range(1, n_lays):
                self.cg_layers.append(interm_lay)
        else:
            for layer_idx in range(1, n_lays):
                debug_this_layer = debug if n_lays - 1 == layer_idx else False
                self.cg_layers.append(
                    Coarse_Grain_3DLayer(invar_feats_dim_h=latent_dim,
                                coord_F_dim=coord_F_dim,
                                cross_msgs=self.cross_msgs,
                                nonlin=nonlin,
                                leakyrelu_neg_slope=leakyrelu_neg_slope,
                                debug=debug_this_layer,
                                device=device,
                                dropout=dropout,
                                save_trajectories=save_trajectories,
                                weight_sharing = self.weight_sharing, **kwargs))

    def forward(self, A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool,
                  A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch = 0, prior_only = False):
        
        if prior_only: #! Quick fix to only pass in B and discard the A
             A_graph = B_graph
             geometry_graph_A = geometry_graph_B
             A_pool = B_pool
             A_cg = B_cg
             geometry_graph_A_cg = geometry_graph_B_cg
             
        orig_coords_A = A_graph.ndata['x']
        orig_coords_B = B_graph.ndata['x']
        orig_coords_A_pool = A_pool.ndata['x']
        orig_coords_B_pool = B_pool.ndata['x']
        orig_coords_A_cg = A_cg.ndata['x']
        orig_coords_B_cg = B_cg.ndata['x']
        
        coords_A = A_graph.ndata['x']
        coords_B = B_graph.ndata['x']
        # import ipdb; ipdb.set_trace()
        # print("EMB CHECK", next(self.atom_embedder.parameters()).is_cuda)
        # print(A_graph.ndata['feat'].device)
        h_feats_A = self.atom_embedder(A_graph.ndata['feat'])
        h_feats_B = self.atom_embedder(B_graph.ndata['feat'])

        coords_A_pool = A_pool.ndata['x']
        coords_B_pool = B_pool.ndata['x']
        h_feats_A_pool = A_pool.ndata['feat']
        h_feats_B_pool = B_pool.ndata['feat']

        # coords_A_cg = A_cg.ndata['x']
        # coords_B_cg = B_cg.ndata['x']
        v_A_cg = A_cg.ndata['v']
        v_B_cg = B_cg.ndata['v']
        h_feats_A_cg = A_cg.ndata['feat']
        h_feats_B_cg = B_cg.ndata['feat']

        rand_dist = torch.distributions.normal.Normal(loc=0, scale=self.random_vec_std)
        rand_h_A = rand_dist.sample([h_feats_A.size(0), self.random_vec_dim]).to(self.device)
        rand_h_B = rand_dist.sample([h_feats_B.size(0), self.random_vec_dim]).to(self.device)
        h_feats_A = torch.cat([h_feats_A, rand_h_A], dim=1)
        h_feats_B = torch.cat([h_feats_B, rand_h_B], dim=1)

        # random noise: #! I do not think we should be adding noise
        # if self.noise_initial > 0:
        #     noise_level = self.noise_initial * self.noise_decay_rate ** (epoch + 1)
        #     h_feats_A = h_feats_A + noise_level * torch.randn_like(h_feats_A)
        #     h_feats_B = h_feats_B + noise_level * torch.randn_like(h_feats_B)
        #     coords_A = coords_A + noise_level * torch.randn_like(coords_A)
        #     coords_B = coords_B + noise_level * torch.randn_like(coords_B)

        if self.use_mean_node_features:
            h_feats_A = torch.cat([h_feats_A, torch.log(A_graph.ndata['mu_r_norm'])],dim=1)
            h_feats_B = torch.cat([h_feats_B, torch.log(B_graph.ndata['mu_r_norm'])], dim=1)

        original_A_node_features = h_feats_A
        original_B_node_features = h_feats_B
        A_graph.edata['feat'] *= self.use_edge_features_in_gmn
        B_graph.edata['feat'] *= self.use_edge_features_in_gmn
        A_pool.edata['feat'] *= self.use_edge_features_in_gmn
        B_pool.edata['feat'] *= self.use_edge_features_in_gmn
        A_cg.edata['feat'] *= self.use_edge_features_in_gmn
        B_cg.edata['feat'] *= self.use_edge_features_in_gmn

        fine_mask = get_mask(A_graph.batch_num_nodes(), B_graph.batch_num_nodes(), self.device)
        coarse_mask = get_mask(A_cg.batch_num_nodes(), B_cg.batch_num_nodes(), self.device)
        # print("Coarse Mask", coarse_mask)

        # full_trajectory = [coords_A.detach().cpu()]
        # full_trajectory_cg = [A_cg.ndata['x'].detach().cpu()]
        geom_losses, geom_losses_cg = 0, 0
        for i, layer in enumerate(self.fine_grain_layers):
            # Fine Grain Layer
            coords_A, \
            h_feats_A, \
            coords_B, \
            h_feats_B, trajectory, geom_loss = layer(A_graph=A_graph,
                                B_graph=B_graph,
                                coords_A=coords_A,
                                h_feats_A=h_feats_A,
                                original_A_node_features=original_A_node_features,
                                orig_coords_A=orig_coords_A,
                                coords_B=coords_B,
                                h_feats_B=h_feats_B,
                                original_B_node_features=original_B_node_features,
                                orig_coords_B=orig_coords_B,
                                mask=fine_mask,
                                geometry_graph_A=geometry_graph_A,
                                geometry_graph_B=geometry_graph_B,
                                )
            # Pooling Layer
            coords_A_pool, h_feats_A_pool, coords_B_pool, h_feats_B_pool, trajectory_cg, geom_loss_cg = self.pooling_layers[i](A_pool=A_pool,
                            B_pool = B_pool,
                            fine_h_A=h_feats_A,
                            fine_h_B=h_feats_B,
                            coarse_h_A=h_feats_A_cg,
                            coarse_h_B=h_feats_B_cg,
                            fine_x_A=coords_A,
                            fine_x_B=coords_B,
                            # coarse_x_A=coords_A_cg,
                            # coarse_x_B=coords_B_cg,
                            pool_h_A=h_feats_A_pool,
                            pool_h_B=h_feats_B_pool,
                            pool_x_A=coords_A_pool,
                            pool_x_B=coords_B_pool,
                            og_pool_x_A=orig_coords_A_pool,
                            og_pool_x_B=orig_coords_B_pool,
                            geometry_graph_A=geometry_graph_A_cg,
                            geometry_graph_B=geometry_graph_B_cg)
            # CG layer
            v_A_cg, \
            h_feats_A_cg, \
            v_B_cg, \
            h_feats_B_cg= self.cg_layers[i](A_graph=A_cg,
                                B_graph=B_cg,
                                v_A=v_A_cg,
                                h_feats_A=h_feats_A_cg,
                                v_B=v_B_cg,
                                h_feats_B=h_feats_B_cg,
                                mask=coarse_mask,
                                pool_coords_A=coords_A_pool,
                                pool_coords_B=coords_B_pool,
                                pool_feats_A=h_feats_A_pool,
                                pool_feats_B=h_feats_B_pool)
            # if not self.separate_lig:
            # geom_losses = geom_losses + geom_loss
            # geom_losses_cg = geom_losses_cg + geom_loss_cg
            # full_trajectory.extend(trajectory)
            # full_trajectory_cg.extend(trajectory_cg)
        # return (coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, full_trajectory), (coords_A_cg, h_feats_A_cg, coords_B_cg, h_feats_B_cg, geom_losses_cg, full_trajectory_cg)
        # if prior_only:
        #     return None, (v_B_cg, h_feats_B_cg), None, None, None, None

        return (v_A_cg, h_feats_A_cg), (v_B_cg, h_feats_B_cg), 0, 0, [], [] #geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg

    def __repr__(self):
        return "ECN " + str(self.__dict__)