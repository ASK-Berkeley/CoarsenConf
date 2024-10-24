from encoder import *
from decoder import *

class VAE(nn.Module):
    def __init__(self, kl_params, encoder_params, decoder_params, loss_params, coordinate_type, device = "cuda"):
        super(VAE, self).__init__()
        self.encoder = Encoder(**encoder_params) #.to(device)
        self.decoder = Decoder(self.encoder.atom_embedder, coordinate_type, **decoder_params) #.to(device)
        self.mse = nn.MSELoss()
        self.mse_none = nn.MSELoss(reduction ='none')
        self.device = device
        F = encoder_params["coord_F_dim"]
        D = encoder_params["latent_dim"]
        
        self.kl_free_bits = kl_params['kl_free_bits']
        self.kl_prior_logvar_clamp = kl_params['kl_prior_logvar_clamp']
        self.kl_softplus = kl_params['kl_softplus']
        self.use_mim = kl_params['use_mim']

        self.kl_v_beta = loss_params['kl_weight']
        self.kl_h_beta = 0
        # self.kl_reg_beta = 1
        self.lambda_global_mse = loss_params['global_mse_weight']
        self.lambda_ar_mse = loss_params['ar_mse_weight']
        self.lambda_x_cc = loss_params['x_cc_weight']
        self.lambda_h_cc = loss_params['h_cc_weight']
        self.lambda_distance = loss_params['distance_weight']
        self.lambda_ar_distance = loss_params['ar_distance_weight']
        self.ar_loss_direction = loss_params['ar_loss_bottom_up']
        self.loss_params = loss_params

        # self.posterior_mean_V = VN_MLP(2*F, F, F, F, use_batchnorm = False)
        self.posterior_mean_V = nn.Sequential(VN_MLP(2*F, 2*F, 2*F, 2*F, use_batchnorm = False), VN_MLP(2*F, F, F, F, use_batchnorm = False))
        # self.posterior_mean_V = Vector_MLP(2*F, 2*F, 2*F, F, use_batchnorm = False) 
        # self.posterior_mean_h = Scalar_MLP(2*D, 2*D, D, use_batchnorm = False)
        self.posterior_logvar_V = Scalar_MLP(2*F*3, 2*F*3, F, use_batchnorm = False)# need to flatten to get equivariant noise N x F x 1
        # self.posterior_logvar_h = Scalar_MLP(2*D, 2*D, D, use_batchnorm = False)

        # self.prior_mean_V = VN_MLP(F,F,F,F, use_batchnorm = False)
        self.prior_mean_V = nn.Sequential(VN_MLP(F, F, F, F, use_batchnorm = False), VN_MLP(F, F, F, F, use_batchnorm = False))
        # self.prior_mean_V = Vector_MLP(F,F,F,F, use_batchnorm = False)
        # self.prior_mean_h = Scalar_MLP(D, D, D, use_batchnorm = False)
        # self.prior_logvar_V = Scalar_MLP(F*3, F*3, F, use_batchnorm = False) #! No longer need since hard coding to 0
        # self.prior_logvar_h = Scalar_MLP(D, D, D, use_batchnorm = False)
        # self.bn = VNBatchNorm(F)

    # def flip_teacher_forcing(self):
    #     self.decoder.teacher_forcing = not self.decoder.teacher_forcing

    def forward(self, frag_ids, A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch, validation = False):
        enc_out = self.forward_vae(A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch, validation)
        results, geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = enc_out
        # print("[ENC] encoder output geom loss adn geom cg loss", geom_losses, geom_loss_cg)
        natoms = A_graph.batch_num_nodes()
        nbeads = A_cg.batch_num_nodes()
        kl_v, kl_v_un_clamped = torch.tensor([0]).to(results["prior_mean_V"].device), torch.tensor([0]).to(results["prior_mean_V"].device)
        mim = torch.tensor([0]).to(results["prior_mean_V"].device)
        if not validation:
            kl_v, kl_v_un_clamped = self.kl(results["posterior_mean_V"], results["posterior_logvar_V"], results["prior_mean_V"], results["prior_logvar_V"], natoms, nbeads, coordinates = True)
            # kl_v2, kl_v_un_clamped2 = self.kl_built_in(results["posterior_mean_V"], results["posterior_logvar_V"], results["prior_mean_V"], results["prior_logvar_V"], natoms, nbeads, coordinates = True)
            # ignoring MIM for now
            if self.use_mim:
                mim = self.mim(results["Z_V"], results["posterior_mean_V"], results["posterior_logvar_V"], results["prior_mean_V"], results["prior_logvar_V"], natoms, nbeads, coordinates = True)
            dec_out = self.decoder(A_cg, B_graph, frag_ids, geometry_graph_A)
        else:
            prev_force = self.decoder.teacher_forcing
            self.decoder.teacher_forcing = False
            dec_out = self.decoder(B_cg, B_graph, frag_ids, geometry_graph_B)
            self.decoder.teacher_forcing = prev_force
        kl_h = 0
        generated_molecule, rdkit_reference, dec_results, channel_selection_info, _ = dec_out
        return generated_molecule, rdkit_reference, dec_results, channel_selection_info, (kl_v, kl_h, kl_v_un_clamped, mim), enc_out, 0 #, kl_v_reg), enc_out
    
    def distance_loss(self, generated_molecule, geometry_graphs):
        geom_loss = []
        for geometry_graph, generated_mol in zip(dgl.unbatch(geometry_graphs), dgl.unbatch(generated_molecule)):
            src, dst = geometry_graph.edges()
            src = src.long()
            dst = dst.long()
            generated_coords = generated_mol.ndata['x_cc']
            d_squared = torch.sum((generated_coords[src] - generated_coords[dst]) ** 2, dim=1)
            geom_loss.append(1/len(src) * torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2).unsqueeze(0)) #TODO: scaling hurt performance
            # geom_loss.append(torch.sum((d_squared - geometry_graph.edata['feat'] ** 2) ** 2))
        # print("[Distance Loss]", geom_loss)
        return torch.mean(torch.cat(geom_loss))

    def loss_function(self, generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, geometry_graph, AR_loss, step = 0, log_latent_stats = True):
        kl_v, kl_h, kl_v_unclamped, mim = KL_terms
        if self.use_mim:
            kl_loss = 0.1*mim
        else:
            kl_loss = self.kl_v_beta*kl_v # + self.kl_h_beta*kl_h # + self.kl_reg_beta*kl_v_reg

        x_cc, h_cc = channel_selection_info
        x_true = rdkit_reference.ndata['x_true']

        x_cc_loss = torch.tensor([0]) #[]
        ar_mse = AR_loss
        # ar_mse, global_mse, ar_dist_loss = self.coordinate_loss(dec_results, generated_molecule, align =  True)
        # import ipdb; ipdb.set_trace()
        _, global_mse = self.coordinate_loss(dec_results, generated_molecule, align =  True)

        loss =  self.lambda_ar_mse*ar_mse + self.lambda_global_mse*global_mse + kl_loss #+ cc_loss
        results, geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = enc_out
        
        if log_latent_stats:
            l2_v = torch.norm(self.std(results["posterior_logvar_V"]), 2)**2
            l2_v2 = torch.norm(results["posterior_mean_V"], 2)**2
            l2_vp = torch.norm(self.std(results["prior_logvar_V"]), 2)**2
            l2_vp2 = torch.norm(results["prior_mean_V"], 2)**2
            l2_d = torch.norm(results["posterior_mean_V"]-results["prior_mean_V"], 2)**2

        rdkit_loss = [self.rmsd(m.ndata['x_ref'], m.ndata['x_true'], align = True).unsqueeze(0) for m in dgl.unbatch(rdkit_reference)]
        rdkit_loss = self.lambda_global_mse*torch.cat(rdkit_loss).mean()

        distance_loss = self.lambda_distance*self.distance_loss(generated_molecule, geometry_graph)
        loss += distance_loss #+ ar_dist_loss
        
        if log_latent_stats:
            loss_results = {
                'latent reg loss': kl_loss.item(),
                'kl_unclamped': self.kl_v_beta*kl_v_unclamped.item(),
                'global_distance': distance_loss.item(),
                # 'ar_distance': ar_dist_loss.item(),
                'global_mse': self.lambda_global_mse*global_mse.item(),
                # 'ar_mse': self.lambda_ar_mse*ar_mse.item(),
                'channel_selection_coords_align': x_cc_loss.item(),
                'rdkit_aligned_mse': rdkit_loss.item(),
                'L2 Norm Squared Posterior LogV': l2_v.item(),
                'L2 Norm Squared Posterior Mean': l2_v2.item(),
                'L2 Norm Squared Prior LogV': l2_vp.item(),
                'L2 Norm Squared Prior Mean': l2_vp2.item(),
                'L2 Norm Squared (Posterior - Prior) Mean': l2_d.item(),
                'unscaled kl': kl_v.item(),
                'unscaled unclamped kl': kl_v_unclamped.item(),
                'unscaled mim': mim.item(),
                'mim': 0.1*mim.item(),
                'beta_kl': self.kl_v_beta,
            }
        else:
            loss_results = {
                'latent reg loss': kl_loss.item(),
                'kl_unclamped': self.kl_v_beta*kl_v_unclamped.item(),
                'global_mse': self.lambda_global_mse*global_mse.item(),
                # 'ar_mse': self.lambda_ar_mse*ar_mse.item(),
                'channel_selection_coords_align': x_cc_loss.item(),
                'rdkit_aligned_mse': rdkit_loss.item(),
                'unscaled kl': kl_v.item(),
                'unscaled unclamped kl': kl_v_unclamped.item(),
                'unscaled mim': mim.item(),
                'mim': 0.1*mim.item(),
                'beta_kl': self.kl_v_beta,
            }
        return loss, loss_results #(ar_mse.cpu(), final_align_rmsd.cpu(), kl_loss.cpu(), x_cc_loss.cpu(), h_cc_loss.cpu(),l2_v.cpu(), l2_v2.cpu(), l2_vp.cpu(), l2_vp2.cpu(), l2_d.cpu(), rdkit_loss.cpu(), distance_loss.cpu(), ar_dist_loss.cpu())

    def std(self, input):
        if self.kl_softplus:
            return 1e-6 + F.softplus(input / 2)
        return 1e-12 + torch.exp(input / 2)

    def align(self, source, target):
        with torch.no_grad():
            lig_coords_pred = target
            lig_coords = source
            if source.shape[0] == 1:
                return source
            lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
            lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

            A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean) 
            A = A + torch.eye(A.shape[0]).to(A.device) * 1e-5 #added noise to help with gradients
            if torch.isnan(A).any() or torch.isinf(A).any():
                print("\n\n\n\n\n\n\n\n\n\nThe SVD tensor contains NaN or Inf values")
                # import ipdb; ipdb.set_trace()
            U, S, Vt = torch.linalg.svd(A)
            # corr_mat = torch.diag(1e-7 + torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            rotation = (U @ corr_mat) @ Vt
            translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
        return (rotation @ lig_coords.t()).t() + translation
    
    def rmsd(self, generated, true, align = False, no_reduction = False):
        if align:
            true = self.align(true, generated)
        if no_reduction:
            loss = self.mse_none(true, generated)
        else:
            loss = self.mse(true, generated) #TODO this should have reduction sum change also for rdkit
        return loss
    
    # def ar_loss_step(self, coords, coords_ref, chunks, condition_coords, condition_coords_ref, chunk_condition, align = False, step = 1, first_step = 1):
    #     loss = 0
    #     start = 0
    #     bottom_up = self.loss_params['ar_loss_bottom_up']
    #     if condition_coords is not None and bottom_up:
    #         start_A, start_B = 0, 0
    #         for chunk_A, chunk_B in zip(chunks, chunk_condition):
    #             A, A_true = coords[start_A: start_A + chunk_A, :], coords_ref[start_A:start_A+chunk_A, :]
    #             B, B_true = condition_coords[start_B: start_B + chunk_B, :], condition_coords_ref[start_B:start_B+chunk_B, :]
    #             if A.shape[0] == 2: # when we force reference we can remove the reference form B since its in A
    #                 b_rows = B.shape[0]
    #                 common_rows = torch.all(torch.eq(B_true[:, None, :], A_true[None, :, :]), dim=-1).any(dim=-1)
    #                 B, B_true = B[~common_rows], B_true[~common_rows]
    #                 assert(B.shape[0] == B_true.shape[0] and (B.shape[0] == b_rows - 1 or B.shape[0] == b_rows))
    #             AB = torch.cat([A, B], dim = 0)
    #             AB_true = torch.cat([A_true, B_true], dim = 0)
    #             unmasked_loss = self.rmsd(AB, AB_true, align, True)
    #             mask = torch.cat([torch.ones_like(A), torch.zeros_like(B)], dim=0)
    #             masked_loss = torch.masked_select(unmasked_loss, mask.bool()).mean() # TODO change to sum then maybe no mean or mean over batch
    #             loss += masked_loss
    #             start_A += chunk_A
    #             start_B += chunk_B
    #             print("       AR loss and A shape, B shape", masked_loss.cpu().item(), A.shape, B.shape)
    #     else:
    #         for chunk in chunks:
    #             sub_loss = self.rmsd(coords[start: start + chunk, :], coords_ref[start:start+chunk, :], align)
    #             print("       ", sub_loss.cpu().item(), coords[start: start + chunk, :].shape)
    #             if coords[start: start + chunk, :].shape[0] == 1 or sub_loss.cpu().item()>3:
    #                 print("       \n", coords[start: start + chunk, :], coords_ref[start: start + chunk, :])
    #             loss += sub_loss
    #             start += chunk
    #     # if step == 0:
    #     #     loss *= first_step
    #     return loss

    def coordinate_loss(self, dec_results, generated_molecule, align = False):# = None, final_gen_coords = None, true_coords = None):
        #TODO: implement new losses with correct hydra control
        ar_loss = 0
        # ar_dist_losses = 0
        # ! Fix AR distance losses if we need them. They should be the same as global summed though since NO alignment
        # for step, info in enumerate(dec_results):
        #     coords_A, h_feats_A, coords_B, h_feats_B, geom_losses, _, id_batch, ref_coords_A, ref_coords_B, ref_coords_B_split, gen_input, model_predicted_B, model_predicted_B_split, dist_loss = info
        #     num_molecule_chunks = [len(x) for x in id_batch if x is not None]
        #     num_molecule_chunks_condition = [x.shape[0] for x in ref_coords_B_split] if step > 0 else None # first step has no conditioning
        #     # import ipdb; ipdb.set_trace()
        #     print("Chunks", num_molecule_chunks, num_molecule_chunks_condition)
        #     ga = self.ar_loss_step(coords_A, ref_coords_A, num_molecule_chunks, model_predicted_B, ref_coords_B, num_molecule_chunks_condition, align, step)
        #     print("Generative", coords_A.shape, ga)
        #     print("Dist Loss", dist_loss, "\n")
        #     loss += ga
        #     dist_losses += dist_loss
        #     # TODO: do we want to do this kind of forced identity function? 
        #     # if coords_B is None:
        #     #     assert(step == 0)
        #     #     print()
        #     #     continue
        #     # num_molecule_chunks = [x.shape[0] for x in ref_coords_B_split]
        #     # arc = self.ar_loss_step(coords_B, ref_coords_B, num_molecule_chunks, align)
        #     # print("AR Consistency", coords_B.shape, arc)
        #     # if torch.gt(arc, 1000).any() or torch.isinf(arc).any() or torch.isnan(arc).any():
        #     #     print("Chunks", num_molecule_chunks)
        #     #     print("conditional input", ar_con_input)
        #     #     print("Bad Coordinate Check B", coords_B)
        #     # print()
        #     # loss += arc
        global_mse_loss = [self.rmsd(m.ndata['x_cc'],m.ndata['x_true'], align = True).unsqueeze(0) for m in dgl.unbatch(generated_molecule)]
        # print("Align MSE Loss", global_mse_loss)
        global_mse_loss = torch.cat(global_mse_loss).mean() #sum(global_mse_loss)
        return ar_loss, global_mse_loss #, ar_dist_losses
    
    def forward_vae(self, A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch, validation = False):
        (v_A, h_A), (v_B, h_B), geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg = self.encoder(A_graph, B_graph, geometry_graph_A, geometry_graph_B, A_pool, B_pool, A_cg, B_cg, geometry_graph_A_cg, geometry_graph_B_cg, epoch, prior_only = validation)
        # ipdb.set_trace()
        if not validation:
            # print("[Encoder] ecn output V A", torch.min(v_A).item(), torch.max(v_A).item())
            posterior_input_V = torch.cat((v_A, v_B), dim = 1) # N x 2F x 3
            # posterior_input_h = torch.cat((h_A, h_B), dim = 1) # N x 2D
            posterior_mean_V = self.posterior_mean_V(posterior_input_V)
            # print("[Encoder] pre BN posterior mean V", torch.min(posterior_mean_V).item(), torch.max(posterior_mean_V).item()) #, torch.sum(posterior_mean_V, dim = 1))
            # posterior_mean_V = self.bn(posterior_mean_V) #made blow up worse: Trying VN Batch Norm
            # posterior_mean_h = self.posterior_mean_h(posterior_input_h)
            posterior_logvar_V = self.posterior_logvar_V(posterior_input_V.reshape((posterior_input_V.shape[0], -1))).unsqueeze(2)
            # posterior_logvar_h = self.posterior_logvar_h(posterior_input_h)
            # posterior_logvar_V = torch.clamp(posterior_logvar_V, max = ) #TODO do I want to clamp
            # print("[Encoder] posterior mean V", torch.min(posterior_mean_V).item(), torch.max(posterior_mean_V).item()) #, torch.sum(posterior_mean_V, dim = 1))
            # print("[Encoder] posterior logvar V", torch.min(posterior_logvar_V).item(), torch.max(posterior_logvar_V).item()) #, torch.sum(posterior_logvar_V,  dim = 1))
            # print("[Encoder] posterior mean h", torch.min(posterior_mean_h).item(), torch.max(posterior_mean_h).item(), torch.sum(posterior_mean_h).item())
            # print("[Encoder] posterior logvar h", torch.min(posterior_logvar_h).item(), torch.max(posterior_logvar_h).item(), torch.sum(posterior_logvar_h).item())
        
        # print("[Encoder] ecn output V B", torch.min(v_B).item(), torch.max(v_B).item())
        prior_mean_V = self.prior_mean_V(v_B)
        # prior_mean_h = self.prior_mean_h(h_B)
        # prior_logvar_V = self.prior_logvar_V(v_B.reshape((v_B.shape[0], -1))).unsqueeze(2)
        # ! Setting Prior log var to 0 so std = 1 no more clamping
        prior_logvar_V = torch.zeros((v_B.shape[0], v_B.shape[1])).unsqueeze(2).to(v_B.device) # N x F x 1
        # prior_logvar_h = self.prior_logvar_h(h_B)

        if validation:
            Z_V = self.reparameterize(prior_mean_V, prior_logvar_V, mean_only=True)
            # Z_h = self.reparameterize(prior_mean_h, prior_logvar_h, mean_only=True)
        else:
            Z_V = self.reparameterize(posterior_mean_V, posterior_logvar_V)
            # Z_h = self.reparameterize(posterior_mean_h, posterior_logvar_h)

        A_cg.ndata["Z_V"] = Z_V
        # A_cg.ndata["Z_h"] = Z_h
        B_cg.ndata["Z_V"] = Z_V
        # B_cg.ndata["Z_h"] = Z_h

        results = {
            "Z_V": Z_V,
            # "Z_h": Z_h,
            "v_A": v_A,
            "v_B": v_B,
            "h_A": h_A,
            "h_B": h_B,

            "prior_mean_V": prior_mean_V,
            # "prior_mean_h": prior_mean_h,
            "prior_logvar_V": prior_logvar_V,
            # "prior_logvar_h": prior_logvar_h,
        }
        if not validation:
            results.update({
            "posterior_mean_V": posterior_mean_V,
            # "posterior_mean_h": posterior_mean_h,
            "posterior_logvar_V": posterior_logvar_V,
            # "posterior_logvar_h": posterior_logvar_h,
            })
        return results, geom_losses, geom_loss_cg, full_trajectory, full_trajectory_cg

    def reparameterize(self, mean, logvar, scale = 1.0, mean_only = False):
        if mean_only:
            return mean
        if self.kl_softplus:
            sigma = 1e-6 + F.softplus(scale*logvar / 2)
        else:
            sigma = 1e-12 + torch.exp(scale*logvar / 2)
        eps = torch.randn_like(mean)
        return mean + eps*sigma

    # https://github.com/NVIDIA/NeMo/blob/b9cf05cf76496b57867d39308028c60fef7cb1ba/nemo/collections/nlp/models/machine_translation/mt_enc_dec_bottleneck_model.py#L217
    def kl(self, z_mean, z_logvar, z_mean_prior, z_logvar_prior, natoms, nbeads, coordinates = False):
        assert len(natoms) == len(nbeads)
        # ! Look into budget per molecule
        free_bits_per_dim = self.kl_free_bits/z_mean[0].numel()
        if self.kl_softplus:
            p_std = 1e-6 + F.softplus(z_logvar / 2)
            q_std = 1e-6 + F.softplus(z_logvar_prior / 2)
        else:
            p_std = 1e-12 + torch.exp(z_logvar / 2)
            q_std = 1e-12 + torch.exp(z_logvar_prior / 2)
        q_mean = z_mean_prior
        p_mean = z_mean
        var_ratio = (p_std / q_std).pow(2)
        t1 = ((p_mean - q_mean) / q_std).pow(2)
        kl =  0.5 * (var_ratio + t1 - 1 - var_ratio.log()) # shape = number of CG beads
        pre_clamp_kl = kl
        kl = torch.clamp(kl, min = free_bits_per_dim)
        kl = kl.sum(-1)
        if coordinates:
            kl = kl.sum(-1)
        # Here kl is [N]
        # return kl.mean()
        return self.kl_loss(kl, natoms, nbeads), self.kl_loss(pre_clamp_kl, natoms, nbeads)
    
    def kl_loss(self, kl, natoms, nbeads):
        B = len(natoms)
        start = 0
        loss = []
        for atom, coarse in zip(natoms, nbeads):
            kl_chunk = kl[start: start + coarse].sum().unsqueeze(0)
            loss.append(1/atom * kl_chunk)
            start += coarse
        # import ipdb; ipdb.set_trace()
        total_loss = 1/B * torch.sum(torch.cat(loss))
        return total_loss
    
    def kl_built_in(self, z_mean, z_logvar, z_mean_prior, z_logvar_prior, natoms, nbeads, coordinates = False):
        assert len(natoms) == len(nbeads)
        free_bits_per_dim = self.kl_free_bits/z_mean[0].numel()
        if self.kl_softplus:
            posterior_std = 1e-12 + F.softplus(z_logvar / 2)
            prior_std = 1e-12 + F.softplus(z_logvar_prior / 2)
        else:
            posterior_std = 1e-12 + torch.exp(z_logvar / 2)
            prior_std = 1e-12 + torch.exp(z_logvar_prior / 2)
        posterior = torch.distributions.Normal(loc = z_mean, scale = posterior_std)
        prior = torch.distributions.Normal(loc = z_mean_prior, scale = prior_std)
        pre_clamp_kl = torch.distributions.kl.kl_divergence(posterior, prior)
        kl = torch.clamp(pre_clamp_kl, min = free_bits_per_dim)
        kl = kl.sum(-1)
        if coordinates:
            kl = kl.sum(-1)
        # Here kl is [N]
        return self.kl_loss(kl, natoms, nbeads), self.kl_loss(pre_clamp_kl, natoms, nbeads)
    
    def mim(self, z, z_mean, z_logvar, z_mean_prior, z_logvar_prior, natoms, nbeads, coordinates = False):
        assert len(natoms) == len(nbeads)
        free_bits_per_dim = self.kl_free_bits/z_mean[0].numel()
        z_logvar = torch.clamp(z_logvar, min = -6) #! minimum uncertainty
        if self.kl_softplus:
            posterior_std = 1e-12 + F.softplus(z_logvar / 2)
            prior_std = 1e-12 + F.softplus(z_logvar_prior / 2)
        else:
            posterior_std = 1e-12 + torch.exp(z_logvar / 2)
            prior_std = 1e-12 + torch.exp(z_logvar_prior / 2)
        posterior = torch.distributions.Normal(loc = z_mean, scale = posterior_std)
        prior = torch.distributions.Normal(loc = z_mean_prior, scale = prior_std)
        log_q_z_given_x = self.kl_loss(posterior.log_prob(z).sum(-1).sum(-1), natoms, nbeads) #.sum(-1).sum(-1).mean()
        log_p_z = self.kl_loss(prior.log_prob(z).sum(-1).sum(-1), natoms, nbeads)
        loss_terms = -0.5 * (log_q_z_given_x + log_p_z)
        return loss_terms
