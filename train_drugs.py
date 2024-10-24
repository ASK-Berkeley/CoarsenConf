import sys
# Print the current Python path
# print(sys.path)
import os
# print(os.environ['PYTHONPATH'])
import hydra
from omegaconf import DictConfig, OmegaConf
import ipdb
import torch
import wandb
import random
import logging
from utils.torsional_diffusion_data_all import load_drugs_conformers
from model.vae import VAE
import datetime
from model.benchmarker import *
import glob


def load_data(cfg):
    print("Loading DRUGS...")
    train_loader, train_data = load_drugs_conformers(batch_size=cfg['train_batch_size'], mode='train', limit_mols=cfg['train_data_limit'])
    val_loader, val_data = load_drugs_conformers(batch_size=cfg['val_batch_size'], mode='val', limit_mols=cfg['val_data_limit'])
    print("Loading DRUGS --> Done")
    return train_loader, train_data, val_loader, val_data

def save_code(wandb_run): 
    code_dir = "./model" 
    # Create an artifact from the code directory
    code_artifact = wandb.Artifact("model", type="code") 
    # Add all the .py files in the code directory to the artifact using glob
    for file_path in glob.glob(code_dir + '/*.py'):
        code_artifact.add_file(file_path)
    wandb_run.log_artifact(code_artifact)
    
    code_artifact = wandb.Artifact("utils", type="code") 
    code_dir = "./utils" 
    # Add all the .py files in the utils directory to the artifact using glob
    for file_path in glob.glob(code_dir + '/*.py'):
        code_artifact.add_file(file_path) 
    wandb_run.log_artifact(code_artifact)

import subprocess

def print_gpu_usage():
    command = "nvidia-smi"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

def get_memory_usage():
    # cmd = "nvidia-smi | grep python | awk '{ print $6 }'"
    cmd = "nvidia-smi | grep python | awk '{ print $8 }' | sed 's/MiB//'"
    output = subprocess.check_output(cmd, shell=True)
    return int(output.strip().decode()) #int(output.strip())
    

@hydra.main(config_path="../configs", config_name="config_drugs.yaml")
def main(cfg: DictConfig): #['encoder', 'decoder', 'vae', 'optimizer', 'losses', 'data', 'coordinates', 'wandb']
    import datetime
    now = datetime.datetime.now()
    suffix = f"_{now.strftime('%m-%d_%H-%M-%S')}"
    coordinate_type = cfg.coordinates
    NAME = cfg.wandb['name'] + suffix
    wandb_run = wandb.init(
        project=cfg.wandb.project,
        name=NAME,
        notes=cfg.wandb.notes,
        config = cfg,
        save_code = True
    )
    save_code(wandb_run)
    train_loader, train_data, val_loader, val_data = load_data(cfg.data)
    # BENCHMARK = BenchmarkRunner(batch_size = cfg.data['train_batch_size'])
    F = cfg.encoder["coord_F_dim"]
    D = cfg.encoder["latent_dim"]
    model = VAE(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, coordinate_type, device = "cuda").cuda()
    
    print("CUDA CHECK", next(model.parameters()).is_cuda)
    print("# of Encoder Params = ", sum(p.numel()
          for p in model.encoder.parameters() if p.requires_grad))
    print("# of Decoder Params = ", sum(p.numel()
          for p in model.decoder.parameters() if p.requires_grad))
    print("# of VAE Params = ", sum(p.numel()
          for p in model.parameters() if p.requires_grad))
    if cfg.optimizer.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr= cfg.optimizer.lr)
    elif cfg.optimizer.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr= cfg.optimizer.lr)
    else:
        assert(1 == 0)
    # self.optim.step()
    # self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
    # self.lr_scheduler.total_warmup_steps > self.lr_scheduler._step)):  # step per batch if that is what we want to do or if we are using a warmup schedule and are still in the warmup period
    #         self.step_schedulers() --> self.lr_scheduler.step()
    # self.optim.zero_grad()
    # self.optim_steps += 1
    # torch.autograd.set_detect_anomaly(True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=1, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.8)


    
    kl_annealing = True
    kl_weight = 1e-5
    kl_annealing_rate = 1e-3
    kl_annealing_interval = 1
    kl_cap = 1e-1
    
    dist_weight = 1e-6
    dist_annealing_rate = 0.05
    dist_cap = 0.5
    for epoch in range(cfg.data['epochs']):
        print("Epoch", epoch)
        if kl_annealing and epoch > 0 and epoch % kl_annealing_interval == 0:
            kl_weight += kl_annealing_rate
            kl_weight = min(kl_weight, kl_cap)
            
            dist_weight += dist_annealing_rate
            dist_weight = min(dist_weight, dist_cap)
        if kl_annealing:
            model.kl_v_beta = kl_weight
            model.lambda_distance = dist_weight
        train_loss_log, val_loss_log = [], []
        count = 0
        for A_batch, B_batch in train_loader:
            A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
            B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch
            A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to(
                'cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
            B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to(
                'cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')

            generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, AR_loss = model(
                frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=epoch)
            loss, losses = model.loss_function(generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, geo_A, AR_loss, step=epoch)
            print(f"Train LOSS = {loss}")
            loss.backward()
            losses['Train Loss'] = loss.cpu()
            wandb.log(losses)
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            norm_type = 2
            total_norm = 0.0
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Train LOSS = {loss}")
            print("TOTAL GRADIENT NORM", total_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.optimizer.clip_grad_norm, norm_type=2) 
            optim.step()
            optim.zero_grad()
            
            del A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids
            del B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids
            del generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, AR_loss, losses
            if count > 0 and count %10 == 0:
                torch.cuda.empty_cache()
                model_path = f'./model_ckpt/{NAME}_{epoch}_temp.pt'
                torch.save(model.state_dict(), model_path)
            count+=1

        print("Validation")
        val_loss = 0
        with torch.no_grad():
            for A_batch, B_batch in val_loader:
                A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
                B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch

                A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
                B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')

                generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, AR_loss = model(
                        B_frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=epoch, validation = True)
                # ipdb.set_trace()
                loss, losses = model.loss_function(generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, geo_A, AR_loss, step=epoch, log_latent_stats = False)
                # train_loss_log.append(losses)
                losses['Val Loss'] = loss.cpu()
                # val_loss += losses['Val Loss']
                wandb.log({'val_' + key: value for key, value in losses.items()})
                print(f"Val LOSS = {loss}")
            
            # print("Test Benchmarks")
            # BENCHMARK.generate(model)
            
        # scheduler.step(val_loss)
        scheduler.step()
        model_path = f'./model_ckpt/{NAME}_{epoch}.pt'
        torch.save(model.state_dict(), model_path)
        


    print("Training Complete")


if __name__ == "__main__":
    main()
