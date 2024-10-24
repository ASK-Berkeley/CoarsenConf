import sys
import os
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
import torch.multiprocessing as mp
from model.benchmarker import *

def load_data(cfg):
    #mp.set_start_method('spawn') # use 'spawn' method instead of 'fork'
    #mp.set_sharing_strategy('file_system') 
    print("Loading DRUGs...")
    train_loader, train_data = load_drugs_conformers(batch_size=cfg['train_batch_size'], mode='train', num_workers= 16) 
    val_loader, val_data = load_drugs_conformers(batch_size=cfg['val_batch_size'], mode='val', num_workers = 16)
    print("Loading DRUGS --> Done")
    return train_loader, train_data, val_loader, val_data

@hydra.main(config_path="../configs", config_name="config_drugs.yaml")
def main(cfg: DictConfig): #['encoder', 'decoder', 'vae', 'optimizer', 'losses', 'data', 'coordinates', 'wandb']
    import datetime
    now = datetime.datetime.now()
    suffix = f"_{now.strftime('%m-%d_%H-%M-%S')}"
    coordinate_type = cfg.coordinates
    NAME = cfg.wandb['name'] + suffix
    F = cfg.encoder["coord_F_dim"]
    D = cfg.encoder["latent_dim"]
    model = VAE(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, coordinate_type, device = "cuda").cuda()
    
    runner = BenchmarkRunner(batch_size = cfg.data['train_batch_size'], 
                             true_mols = '/data/DRUGS/test_mols.pkl', 
                             valid_mols = '/data/DRUGS/test_smiles.csv', 
                             dataset = 'drugs', 
                             save_dir = '/data/test_set')
    runner.generate(model, rdkit_only = False, use_wandb=False)


if __name__ == "__main__":
    # mp.set_sharing_strategy('file_system')
    main()
