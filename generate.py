import os
import sys
sys.path.insert(0, './model')
sys.path.insert(0, './utils')
import pickle, random
from argparse import ArgumentParser
from multiprocessing import Pool
import numpy as np
import pandas as pd
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from tqdm import tqdm
import wandb, copy, torch, random
from utils.torsional_diffusion_data_all import featurize_mol, featurize_mol_new, qm9_types, drugs_types, get_transformation_mask, check_distances
from molecule_utils import *
import dgl
from collections import defaultdict
from tqdm import tqdm

from hydra.experimental import compose, initialize_config_dir
from omegaconf import DictConfig
from model.vae import VAE
import multiprocessing as mp
from functools import partial
import time

# Load the config using Hydra
def dgl_to_mol(mol, data, mmff=False, rmsd=False, copy=True, key = 'x_cc'):
    if not mol.GetNumConformers():
        conformer = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conformer)
    coords = data.ndata[key]
    if type(coords) is not np.ndarray:
        coords = coords.double().numpy()
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
    if mmff:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        except Exception as e:
            pass
    if not copy: return mol
    return deepcopy(mol)

def lazy_process_data(dataset):
    for data in dataset:
        yield data
            
def collate(samples):
    # A, B = map(list, zip(*samples))
    A = samples
    A_graph = dgl.batch([x[0] for x in A])
    geo_A = dgl.batch([x[1] for x in A])
    Ap = dgl.batch([x[2] for x in A])
    A_cg = dgl.batch([x[3] for x in A])
    geo_A_cg = dgl.batch([x[4] for x in A])
    frag_ids = [x[5] for x in A]
    #
    B_graph = dgl.batch([x[0] for x in A])
    geo_B = dgl.batch([x[1] for x in A])
    Bp = dgl.batch([x[2] for x in A])
    B_cg = dgl.batch([x[3] for x in A])
    geo_B_cg = dgl.batch([x[4] for x in A])
    B_frag_ids = [x[5] for x in A]
    return (A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids), (B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids)

def parallel_generate_smi(wrapper, smi):
    return wrapper.generate_smi(smi)

def generate_smi(args):
        conf_id, mol = args
        # print(conf_id)
        try:
            feats = featurize_mol_new(mol=mol, types=drugs_types, conf_id = conf_id)
            A_frags, A_frag_ids, A_adj, A_out, A_bond_break, A_cg_bonds, A_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = False)
            A_cg = conditional_coarsen_3d(feats, A_frag_ids, A_cg_map, A_bond_break, radius=4, max_neighbors=None, latent_dim_D = 64, latent_dim_F = 32)
            geometry_graph_A = get_geometry_graph(mol)
            Ap = create_pooling_graph(feats, A_frag_ids)
            geometry_graph_A_cg = get_coarse_geometry_graph(A_cg, A_cg_map)
            return mol, conf_id, feats, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids
        except Exception as e:
            print(e)
            return None
        # return conf_id

class CoarsenConf():
    def __init__(self, weights_path = "your_coarsenconf.pt", batch_size = 100, name = "test", save_dir = './output'):
        config_dir = "./configs"
        config_name = "config_drugs.yaml"

        # Initialize the Hydra config system
        initialize_config_dir(config_dir)
        self.name = name
        self.batch_size = batch_size

        self.save_dir = save_dir
        self.types = drugs_types
        # torch.multiprocessing.set_sharing_strategy('file_system')
        self.dataloader = None
        cfg = compose(config_name)
        model = VAE(cfg.vae, cfg.encoder, cfg.decoder, cfg.losses, cfg.coordinates, device = "cuda").cuda()
        chkpt = torch.load(weights_path)
        if "model" in chkpt:
            chkpt = chkpt['model']
        model.load_state_dict(chkpt, strict = False)
        self.model = model

    def load_data(self, batch_size = None):
        conf_save_path = os.path.join(self.save_dir, f'{self.name}_rdkit_all.pkl')
        with open(conf_save_path, 'rb') as handle:
            self.all_moles = pickle.load(handle)
            
        conf_save_path_loader = os.path.join(self.save_dir, f'{self.name}_data_loader.pkl')
        with open(conf_save_path_loader, 'rb') as handle:
            self.model_preds = pickle.load(handle)
            
        self.datapoints = []
        for k, v in self.model_preds.items():
            if v[0] == None:
                continue
            self.datapoints.extend([(k, vv) for vv in v])
        self.smiles = [x[0] for x in self.datapoints]
        data = [x[1] for x in self.datapoints]
        print(f"Generating {len(data)} molecules")
        if batch_size is None:
            batch_size = self.batch_size
        self.dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size= batch_size, shuffle=False, drop_last=False, num_workers=0 ,collate_fn = collate)
            
    def generate_parallel(self, smile_list = None):
        if self.dataloader == None:
            self.model_preds = defaultdict(list)
            self.all_moles = defaultdict(list)
            # dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
            if type(smile_list) == str:
                smile_list = {smile_list: 1}
            countx = 1
            total = len(smile_list)
            # wrapper = partial(self.generate_smi)
            for smi, count in smile_list.items():
                print(countx, total, time.asctime( time.localtime(time.time()) ))
                countx += 1
                mol = Chem.MolFromSmiles(smi)
                atom_types = set([atom.GetSymbol() for atom in mol.GetAtoms()])
                if any (atom_type not in drugs_types for atom_type in atom_types):
                    print("Not DRUGS molecule")
                    continue
                mol, info = self.get_rdkit_coords_bulk(smi, count*2)
                if mol is None or info is None:
                    continue
                indices = [(idx, copy.deepcopy(mol)) for idx, val in enumerate(info[1]) if val[0] == 0]
                if len(indices) > count:
                    energies = [(idx, val[1]) for idx, val in enumerate(info[1]) if val[0] == 0]
                    # import ipdb; ipdb.set_trace()
                    energies.sort(key=lambda x: x[1])
                    ids = [x[0] for x in energies[:count]]
                    indices = [(idx, m) for idx, m in indices if idx in ids]
                elif len(indices) < count:
                    new_mol = []
                    for _ in range(count - len(indices)):
                        idx_, mol_ = random.choice(indices)
                        new_mol.append(copy.deepcopy(mol_))
                    for nidx, m in enumerate(new_mol):
                        indices.append((len(indices)+ nidx, m))
                assert(len(indices) == count)
                    
                # pdata = lazy_process_data(indices)
                # with mp.Pool(processes=16) as pool:
                #     results = pool.map(partial(parallel_generate_smi, wrapper), pdata)
                # with mp.Pool(processes=16) as pool:
                #     results = pool.map(generate_smi, indices)
                results = [generate_smi(arg) for arg in indices]
                # import ipdb; ipdb.set_trace()
    
                # 'results' will contain the output of the generate_smi function for each SMILES
                for result in results:
                    if result is not None:
                        nmol, cid, feats, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids = result
                        self.model_preds[smi].append((feats, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
                        self.all_moles[smi].append((nmol, cid))
            
            conf_save_path = os.path.join(self.save_dir, f'{self.name}_rdkit_all.pkl')
            with open(conf_save_path, 'wb') as handle:
                pickle.dump(self.all_moles, handle)
                
            conf_save_path_loader = os.path.join(self.save_dir, f'{self.name}_data_loader.pkl')
            with open(conf_save_path_loader, 'wb') as handle:
                pickle.dump(self.model_preds, handle)
                
            # import ipdb; ipdb.set_trace()
            self.datapoints = []
            for k, v in self.model_preds.items():
                if v[0] == None:
                    continue
                self.datapoints.extend([(k, vv) for vv in v])
            self.smiles = [x[0] for x in self.datapoints]
            data = [x[1] for x in self.datapoints]
            print(f"Generating {len(data)} molecules")
            self.dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size= self.batch_size, shuffle=False, drop_last=False, num_workers=0 ,collate_fn = collate)
            
        molecules = []
        distances = []
        xcount = 1
        total = len(self.dataloader)
        with torch.no_grad():
            for A_batch, B_batch in self.dataloader:
                A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
                B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids= B_batch
                print(f"Batch {xcount} out of {total}")
                xcount += 1
                A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
                B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')
                generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, AR_loss = self.model(
                        B_frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=0, validation = True)
                molecules.extend(dgl.unbatch(generated_molecule.cpu()))
                distances.extend(dgl.unbatch(geo_A.cpu()))
        self.final_confs = defaultdict(list)
        self.generated_molecules = molecules
        self.generated_molecules_distances = distances
        try:
            for smi, data in zip(self.smiles, molecules):
                self.final_confs[smi].append(dgl_to_mol(copy.deepcopy(self.all_moles[smi][0][0]), data, mmff=False, rmsd=False, copy=True))
            conf_save_path = os.path.join(self.save_dir, f'{self.name}_coarsen_conf_all.pkl')
            with open(conf_save_path, 'wb') as handle:
                pickle.dump(self.final_confs, handle)
        except:
            import ipdb; ipdb.set_trace()
        return self.final_confs

    def generate(self, smile_list = None):
        if self.dataloader == None:
            self.model_preds = defaultdict(list)
            self.all_moles = defaultdict(list)
            # dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
            if type(smile_list) == str:
                smile_list = {smile_list: 1}
            for smi, count in smile_list.items():
                mol = Chem.MolFromSmiles(smi)
                atom_types = set([atom.GetSymbol() for atom in mol.GetAtoms()])
                if any (atom_type not in drugs_types for atom_type in atom_types):
                    print("Not DRUGS molecule")
                    continue
                # smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                # TODO: OVERHAUL THE DATALOADING PROCESS SINCE THIS IS SO SLOW!!!!!!!
                for k in range(count):
                    mol = self.get_rdkit_coords(smi)
                    # print(mol.GetConformer().GetPositions())
                    # import ipdb; ipdb.set_trace()
                    if mol == None:
                        print("Error in embedding", smi)
                        break
                    feats = featurize_mol(mol=mol, types=self.types)
                    A_frags, A_frag_ids, A_adj, A_out, A_bond_break, A_cg_bonds, A_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = False)
                    A_cg = conditional_coarsen_3d(feats, A_frag_ids, A_cg_map, A_bond_break, radius=4, max_neighbors=None, latent_dim_D = 64, latent_dim_F = 32)
                    geometry_graph_A = get_geometry_graph(mol)
                    Ap = create_pooling_graph(feats, A_frag_ids)
                    geometry_graph_A_cg = get_coarse_geometry_graph(A_cg, A_cg_map)
                    self.model_preds[smi].append((feats, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
                    self.all_moles[smi].append(mol)
                    
            conf_save_path = os.path.join(self.save_dir, f'{self.name}_rdkit_all.pkl')
            with open(conf_save_path, 'wb') as handle:
                pickle.dump(self.all_moles, handle)
            self.datapoints = []
            for k, v in self.model_preds.items():
                if v[0] == None:
                    continue
                self.datapoints.extend([(k, vv) for vv in v])
            self.smiles = [x[0] for x in self.datapoints]
            data = [x[1] for x in self.datapoints]
            print(f"Generating {len(data)} molecules")
            self.dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size= self.batch_size, shuffle=False, drop_last=False, num_workers=0 ,collate_fn = collate)
            
        molecules = []
        distances = []
        count = 1
        total = len(self.dataloader)
        with torch.no_grad():
            for A_batch, B_batch in self.dataloader:
                A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
                B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids= B_batch
                print(f"Batch {count} out of {total}")
                count+= 1
                # A_cg.ndata['v'] = torch.zeros((A_cg.ndata['v'].shape[0], self.F, 3))
                # B_cg.ndata['v'] = torch.zeros((B_cg.ndata['v'].shape[0], self.F, 3))
                A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
                B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')
                # import ipdb; ipdb.set_trace()
                generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, AR_loss = self.model(
                        B_frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=0, validation = True)
                molecules.extend(dgl.unbatch(generated_molecule.cpu()))
                distances.extend(dgl.unbatch(geo_A.cpu()))
        self.final_confs = defaultdict(list)
        # self.final_confs_rdkit = defaultdict(list)
        self.generated_molecules = molecules
        self.generated_molecules_distances = distances
        for smi, data in zip(self.smiles, molecules):
            self.final_confs[smi].append(dgl_to_mol(copy.deepcopy(self.all_moles[smi][0]), data, mmff=False, rmsd=False, copy=True))
        conf_save_path = os.path.join(self.save_dir, f'{self.name}_coarsen_conf_all.pkl')
        with open(conf_save_path, 'wb') as handle:
            pickle.dump(self.final_confs, handle)
        return self.final_confs

        
                    
    def get_rdkit_coords(self, smi, seed = None, use_mmff = True):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
        except:
            print(f"Could not use mmff {smi}")
            return None
        return mol
    
    def get_rdkit_coords_bulk(self, smi, count, seed = None, use_mmff = True, threads = 16):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        # AllChem.EmbedMolecule(mol)
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=count, numThreads=threads)
        try:
            # AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s')
            res = AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s', numThreads = threads)
        except:
            print(f"Could not use mmff {smi}")
            return None, None
        # rmslist = []
        # AllChem.AlignMolConformers(mol, RMSlist=rmslist)
        # print(rmslist)
        return mol, (cids, res)

if __name__ == "__main__":
    coarsenConf = CoarsenConf()
    with open("your_smiles.pkl", 'rb') as f:
        smiles_set = pickle.load(f)
    frequency = 100
    smiles_dict = {}
    for smi in smiles_set:
        smiles_dict[smi] = frequency
    results = coarsenConf.generate_parallel(smiles_dict)
    print("Done")
    