import pickle, random
from argparse import ArgumentParser
from multiprocessing import Pool
import numpy as np
import pandas as pd
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from tqdm import tqdm
import wandb, copy
from utils.torsional_diffusion_data_all import featurize_mol, qm9_types, drugs_types, get_transformation_mask, check_distances
from molecule_utils import *
import dgl
from collections import defaultdict
from tqdm import tqdm

# parser = ArgumentParser()
# parser.add_argument('--confs', type=str, required=True, help='Path to pickle file with generated conformers')
# parser.add_argument('--test_csv', type=str, default='./data/DRUGS/test_smiles_corrected.csv', help='Path to csv file with list of smiles')
# parser.add_argument('--true_mols', type=str, default='./data/DRUGS/test_mols.pkl', help='Path to pickle file with ground truth conformers')
# parser.add_argument('--n_workers', type=int, default=1, help='Numer of parallel workers')
# parser.add_argument('--limit_mols', type=int, default=0, help='Limit number of molecules, 0 to evaluate them all')
# parser.add_argument('--dataset', type=str, default="drugs", help='Dataset: drugs, qm9 and xl')
# parser.add_argument('--filter_mols', type=str, default=None, help='If set, is path to list of smiles to test')
# parser.add_argument('--only_alignmol', action='store_true', default=False, help='If set instead of GetBestRMSD, it uses AlignMol (for large molecules)')
# args = parser.parse_args()

"""
    Evaluates the RMSD of some generated conformers w.r.t. the given set of ground truth
    Part of the code taken from GeoMol https://github.com/PattanaikL/GeoMol
"""
# with open(args.confs, 'rb') as f:
#     model_preds = pickle.load(f)

# test_data = pd.read_csv(args.test_csv)  # this should include the corrected smiles
# with open(args.true_mols, 'rb') as f:
#     true_mols = pickle.load(f)
# threshold = threshold_ranges = np.arange(0, 2.5, .125)
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
    # try:
    #     if rmsd:
    #         mol.rmsd = AllChem.GetBestRMS(
    #             Chem.RemoveHs(data.seed_mol),
    #             Chem.RemoveHs(mol)
    #         )
    #     # mol.total_perturb = data.total_perturb
    # except:
    #     pass
    # mol.n_rotable_bonds = data.edge_mask.sum()
    return mol
    # if not copy: return mol
    # import ipdb; ipdb.set_trace()
    # return copy.deepcopy(mol)

def collate(samples):
    A, B = map(list, zip(*samples))
    A_graph = dgl.batch([x[0] for x in A])
    geo_A = dgl.batch([x[1] for x in A])
    Ap = dgl.batch([x[2] for x in A])
    A_cg = dgl.batch([x[3] for x in A])
    geo_A_cg = dgl.batch([x[4] for x in A])
    frag_ids = [x[5] for x in A]
    #
    B_graph = dgl.batch([x[0] for x in B])
    geo_B = dgl.batch([x[1] for x in B])
    Bp = dgl.batch([x[2] for x in B])
    B_cg = dgl.batch([x[3] for x in B])
    geo_B_cg = dgl.batch([x[4] for x in B])
    B_frag_ids = [x[5] for x in B]
    return (A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids), (B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids)

class BenchmarkRunner():
    def __init__(self, true_mols = '/data/QM9/test_mols.pkl', valid_mols = '/data/QM9/test_smiles.csv', n_workers = 1, dataset = 'qm9',
                 D = 64, F = 32, save_dir = '/data/QM9/test_set', batch_size = 2000, name = None):
    # def __init__(self, true_mols = '/data/torsional_diffusion/QM9/test_mols.pkl', valid_mols = '/data/torsional_diffusion/QM9/test_smiles.csv', n_workers = 1, dataset = 'qm9',
    #              D = 64, F = 32, save_dir = '/data/torsional_diffusion/QM9', batch_size = 2000):
        with open(true_mols, 'rb') as f:
            self.true_mols = pickle.load(f)
        self.threshold = np.arange(0, 2.5, .125)
        self.test_data = pd.read_csv(valid_mols)
        self.dataset = dataset
        self.n_workers = 1
        self.D, self.F = D, F
        if name is None:
            self.name = f'{dataset}_full'#_full_V3_check # dataste _80
        else:
            self.name = name
        self.batch_size = batch_size
        self.only_alignmol = False
        if dataset == 'xl':
            self.only_alignmol = True
        self.save_dir = save_dir
        self.types = qm9_types if dataset == 'qm9' else drugs_types
        self.use_diffusion_angle_def = False
        if self.dataset == 'qm9':
            self.use_check = False
        if self.has_cache():
            self.clean_true_mols()
            self.smiles, data = self.load()
            print(f"{len(data)} Conformers Loaded")
        else:
            print("NO CACHE \n\n\n\n\n")
            self.build_test_dataset_V2()
            self.save() 
            self.smiles = [x[0] for x in self.datapoints]
            data = [x[1] for x in self.datapoints]
        # torch.multiprocessing.set_sharing_strategy('file_system')
        self.dataloader = dgl.dataloading.GraphDataLoader(data, use_ddp=False, batch_size= self.batch_size, shuffle=False, drop_last=False, num_workers=0 ,collate_fn = collate)
    
    def clean_true_mols(self):
        for smi_idx, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
            # if smi_idx < 950:
                # continue
            raw_smi = row['smiles']
            n_confs = row['n_conformers']
            smi = row['corrected_smiles']
            if self.dataset == 'xl':
                raw_smi = smi
            # self.true_mols[raw_smi] = true_confs = self.clean_confs(smi, self.true_mols[raw_smi])
            self.true_mols[smi] = true_confs = self.clean_confs(smi, self.true_mols[raw_smi])
    
    def build_test_dataset_V2(self, confs_per_mol = None):
        self.model_preds = defaultdict(list)
        self.problem_smiles = set()
        dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
        print("Buidling Test Set ...")
        for smi_idx, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
            raw_smi = row['smiles']
            n_confs = row['n_conformers']
            smi = row['corrected_smiles']
            if self.dataset == 'xl':
                raw_smi = smi
            self.true_mols[smi] = true_confs = self.clean_confs(smi, self.true_mols[raw_smi])
            if self.use_check and smi in self.check:
                print(smi, " in precalc'd errors of me and torsional diffusion")
                continue
            if len(true_confs) == 0:
                print(f'poor ground truth conformers: {smi}')
                self.model_preds[smi] = [None]
                self.problem_smiles.add(smi)
                continue
            # if true_confs[0].GetNumAtoms() < 80: # skip and not punish the small molecules
            #     self.model_preds[smi] = [None]
            #     self.problem_smiles.add(smi)
            #     continue
            
            datas = self.featurize_mol(smi, true_confs)
            bad_idx_A = []
            results_A = []
            # print("\n\n\n Generating:", smi)
            for idx, data in enumerate(datas):
                mol = true_confs[idx]
                edge_mask, mask_rotate = get_transformation_mask(mol, data)
                if not mol.HasSubstructMatch(dihedral_pattern):# or np.sum(edge_mask) < 0.5: # if no rotatable bonds
                    bad_idx_A.append(idx)
                    results_A.append(None)
                    print("No rotatable bonds", idx, len(true_confs), smi)
                    continue
                try:
                    A_frags, A_frag_ids, A_adj, A_out, A_bond_break, A_cg_bonds, A_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = self.use_diffusion_angle_def)
                except Exception as e:
                    print(e)
                    print("Coarsen Issue", idx, smi)
                    bad_idx_A.append(idx)
                    results_A.append(None)
                try:
                    A_cg = conditional_coarsen_3d(data, A_frag_ids, A_cg_map, A_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
                except Exception as e:
                    print(e)
                    print("CG Graph Creation Error", idx, smi)
                    bad_idx_A.append(idx)
                    results_A.append(None)
                    continue
                geometry_graph_A = get_geometry_graph(mol)
                Ap = create_pooling_graph(data, A_frag_ids)
                geometry_graph_A_cg = get_coarse_geometry_graph(A_cg, A_cg_map)
                results_A.append((data, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
            if len(results_A) == len(bad_idx_A):
                continue
            # import ipdb; ipdb.set_trace()
            B_confs = copy.deepcopy(true_confs) + copy.deepcopy(true_confs)
            B_confs = [Chem.AddHs(x) for x in B_confs]
            try:
                assert(sum([x.ndata['x'].shape[0] for x in datas]) == sum([x.GetNumAtoms() for x in B_confs])/2)
            except:
                ca = sum([x.ndata['x'].shape[0] for x in datas])
                cb = sum([x.GetNumAtoms() for x in B_confs])
                import ipdb; ipdb.set_trace()
                
            data_Bs = self.featurize_mol(smi, B_confs, use_rdkit_coords = True)
            bad_idx_B = []
            results_B = []
            for idx, data_B in enumerate(data_Bs):
                if idx in set(bad_idx_A):
                    bad_idx_B.append(idx)
                    results_B.append(None)
                    continue
                if not data_B:
                    print('B Graph Cannot RDKit Featurize', idx, len(B_confs), smi)
                    bad_idx_B.append(idx)
                    results_B.append(None)
                    continue
                mol = B_confs[idx] #true_confs[idx]
                B_frags, B_frag_ids, B_adj, B_out, B_bond_break, B_cg_bonds, B_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = self.use_diffusion_angle_def)
                B_cg = conditional_coarsen_3d(data_B, B_frag_ids, B_cg_map, B_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
                geometry_graph_B = get_geometry_graph(mol)
                Bp = create_pooling_graph(data_B, B_frag_ids)
                geometry_graph_B_cg = get_coarse_geometry_graph(B_cg, B_cg_map)
                err = check_distances(data_B, geometry_graph_B, True)
                assert( err < 1e-3 )
                results_B.append((data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
                
            try:
                assert(2*len(results_A) == len(results_B))
            except:
                print("mismatch A and B")
                import ipdb; ipdb.set_trace()
                
            bad_idx = set(bad_idx_A) | set(bad_idx_B)
            # import ipdb; ipdb.set_trace()
            results_A = [x for idx, x in enumerate(results_A) if idx not in bad_idx]
            results_B = [x for idx, x in enumerate(results_B) if idx not in bad_idx]
            
            point_clouds_array = np.array([x[0].ndata['x_ref'].numpy() for x in results_B])
            unique_point_clouds_array = np.unique(point_clouds_array, axis=0)
            num_unique_point_clouds = unique_point_clouds_array.shape[0]
            # print("Unique RDKit", num_unique_point_clouds)
            if num_unique_point_clouds != len(results_B):
                import ipdb; ipdb.set_trace()

            count = 0
            first = results_B[:len(results_A)]
            second = results_B[len(results_A):]
            # print(len(results_A), len(results_B))
            for a,b in zip(results_A, first):
                assert(a is not None and b is not None)
                self.model_preds[smi].append((a,b))
                if count >= len(second):
                    c = copy.deepcopy(first[0])
                else:
                    c = second[count]
                self.model_preds[smi].append((copy.deepcopy(a), c))
                count += 1
                
        self.datapoints = []
        for k, v in self.model_preds.items():
            if v[0] == None:
                continue
            self.datapoints.extend([(k, vv) for vv in v])
        print('Fetched', len(self.datapoints), 'mols successfully')
        # print('Example', self.datapoints[0])
         
    # def build_test_dataset(self, confs_per_mol = None):
    #     self.model_preds = defaultdict(list)
    #     self.problem_smiles = set()
    #     dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
    #     # import ipdb; ipdb.set_trace()
    #     print("Buidling Test Set ...")
    #     for smi_idx, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
    #         # if smi_idx < 950:
    #             # continue
    #         raw_smi = row['smiles']
    #         n_confs = row['n_conformers']
    #         smi = row['corrected_smiles']
    #         if self.dataset == 'xl':
    #             raw_smi = smi
    #         # self.true_mols[raw_smi] = true_confs = self.clean_confs(smi, self.true_mols[raw_smi])
    #         self.true_mols[smi] = true_confs = self.clean_confs(smi, self.true_mols[raw_smi])
    #         # import ipdb; ipdb.set_trace()
    #         if len(true_confs) == 0:
    #             print(f'poor ground truth conformers: {smi}')
    #             self.model_preds[smi] = [None]
    #             self.problem_smiles.add(smi)
    #             continue
    #         # if confs_per_mol:
    #         #     duplicate = confs_per_mol
    #         # else:
    #         #     duplicate = 2*len(true_confs) #n_confs
    #         # moles = [self.true_mols[raw_smi]] * duplicate
    #         # moles = []
    #         # for mol in true_confs:
    #         #     moles.extend([mol, mol])
    #         # datas = self.featurize_mol(smi, moles)
    #         datas = self.featurize_mol(smi, true_confs)
    #         # datas = []
    #         # for d in datas_:
    #         #     datas.extend([d, copy.deepcopy(d)])
    #         # import ipdb; ipdb.set_trace()
    #         bad_idx_A = []
    #         results_A = []
    #         print("\n\n\n Generating:", smi)
    #         for idx, data in enumerate(datas):
    #             mol = true_confs[idx]
    #             edge_mask, mask_rotate = get_transformation_mask(mol, data)
    #             if not mol.HasSubstructMatch(dihedral_pattern) or np.sum(edge_mask) < 0.5: # if no rotatable bonds
    #             # if np.sum(edge_mask) < 0.5: 
    #                 bad_idx_A.append(idx)
    #                 results_A.append(None)
    #                 # self.problem_smiles.add(smi)
    #                 # ok_check = mol.HasSubstructMatch(dihedral_pattern)
    #                 # if ok_check:
    #                 #     print(idx, smi, "should not skip")
    #                 print("No rotatable bonds", idx, smi)
    #                 continue
    #             try:
    #                 A_frags, A_frag_ids, A_adj, A_out, A_bond_break, A_cg_bonds, A_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = self.use_diffusion_angle_def)
    #             except Exception as e:
    #                 print(e)
    #                 print("Coarsen Issue", idx, smi)
    #                 bad_idx_A.append(idx)
    #                 results_A.append(None)
    #             try:
    #                 A_cg = conditional_coarsen_3d(data, A_frag_ids, A_cg_map, A_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
    #             except Exception as e:
    #                 print(e)
    #                 print("CG Graph Creation Error", idx, smi)
    #                 bad_idx_A.append(idx)
    #                 results_A.append(None)
    #                 # self.problem_smiles.add(smi)
    #                 continue
    #                 # import ipdb; ipdb.set_trace()
    #                 # A_frags, A_frag_ids, A_adj, A_out, A_bond_break, A_cg_bonds, A_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = self.use_diffusion_angle_def)
    #                 # A_cg = conditional_coarsen_3d(data, A_frag_ids, A_cg_map, A_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
    #             geometry_graph_A = get_geometry_graph(mol)
    #             # err = check_distances(data, geometry_graph_A)
    #             # if err.item() > 1e-3:
    #             #     import ipdb; ipdb.set_trace()
    #             #     data = self.featurize_mol(mol_dic)
    #             Ap = create_pooling_graph(data, A_frag_ids)
    #             geometry_graph_A_cg = get_coarse_geometry_graph(A_cg, A_cg_map)
    #             results_A.append((data, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids))
    #         # import ipdb; ipdb.set_trace()
    #         if len(results_A) == len(bad_idx_A):
    #             # self.model_preds[smi] = [None]
    #             # self.problem_smiles.add(smi)
    #             continue
    #         # TODO: featurize with 2* true confs then filter for 2x then split into two final results
    #         # TODO if not generated 2x try again
    #         B_confs = copy.deepcopy(true_confs)
    #         data_Bs = self.featurize_mol(smi, B_confs, use_rdkit_coords = True)
    #         bad_idx_B = []
    #         results_B = []
    #         for idx, data_B in enumerate(data_Bs):
    #             if idx in set(bad_idx_A):
    #                 bad_idx_B.append(idx)
    #                 results_B.append(None)
    #                 continue
    #             if not data_B:
    #                 print('B Graph Cannot RDKit Featurize', idx, smi)
    #                 bad_idx_B.append(idx)
    #                 results_B.append(None)
    #                 # self.problem_smiles.add(smi)
    #                 continue
    #                 # return False
    #             mol = B_confs[idx] #true_confs[idx]
    #             B_frags, B_frag_ids, B_adj, B_out, B_bond_break, B_cg_bonds, B_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = self.use_diffusion_angle_def)
    #             B_cg = conditional_coarsen_3d(data_B, B_frag_ids, B_cg_map, B_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
    # #             geometry_graph_B = copy.deepcopy(geometry_graph_A) #get_geometry_graph(mol)
    #             geometry_graph_B = get_geometry_graph(mol)
    #             Bp = create_pooling_graph(data_B, B_frag_ids)
    #             geometry_graph_B_cg = get_coarse_geometry_graph(B_cg, B_cg_map)
    #             err = check_distances(data_B, geometry_graph_B, True)
    #             assert( err < 1e-3 )
    #             # if err.item() > 1e-3:
    #             #     import ipdb; ipdb.set_trace()
    #             #     data_B = self.featurize_mol(mol_dic, use_rdkit_coords = True)
    #             results_B.append((data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
            
    #         # bad_idx_B2 = []
    #         # results_B2 = []
    #         # B2_confs = copy.deepcopy(true_confs)
    #         # data_Bs2 = self.featurize_mol(smi, B2_confs, use_rdkit_coords = True)
    #         # for idx, data_B in enumerate(data_Bs2):
    #         #     if idx in set(bad_idx_A):
    #         #         results_B2.append(None)
    #         #         continue
    #         #     if idx in set(bad_idx_B):
    #         #         bad_idx_B2.append(idx)
    #         #         results_B2.append(None)
    #         #         continue
    #         #     if not data_B:
    #         #         print('B Graph2 Cannot RDKit Featurize [CHECK]', idx, smi)
    #         #         bad_idx_B2.append(idx)
    #         #         results_B2.append(None)
    #         #         continue
    #         #     mol = B2_confs[idx]
    #         #     B_frags, B_frag_ids, B_adj, B_out, B_bond_break, B_cg_bonds, B_cg_map = coarsen_molecule(mol, use_diffusion_angle_def = self.use_diffusion_angle_def)
    #         #     B_cg = conditional_coarsen_3d(data_B, B_frag_ids, B_cg_map, B_bond_break, radius=4, max_neighbors=None, latent_dim_D = self.D, latent_dim_F = self.F)
    #         #     geometry_graph_B = get_geometry_graph(mol)
    #         #     Bp = create_pooling_graph(data_B, B_frag_ids)
    #         #     geometry_graph_B_cg = get_coarse_geometry_graph(B_cg, B_cg_map)
    #         #     results_B2.append((data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids))
                
    #         try:
    #             assert(len(results_A) == len(results_B))
    #         except:
    #             import ipdb; ipdb.set_trace()
            
    #         # try:
    #         #     assert(set(bad_idx_B) == set(bad_idx_B2))
    #         # except:
    #         #     import ipdb; ipdb.set_trace()
                
    #         bad_idx = set(bad_idx_A) | set(bad_idx_B)
    #         # import ipdb; ipdb.set_trace()
    #         results_A = [x for idx, x in enumerate(results_A) if idx not in bad_idx]
    #         results_B = [x for idx, x in enumerate(results_B) if idx not in bad_idx]
    #         # results_B2 = [x for idx, x in enumerate(results_B2) if idx not in bad_idx]
    #         assert(len(results_A) == len(results_B))
    #         # if len(results_A) == 0 or len(results_B) == 0:
    #         #     self.model_preds[smi] = [None]

    #         count = 0
    #         for a,b in zip(results_A, results_B):
    #             self.model_preds[smi].append((a,b))
    #             self.model_preds[smi].append((copy.deepcopy(a), copy.deepcopy(b)))
    #             count += 1
                
    #     self.datapoints = []
    #     for k, v in self.model_preds.items():
    #         if v[0] == None:
    #             continue
    #         self.datapoints.extend([(k, vv) for vv in v])
    #     print('Fetched', len(self.datapoints), 'mols successfully')
    #     # print('Example', self.datapoints[0])
        
    def featurize_mol(self, smi, moles, use_rdkit_coords = False):
        name = smi
        mol_ = Chem.MolFromSmiles(name)
        canonical_smi = Chem.MolToSmiles(mol_, isomericSmiles=False)
        datas = []
        early_kill = False
        for mol in moles:
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
            except Exception as e:
                print(e)
                continue
            if conf_canonical_smi != canonical_smi or early_kill:
                datas.append(None)
                continue
            # pos.append(torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float))
            correct_mol = mol
            check_ = correct_mol.GetConformer().GetPositions()
            mol_features = featurize_mol(correct_mol, self.types, use_rdkit_coords = use_rdkit_coords)
            # mol_features = featurize_mol(correct_mol, self.types, use_rdkit_coords = use_rdkit_coords, use_mmff = False)
            if mol_features is not None:
                try:
                    cmpt = correct_mol.GetConformer().GetPositions()
                    if use_rdkit_coords:
                        # a = mol_features.ndata['x_true'].numpy() #cmpt-np.mean(cmpt, axis = 0)
                        # b = mol_features.ndata['x_ref'].numpy()
                        # if np.mean((a - b) ** 2) < 1e-5:
                        #     import ipdb; ipdb.set_trace()
                        # assert(np.mean((a - b) ** 2) < 1e-7 ) # This fails since the featurization aligns the rdkit so the MSE is not preserved
                        cmpt = check_
                    a = cmpt-np.mean(cmpt, axis = 0)
                    b = mol_features.ndata['x_true'].numpy()
                    assert(np.mean((a - b) ** 2) < 1e-7 )
                except:
                    import ipdb; ipdb.set_trace()
                    mol_features = featurize_mol(correct_mol, self.types, use_rdkit_coords = use_rdkit_coords)
                
            datas.append(mol_features)
            # if mol_features is None:
            #     print(f"Skipping {len(moles)-1} since I am {use_rdkit_coords} using RDKit and I am getting a Featurization error")
            #     early_kill = True
        # if use_rdkit_coords:
        #     import ipdb; ipdb.set_trace()
        return datas
            
    def generate(self, model, rdkit_only = False, save = False, use_wandb = True):
        if not rdkit_only:
            if save and os.path.exists(os.path.join(self.save_dir, f'{self.name}_random_weights_gen.bin')):
                molecules, _ = dgl.data.utils.load_graphs(self.save_dir + f'/{self.name}_random_weights_gen.bin')
            else:
                molecules = []
                distances = []
                with torch.no_grad():
                    for A_batch, B_batch in self.dataloader:
                        A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
                        B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids= B_batch
                        # A_cg.ndata['v'] = torch.zeros((A_cg.ndata['v'].shape[0], self.F, 3))
                        # B_cg.ndata['v'] = torch.zeros((B_cg.ndata['v'].shape[0], self.F, 3))
                        A_graph, geo_A, Ap, A_cg, geo_A_cg = A_graph.to('cuda:0'), geo_A.to('cuda:0'), Ap.to('cuda:0'), A_cg.to('cuda:0'), geo_A_cg.to('cuda:0')
                        B_graph, geo_B, Bp, B_cg, geo_B_cg = B_graph.to('cuda:0'), geo_B.to('cuda:0'), Bp.to('cuda:0'), B_cg.to('cuda:0'), geo_B_cg.to('cuda:0')
                        # import ipdb; ipdb.set_trace()
                        generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, AR_loss = model(
                                B_frag_ids, A_graph, B_graph, geo_A, geo_B, Ap, Bp, A_cg, B_cg, geo_A_cg, geo_B_cg, epoch=0, validation = True)
                        # ipdb.set_trace()
                        # loss, losses = model.loss_function(generated_molecule, rdkit_reference, dec_results, channel_selection_info, KL_terms, enc_out, geo_A, AR_loss, step=0, log_latent_stats = False)
                        # train_loss_log.append(losses)
                        # losses['Test Loss'] = loss.cpu()
                        # wandb.log({'test_' + key: value for key, value in losses.items()})
                        molecules.extend(dgl.unbatch(generated_molecule.cpu()))
                        distances.extend(dgl.unbatch(geo_A.cpu()))
                if save:
                    dgl.data.utils.save_graphs(self.save_dir + f'/{self.name}_random_weights_gen.bin', molecules)
            self.final_confs = defaultdict(list)
            # self.final_confs_rdkit = defaultdict(list)
            self.generated_molecules = molecules
            self.generated_molecules_distances = distances
            for smi, data in zip(self.smiles, molecules):
                self.final_confs[smi].append(dgl_to_mol(copy.deepcopy(self.true_mols[smi][0]), data, mmff=False, rmsd=False, copy=True))
                # self.final_confs_rdkit[smi].append(dgl_to_mol(copy.deepcopy(self.true_mols[smi][0]), data, mmff=False, rmsd=False, copy=True, key = 'x_ref'))
            self.results_model = self.calculate(self.final_confs, use_wandb = use_wandb)
            # self.results_rdkit = self.calculate(self.final_confs_rdkit)
        else:
            molecules = []
            distances = []
            for A_batch, B_batch in self.dataloader:
                # A_graph, geo_A, Ap, A_cg, geo_A_cg, frag_ids = A_batch
                B_graph, geo_B, Bp, B_cg, geo_B_cg, B_frag_ids = B_batch
                # B_cg.ndata['v'] = torch.zeros((B_cg.ndata['v'].shape[0], self.F, 3))
                molecules.extend(dgl.unbatch(B_graph.cpu()))
                distances.extend(dgl.unbatch(geo_B.cpu()))
            self.rdkit_molecules = molecules
            self.rdkit_molecules_distances = distances
            self.final_confs_rdkit = defaultdict(list)
            for smi, data in zip(self.smiles, molecules):
                # if self.flag_molecule(smi, data):
                #     continue
                self.final_confs_rdkit[smi].append(dgl_to_mol(copy.deepcopy(self.true_mols[smi][0]), data, mmff=False, rmsd=False, copy=True, key = 'x_ref'))
            self.results_rdkit = self.calculate(self.final_confs_rdkit, use_wandb= use_wandb)
                
            
        
        
    def calculate(self, final_confs, use_wandb = True):
        rdkit_smiles = self.test_data.smiles.values
        corrected_smiles = self.test_data.corrected_smiles.values
        self.num_failures = 0
        results = {}
        jobs = []
        # for smi, corrected_smi in tqdm(zip(rdkit_smiles, corrected_smiles)):
        for smi_idx, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
            # if smi_idx < 950:
                # continue
            smi = row['smiles']
            n_confs = row['n_conformers']
            corrected_smi = row['corrected_smiles']
            if self.dataset == 'xl':
                smi = corrected_smi
            
            true_confs = self.true_mols[corrected_smi]
            if len(true_confs) == 0:
                print(f'poor ground truth conformers: {corrected_smi}')
                continue
            
            if corrected_smi not in final_confs:
                if corrected_smi not in self.problem_smiles:
                    self.num_failures += 1
                    print('Cannot feed into Model', corrected_smi)
                else:
                    print('problematic ground truth not caught early on [CHECK]', corrected_smi)
                continue

            # true_mols[smi] = true_confs = self.clean_confs(corrected_smi, true_mols[smi])
            # true_confs = self.true_mols[smi]

            n_true = len(true_confs)
            n_model = len(final_confs[corrected_smi])
            results[(smi, corrected_smi)] = {
                'n_true': n_true,
                'n_model': n_model,
                'rmsd': np.nan * np.ones((n_true, n_model))
            }
            # self.results = results
            for i_true in range(n_true):
                jobs.append((smi, corrected_smi, i_true))
                
        self.results = results
        random.shuffle(jobs)
        if self.n_workers > 1:
            p = Pool(self.n_workers)
            map_fn = p.imap_unordered
            p.__enter__()
        else:
            map_fn = map
        self.final_confs_temp = final_confs
        conf_save_path = os.path.join(self.save_dir, f'{self.name}_final_confs_gen3_rmsd_fix_2.pkl')
        with open(conf_save_path, 'wb') as handle:
            pickle.dump(final_confs, handle)
            
        for res in tqdm(map_fn(self.worker_fn, jobs), total=len(jobs)):
            self.populate_results(res)

        if self.n_workers > 1:
            p.__exit__(None, None, None)
        # ! previous code that worked below
        # self.final_confs_temp = final_confs
        # for job in tqdm(jobs, total=len(jobs)):
        #     self.populate_results(self.worker_fn(job))
        self.run(results, reduction='min', use_wandb = use_wandb)
        self.run(results, reduction='max', use_wandb = use_wandb)
        self.run(results, reduction='mean', use_wandb = use_wandb)
        self.run(results, reduction='std', use_wandb = use_wandb)
        # import ipdb; ipdb.set_trace()
        # self.run(results, reduction='min', use_wandb = use_wandb)
        return results
        

    def calc_performance_stats(self, rmsd_array):
        coverage_recall = np.mean(rmsd_array.min(axis=1, keepdims=True) < self.threshold, axis=0)
        amr_recall = rmsd_array.min(axis=1).mean()
        coverage_precision = np.mean(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(self.threshold, 1), axis=1)
        amr_precision = rmsd_array.min(axis=0).mean()
        return coverage_recall, amr_recall, coverage_precision, amr_precision
    
    def calc_performance_stats_max(self, rmsd_array):
        coverage_recall = np.mean(rmsd_array.max(axis=1, keepdims=True) < self.threshold, axis=0)
        amr_recall = rmsd_array.max(axis=1).mean()
        coverage_precision = np.mean(rmsd_array.max(axis=0, keepdims=True) < np.expand_dims(self.threshold, 1), axis=1)
        amr_precision = rmsd_array.max(axis=0).mean()
        return coverage_recall, amr_recall, coverage_precision, amr_precision
    
    def calc_performance_stats_mean(self, rmsd_array):
        coverage_recall = np.mean(rmsd_array.mean(axis=1, keepdims=True) < self.threshold, axis=0)
        amr_recall = rmsd_array.mean(axis=1).mean()
        coverage_precision = np.mean(rmsd_array.mean(axis=0, keepdims=True) < np.expand_dims(self.threshold, 1), axis=1)
        amr_precision = rmsd_array.mean(axis=0).mean()
        return coverage_recall, amr_recall, coverage_precision, amr_precision
    
    def calc_performance_stats_std(self, rmsd_array):
        coverage_recall = np.mean(rmsd_array.std(axis=1, keepdims=True) < self.threshold, axis=0)
        amr_recall = rmsd_array.std(axis=1).mean()
        coverage_precision = np.mean(rmsd_array.std(axis=0, keepdims=True) < np.expand_dims(self.threshold, 1), axis=1)
        amr_precision = rmsd_array.std(axis=0).mean()
        return coverage_recall, amr_recall, coverage_precision, amr_precision

    def clean_confs(self, smi, confs):
        good_ids = []
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
        for i, c in enumerate(confs):
            conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
            if conf_smi == smi:
                good_ids.append(i)
        return [confs[i] for i in good_ids]

    def run(self, results, reduction = 'min', use_wandb = True):
        stats = []
        for res in results.values():
            if reduction == 'min':
                stats_ = self.calc_performance_stats(res['rmsd'])
            elif reduction == 'max':
                stats_ = self.calc_performance_stats_max(res['rmsd'])
            elif reduction == 'mean':
                stats_ = self.calc_performance_stats_mean(res['rmsd'])
            elif reduction == 'std':
                stats_ = self.calc_performance_stats_std(res['rmsd'])
            cr, mr, cp, mp = stats_
            stats.append(stats_)
        coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)
        for i, thresh in enumerate(self.threshold):
            coverage_recall_vals = [stat[i] for stat in coverage_recall] + [0] * self.num_failures
            coverage_precision_vals = [stat[i] for stat in coverage_precision] + [0] * self.num_failures
            if use_wandb:
                wandb.log({
                    f'{reduction} {thresh} Recall Coverage Mean': np.mean(coverage_recall_vals) * 100,
                    f'{reduction} {thresh} Recall Coverage Median': np.median(coverage_recall_vals) * 100,
                    f'{reduction} {thresh} Recall AMR Mean': np.nanmean(amr_recall),
                    f'{reduction} {thresh} Recall AMR Median': np.nanmedian(amr_recall),
                    f'{reduction} {thresh} Precision Coverage Mean': np.mean(coverage_precision_vals) * 100,
                    f'{reduction} {thresh} Precision Coverage Median': np.median(coverage_precision_vals) * 100,
                    f'{reduction} {thresh} Precision AMR Mean': np.nanmean(amr_precision),
                    f'{reduction} {thresh} Precision AMR Median': np.nanmedian(amr_precision),
                })
            else:
                print({
                    f'{reduction} {thresh} Recall Coverage Mean': np.mean(coverage_recall_vals) * 100,
                    f'{reduction} {thresh} Recall Coverage Median': np.median(coverage_recall_vals) * 100,
                    f'{reduction} {thresh} Recall AMR Mean': np.nanmean(amr_recall),
                    f'{reduction} {thresh} Recall AMR Median': np.nanmedian(amr_recall),
                    f'{reduction} {thresh} Precision Coverage Mean': np.mean(coverage_precision_vals) * 100,
                    f'{reduction} {thresh} Precision Coverage Median': np.median(coverage_precision_vals) * 100,
                    f'{reduction} {thresh} Precision AMR Mean': np.nanmean(amr_precision),
                    f'{reduction} {thresh} Precision AMR Median': np.nanmedian(amr_precision),
                })
        if use_wandb:
            wandb.log({
                f'{reduction} Conformer Sets Compared': len(results),
                f'{reduction} Model Failures': self.num_failures,
                f'{reduction} Additional Failures': np.isnan(amr_recall).sum()
            }) # can replace wandb log with
        else:
            print({
                f'{reduction} Conformer Sets Compared': len(results),
                f'{reduction} Model Failures': self.num_failures,
                f'{reduction} Additional Failures': np.isnan(amr_recall).sum()
            }) # can replace wandb log with
        return True
    
    def save(self):
        graphs, infos = [], []
        smiles = [x[0] for x in self.datapoints]
        data = [x[1] for x in self.datapoints]
        for A, B in data:
            data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, A_frag_ids = A
            data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg, B_frag_ids = B
            infos.append((A_frag_ids, B_frag_ids))
            graphs.extend([data_A, geometry_graph_A, Ap, A_cg, geometry_graph_A_cg, data_B, geometry_graph_B, Bp, B_cg, geometry_graph_B_cg])
        dgl.data.utils.save_info(self.save_dir + f'/{self.name}_infos.bin', infos)
        dgl.data.utils.save_info(self.save_dir + f'/{self.name}_smiles.bin', smiles)
        dgl.data.utils.save_info(self.save_dir + f'/{self.name}_problem_smiles.bin', self.problem_smiles)
        dgl.data.utils.save_graphs(self.save_dir + f'/{self.name}_graphs.bin', graphs)
        print("Saved Successfully", self.save_dir, self.name, len(self.datapoints))
    
    def load(self):
        print('Loading data ...')
        graphs, _ = dgl.data.utils.load_graphs(self.save_dir + f'/{self.name}_graphs.bin')
        info = dgl.data.utils.load_info(self.save_dir + f'/{self.name}_infos.bin')
        smiles = dgl.data.utils.load_info(self.save_dir + f'/{self.name}_smiles.bin')
        self.problem_smiles = dgl.data.utils.load_info(self.save_dir + f'/{self.name}_problem_smiles.bin')
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
        data = [(a,b) for a,b in zip(results_A, results_B)]
        print("Loaded Successfully",  self.save_dir, self.name, len(data))
        return smiles, data
    
    def has_cache(self):
         return os.path.exists(os.path.join(self.save_dir, f'{self.name}_graphs.bin'))
        
    def populate_results(self, res):
            smi, correct_smi, i_true, rmsds = res
            self.results[(smi, correct_smi)]['rmsd'][i_true] = rmsds
            
    def worker_fn(self, job):
            smi, correct_smi, i_true = job
            # true_confs = self.true_mols[smi]
            true_confs = self.true_mols[correct_smi]
            model_confs = self.final_confs_temp[correct_smi]
            tc = true_confs[i_true]
            rmsds = []
            # import ipdb; ipdb.set_trace()
            for mc in model_confs:
                try:
                    if self.only_alignmol:
                        rmsd = AllChem.AlignMol(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
                    else:
                        # rmsd = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
                        # tc_coords = torch.tensor(tc.GetConformer().GetPositions())
                        # mc_coords = torch.tensor(mc.GetConformer().GetPositions())
                        # tc_coords = self.align(tc_coords, mc_coords)
                        # rmsd = self.calculate_rmsd(tc_coords.numpy(), mc_coords.numpy())
                        a = tc.GetConformer().GetPositions()
                        b = mc.GetConformer().GetPositions()
                        err = np.mean((a - b) ** 2)
                        if err < 1e-7 :
                            print(f"[RMSD low crude error] {smi} {correct_smi} {i_true} = {err}")
                            import ipdb; ipdb.set_trace()
                        rmsd = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
                        if rmsd < 1e-2:
                            import ipdb; ipdb.set_trace()
                        # print("[Best RMSD , MSE ]", rmsd, err)
                    rmsds.append(rmsd)
                except:
                    print('Additional failure', smi, correct_smi)
                    rmsds = [np.nan] * len(model_confs)
                    break
            return smi, correct_smi, i_true, rmsds
    
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
                return source
                # import ipdb; ipdb.set_trace()
            U, S, Vt = torch.linalg.svd(A)
            # corr_mat = torch.diag(1e-7 + torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            rotation = (U @ corr_mat) @ Vt
            translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
        return (rotation @ lig_coords.t()).t() + translation
    
    def calculate_rmsd(self, array1, array2):
        # Calculate the squared differences
        squared_diff = np.square(array1 - array2)
        
        # Sum the squared differences along the axis=1
        sum_squared_diff = np.sum(squared_diff, axis=1)
        
        # Calculate the mean of the squared differences
        mean_squared_diff = np.mean(sum_squared_diff)
        
        # Calculate the square root of the mean squared differences
        rmsd = np.sqrt(mean_squared_diff)
        
        return rmsd
