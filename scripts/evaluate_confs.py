import pickle, random
from argparse import ArgumentParser
from multiprocessing import Pool
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import copy

parser = ArgumentParser()
parser.add_argument('--confs', type=str, required=True, help='Path to pickle file with generated conformers')
parser.add_argument('--test_csv', type=str, default='./data/DRUGS/test_smiles_corrected.csv', help='Path to csv file with list of smiles')
parser.add_argument('--true_mols', type=str, default='./data/DRUGS/test_mols.pkl', help='Path to pickle file with ground truth conformers')
parser.add_argument('--n_workers', type=int, default=1, help='Numer of parallel workers')
parser.add_argument('--limit_mols', type=int, default=0, help='Limit number of molecules, 0 to evaluate them all')
parser.add_argument('--dataset', type=str, default="drugs", help='Dataset: drugs, qm9 and xl')
parser.add_argument('--filter_mols', type=str, default=None, help='If set, is path to list of smiles to test')
parser.add_argument('--only_alignmol', action='store_true', default=False, help='If set instead of GetBestRMSD, it uses AlignMol (for large molecules)')
args = parser.parse_args()

"""
    Evaluates the RMSD of some generated conformers w.r.t. the given set of ground truth
    Part of the code taken from GeoMol https://github.com/PattanaikL/GeoMol
"""

with open(args.confs, 'rb') as f:
    model_preds = pickle.load(f)

test_data = pd.read_csv(args.test_csv)  # this should include the corrected smiles
with open(args.true_mols, 'rb') as f:
    true_mols = pickle.load(f)
threshold = threshold_ranges = np.arange(0, 2.5, .125)

def calc_performance_stats(rmsd_array):
    coverage_recall = np.mean(rmsd_array.min(axis=1, keepdims=True) < threshold, axis=0)
    amr_recall = rmsd_array.min(axis=1).mean()
    coverage_precision = np.mean(rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(threshold, 1), axis=1)
    amr_precision = rmsd_array.min(axis=0).mean()

    return coverage_recall, amr_recall, coverage_precision, amr_precision


def clean_confs(smi, confs):
    good_ids = []
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi, sanitize=False), isomericSmiles=False)
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c, sanitize=False), isomericSmiles=False)
        if conf_smi == smi:
            good_ids.append(i)
    return [confs[i] for i in good_ids]


rdkit_smiles = test_data.smiles.values
corrected_smiles = test_data.corrected_smiles.values

if args.limit_mols:
    rdkit_smiles = rdkit_smiles[:args.limit_mols]
    corrected_smiles = corrected_smiles[:args.limit_mols]

num_failures = 0
results = {}
jobs = []

filter_mols = None
if args.filter_mols:
    with open(args.filter_mols, 'rb') as f:
        filter_mols = pickle.load(f)
        
count = 0
for smi, corrected_smi in tqdm(zip(rdkit_smiles, corrected_smiles)):
    # import ipdb; ipdb.set_trace()
    if filter_mols is not None and corrected_smi not in filter_mols:
        continue
    
    # if smi == bad_smile or corrected_smi == bad_smile:
    #     print("BAD SMILE SKIP")
    #     continue
    
    if args.dataset == 'xl':
        smi = corrected_smi
    

    # if corrected_smi in check:
    #     print('precalculated failures', corrected_smi)
    #     num_failures += 1
    #     continue
    
    if corrected_smi not in model_preds:
        print('model failure', corrected_smi)
        num_failures += 1
        continue
    # continue #! check fialures
    true_mols[smi] = true_confs = clean_confs(corrected_smi, true_mols[smi])

    if len(true_confs) == 0:
        print(f'poor ground truth conformers: {corrected_smi}')
        continue
    
    # if true_confs[0].GetNumAtoms() < 80:
    #     print(f'too small: {corrected_smi}')
    #     continue

    n_true = len(true_confs)
    n_model = len(model_preds[corrected_smi])
    results[(smi, corrected_smi)] = {
        'n_true': n_true,
        'n_model': n_model,
        'rmsd': np.nan * np.ones((n_true, n_model)),
        'coords': {}
    }
    for i_true in range(n_true):
        jobs.append((smi, corrected_smi, i_true))
    
    # if count > 10:
    #     break
    # count += 1

print("Nim Failures", num_failures)
def worker_fn(job):
    smi, correct_smi, i_true = job
    true_confs = true_mols[smi]
    model_confs = model_preds[correct_smi]
    tc = true_confs[i_true]

    rmsds = []
    coords = []
    for mc in model_confs:
        try:
            if args.only_alignmol:
                rmsd = AllChem.AlignMol(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
            else:
                rmsd = AllChem.GetBestRMS(Chem.RemoveHs(tc), Chem.RemoveHs(mc))
            rmsds.append(rmsd)
            # coords.append((copy.deepcopy(tc), tc.GetConformer().GetPositions(), mc.GetConformer().GetPositions()))
        except:
            print('Additional failure', smi, correct_smi)
            rmsds = [np.nan] * len(model_confs)
            coords = [np.nan] * len(model_confs)
            break
        # coords.append((rmsd, tc.GetConformer().GetPositions(), mc.GetConformer().GetPositions()))
    return smi, correct_smi, i_true, rmsds, coords


def populate_results(res):
    smi, correct_smi, i_true, rmsds, coords = res
    results[(smi, correct_smi)]['rmsd'][i_true] = rmsds
    # results[(smi, correct_smi)]['coords'][i_true] = coords #! trying to debig memory issues


random.shuffle(jobs)
if args.n_workers > 1:
    p = Pool(args.n_workers)
    map_fn = p.imap_unordered
    p.__enter__()
else:
    map_fn = map

for res in tqdm(map_fn(worker_fn, jobs), total=len(jobs)):
    populate_results(res)

if args.n_workers > 1:
    p.__exit__(None, None, None)

stats = []
for res in results.values():
    stats_ = calc_performance_stats(res['rmsd'])
    cr, mr, cp, mp = stats_
    stats.append(stats_)
coverage_recall, amr_recall, coverage_precision, amr_precision = zip(*stats)

for i, thresh in enumerate(threshold_ranges):
    print('Min threshold', thresh)
    coverage_recall_vals = [stat[i] for stat in coverage_recall] + [0] * num_failures
    coverage_precision_vals = [stat[i] for stat in coverage_precision] + [0] * num_failures
    print(f'Min Recall Coverage: Mean = {np.mean(coverage_recall_vals) * 100:.2f}, Median = {np.median(coverage_recall_vals) * 100:.2f}')
    print(f'Min Recall AMR: Mean = {np.nanmean(amr_recall):.4f}, Median = {np.nanmedian(amr_recall):.4f}')
    print(f'Min Precision Coverage: Mean = {np.mean(coverage_precision_vals) * 100:.2f}, Median = {np.median(coverage_precision_vals) * 100:.2f}')
    print(f'Min Precision AMR: Mean = {np.nanmean(amr_precision):.4f}, Median = {np.nanmedian(amr_precision):.4f}')

print(len(results), 'conformer sets compared', num_failures, 'model failures', np.isnan(amr_recall).sum(),
      'additional failures')

print("\n\n\nL")
stats2 = []
for res in results.values():
    stats_ = calc_performance_stats(res['rmsd'][:, :max(1,res['rmsd'].shape[1]//2)])
    cr, mr, cp, mp = stats_
    stats2.append(stats_)
coverage_recall2, amr_recall2, coverage_precision2, amr_precision2 = zip(*stats2)

for i, thresh in enumerate(threshold_ranges):
    print('Min threshold', thresh)
    coverage_recall_vals = [stat[i] for stat in coverage_recall2] + [0] * num_failures
    coverage_precision_vals = [stat[i] for stat in coverage_precision2] + [0] * num_failures
    print(f'Min Recall Coverage: Mean = {np.mean(coverage_recall_vals) * 100:.2f}, Median = {np.median(coverage_recall_vals) * 100:.2f}')
    print(f'Min Recall AMR: Mean = {np.nanmean(amr_recall2):.4f}, Median = {np.nanmedian(amr_recall2):.4f}')
    print(f'Min Precision Coverage: Mean = {np.mean(coverage_precision_vals) * 100:.2f}, Median = {np.median(coverage_precision_vals) * 100:.2f}')
    print(f'Min Precision AMR: Mean = {np.nanmean(amr_precision2):.4f}, Median = {np.nanmedian(amr_precision2):.4f}')
    
print("\n\n\nL/2")
stats2 = []
for res in results.values():
    stats_ = calc_performance_stats(res['rmsd'][:, :max(1,res['rmsd'].shape[1]//4)])
    cr, mr, cp, mp = stats_
    stats2.append(stats_)
coverage_recall2, amr_recall2, coverage_precision2, amr_precision2 = zip(*stats2)

for i, thresh in enumerate(threshold_ranges):
    print('Min threshold', thresh)
    coverage_recall_vals = [stat[i] for stat in coverage_recall2] + [0] * num_failures
    coverage_precision_vals = [stat[i] for stat in coverage_precision2] + [0] * num_failures
    print(f'Min Recall Coverage: Mean = {np.mean(coverage_recall_vals) * 100:.2f}, Median = {np.median(coverage_recall_vals) * 100:.2f}')
    print(f'Min Recall AMR: Mean = {np.nanmean(amr_recall2):.4f}, Median = {np.nanmedian(amr_recall2):.4f}')
    print(f'Min Precision Coverage: Mean = {np.mean(coverage_precision_vals) * 100:.2f}, Median = {np.median(coverage_precision_vals) * 100:.2f}')
    print(f'Min Precision AMR: Mean = {np.nanmean(amr_precision2):.4f}, Median = {np.nanmedian(amr_precision2):.4f}')
    
print("\n\n\nL/4")
stats2 = []
for res in results.values():
    stats_ = calc_performance_stats(res['rmsd'][:, :max(1,res['rmsd'].shape[1]//8)])
    cr, mr, cp, mp = stats_
    stats2.append(stats_)
coverage_recall2, amr_recall2, coverage_precision2, amr_precision2 = zip(*stats2)

for i, thresh in enumerate(threshold_ranges):
    print('Min threshold', thresh)
    coverage_recall_vals = [stat[i] for stat in coverage_recall2] + [0] * num_failures
    coverage_precision_vals = [stat[i] for stat in coverage_precision2] + [0] * num_failures
    print(f'Min Recall Coverage: Mean = {np.mean(coverage_recall_vals) * 100:.2f}, Median = {np.median(coverage_recall_vals) * 100:.2f}')
    print(f'Min Recall AMR: Mean = {np.nanmean(amr_recall2):.4f}, Median = {np.nanmedian(amr_recall2):.4f}')
    print(f'Min Precision Coverage: Mean = {np.mean(coverage_precision_vals) * 100:.2f}, Median = {np.median(coverage_precision_vals) * 100:.2f}')
    print(f'Min Precision AMR: Mean = {np.nanmean(amr_precision2):.4f}, Median = {np.nanmedian(amr_precision2):.4f}')
  
    
   

with open('./eval_results.pkl', 'wb') as handle:
    pickle.dump(results, handle)
 