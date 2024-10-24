# CoarsenConf

Implementation of [CoarsenConf](https://arxiv.org/pdf/2306.14852.pdf) by D. Reidenbach* and A. Krishnapriyan.

CoarsenConf is a coarse grained variational auto encoder for molecular conformer generation.

If you have questions, don't hesitate to open an issue or send us an email at dreidenbach@berkeley.edu


## Setting up Conda environment

Create new [Conda](https://docs.anaconda.com/anaconda/install/index.html) environment using `mcg_environment.yml`. You might need to adjust the `cudatoolkit` version to match your cuda version or set `cpuonly`.

    conda env create -f mcg_environment.yml
    conda activate mcg


## Generate conformers from SMILES

 To generate conformers using the trained model, create a `smiles.csv`  or `.pkl` file containing at every line `smile_str, num_conformers, smile_str` (for example `CN1C=NC2=C1C(=O)N(C(=O)N2C)C, 10, CN1C=NC2=C1C(=O)N(C(=O)N2C)C`) where `smile_str` is the SMILE representation of the molecule (note: technically the first is the one used as identifier of the molecule and the second the one used to create it but we suggest to keep them the same). Then you can generate the conformers running:

    python generate.py


## Training model

Following the instruction from [Torsional Diffusion](https://github.com/gcorso/torsional-diffusion) download and extract all the relevant data from the compressed `.tar.gz` folders from [this shared Drive](https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7?usp=sharing) putting them in the subdirectory `data`. These contain the GEOM datasets used in the project (license CC0 1.0), the splits from GeoMol and the pickle files with preprocessed molecules (see below to recreate them) and are divided based on the dataset they refer to. Then, you can start training:

    python train_drugs.py

Details on all  hyperparameters or how to update to different datasets can be found in  `configs`. The first time the training is run, a featurisation procedure starts and caches the result so that it won't be required the next time training is run.

## Running evaluation

In order to evaluate a model on the test set of one of the datasets you need to first download the data (see section above, but the only files needed are `test_smiles.csv`, list of SMILES strings and the number of conformers, and `test_mols.pkl`, dictionary of ground truth conformers). Locate the work directory of your trained model and, then, you can generate the conformers with the model via:

    python scripts/generate_confs.py 

Finally, evaluate the error of the conformers using the following command:

    python evaluate_confs.py 


## Citation

If you use this code, please cite:

    @article{reidenbach2023coarsenconf,
      title={CoarsenConf: Equivariant Coarsening with Aggregated Attention for Molecular Conformer Generation},
      author={Danny Reidenbach and Aditi S. Krishnapriyan},
      journal={arXiv preprint arXiv:2306.14852},
      year={2023},
      }

## License
MIT