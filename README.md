# GatorST: A Versatile Contrastive Meta-Learning Framework for Spatial Transcriptomic Data Analysis
![model](GatorST.png)

## Requirements
- python : 3.9.12
- scanpy : 1.10.3
- sklearn : 1.1.1
- scipy : 1.9.0
- torch : 1.11.2
- torch-geometric : 2.1.0
- numpy : 1.24.4
- pandas : 2.2.3


## Project Structure

```bash
.
├── main.py            # Main training and evaluation loop
├── model.py           # Model architecture and loss functions
├── data_loader.py     # Data loading and graph construction utilities
├── utils.py            # Utility functions (seed setup, metrics, dropout)
├── data/              # Folder for .h5ad input files
├── saved_models/      # Folder to save trained models
└── result.json        # Evaluation results output
```

## Usage

### **1. Prepare your input data**

Place your **.h5ad** spatial transcriptomics datasets in the `./data/` directory.

Each `.h5ad` file should contain:

* **`adata.X`** – Gene expression matrix
* **`adata.obs`** – Cell/Spot metadata.
  The loader automatically searches typical label fields:

  ```
  ['ground_truth']
  ```

  and maps them to integer class indices. 
* **`adata.obsm["spatial"]`** – Spatial coordinates (N × 2)

Example dataset folder:

```bash
data/
 ├── 151507.h5ad
 ├── human_breast_cancer.h5ad
 └── mouse_brain_anterior.h5ad
```

---

### **2. Run training and evaluation**

Execute the main script to train and evaluate across datasets:

```bash
python main.py
```

#### Optional configuration inside `main.py`:

* `epochs`: number of training epochs per run 
* `batch_size`: number of samples per batch
* `lr`: learning rate
* `alpha`: balancing the contributions of two losses
* `N_way`: number of classes to sample
* `M_shot`: number of support samples per class
* `Q_query`: number of query samples per class

You can modify these directly in `main.py`:

```python
epochs = 50
batch_size = 20
lr = 0.001
alpha = 0.5
N_way = 5
M_shot = 5
Q_query = 5
```

---

### **3. Output files**

After training completed:

* Trained model checkpoints: `saved_models/`

  ```
  saved_models/
   ├── 151507_model_run_0
   ├── 151507_model_run_1
   ...
  ```
* Evaluation results (accuracy, clustering, etc.): `result.json`

  ```json
  {
      "151507": ["ARI":,...],
      "human_breast_cancer": ["ARI":,...],
          ...
  }
  ```

---

## Datasets
The spatial transcriptomics datasets analyzed in this study are publicly available from the following sources: the LIBD human dorsolateral prefrontal cortex (DLPFC) dataset, which was obtained using the 10x Visium platform (http://research.libd.org/spatialLIBD/); human lymph node Visium dataset acquired from tissue containing germinal centers (GCs) and obtained from GEO (accession no. GSE263617); the human breast cancer dataset (https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0) and the mouse brain tissue dataset (https://www.10xgenomics.com/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-1-0), both obtained from the 10x Genomics Data Repository. In addition, we used an E9.5 mouse embryo dataset generated with Stereo-seq and downloaded from the MOSTA resource (https://db.cngb.org/stomics/mosta/), a Stereo-seq dataset of mouse olfactory bulb (https://github.com/JinmiaoChenLab/SEDR_analyses), and a mouse hippocampus dataset profiled with Slide-seqV2 (https://portals.broadinstitute.org/single_cell/study/slide-seq-study).

