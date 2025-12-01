## Usage
1. Download raw dataset from the official site: 
    - MIMIC-4: https://physionet.org/content/mimiciv/3.1/
    - ADNI: https://tadpole.grand-challenge.org/
2. Data pre-process, run the corresponding jupyter notebook:
    - MIMIC-4: ```process_mimic4.ipynb```
    - ADNI:```process_adni.ipynb```
3. Run the start.sh
## Dependency
  - mpi4py=3.1.4
  - python=3.8.18
  - huggingface-hub=0.28.1
  - matplotlib=3.7.5
  - networkx=3.0
  - numpy=1.24.4
  - pandas=2.0.3
  - pillow=10.4.0
  - pyg-lib=0.4.0+pt20cu118
  - scikit-learn=1.3.2
  - scipy=1.10.1
  - seaborn=0.13.2
  - setproctitle=1.3.4
  - torch=2.0.1+cu118
  - torch-cluster=1.6.3+pt20cu118
  - torch-geometric=2.6.1
  - torch-scatter=2.1.2+pt20cu118
  - torch-sparse=0.6.18+pt20cu118
  - torch-spline-conv=1.2.2+pt20cu118
  - tqdm=4.67.1
  - transformers=4.46.3
  - wandb=0.19.6