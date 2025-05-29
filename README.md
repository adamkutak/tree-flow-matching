# Flow Matching Inference Time Compute Scaling
## Installation Instructions (Python 3)

### 1. Clone Repository
```bash
git clone git@github.com:adamkutak/tree-flow-matching.git
cd tree-flow-matching
```

### 2. Install SiT as a Submodule
```bash
git submodule add https://github.com/willisma/SiT.git third_party/SiT
```

### 3. Get the SiT-XL/2 Pretrained Model
```bash
mkdir saved_models
wget -O saved_models/SiT-XL-2-256.pt "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0"
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Setup ImageNet Validation Dataset

Create data directory and download ImageNet validation set (for calculating FID):

```bash
mkdir data
cd data
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate
```

Extract and organize the dataset:

```bash
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

### 6. Compute Dataset Statistics
```bash
python compute_dataset_stats.py --dataset imagenet256
```

## Usage

Run experiments from the root directory:

```bash
python run_experiments.py
```
```