# Tree Flow Matching
This is the first implementation of inference time scaling flow matching for image generation

installation instructions (python3)
clone repository:
`git@github.com:adamkutak/tree-flow-matching.git`

install SiT as a submodule:
`git submodule add https://github.com/willisma/SiT.git third_party/SiT`

get the SiT-XL/2 pretrained model:
`mkdir saved_models`
`wget -O saved_models/SiT-XL-2-256.pt "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0"`


install pip requirements:
`pip install -r requirements.txt`

add imagenet1k validation set (for calculating FID) in `data/` folder:
`mkdir data`
`cd data`
`wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate`

`wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate`

extract imagenet dataset and organize it into correct subfolders:
`mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar`
`wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash`

compute datset statistics:
`python compute_dataset_stats.py --dataset imagenet256`

run experiments (from `/` root directory):
`python run_experiments.py`
