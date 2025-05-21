# Tree Flow Matching
This is the first implementation of inference time scaling flow matching for image generation

installation instructions (python3)
clone repository:
`git@github.com:adamkutak/tree-flow-matching.git`

install SiT as a submodule:
`git submodule add https://github.com/willisma/SiT.git third_party/SiT`

install pip requirements:
`pip install -r requirements.txt`

add imagenet1k validation set (for calculating FID) in `data/` folder:
`mkdir data`
`cd data`
`wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate`

`wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate`

run experiments (from `/` root directory):
`python run_experiments.py`
