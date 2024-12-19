mkvirtualenv sam2

pip3 install "torch>=2.5.1" "torchvision>=0.20.1" --index-url https://download.pytorch.org/whl/cpu

# Download and install SAM-2
cd
git clone git@github.com:facebookresearch/sam2.git
cd sam2
pip install .

# Download checkpoints: takes a couple of minutes
cd checkpoints && \
./download_ckpts.sh && \
cd $HOME

# Install extra requirements
cd adventures-with-sam
pip install -r requirements.txt

# Example run command
PYTHONPATH=. python -m sam_src.run_sam2 --img-dir $HOME/adventures-with-sam/data/img/xmas/small -m tiny --save-segment-imgs
