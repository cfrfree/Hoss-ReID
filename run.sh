#! /bin/bash
#SBATCH --job-name=reid
#SBATCH --output reid_b.out
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH -p 3090 --gres=gpu:8 --nodelist=gpu9
#SBATCH --time=100:00:00

# Source global definitions
if [ -f /etc/bashrc ]; then
				. /etc/bashrc
		fi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/chenfree2002/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"  
if [ $? -eq 0 ]; then
	eval "$__conda_setup"
else
	if [ -f "/home/chenfree2002/anaconda3/etc/profile.d/conda.sh" ]; then
		. "/home/chenfree2002/anaconda3/etc/profile.d/conda.sh"
	else
		export PATH="/home/chenfree2002/anaconda3/bin:$PATH"
	fi
fi
unset __conda_setup
# <<< conda initialize <<<

cd /home/chenfree2002/Python/Hoss-ReID
conda activate hoss

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 6669 train.py MODEL.DIST_TRAIN True

# python train.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 6699 train_pair.py --config_file configs/pretrain_transoss.yml MODEL.DIST_TRAIN True