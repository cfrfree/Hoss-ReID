#! /bin/bash
#SBATCH --job-name=hjj
#SBATCH --output hjj.out
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH -p 3090 --gres=gpu:4 --nodelist=gpu19
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

# python train_hjj.py


# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 6667 train_hjj.py --config_file configs/hjj.yml MODEL.DIST_TRAIN True


python inference.py --test_dir /home/share/chenfree/ReID/Aircraft_new/val_set --output_path ./result.xml