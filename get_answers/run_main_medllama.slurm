#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --tmp=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yang8597@umn.edu
#SBATCH -p a100-8  
#SBATCH --gres=gpu:a100:8


#SBATCH --job-name=run_main_PubMedQA_200_medllama
#SBATCH --output=/home/zhan1386/yang8597/QA/get_answers/run_main_PubMedQA_200_medalpaca_%j.log

# Initialize Conda environment
source /home/zhan1386/yang8597/.bashrc
conda activate base

# Run the Python script
cd /home/zhan1386/yang8597/QA/get_answers
# /home/zhan1386/yang8597/anaconda3/bin/python /home/zhan1386/yang8597/QA/get_answers/main.py --input_file /home/zhan1386/yang8597/QA/data/PubMedQA_200.csv --qa_dataset PubMedQA --llm Llama_13B --output_path /home/zhan1386/yang8597/QA/results/result --random_seed 42
# /home/zhan1386/yang8597/anaconda3/bin/python /home/zhan1386/yang8597/QA/get_answers/main.py --input_file /home/zhan1386/yang8597/QA/data/PubMedQA_200.csv --qa_dataset PubMedQA --llm Vicuna_13B --output_path /home/zhan1386/yang8597/QA/results/result --random_seed 42
# /home/zhan1386/yang8597/anaconda3/bin/python /home/zhan1386/yang8597/QA/get_answers/main.py --input_file /home/zhan1386/yang8597/QA/data/PubMedQA_200.csv --qa_dataset PubMedQA --llm Medalpaca_13B --output_path /home/zhan1386/yang8597/QA/results/result --random_seed 42
/home/zhan1386/yang8597/anaconda3/bin/python /home/zhan1386/yang8597/QA/get_answers/main.py --input_file /home/zhan1386/yang8597/QA/data/PubMedQA_200.csv --qa_dataset PubMedQA --llm Medllama_13B --output_path /home/zhan1386/yang8597/QA/results/result --random_seed 42