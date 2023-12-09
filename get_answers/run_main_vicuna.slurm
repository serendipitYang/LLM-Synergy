#!/bin/bash

#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --tmp=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yang8597@umn.edu
#SBATCH -p a100-8  
#SBATCH --gres=gpu:a100:8


#SBATCH --job-name=run_main_MedMCQA10000_Vicuna
#SBATCH --output=run_main_MedMCQA10000_Vicuna_output_%j.log

# Initialize Conda environment
source /home/zhan1386/yang8597/.bashrc
conda activate base

# Run the Python script
cd /home/zhan1386/yang8597/QA/get_answers
# /home/zhan1386/yang8597/anaconda3/bin/python /home/zhan1386/yang8597/QA/get_answers/main.py --input_file /home/zhan1386/yang8597/QA/data/MedMCQA_10000.csv --qa_dataset MedMCQA --llm Llama_13B --output_path /home/zhan1386/yang8597/QA/results/result --random_seed 42
/home/zhan1386/yang8597/anaconda3/bin/python /home/zhan1386/yang8597/QA/get_answers/main.py --input_file /home/zhan1386/yang8597/QA/data/MedMCQA_10000.csv --qa_dataset MedMCQA --llm Vicuna_13B --output_path /home/zhan1386/yang8597/QA/results/result --random_seed 42
# /home/zhan1386/yang8597/anaconda3/bin/python /home/zhan1386/yang8597/QA/get_answers/main.py --input_file /home/zhan1386/yang8597/QA/data/MedMCQA_10000.csv --qa_dataset MedMCQA --llm Medalpaca_13B --output_path /home/zhan1386/yang8597/QA/results/result --random_seed 42
# /home/zhan1386/yang8597/anaconda3/bin/python /home/zhan1386/yang8597/QA/get_answers/main.py --input_file /home/zhan1386/yang8597/QA/data/MedMCQA_10000.csv --qa_dataset MedMCQA --llm Medllama_13B --output_path /home/zhan1386/yang8597/QA/results/result --random_seed 42