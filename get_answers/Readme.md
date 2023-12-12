Usage:

# Run Llama2-13B on PubMedQA dataset:
```
python main.py --input_file PubMedQA_10000.csv --qa_dataset PubMedQA --llm Llama_13B --output_path /home/zhan1386/yang8597/QA/results/result --random_seed 42
```


# MedMCQA
```
python main.py --input_file /data/MedMCQA/dev.csv --qa_dataset MedMCQA --llm Medllama_13B --output_path /dev_results --random_seed 42
```

# PubmedQA
```
python main.py --input_file /data/PubMedQA/dev.csv --qa_dataset PubmedQA --llm Medllama_13B --output_path /dev_results --random_seed 42
```

# MedQA-USMLE
```
python main.py --input_file /data/MedMC_USMLE/dev.csv --qa_dataset MedQA_USMLE --llm Medllama_13B --output_path /dev_results --random_seed 42
```



