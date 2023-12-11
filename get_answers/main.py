import argparse
import pandas as pd
import random
import numpy as np
from config import CONFIG, GPU_info
import transformers
from huggingface_hub import login
login("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")
from tqdm import tqdm

# medmcqa
def run_medmcqa_llama(input_file, llm, output_path):
    df = pd.read_csv(input_file)
    length, rows = df.shape
    llm = CONFIG.adapter_dict[llm]
    # import model
    pl = transformers.pipeline("text-generation", model=CONFIG.LLMs_path[llm]\
                     , tokenizer=CONFIG.LLMs_path[llm]\
                     , device_map="auto")
    # load instruction
    inst = CONFIG.instructions["MedMCQA"][llm]
    responses, suggested_ans = [], []
    for idx,q in tqdm(enumerate(df['question'])):
        # Get QA pairs
        ipt = f"{q}\n a. {q}; b. {q}; c. {q}; d. {q}\n"
        
        # Generate prompt
        prompt = inst + ipt + "### Response:"
        
        # run model and get response
        max_len = len(pl.tokenizer(prompt)['input_ids'])+100
        responseInFull = pl(prompt, max_length= max_len, do_sample=True, 
                         num_return_sequences=1, temperature=0.3, top_k=40, 
                         top_p=0.5)
        response = responseInFull[0]["generated_text"]
        responses.append(response)
        
        ans = response[len(prompt):]
        suggested_ans.append(ans)
    # Save results appending to the inital DF
    df["responses"] = responses
    df["suggested_answers"] = suggested_ans # post-processing
    df.to_csv(output_path + "_MedMCQA_" + llm + "_" + str(length) + ".csv", index=False)
    return 0

def run_medmcqa_vicuna(input_file, llm, output_path):
    df = pd.read_csv(input_file)
    length, rows = df.shape
    llm = CONFIG.adapter_dict[llm]
    # import model
    pl = transformers.pipeline("text-generation", model=CONFIG.LLMs_path[llm]\
                     , tokenizer=CONFIG.LLMs_path[llm]\
                     , device_map="auto")
    # load instruction
    inst = CONFIG.instructions["MedMCQA"][llm]
    responses, suggested_ans = [], []
    for idx,q in tqdm(enumerate(df['question'])):
        # Get QA pairs
        ipt = f"{q}\n a. {q}; b. {q}; c. {q}; d. {q}\n"
        
        # Generate prompt
        prompt = f"{inst}Context: {ipt}\nYour answer: "
        
        # run model and get response
        max_len = len(pl.tokenizer(prompt)['input_ids'])+100
        responseInFull = pl(prompt, max_length= max_len, do_sample=True, 
                         num_return_sequences=1, temperature=0.3, top_k=40, 
                         top_p=0.5)
        response = responseInFull[0]["generated_text"]
        responses.append(response)
        
        ans = response[len(prompt):]
        suggested_ans.append(ans)
    # Save results appending to the inital DF
    df["responses"] = responses
    df["suggested_answers"] = suggested_ans
    df.to_csv(output_path + "_MedMCQA_" + llm + "_" + str(length) + ".csv" ,index=False)
    return 0

def run_medmcqa_medalp(input_file, llm, output_path):
    df = pd.read_csv(input_file)
    length, rows = df.shape
    llm = CONFIG.adapter_dict[llm]
    # import model
    pl = transformers.pipeline("text-generation", model=CONFIG.LLMs_path[llm]\
                     , tokenizer=CONFIG.LLMs_path[llm]\
                     , device_map="auto")
    
    # load instruction
    inst = CONFIG.instructions["MedMCQA"][llm]
    responses, suggested_ans = [], []
    for idx,q in tqdm(enumerate(df['question'])):
        # Get QA pairs
        ipt = f"{q}\n a. {q}; b. {q}; c. {q}; d. {q}\n"
        
        # Generate prompt
        prompt = f"{inst}Context: {ipt}\nYour answer: "
        
        # run model and get response
        max_len = len(pl.tokenizer(prompt)['input_ids'])+100
        responseInFull = pl(prompt, max_length= max_len, do_sample=True, 
                         num_return_sequences=1, temperature=0.3, top_k=40, 
                         top_p=0.5)
        response = responseInFull[0]["generated_text"]
        responses.append(response)
        
        ans = response[len(prompt):]
        suggested_ans.append(ans)
    # Save results appending to the inital DF
    df["responses"] = responses
    df["suggested_answers"] = suggested_ans
    df.to_csv(output_path + "_" + llm + "_" + str(length) + ".csv" ,index=False)
    return 0

def run_medmcqa_medllama(input_file, llm, output_path):
    df = pd.read_csv(input_file)
    length, rows = df.shape
    llm = CONFIG.adapter_dict[llm]
    # import model
    print("----****----import model----****----")
    tokenizer = transformers.LlamaTokenizer.from_pretrained(CONFIG.LLMs_path[llm])
    model = transformers.LlamaForCausalLM.from_pretrained(CONFIG.LLMs_path[llm])
    pl = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    # load instruction
    inst = CONFIG.instructions["MedMCQA"][llm]
    responses, suggested_ans = [], []
    print("----****----Start iterating----****----")
    for idx,q in tqdm(enumerate(df['question'])):
        # Get QA pairs
        ipt = f"{q}\n### Options: a. {q}; b. {q}; c. {q}; d. {q}\n"
        
        # Generate prompt
        prompt = inst + ipt + "### Answer:"
        
        # run model and get response
        max_len = len(pl.tokenizer(prompt)['input_ids'])+230
        responseInFull = pl(prompt, max_length= max_len, do_sample=True, 
                         num_return_sequences=1, temperature=0.3, top_k=40, 
                         top_p=0.5)
        response = responseInFull[0]["generated_text"]
        responses.append(response)
        
        ans = response[len(prompt):]
        suggested_ans.append(ans)
    # Save results appending to the inital DF
    df["responses"] = responses
    df["suggested_answers"] = suggested_ans
    df.to_csv(output_path + "_MedMCQA_" + llm + "_" + str(length) + ".csv", index=False)
    return 0



# pubmedqa
def run_pubmedqa_llama(input_file, llm, output_path):
    df = pd.read_csv(input_file)
    length, rows = df.shape
    llm = CONFIG.adapter_dict[llm]
    # import model
    print("----****----import model----****----")
    pl = transformers.pipeline("text-generation", model=CONFIG.LLMs_path[llm]\
                     , tokenizer=CONFIG.LLMs_path[llm]\
                     , device_map="auto")
    # load instruction
    inst = CONFIG.instructions["PubMedQA"][llm]
    responses, suggested_ans = [], []
    print("----****----Start iterating----****----")
    # load instruction
    inst = CONFIG.instructions["PubMedQA"][llm]
    responses, suggested_ans = [], []
    for idx,q in tqdm(enumerate(df['QUESTION'])):
        # Get QA pairs
        ipt = f"{df['CONTEXTS'][idx]}\nQuestion: {q}\n"
        # Generate prompt
        prompt = inst + ipt + "### Response (only yes or no):"
        
        # run model and get response
        max_len = len(pl.tokenizer(prompt)['input_ids'])+100
        responseInFull = pl(prompt, max_length= max_len, do_sample=True, 
                         num_return_sequences=1, temperature=0.3, top_k=40, 
                         top_p=0.5)
        response = responseInFull[0]["generated_text"]
        responses.append(response)
        
        ans = response[len(prompt):]
        suggested_ans.append(ans)
    # Save results appending to the inital DF
    df["responses"] = responses
    df["suggested_answers"] = suggested_ans # post-processing
    df.to_csv(output_path + "_PubMedQA_" + llm + "_" + str(length) + ".csv", index=False)
    return 0

def run_pubmedqa_medllama(input_file, llm, output_path):
    df = pd.read_csv(input_file)
    length, rows = df.shape
    llm = CONFIG.adapter_dict[llm]
    # import model
    tokenizer = transformers.LlamaTokenizer.from_pretrained(CONFIG.LLMs_path[llm])
    model = transformers.LlamaForCausalLM.from_pretrained(CONFIG.LLMs_path[llm])
    pl = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    # load instruction
    inst = CONFIG.instructions["PubMedQA"][llm]
    responses, suggested_ans = [], []
    for idx,q in tqdm(enumerate(df['QUESTION'])):
        # Get QA pairs
        ipt = f"{df['CONTEXTS'][idx]}\nQuestion: {q}\n"
        
        # Generate prompt
        prompt = inst + ipt + "### Response (only yes or no):"
        
        # run model and get response
        max_len = len(tokenizer(prompt)['input_ids'])+100
        responseInFull = pl(prompt, max_length= max_len, do_sample=True, 
                         num_return_sequences=1, temperature=0.3, top_k=40, 
                         top_p=0.5)
        response = responseInFull[0]["generated_text"]
        responses.append(response)
        
        ans = response[len(prompt):]
        suggested_ans.append(ans)
    # Save results appending to the inital DF
    df["responses"] = responses
    df["suggested_answers"] = suggested_ans # post-processing
    df.to_csv(output_path + "_PubMedQA_" + llm + "_" + str(length) + ".csv", index=False)
    return 0

def run_pubmedqa_medalpaca(input_file, llm, output_path):
    df = pd.read_csv(input_file)
    length, rows = df.shape
    llm = CONFIG.adapter_dict[llm]
    # import model
    pl = transformers.pipeline("text-generation", model=CONFIG.LLMs_path[llm]\
                     , tokenizer=CONFIG.LLMs_path[llm]\
                     , device_map="auto")
    # load instruction
    inst = CONFIG.instructions["PubMedQA"][llm]
    responses, suggested_ans = [], []
    for idx,q in tqdm(enumerate(df['QUESTION'])):
        # Get QA pairs
        ipt = f"{df['CONTEXTS'][idx]}\nQuestion: {q}\n"
        
        # Generate prompt
        prompt = inst + ipt + "### Response (only yes or no):"
        
        # run model and get response
        max_len = len(pl.tokenizer(prompt)['input_ids'])+100
        responseInFull = pl(prompt, max_length= max_len, do_sample=True, 
                         num_return_sequences=1, temperature=0.3, top_k=40, 
                         top_p=0.5)
        response = responseInFull[0]["generated_text"]
        responses.append(response)
        
        ans = response[len(prompt):]
        suggested_ans.append(ans)
    # Save results appending to the inital DF
    df["responses"] = responses
    df["suggested_answers"] = suggested_ans # post-processing
    df.to_csv(output_path + "_PubMedQA_" + llm + "_" + str(length) + ".csv", index=False)
    return 0

def run_pubmedqa_vicuna(input_file, llm, output_path):
    df = pd.read_csv(input_file)
    length, rows = df.shape
    llm = CONFIG.adapter_dict[llm]
    # import model
    pl = transformers.pipeline("text-generation", model=CONFIG.LLMs_path[llm]\
                     , tokenizer=CONFIG.LLMs_path[llm]\
                     , device_map="auto")
    # load instruction
    inst = CONFIG.instructions["PubMedQA"][llm]
    responses, suggested_ans = [], []
    for idx,q in tqdm(enumerate(df['QUESTION'])):
        # Get QA pairs
        ipt = f"{df['CONTEXTS'][idx]}\nQuestion: {q}\n"
        
        # Generate prompt
        prompt = inst + ipt + "### Response (only yes or no):"
        
        # run model and get response
        max_len = len(pl.tokenizer(prompt)['input_ids'])+100
        responseInFull = pl(prompt, max_length= max_len, do_sample=True, 
                         num_return_sequences=1, temperature=0.3, top_k=40, 
                         top_p=0.5)
        response = responseInFull[0]["generated_text"]
        responses.append(response)
        
        ans = response[len(prompt):]
        suggested_ans.append(ans)
    # Save results appending to the inital DF
    df["responses"] = responses
    df["suggested_answers"] = suggested_ans # post-processing
    df.to_csv(output_path + "_PubMedQA_" + llm + "_" + str(length) + ".csv", index=False)
    return 0

#medqa-usmle
def run_medqa_usmle_medllama(input_file, llm, output_path):
    df = pd.read_csv(input_file)
    length, rows = df.shape
    llm = CONFIG.adapter_dict[llm]
    # import model
    print("----****----import model----****----")
    tokenizer = transformers.LlamaTokenizer.from_pretrained(CONFIG.LLMs_path[llm])
    model = transformers.LlamaForCausalLM.from_pretrained(CONFIG.LLMs_path[llm])
    pl = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    # load instruction
    inst = CONFIG.instructions["MedQA-USMLE"][llm]
    responses, suggested_ans = [], []
    print("----****----Start iterating----****----")
    for idx,q in tqdm(enumerate(df['question'])):
        # Get QA pairs
        ipt = f"{q}\n### Options: A. {df['option_A'][idx]}; B. {df['option_B'][idx]}"\
        f"; C. {df['option_C'][idx]}; D. {df['option_D'][idx]}; E. {df['option_E'][idx]}\n"

        # Generate prompt
        prompt = inst + ipt + "### Answer (only A, B, C, D, or E) :"
        
        # run model and get response
        max_len = len(tokenizer(prompt)['input_ids'])+210
        responseInFull = pl(prompt, max_length= max_len, do_sample=True, 
                         num_return_sequences=1, temperature=0.3, top_k=40, 
                         top_p=0.5)
        response = responseInFull[0]["generated_text"]
        responses.append(response)
        
        ans = response[len(prompt):]
        suggested_ans.append(ans)
    # Save results appending to the inital DF
    df["responses"] = responses
    df["suggested_answers"] = suggested_ans
    df.to_csv(output_path + "_MedQA-USMLE_" + llm + "_" + str(length) + ".csv", index=False)
    return 0

def run_medqa_usmle_llama(input_file, llm, output_path):
    df = pd.read_csv(input_file)
    length, rows = df.shape
    llm = CONFIG.adapter_dict[llm]
    # import model
    print("----****----import model----****----")
    pl = transformers.pipeline("text-generation", model=CONFIG.LLMs_path[llm]\
                     , tokenizer=CONFIG.LLMs_path[llm]\
                     , device_map="auto")
    # load instruction
    inst = CONFIG.instructions["MedQA-USMLE"][llm]
    responses, suggested_ans = [], []
    print("----****----Start iterating----****----")
    for idx,q in tqdm(enumerate(df['question'])):
        # Get QA pairs
        ipt = f"{q}\n### Options: A. {df['option_A'][idx]}; B. {df['option_B'][idx]}"\
        f"; C. {df['option_C'][idx]}; D. {df['option_D'][idx]}; E. {df['option_E'][idx]}\n"

        # Generate prompt
        prompt = inst + ipt + "### Answer (only A, B, C, D, or E) :"
        
        # run model and get response
        max_len = len(pl.tokenizer(prompt)['input_ids'])+100
        responseInFull = pl(prompt, max_length= max_len, do_sample=True, 
                         num_return_sequences=1, temperature=0.3, top_k=40, 
                         top_p=0.5)
        response = responseInFull[0]["generated_text"]
        responses.append(response)
        
        ans = response[len(prompt):]
        suggested_ans.append(ans)
    # Save results appending to the inital DF
    df["responses"] = responses
    df["suggested_answers"] = suggested_ans
    df.to_csv(output_path + "_MedQA-USMLE_" + llm + "_" + str(length) + ".csv", index=False)
    return 0

def run_medqa_usmle_vicuna(input_file, llm, output_path):
    df = pd.read_csv(input_file)
    length, rows = df.shape
    llm = CONFIG.adapter_dict[llm]
    # import model
    print("----****----import model----****----")
    pl = transformers.pipeline("text-generation", model=CONFIG.LLMs_path[llm]\
                     , tokenizer=CONFIG.LLMs_path[llm]\
                     , device_map="auto")
    # load instruction
    inst = CONFIG.instructions["MedQA-USMLE"][llm]
    responses, suggested_ans = [], []
    print("----****----Start iterating----****----")
    for idx,q in tqdm(enumerate(df['question'])):
        # Get QA pairs
        ipt = f"{q}\n### Options: A. {df['option_A'][idx]}; B. {df['option_B'][idx]}"\
        f"; C. {df['option_C'][idx]}; D. {df['option_D'][idx]}; E. {df['option_E'][idx]}\n"

        # Generate prompt
        prompt = inst + ipt + "### Answer (only A, B, C, D, or E) :"
        
        # run model and get response
        max_len = len(pl.tokenizer(prompt)['input_ids'])+100
        responseInFull = pl(prompt, max_length= max_len, do_sample=True, 
                         num_return_sequences=1, temperature=0.3, top_k=40, 
                         top_p=0.5)
        response = responseInFull[0]["generated_text"]
        responses.append(response)
        
        ans = response[len(prompt):]
        suggested_ans.append(ans)
    # Save results appending to the inital DF
    df["responses"] = responses
    df["suggested_answers"] = suggested_ans
    df.to_csv(output_path + "_MedQA-USMLE_" + llm + "_" + str(length) + ".csv", index=False)
    return 0

def run_medqa_usmle_medalpaca(input_file, llm, output_path):
    df = pd.read_csv(input_file)
    length, rows = df.shape
    llm = CONFIG.adapter_dict[llm]
    # import model
    print("----****----import model----****----")
    pl = transformers.pipeline("text-generation", model=CONFIG.LLMs_path[llm]\
                     , tokenizer=CONFIG.LLMs_path[llm]\
                     , device_map="auto")
    # load instruction
    inst = CONFIG.instructions["MedQA-USMLE"][llm]
    responses, suggested_ans = [], []
    print("----****----Start iterating----****----")
    for idx,q in tqdm(enumerate(df['question'])):
        # Get QA pairs
        ipt = f"{q}\n### Options: A. {df['option_A'][idx]}; B. {df['option_B'][idx]}"\
        f"; C. {df['option_C'][idx]}; D. {df['option_D'][idx]}; E. {df['option_E'][idx]}\n"
        # Generate prompt
        prompt = inst + ipt + "### Answer (only A, B, C, D, or E) :"
        
        # run model and get response
        max_len = len(pl.tokenizer(prompt)['input_ids'])+100
        responseInFull = pl(prompt, max_length= max_len, do_sample=True, 
                         num_return_sequences=1, temperature=0.3, top_k=40, 
                         top_p=0.5)
        response = responseInFull[0]["generated_text"]
        responses.append(response)
        ans = response[len(prompt):]
        suggested_ans.append(ans)
    # Save results appending to the inital DF
    df["responses"] = responses
    df["suggested_answers"] = suggested_ans
    df.to_csv(output_path + "_MedQA-USMLE_" + llm + "_" + str(length) + ".csv", index=False)
    return 0

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run QA tasks on 3 Medical QA dataset.')
    parser.add_argument('--input_file', type=str, help='Input csv file related to the QA dataset.')
    parser.add_argument('--qa_dataset', type=str, help='Medical QA dataset, options: {MedMCQA, PubMedQA, MedQA_USMLE}')
    parser.add_argument('--llm', type=str, help='Specify the LLM you want to use, options: {Llama_13B, Vicuna_13B, Medalpaca_13B, Medllama_13B}')
    parser.add_argument('--output_path', type=str, help='Path you want to save your answer sheets',
                       default='/home/zhan1386/yang8597/QA/results')
    parser.add_argument('--random_seed', type=int, help='Random state number for reproducibility',
                       default=42)
    
    # Parse arguments
    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    GPU_info()
    # nested run
    if args.qa_dataset == "MedMCQA" and args.llm == "Llama_13B":
        run_medmcqa_llama(args.input_file, args.llm, args.output_path)
    elif args.qa_dataset == "MedMCQA" and args.llm == "Vicuna_13B":
        run_medmcqa_vicuna(args.input_file, args.llm, args.output_path)
    elif args.qa_dataset == "MedMCQA" and args.llm == "Medalpaca_13B":
        run_medmcqa_medalp(args.input_file, args.llm, args.output_path)
    elif args.qa_dataset == "MedMCQA" and args.llm == "Medllama_13B":
        run_medmcqa_medllama(args.input_file, args.llm, args.output_path)
    elif args.qa_dataset == "PubMedQA" and args.llm == "Llama_13B":
        run_pubmedqa_llama(args.input_file, args.llm, args.output_path)
    elif args.qa_dataset == "PubMedQA" and args.llm == "Vicuna_13B":
        run_pubmedqa_vicuna(args.input_file, args.llm, args.output_path)
    elif args.qa_dataset == "PubMedQA" and args.llm == "Medalpaca_13B":
        run_pubmedqa_medalpaca(args.input_file, args.llm, args.output_path)
    elif args.qa_dataset == "PubMedQA" and args.llm == "Medllama_13B":
        run_pubmedqa_medllama(args.input_file, args.llm, args.output_path)
    elif args.qa_dataset == "MedQA_USMLE" and args.llm == "Llama_13B":
        run_medqa_usmle_llama(args.input_file, args.llm, args.output_path)
    elif args.qa_dataset == "MedQA_USMLE" and args.llm == "Vicuna_13B":
        run_medqa_usmle_vicuna(args.input_file, args.llm, args.output_path)
    elif args.qa_dataset == "MedQA_USMLE" and args.llm == "Medalpaca_13B":
        run_medqa_usmle_medalpaca(args.input_file, args.llm, args.output_path)
    elif args.qa_dataset == "MedQA_USMLE" and args.llm == "Medllama_13B":
        run_medqa_usmle_medllama(args.input_file, args.llm, args.output_path)
    else:
        # Handle other cases or invalid inputs
        print("Wrong input, check your cmd inputs.")

if __name__ == '__main__':
    main()
