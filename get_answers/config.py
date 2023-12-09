import transformers
import torch
import numpy as np
import os

class CONFIG:
    seed = 42
    min_length=10
    do_sample=True
    num_return_sequences=1
    temperature=0.3 # Set to a lower value (e.g., 0.2 or 0.3) to reduce randomness and increase predictability in responses.
    top_k=40 # Set to a moderate value (e.g., 30 or 40) to maintain a balance between variety and relevance.
    top_p=0.5 # Set to a lower value (e.g., 0.4 or 0.5) to focus on more likely responses while still allowing for some variability.
    LLMs_path = {
        "Llama-7B": "meta-llama/Llama-2-7b-hf",
        "Llama-13B": "meta-llama/Llama-2-13b-hf",
        "Vicuna-7B": "lmsys/vicuna-7b-v1.5",
        "Vicuna-13B": "lmsys/vicuna-13b-v1.5",
        "Medalpaca-7B": "medalpaca/medalpaca-7b",
        "Medalpaca-13B": "medalpaca/medalpaca-13b",
        "Medllama-13B": "axiong/PMC_LLaMA_13B"
    }
    adapter_dict = {
        "Llama_13B": "Llama-13B",
        "Vicuna_13B": "Vicuna-13B",
        "Medalpaca_13B": "Medalpaca-13B",
        "Medllama_13B": "Medllama-13B" 
    }
    instructions={
        "MedMCQA": {
            "Medalpaca-13B": "You are an excellent doctor. The task is to answer the following question choosing options within {a,b,c,d, NA}."\
            " DO NOT BE VERBOSE, and NA is for the situation when you are not sure or none of them is correct"\
            " (However, I am 100% sure the correct answer MUST BE one of {a,b,c,d})."
            " Here is an example of the answering format:\n"\
            "Which of the following is not true for myelinated nerve fibers:\n"\
            "a. Impulse through myelinated fibers is slower than non-myelinated fibers; "\
            "b. Membrane currents are generated at nodes of Ranvier;"\
            "c. Saltatory conduction of impulses is seen; "\
            "d. Local anesthesia is effective only when the nerve is not covered by myelin sheath.\n"\
            "(Your answer:) d.\n\n",
            "Llama-13B": """### Instruction: As a knowledgeable medical professional, your task is to identify the correct answer from the provided options {a, b, c, d}. There is no 'NA' option as one of these choices is definitively correct. Please be concise in your response, , NO VERBOSITY. An example is provided for guidance:
            Example Question: Which of the following is not true for myelinated nerve fibers?
            a. Impulse through myelinated fibers is slower than non-myelinated fibers;
            b. Membrane currents are generated at nodes of Ranvier;
            c. Saltatory conduction of impulses is seen;
            d. Local anesthesia is effective only when the nerve is not covered by myelin sheath.
            Example Answer: d.
            ### Input: Question: """,
            "Vicuna-13B": "You are an excellent medical professional. The task is to answer the following question choosing options within {a,b,c,d, NA}."\
            " DO NOT BE VERBOSE, and NA is for the situation when you are not sure or none of them is correct"\
            " (However, I am 100% sure the correct answer MUST BE one of {a,b,c,d})."\
            " Here is an example of the answering format:\n"\
            "Which of the following is not true for myelinated nerve fibers:\n"\
            "a. Impulse through myelinated fibers is slower than non-myelinated fibers; "\
            "b. Membrane currents are generated at nodes of Ranvier;"\
            "c. Saltatory conduction of impulses is seen; "\
            "d. Local anesthesia is effective only when the nerve is not covered by myelin sheath.\n"\
            "(Your answer:) d.\n\n",
            "Medllama-13B": """### Instruction: As a knowledgeable medical professional, your task is to identify the correct answer from the provided options {a, b, c, d}. There is no 'NA' option as one of these choices is definitively correct. Please be concise in your response, NO VERBOSITY. An example is provided for guidance:
Example Question: Which of the following is not true for myelinated nerve fibers?
a. Impulse through myelinated fibers is slower than non-myelinated fibers;
b. Membrane currents are generated at nodes of Ranvier;
c. Saltatory conduction of impulses is seen;
d. Local anesthesia is effective only when the nerve is not covered by myelin sheath.
Example Answer (strictly follow): d.
Input: ### Question:"""
        },
        "PubMedQA": {
            "Llama-13B":"""### Instruction: As a knowledgeable medical professional, based on the given context, your task is to identify the correct answer and directly provide it answering yes or no. There is no 'NA' option as one of 2 choices is definitively correct. Please be concise in your response, NO VERBOSITY. An example is provided for guidance:
Example: Context: Early appearance of antibodies specific for native human type II collagen (anti-CII) characterizes an early inflammatory and destructive phenotype in adults with rheumatoid arthritis (RA). The objective of this study was to investigate the occurrence of anti-CII, IgM RF, IgA RF and anti-CCP in serum samples obtained early after diagnosis, and to relate the occurrence of autoantibodies to outcome after eight years of disease in children with juvenile idiopathic arthritis (JIA). The Nordic JIA database prospectively included JIA patients followed for eight years with data on remission and joint damage. From this database, serum samples collected from 192 patients, at a median of four months after disease onset, were analysed for IgG anti-CII, IgM RF, IgA RF and IgG anti-CCP. Joint damage was assessed based on Juvenile Arthritis Damage Index for Articular damage (JADI-A), a validated clinical instrument for joint damage. Elevated serum levels of anti-CII occurred in 3.1%, IgM RF in 3.6%, IgA RF in 3.1% and anti-CCP in 2.6% of the patients. Occurrence of RF and anti-CCP did to some extent overlap, but rarely with anti-CII. The polyarticular and oligoarticular extended categories were overrepresented in patients with two or more autoantibodies. Anti-CII occurred in younger children, usually without overlap with the other autoantibodies and was associated with high levels of C-reactive protein (CRP) early in the disease course. All four autoantibodies were significantly associated with joint damage, but not with active disease at the eight-year follow up.
Question: Are anti-type II collagen antibodies , anti-CCP , IgA RF and IgM RF associated with joint damage , assessed eight years after onset of juvenile idiopathic arthritis ( JIA )?
Example Answer: Yes.
### Input: Context: """,
            "Medllama-13B": """### Instruction: As a knowledgeable medical professional, based on the given context, your task is to identify the correct answer and directly provide it answering yes or no. There is no 'NA' option as one of 2 choices is definitively correct. Please be concise in your response, NO VERBOSITY. An example is provided for guidance:
Example: Context: Early appearance of antibodies specific for native human type II collagen (anti-CII) characterizes an early inflammatory and destructive phenotype in adults with rheumatoid arthritis (RA). The objective of this study was to investigate the occurrence of anti-CII, IgM RF, IgA RF and anti-CCP in serum samples obtained early after diagnosis, and to relate the occurrence of autoantibodies to outcome after eight years of disease in children with juvenile idiopathic arthritis (JIA). The Nordic JIA database prospectively included JIA patients followed for eight years with data on remission and joint damage. From this database, serum samples collected from 192 patients, at a median of four months after disease onset, were analysed for IgG anti-CII, IgM RF, IgA RF and IgG anti-CCP. Joint damage was assessed based on Juvenile Arthritis Damage Index for Articular damage (JADI-A), a validated clinical instrument for joint damage. Elevated serum levels of anti-CII occurred in 3.1%, IgM RF in 3.6%, IgA RF in 3.1% and anti-CCP in 2.6% of the patients. Occurrence of RF and anti-CCP did to some extent overlap, but rarely with anti-CII. The polyarticular and oligoarticular extended categories were overrepresented in patients with two or more autoantibodies. Anti-CII occurred in younger children, usually without overlap with the other autoantibodies and was associated with high levels of C-reactive protein (CRP) early in the disease course. All four autoantibodies were significantly associated with joint damage, but not with active disease at the eight-year follow up.
Question: Are anti-type II collagen antibodies , anti-CCP , IgA RF and IgM RF associated with joint damage , assessed eight years after onset of juvenile idiopathic arthritis ( JIA )?
Example Answer: Yes.
### Input: Context: """,
            "Vicuna-13B": """### Instruction: As a knowledgeable medical professional, based on the given context, your task is to identify the correct answer and directly provide it answering yes or no. There is no 'NA' option as one of 2 choices is definitively correct. Please be concise in your response, NO VERBOSITY. An example is provided for guidance:
Example: Context: Early appearance of antibodies specific for native human type II collagen (anti-CII) characterizes an early inflammatory and destructive phenotype in adults with rheumatoid arthritis (RA). The objective of this study was to investigate the occurrence of anti-CII, IgM RF, IgA RF and anti-CCP in serum samples obtained early after diagnosis, and to relate the occurrence of autoantibodies to outcome after eight years of disease in children with juvenile idiopathic arthritis (JIA). The Nordic JIA database prospectively included JIA patients followed for eight years with data on remission and joint damage. From this database, serum samples collected from 192 patients, at a median of four months after disease onset, were analysed for IgG anti-CII, IgM RF, IgA RF and IgG anti-CCP. Joint damage was assessed based on Juvenile Arthritis Damage Index for Articular damage (JADI-A), a validated clinical instrument for joint damage. Elevated serum levels of anti-CII occurred in 3.1%, IgM RF in 3.6%, IgA RF in 3.1% and anti-CCP in 2.6% of the patients. Occurrence of RF and anti-CCP did to some extent overlap, but rarely with anti-CII. The polyarticular and oligoarticular extended categories were overrepresented in patients with two or more autoantibodies. Anti-CII occurred in younger children, usually without overlap with the other autoantibodies and was associated with high levels of C-reactive protein (CRP) early in the disease course. All four autoantibodies were significantly associated with joint damage, but not with active disease at the eight-year follow up.
Question: Are anti-type II collagen antibodies , anti-CCP , IgA RF and IgM RF associated with joint damage , assessed eight years after onset of juvenile idiopathic arthritis ( JIA )?
Example Answer: Yes.
### Input: Context: """,
            "Medalpaca-13B": """### Instruction: As a knowledgeable medical professional, based on the given context, your task is to identify the correct answer and directly provide it answering yes or no. There is no 'NA' option as one of 2 choices is definitively correct. Please be concise in your response, NO VERBOSITY. An example is provided for guidance:
Example: Context: Early appearance of antibodies specific for native human type II collagen (anti-CII) characterizes an early inflammatory and destructive phenotype in adults with rheumatoid arthritis (RA). The objective of this study was to investigate the occurrence of anti-CII, IgM RF, IgA RF and anti-CCP in serum samples obtained early after diagnosis, and to relate the occurrence of autoantibodies to outcome after eight years of disease in children with juvenile idiopathic arthritis (JIA). The Nordic JIA database prospectively included JIA patients followed for eight years with data on remission and joint damage. From this database, serum samples collected from 192 patients, at a median of four months after disease onset, were analysed for IgG anti-CII, IgM RF, IgA RF and IgG anti-CCP. Joint damage was assessed based on Juvenile Arthritis Damage Index for Articular damage (JADI-A), a validated clinical instrument for joint damage. Elevated serum levels of anti-CII occurred in 3.1%, IgM RF in 3.6%, IgA RF in 3.1% and anti-CCP in 2.6% of the patients. Occurrence of RF and anti-CCP did to some extent overlap, but rarely with anti-CII. The polyarticular and oligoarticular extended categories were overrepresented in patients with two or more autoantibodies. Anti-CII occurred in younger children, usually without overlap with the other autoantibodies and was associated with high levels of C-reactive protein (CRP) early in the disease course. All four autoantibodies were significantly associated with joint damage, but not with active disease at the eight-year follow up.
Question: Are anti-type II collagen antibodies , anti-CCP , IgA RF and IgM RF associated with joint damage , assessed eight years after onset of juvenile idiopathic arthritis ( JIA )?
Example Answer: Yes.
### Input: Context: """
        },
        "MedQA-USMLE": {}
    }

def GPU_info():
    from rich.console import Console
    from rich.table import Table
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("GPU ID", justify="right", style="dim", width=8)
    table.add_column("Name", min_width=20)
    table.add_column("Compute Capability")
    table.add_column("Total Memory (GB)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info = torch.cuda.get_device_properties(i)
            table.add_row(str(i), gpu_info.name, f"{gpu_info.major}.{gpu_info.minor}",
                          str(round(gpu_info.total_memory / 1e9, 2)))
    else:
        table.add_row("N/A", "CUDA not available", "-", "-")
    console.print(table)


# def set_seed(seed = CONFIG.seed):
#     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed(seed)

