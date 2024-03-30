from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os
import tqdm

path = './LogBench-O_prefix_1point'
ground_truth_folder = './LogBench-O_prefix_1point'
output_path= './StarCoder_LogBench-O_prefix_1point'
FIM_INDICATOR = "<FILL_HERE>"
FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"

checkpoint = "bigcode/starcoder"
device = "cuda" 
auth_token = "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"

# Check if output_path exists, if not, create it
if not os.path.exists(output_path):
    os.makedirs(output_path)


def insert_text_to_java_file(file_name, line_number):
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    if line_number > len(lines):
        print("out of range")
    lines[line_number - 1] = lines[line_number - 1].rstrip() + FIM_INDICATOR +'\n'
    with open(file_name, 'w', encoding='utf-8') as file:
        file.writelines(lines)

        
def extract_numbers(s):
    return re.findall(r'\d+', s)

def parse_directory(dir_path, ground_truth_folder):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.java'):
            ground_truth_path = os.path.join(ground_truth_folder, file_path.split('/')[-1][:-5] + '_config.txt')
            try:
                with open(ground_truth_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        line_number = int(extract_numbers(lines[0].strip(' ')[:-1])[0])
                        insert_text_to_java_file(file_path, line_number)
            except FileNotFoundError:
                pass
        elif os.path.isdir(file_path):
            parse_directory(file_path, ground_truth_folder)

parse_directory(path,ground_truth_folder)

tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_auth_token=auth_token)
model = AutoModelForCausalLM.from_pretrained(checkpoint, use_auth_token=auth_token).to(device)

def generate(input_text):
    if FIM_INDICATOR in input_text:
        try:
            prefix, suffix = input_text.split(FIM_INDICATOR)
        except:
            raise ValueError(f"Only one {FIM_INDICATOR} allowed in prompt!")
        input_text = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=1024,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id
    )
    return (tokenizer.decode(outputs[0]))

for filename in os.listdir(path):
    if filename.endswith(".java"):
        print(filename)
        input_file_path = os.path.join(path, filename)
        
        try:
            with open(input_file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                example = f"'''\\\n{file_content}\n'''"
                processed_content = generate(example)
                output_file_path = os.path.join(output_path, filename)
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(f"{processed_content}\n")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
