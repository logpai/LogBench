import os
import javalang
import re
from typing import List
import torch
import tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

path = ''
ground_truth_folder = ''

def insert_text_to_java_file(file_name, line_number):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    if line_number > len(lines):
        print("out of range")

    lines[line_number - 1] = lines[line_number - 1].rstrip() + '<insert>\n'

    with open(file_name, 'w') as file:
        file.writelines(lines)

        
def extract_numbers(s):
    return re.findall(r'\d+', s)


def parse_directory(dir_path, ground_truth_folder):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path) and file_path.endswith('.java'):
            ground_truth_path = ground_truth_folder + file_path.split('/')[-1][:-5] + '_config.txt'
            try:
                with open(ground_truth_path) as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        line_number = int(extract_numbers(lines[0].strip(' ')[:-1])[0])
                        insert_text_to_java_file(file_path, line_number)
            except FileNotFoundError:
                pass
        elif os.path.isdir(file_path):
            parse_directory(file_path, ground_truth_folder)

parse_directory(path,ground_truth_folder)
# Data procession done.


tokenizers_version = tuple(int(n) for n in tokenizers.__version__.split('.'))
if tokenizers_version < (0, 12, 1):
    print("warning: Your tokenizers version looks old and you will likely have formatting issues. We recommend installing tokenizers >= 0.12.1")

# set BIG_MODEL to use the 6.7B parameter model
BIG_MODEL = True

# use a GPU
CUDA = True

# print intermediate outputs of infilling
VERBOSE = False

if BIG_MODEL:
    model_name = "facebook/incoder-6B"
    if CUDA:
        kwargs = dict(
            revision="float16", 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        )
    else:
        kwargs = dict(
            low_cpu_mem_usage=False,
        )
else:
    model_name = "facebook/incoder-1B"
    kwargs = {}

print("loading model")
model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("loading complete")

if CUDA:
    # if you plan to fine-tune the model, you should not use half precision.
    model = model.half().cuda()

# signals the start of a document
BOS = "<|endoftext|>"
# signals the end of a generated infill
EOM = "<|endofmask|>"

def make_sentinel(i):
    # signals (1) a location to insert an infill and (2) the start of the infill generation
    return f"<|mask:{i}|>"

def generate(input: str, max_to_generate: int=128, temperature: float=0.2):
    """
    Do standard left-to-right completion of the prefix `input` by sampling from the model
    """
    input_ids = tokenizer(input, return_tensors="pt").input_ids
    if CUDA:
        input_ids = input_ids.cuda()
    max_length = max_to_generate + input_ids.flatten().size(0)
    if max_length > 2048:
        print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, do_sample=True, top_p=0.95, temperature=temperature, max_length=max_length)
    # pass clean_up_tokenization_spaces=False to avoid removing spaces before punctuation, e.g. "from ." -> "from."
    detok_hypo_str = tokenizer.decode(output.flatten(), clean_up_tokenization_spaces=False)
    if detok_hypo_str.startswith(BOS):
        detok_hypo_str = detok_hypo_str[len(BOS):]
    return detok_hypo_str

def infill(parts: List[str], max_to_generate: int=50, temperature: float=0.2, extra_sentinel: bool=True, max_retries: int=1):
    """
    Generate infills to complete a partial document, e.g.
    [A C E] -> [A B C D E], where B and D are infills that have been generated.

    parts: List[str]. list of parts of the document. One string will be
            inserted in between each element, i.e. infilling N-1 locations for a list
            of length N.
    max_to_generate: int. maximum number of tokens to generate. Keep in mind
            that the model context size is 2048.
    temperature: float. temperature parameter for sampling.
    extra_sentinel: bool. we recommend setting this to True, as it makes it
            easier for the model to end generated infills. See the footnote in 
            section 2.2 of our paper for details.
    max_retries: int. if > 1, use rejection sampling to keep sampling infills until
            all infills sample a completion token.

    returns a dictionary containing the following:
        text:  str, the completed document (with infills inserted)
        parts:  List[str], length N. Same as passed to the method
        infills:  List[str], length N-1. The list of infills generated
        retries_attempted:  number of retries used (if max_retries > 1)
    """
    assert isinstance(parts, list)
    retries_attempted = 0
    done = False

    while (not done) and (retries_attempted < max_retries):
        retries_attempted += 1

        if VERBOSE:
            print(f"retry {retries_attempted}")
        
        ## (1) build the prompt
        if len(parts) == 1:
            prompt = parts[0]
        else:
            prompt = ""
            # encode parts separated by sentinel
            for sentinel_ix, part in enumerate(parts):
                prompt += part
                if extra_sentinel or (sentinel_ix < len(parts) - 1):
                    prompt += make_sentinel(sentinel_ix)
        
        infills = []
        complete = []

        done = True

        ## (2) generate infills
        for sentinel_ix, part in enumerate(parts[:-1]):
            complete.append(part)
            prompt += make_sentinel(sentinel_ix)
            # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
            completion = generate(prompt, max_to_generate, temperature)
            completion = completion[len(prompt):]
            if EOM not in completion:
                if VERBOSE:
                    print(f"warning: {EOM} not found")
                completion += EOM
                done = False
            completion = completion[:completion.index(EOM) + len(EOM)]
            infilled = completion[:-len(EOM)]
            infills.append(infilled)
            complete.append(infilled)
            prompt += completion
        complete.append(parts[-1])
        text = ''.join(complete)

    if VERBOSE:
        print("generated text:")
        print(prompt)
        print()
        print("parts:")
        print(parts)
        print()
        print("infills:")
        print(infills)
        print()
        print("restitched text:")
        print(text)
        print()
    
    return {
        'text': text, # str, the completed document (with infills inserted)
        'parts': parts, # List[str], length N. Same as passed to the method
        'infills': infills, # List[str], length N-1. The list of infills generated
        'retries_attempted': retries_attempted, # number of retries used (if max_retries > 1)
    } 

def docstring_to_code(code, max_to_generate=50, temperature=0.2):

    parts = code.split("<insert>")
    result = infill(parts, max_to_generate=max_to_generate, temperature=temperature)
    return result

input_path = path
output_path= ''

if not os.path.exists():
    os.makedirs(output_path)

for filename in os.listdir(input_path):
    if filename.endswith(".java"):
        print(filename)
        input_file_path = os.path.join(input_path, filename)
        
        with open(input_file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            example = f"'''\\\n{file_content}\n'''"
            
            processed_content = docstring_to_code(example)
            
            output_file_path = os.path.join(output_path, filename)
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for item in processed_content['infills']:
                    output_file.write(f"{item}\n")
