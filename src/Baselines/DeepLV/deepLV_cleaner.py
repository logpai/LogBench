import pandas
import numpy as np
import os
import javalang
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge 
import re
import numpy as np
from sklearn import metrics
import pandas as pd

def level_acc(classification_pred, classification_label) -> float:
    level_map = {'trace':0., 'debug':1., 'info':2., 'warn':3., 'error':4.}
    new_pred = []
    new_label = []
    length = len(classification_pred)
    for idx in range(length):
        predict = classification_pred[idx]
        label = classification_label[idx]
        if predict in level_map.keys() and label in level_map.keys():
            pred_sum = level_map[predict]
            label_sum = level_map[label]
            new_pred.append(pred_sum)
            new_label.append(label_sum)
    matches = sum(x == y for x, y in zip(new_pred, new_label))
    total_elements = len(new_pred)
    accuracy = matches / total_elements
    return accuracy

def query_level(level: float) -> str:
    if level == 1.:
        return 'trace'
    elif level == 2.:
        return 'debug'
    elif level == 3.:
        return 'info'
    elif level == 4.:
        return 'warn'
    elif level == 5.:
        return 'error'
    else:
        return ''
        
def aod(classification_pred, classification_label) -> float:
    level_map = {'trace':1., 'debug':2., 'info':3., 'warn':4., 'error':5.}
    max_distance = {'trace':4., 'debug':3., 'info':2., 'warn':3., 'error':4.}

    distance_sum = 0.
    noise = 0.
    length = len(classification_pred)
    
    for idx in range(length):
        try:
            predict = classification_pred[idx]
            label = classification_label[idx]
            pred_sum = level_map[predict]
            label_sum = level_map[label]
            level = query_level(label_sum)
            _distance = abs(label_sum - pred_sum)
            distance_sum = distance_sum + (1 - _distance / max_distance[level])
        except Exception as e:
            noise = noise+1
    aod = distance_sum / (length-noise)    
    return aod

def extract_quoted_strings(s):
    quoted_strings = re.findall(r'"([^"]*)"', s)
    " ".join(quoted_strings)
    remaining = re.sub(r'"[^"]*"', '', s)
    char_to_remove = ['+', ',']
    for char in char_to_remove:
        remaining = remaining.replace(char, '')
    var_list_origin = remaining.split(' ')
    var_list = [item for item in var_list_origin if (not item == ' ')]
    var_list = [item for item in var_list if item]
    return quoted_strings, var_list

def extract_outer_brackets(s):
    stack = []
    result = []

    for m in re.finditer(r"[()]", s):
        char, pos = m.group(0), m.start(0)
        if char == "(":
            stack.append(pos)
        elif char == ")":
            if len(stack) == 1:
                result.append(s[stack.pop() + 1:pos])
            else:
                stack.pop()
    return result

def extract_level(statement):
    parts = statement.split('.')
    for part in parts:
        if '(' in part:
            level = part.split('(')[0]
            return level.strip()
    return ''



def extract_text(statement):
    bracket_contents = extract_outer_brackets(statement)
    if bracket_contents:  # Check if the list is not empty
        # Pass the first item (contents of the first set of brackets) to extract_quoted_strings
        quoted_strings, remaining = extract_quoted_strings(bracket_contents[0])
        quoted_strings_combined = ' '.join(quoted_strings)
        return quoted_strings_combined
    else:
        return ''  # Return an empty string if no brackets are found

df = pd.read_csv('logbench.csv')
df = df[df['Statement'].apply(lambda x: len(x.splitlines()) == 1)]
df['level'] = df['Statement'].apply(extract_level)
df['text'] = df['Statement'].apply(extract_text)


df.to_csv('logbench_cleaned.csv', index=False)