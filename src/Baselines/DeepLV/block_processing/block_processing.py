import sys
import multiprocessing
import numpy as np
import pandas as pd
import csv
import re

block_set = {"DoStatement", "WhileStatement", "SynchronizedStatement", "IfStatement", "SwitchStatement", "TryStatement", "EnhancedForStatement", "ForStatement", "MethodDeclaration", "CatchClause", "Block" , "SwitchCase"}
syntactic_filter_set = {"Block", "SimpleName", "SimpleType", "QualifiedName", "ParameterizedType", "PrimitiveType", "SingleVariableDeclaration", "ArrayType", "TypeLiteral"}
block_dict = {}
target_dict = {}
methods_dict = {} 
methods_lines = {}
target_dict_logged = {}
level_dict_logged = {}
message_dict_logged = {}
target_dict_nonlogged = {}


def read_logs(filename):
    f = open('original_logs/logs-' + filename+ '.txt')
    lines = f.readlines()
    f.close()
    return lines


def get_classname(method):
    fullpath = method.split('.')
    class_name = fullpath[-3] + '.' + fullpath[-2]+'.java'
    return class_name


def read_AST_file(filename):
    f = open('AST/AST-'+filename+'.txt')
    lines = f.readlines()
    f.close()

    return lines


def parse_ASTlines(ASTlines):
    lines = []
    #parse ASTlines by regex
    for astline in ASTlines:
        
        astType = re.findall(r'<type>([^<]+)</type>', astline)[0]
        location = re.findall(r'<method>([^<]+)</method>', astline)[0]
        begin = re.findall(r'<begin>([^<]+)</begin>', astline)[0]
        end = re.findall(r'<end>([^<]+)</end>', astline)[0]
        #content = re.findall(r'<name>([^<]+)</name>', astline)[0]
        content = re.findall(r'<name>(.*?)</name>', astline)[0]
        lines.append([astType, location, begin, end, content])
        #for every AST line, 0: type, 1: location, 2: beginline, 3: endline, 4: content
    return lines



def parse_Loglines(Loglines):
    loglines = []
    #parse ASTlines by regex
    for logline in Loglines:   
        callsite = re.findall(r'<callsite>([^<]+)</callsite>', logline)[0]
        level = re.findall(r'<level>([^<]+)</level>', logline)[0]
        line = re.findall(r'<line>([^<]+)</line>', logline)[0]     
        if(re.findall(r'<constant>([^<]+)</constant>', logline)):
            content = re.findall(r'<constant>([^<]+)</constant>', logline)[0]
            loglines.append([level, line, content, callsite])
        else:
            loglines.append([level, line, 'No message', callsite])      
        #0: level, 1: line number, 2: content, 3: callsite

    return loglines


def if_log_line(ast, loglines):
    for log in loglines:
        #print (get_classname(log[3]), get_classname(astlist[1]))
        #print (log[1], astlist[2])
        if(get_classname(log[3]) == get_classname(astlist[1]) and int(log[1]) == int(astlist[2])):
            #print ('1')
            return True
    return False



def if_diff_levels(value_list):
    if len(value_list) > 1:
        for i in range (0, len(value_list)-1):
            for j in range (i+1, len(value_list)):
                if value_list[i][0] != value_list[j][0]:
                    return 2
    else:
        return 0
    return 1

def not_level_guard(string):
    if "enabled" in string:
        if "info" in string or "debug" in string or "trace" in string:
            return False
    return True

    #0: <= 1 log in the block, 1: multiple logs at the same level, 2: multiple logs at different levels


def get_level_id(log, current_level):
    log_level = re.findall(r'<level>([^<]+)</level>', log)[0]
    message = '-'
    if(re.findall(r'<constant>([^<]+)</constant>', log)):
        message = re.findall(r'<constant>([^<]+)</constant>', log)[0]
    if log_level == 'trace':
        level_id = 0
    elif log_level == 'debug':
        level_id = 1
    elif log_level == 'info':
        level_id = 2
    elif log_level == 'warn':
        level_id = 3
    elif log_level == 'error':
        level_id = 4
    else:
        level_id = 5
    if level_id > current_level:
        return level_id, message
    else:
        return current_level, message
    

def get_level_name(level_id):
    if level_id == 0:
        return "trace"
    elif level_id == 1:
        return "debug"
    elif level_id == 2:
        return "info"
    elif level_id == 3:
        return "warn"
    elif level_id == 4:
        return "error"
    elif level_id == 5:
        return "fatal"
    else:
        return "unknown"

def label_blocks(target_dict, loglines):
    for key, value in target_dict.items():
        logged_flag = False
        #level id: 0 - trace, 1 - debug, 2 - info, 3 - warn, 4 - error, 5 - fatal
        level_id = 0
        message = '-'
        for log in loglines:     
            log_class = get_classname(re.findall(r'<callsite>([^<]+)</callsite>', log)[0])
            log_line = int(re.findall(r'<line>([^<]+)</line>', log)[0])
            key_class = re.findall(r'<class>([^<]+)</class>', key)[0]
            key_start = int(re.findall(r'<start>([^<]+)</start>', key)[0])
            key_end = int(re.findall(r'<end>([^<]+)</end>', key)[0])
            if log_line >= key_start and log_line <= key_end and log_class == key_class:
                level_id, message = get_level_id(log, level_id)
                logged_flag = True
        if logged_flag == True:
            target_dict_logged[key] = value
            level_dict_logged[key]=get_level_name(level_id)
            message_dict_logged[key]= message
        else:
            target_dict_nonlogged[key] = value


def get_methods_dict (node): # set the startline of the first node of a method as it's startline
    if node[1] in methods_dict:
        if int(methods_dict[node[1]]) > int(node[2]):
            methods_dict[node[1]] = node[2]
    else:
        methods_dict[node[1]] = node[2]


def get_methods_lines (methods_dict):
    for key, value in methods_dict.items():
        class_name = get_classname(key)
        if class_name in methods_lines:
            methods_lines[class_name].append(int(value))
        else:
            methods_lines[class_name] = []

    for key, value in methods_lines.items():
        value.sort()
        #print (key)
        #print (value)


def get_method_start_line_for_AST (class_name, start_line):
    method_start_line = int(start_line)
    memory_line = 1
    if methods_lines[class_name]:
        for v in methods_lines[class_name]:
            if int(v) >= int(start_line):
                #print (memory_line)
                return int(memory_line)
            else:
                memory_line = int(v)
    else:
        return int(method_start_line)


if __name__=='__main__':

    ASTlines = read_AST_file(sys.argv[1])
    loglines = read_logs(sys.argv[1])

    ASTlists = parse_ASTlines(ASTlines)
    loglists = parse_Loglines(loglines)

    for astlist in ASTlists:
        get_methods_dict(astlist)
        #filter level-guard if statements
        ast_content = astlist[4].lower()[0:40]
        #for every AST line, 0: type, 1: location, 2: beginline, 3: endline, 4: content
        if astlist[0] in block_set and not_level_guard(ast_content):
            if astlist[1] in block_dict:
                if (astlist[2]) not in block_dict[astlist[1]]:
                    block_dict[astlist[1]].append(int(astlist[2]))
                if (astlist[3]) not in block_dict[astlist[1]]:
                    block_dict[astlist[1]].append(int(astlist[3]))
                    
            else:
                block_dict[astlist[1]] = []
    get_methods_lines(methods_dict)

    for key, value in block_dict.items():
        value.sort()




    for key, value in block_dict.items():
        for i in range (0, len(value)-1):
            dict_key = '<class>' + get_classname(key) + '</class>' + '<start>' + str(value[i]) + '</start>' + '<end>' + str((value[i+1])-1) + '</end>' 
            target_dict[dict_key] = []


    m_start_line = 0
    for key, value in target_dict.items():
        class_name = re.findall(r'<class>([^<]+)</class>', key)[0]
        start_line = re.findall(r'<start>([^<]+)</start>', key)[0]
        m_start_line = get_method_start_line_for_AST(class_name, start_line)    
        if m_start_line is not None:
            if int(m_start_line) == 1:
                m_start_line = start_line
        else:
            m_start_line = start_line

        end_line = re.findall(r'<end>([^<]+)</end>', key)[0]
        #print (key)
        for astlist in ASTlists:
            if astlist[0] not in syntactic_filter_set and int(astlist[2]) <= int(end_line) and int(astlist[2]) >= int(m_start_line) and class_name == get_classname(astlist[1]):
                if(if_log_line(astlist, loglists)==False):
                    value.append(astlist[0])
                


    label_blocks(target_dict, loglines)
    result_list_logged = []
    for key, value in target_dict_logged.items():
        result_list_logged.append([key, value, level_dict_logged[key], message_dict_logged[key]])

    result_list_nonlogged = []
    for key, value in target_dict_nonlogged.items():
        result_list_nonlogged.append([key, value])




    header_logged = ['Key', 'Values', 'Level', 'Message']
    logged_dict_to_write=pd.DataFrame(columns=header_logged,data=result_list_logged)
    logged_dict_to_write.to_csv('blocks/logged_syn_' + sys.argv[1] + '.csv')


