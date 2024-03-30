# LoGenText-Plus

The implementation of "LoGenText-Plus: Improving Neural Machine Translation-based Logging Texts Generation with Syntactic Templates"

> This code and dataset are based on [Context-Aware Model on Fairseq](https://github.com/libeineu/Context-Aware) and [LoGenText](https://github.com/conf-202x/experimental-result).

## Requirements and Installation

* Pytorch  >= 1.5.1
* Python version >= 3.6

1. `conda create --name <env> --file requirements.txt`

## Stage 1: template generation


Note: `<root_dir>` is the path to the replication package.

### Train and inference for templates

> 1. Run the following command to start the pre-training: 
```
cd <root_dir>/code/template-gen/pre-train
bash runs/pre-train.sh
```


> 2. Run the following command to train a basic model: 
```
cd <root_dir>/code/template-gen/basic-train
bash runs/basic-train.sh <root_dir> <project>
```
`<project>` is the project name in lowercase, which can be activemq, ambari, etc. 

> 3. Run the following command to train and generate the templates for a certain <project>: 
```
cd <root_dir>/code/template-gen/ast-temp
bash runs/temp-gen.sh <root_dir> <project>
```
`<project>` should be the same with the project in step 2, and the generated templates can be found in `saved_checkpoints/pre-ast-templete/<project>`. 


## Stage 2: template-based logging text generation

Note: `<root_dir>` is the path to the replication package.

### Train and inference for logging texts

> 1. Run the following command to start the pre-training: 
```
cd <root_dir>/code/logging-gen/pre-train
bash runs/pre-train.sh
```

> 2. Run the following command to train a basic model: 
```
cd <root_dir>/code/logging-gen/basic-train
bash runs/basic-train.sh <root_dir> <project>
```
`<project>` is the project name in lowercase, which can be activemq, ambari, etc. 

> 3. Run the following command to train and generate the logging texts for a certain <project>: 
```
cd <root_dir>/code/logging-gen/ast-temp
bash runs/log-gen.sh <root_dir> <project>
```
`<project>` should be the same with the project in step 2, and the generated logging texts can be found in `translations/1/<project>`. 

## Results

The results can be found in the `results` folder, which is organized by project.

## Data

The dataset can be found in the `dataset` folder, which is organized by project. It has the following structure:
```
dataset
├── <project>
│   ├── dev.code.1.templete
│   ├── dev.log
│   ├── dev.log.1.templete
│   ├── dev.pre-ast
│   ├── test.code.1.templete
│   ├── test.code.gen.ast.similar.1.templete
│   ├── test.log
│   ├── test.log.1.templete
│   ├── test.pre-ast
│   ├── train.code.1.templete
│   ├── train.log
│   ├── train.log.1.templete
│   └── train.pre-ast
```
- `<project>` is one of the studied projects, suach as `activemq`.
- `train/dev/test.log` are the files containing the extracted `logging texts` target sequence.
- `train/dev/test.pre-ast` are the files containing the `ASTs` context.
- `train/dev/test.code.1.templete` are the files containing `pre-log code + template from logging text in similar code`.
- `train/dev/test.log.1.template` are the files containing the template extracted from the `logging text`.
- `test.code.gen.ast.similar.1.templete` are the file containing the `pre-log code + predicted template`.