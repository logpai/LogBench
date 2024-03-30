import functools
import os
import gin
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from contextlib import contextmanager
import logging as py_logging
import t5
from t5.data import postprocessors as t5_postprocessors
from t5.seqio import Feature,SentencePieceVocabulary
from mesh_tensorflow.transformer.learning_rate_schedules import slanted_triangular 
from mesh_tensorflow.transformer.learning_rate_schedules import truncated_rsqrt
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from t5 import models

BASE_DIR = "gs://xxxx" #@param { type: "string" }
TPU_TOPOLOGY = "2x2"
tpu = tf.distribute.cluster_resolver.TPUClusterResolver("grpc://xx.xx.xx.xx")  # TPU detection
TPU_ADDRESS = tpu.get_master()
tf.disable_v2_behavior()
tf.get_logger().propagate = False
py_logging.root.setLevel('INFO')

@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)

path_finetuning = BASE_DIR + '/datasets/Fine-tuning/train.tsv' #@param { type: "string" }
path_eval = BASE_DIR + '/datasets/Fine-tuning/eval.tsv' #@param { type: "string" }
path_test = BASE_DIR + '/datasets/Fine-tuning/test.tsv' #@param { type: "string" }

nq_tsv_path = {
    "train":      path_finetuning,
    "validation": path_test
}

num_nq_examples = dict(train=106382, validation=12020)

vocab_model_path = BASE_DIR + '/Code/SP_LOG.model' #@param { type: "string" }
vocab_path = BASE_DIR + '/Code/SP_LOG.vocab' #@param { type: "string" }


TaskRegistry = t5.data.TaskRegistry
TfdsTask = t5.data.TfdsTask


def get_default_vocabulary():
  return SentencePieceVocabulary(vocab_model_path, 100)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": Feature(
        vocabulary=get_default_vocabulary(), add_eos=True, required=False),

    "targets": Feature(
        vocabulary=get_default_vocabulary(), add_eos=True)
}

def nq_dataset_task(split, shuffle_files=True):
  # We only have one file for each split.
  del shuffle_files

  # Load lines from the text file as examples.

  ds = tf.data.TextLineDataset(nq_tsv_path[split])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["string","string"],
                        field_delim="\t", use_quote_delim=True),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  ds = ds.map(lambda *ex: dict(zip(["input", "output"], ex)))
  return ds

print("A few raw train examples...")
for ex in tfds.as_numpy(nq_dataset_task("train").take(5)):
  print(ex)

def preprocessing(ds):
  
  def to_inputs_and_targets(ex):
        x_input = tf.strings.strip(ex['input'])
        y_label = tf.strings.strip(ex['output']) 
        inputs = tf.strings.join([x_input], separator=' ')
        class_label = tf.strings.join([y_label], separator=' ')
        return {'inputs': inputs, 'targets': class_label}
    
  return ds.map(to_inputs_and_targets, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

t5.data.TaskRegistry.remove('log_injection')
t5.data.TaskRegistry.add(
    "log_injection",
    dataset_fn=nq_dataset_task,
    splits=["train","validation"],
    text_preprocessor=[preprocessing],
    output_features = DEFAULT_OUTPUT_FEATURES,
    metric_fns=[t5.evaluation.metrics.accuracy],
    num_input_examples=num_nq_examples
)

nq_task = t5.data.TaskRegistry.get("log_injection")
ds = nq_task.get_dataset(split="train", sequence_length={"inputs": 512, "targets": 512})
print("A few preprocessed training examples...")
for ex in tfds.as_numpy(ds.take(5)):
  print(ex)

starter_learning_rate = 0.01
end_learning_rate = 0.001
decay_steps = 10000

learning_rate_fn = PolynomialDecay(
     starter_learning_rate,
     decay_steps,
     end_learning_rate,
     power=0.5)

MODEL_SIZE = "small" 

MODEL_DIR = BASE_DIR + '/modeltest/'#@param { type: "string" }

PRETRAINED_DIR=BASE_DIR + '/denoising_task_model/'#@param { type: "string" }


model_parallelism, train_batch_size, keep_checkpoint_max = {
    "small": (1, 128, 16),
    "base": (2, 128, 8),
    "large": (8, 64, 4),
    "3B": (8, 16, 1),
    "11B": (8, 16, 1)}[MODEL_SIZE]

tf.io.gfile.makedirs(MODEL_DIR)

model = t5.models.MtfModel(
    model_dir=PRETRAINED_DIR,
    tpu=TPU_ADDRESS,
    #tpu_job_name="node-1",
    #tpu_zone="us-central1-f",
    #gcp_project="lance",
    tpu_topology=TPU_TOPOLOGY,
    model_parallelism=model_parallelism,
    batch_size=train_batch_size,
    learning_rate_schedule = learning_rate_fn, #pick the correct scheduler, according to the model you want to train
    sequence_length={"inputs": 512, "targets": 512},
    save_checkpoints_steps=5000,
    keep_checkpoint_max=keep_checkpoint_max,
    iterations_per_loop=100,
)

PATH_GIN_FILE_NO_PT = BASE_DIR + '/Configs/no_pretraining_operative_config.gin'
PATH_GIN_FILE_MT = BASE_DIR + '/Configs/multi-task_operative_config.gin'
PATH_GIN_FILE_DENOISE = BASE_DIR + '/Configs/denoise_only_operative_config.gin'
PATH_GIN_FILE_LOG_STMT = BASE_DIR + '/Configs/log_stmt_only_operative_config.gin'

#with gin.unlock_config():
#    gin.parse_config_file(PATH_GIN_FILE_DENOISE)
#    #RUN FINE-TUNING
#    TRAIN_STEPS = 200000
#    model.finetune(mixture_or_task_name="log_injection",
#                  finetune_steps=TRAIN_STEPS,
#                pretrained_model_dir=PRETRAINED_DIR)

    # If the no-pretraining experiment is the one you want to run, then, uncomment the following and comment model.finetune
    # Also, make sure to upload the slanted_operative.gin
    #model.train("log_injection", TRAIN_STEPS)
    #model.bach_size=32
    #model.eval(
    #mixture_or_task_name="log_injection",
    #checkpoint_steps=-1
    #)
#dataset_list = ["cassandra","elasticsearch","flink","hbase","wicket","zookeeper"]
dataset_list = ['logstudy']
for item in dataset_list:
    model.batch_size = 256
    input_file = BASE_DIR + f'/datasets/logr_input/lance_function_transformed.txt'#@param { type: "string" }
    output_file = BASE_DIR+ f'/datasets/logr_input/lance_function_transformed_result.txt'#@param { type: "string" }
    model.predict(input_file, output_file, checkpoint_steps=-1, vocabulary=get_default_vocabulary())