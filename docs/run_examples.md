# User Guide
This document details the running examples for different models in `lmms_eval`. We include commandas on how to prepare environments for different model and some commands to run these models

## Environmental Variables

Before running experiments and evaluations, we recommend you to export following environment variables to your environment. Some are necessary for certain tasks to run.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>" 
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
# Other possible environment variables include 
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

## Some common environment issue
Sometimes you might encounter some common issues for example error related to `httpx` or `protobuf`. To solve these issues, you can first try

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26;
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece;
```

# Image Model

### LLaVA
First, you will need to clone repo of `lmms_eval` and repo of [`llava`](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/inference)

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

cd /path/to/LLaVA-NeXT;
python3 -m pip install -e ".[train]";


TASK=$1
CKPT_PATH=$2
CONV_TEMPLATE=$3
MODEL_NAME=$4
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

#mmbench_en_dev,mathvista_testmini,llava_in_the_wild,mmvet
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava \
    --model_args pretrained=$CKPT_PATH,conv_template=$CONV_TEMPLATE,model_name=$MODEL_NAME \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/
```
If you are trying to use large LLaVA models such as LLaVA-NeXT-Qwen1.5-72B, you can try adding `device_map=auto` in model_args and change `num_processes` to 1.

### IDEFICS2

You won't need to clone any other repos to run idefics. Making sure your transformers version supports idefics2 would be enough

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

python3 -m pip install transformers --upgrade;

TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model idefics2 \
    --model_args pretrained=HuggingFaceM4/idefics2-8b \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```

### InternVL2

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;


python3 -m pip install flash-attn --no-build-isolation;
python3 -m pip install torchvision einops timm sentencepiece;


TASK=$1
CKPT_PATH=$2
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12380 -m lmms_eval \
    --model internvl2 \
    --model_args pretrained=$CKPT_PATH \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 
```


### InternVL-1.5
First you need to fork [`InternVL`](https://github.com/OpenGVLab/InternVL)

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

cd /path/to/InternVL/internvl_chat
python3 -m pip install -e .;

python3 -m pip install flash-attn==2.3.6 --no-build-isolation;


TASK=$1
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model internvl \
    --model_args pretrained="OpenGVLab/InternVL-Chat-V1-5"\
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 

```

### Xcomposer-4KHD and Xcomposer-2d5

Both of these two models does not require external repo

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;


python3 -m pip install flash-attn --no-build-isolation;
python3 -m pip install torchvision einops timm sentencepiece;

TASK=$1
MODALITY=$2
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

# For Xcomposer2d5
accelerate launch --num_processes 8 --main_process_port 10000 -m lmms_eval \
    --model xcomposer2d5 \
    --model_args pretrained="internlm/internlm-xcomposer2d5-7b",device="cuda",modality=$MODALITY\
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/ 

# For Xcomposer-4kHD
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model xcomposer2_4khd \
    --model_args pretrained="internlm/internlm-xcomposer2-4khd-7b" \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./logs/

```

### InstructBLIP

```bash
cd /path/to/lmms-eval
python3 -m pip install -e .;

python3 -m pip install transformers --upgrade;

CKPT_PATH=$1
TASK=$2
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model instructblip \
    --model_args pretrained=$CKPT_PATH \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix instructblip \
    --output_path ./logs/ 

```
