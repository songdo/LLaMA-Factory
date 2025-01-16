NNODES=1          # 物理节点数，就是电脑数量
NODE_RANK=0       # 物理节点的序号，每个电脑的序号
NPROC_PER_NODE=2  # 每个物理节点上面进程的数量，等价于每个电脑上GPU的数量，就是可以开几个进程。

MASTER_ADDR=127.0.0.1
MASTER_PORT=520
DS_CONFIG_PATH=/workspace/sy/my_project_v1/llm/mllm/LLaMA-Factory/examples/deepspeed/ds_z2_config.json
MODEL_PATH=/workspace/sy/disk/Qwen/Qwen2-7B-Instruct
OUTPUT_PATH=/workspace/sy/my_project_v1/llm/mllm/LLaMA-Factory/study/output

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
  "

torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --model_name_or_path $MODEL_PATH \
    --dataset alpaca_zh_demo \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 30 \
    # --bf16
    --fp16