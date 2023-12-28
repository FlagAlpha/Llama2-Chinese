# 从社区下载的模型文件, 文件下下包含：checklist.chk  checksum.chk  consolidated.00.pth  params.json  tokenizer_checklist.chk  tokenizer.model
ckpt_dir="/path/llama-2-7b-chat"

token_dir="$ckpt_dir/tokenizer.model"

torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=8999 \
  example_chat_completion.py \
  $ckpt_dir \
  $token_dir
