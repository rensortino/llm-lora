# llm-lora

## Usage

```bash
train_lora.py \
    --base_model {BASE_MODEL_NAME} \
    --data_path {PATH_TO_DATA_JSON} \
    --prompt_template_name {TEMPLATE_NAME} \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 10 \
    --cutoff_len 256 \
    --lora_alpha 16 \
    --learning_rate 1e-4 \
    --group_by_length \
    --output_dir {OUTPUT_DIR} \
    --lora_r 8 
```
