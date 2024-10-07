data_path=data/comfort_car_ref_facing_right
models=(
    "liuhaotian/llava-v1.5-7b"
    "liuhaotian/llava-v1.5-13b"
    "Salesforce/instructblip-vicuna-7b"
    "Salesforce/instructblip-vicuna-13b"
    "Gregor/mblip-bloomz-7b"
    "openbmb/MiniCPM-Llama3-V-2_5"
    "MBZUAI/GLaMM-FullScope"
    "internlm/internlm-xcomposer2-vl-7b"
    "GPT-4o"
)
perspective_prompts=(
    "nop"
    "camera3"
    "addressee3"
    "reference3"
)

for model in "${models[@]}"; do
    for perspective_prompt in "${perspective_prompts[@]}"; do
        python spatial_gen.py \
            --model $model \
            --data_path $data_path \
            --perspective_prompt $perspective_prompt \
            --batch_size 1
    done
done
