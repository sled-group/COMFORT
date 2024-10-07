data_path=data/comfort_ball
models=(
    "liuhaotian/llava-v1.5-7b"
)
perspective_prompts=(
    "camera3"
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
