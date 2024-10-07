data_path=data/comfort_ball

models=(
    "gpt-4o"
)

while true; do
	for model in "${models[@]}"; do
		python3 spatial_gen_api_multilingual.py \
			--model $model \
			--data_path $data_path
	done
done
