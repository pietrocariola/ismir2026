python generate_embeds.py   --dataset_path "/home/gpu1/datasets/fma/fma_small/fma_pool" \
                            --dataset_name "fmasmall" \
                            --models "clap" \
                            --transformations "identity" "pitchshift" "timestretch" \
                            --output_path "./embeds" \
                            