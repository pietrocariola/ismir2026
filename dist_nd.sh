python /home/gpu3/visfma/dist_nd.py --input_path "/home/gpu3/visfma/embeds" \
                    --datasets "fmasmallx" \
                    --models "clap" "audioclip" "wav2clip" \
                    --transformations "pitchshift" "timestretch" \
                    "highpass" "lowpass" "clipper" "noiseadder" "bitcrush" "gain" \
                    --normalize 1 \
                    --calc_mode 'cos' \
                    --output_path "/home/gpu3/visfma/cos_norm_nd" \
                            