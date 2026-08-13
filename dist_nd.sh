python /home/gpu3/visfma/dist_nd.py --input_path "/home/gpu3/visfma/embeds" \
                    --datasets "fmasmallx" \
                    --models "clap" "audioclip" "wav2clip" \
                    --transformations "identity" "pitchshift" "timestretch" \
                    "highpass" "lowpass" "clipper" "noiseadder" "bitcrush" "gain" \
                    --output_path "/home/gpu3/visfma/dist_nd" \
                            