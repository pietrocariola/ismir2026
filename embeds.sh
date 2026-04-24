export PYTHONPATH=/home/gpu1/code/ismir2026:/home/gpu1/code/ismir2026/AudioCLIP:$PYTHONPATH
python /home/gpu1/code/ismir2026/embeds.py  --tracks "tracks.json" \
                    --datasets "fmasmall" \
                    --models "clap" "audioclip" "wav2clip"\
                    --transformations "identity" "pitchshift" "timestretch" \
                    "highpass" "lowpass" "clipper" "noiseadder" "bitcrush" "gain" \
                    --output_path "/home/gpu1/embeds/ismir2026" \
                            