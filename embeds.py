import argparse
import transformations as tf
import os    
import librosa
import laion_clap
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

SR = 48000 # clap requires 48kHz sample rate
CLAP_FUSION = False # fusion or not snipets from the first middle and last third of the sample

def load_clap():
    mdl = laion_clap.CLAP_Module(enable_fusion=CLAP_FUSION)
    mdl.load_ckpt() # download the default pretrained checkpoint.
    return mdl

def clap_embeds(y, mdl, transf, tfparam):
    x, sr = tf.tf_dict[transf](y, SR, tfparam)
    x = x.reshape(1, -1) # (n_channels, embeds_size)
    x = mdl.get_audio_embedding_from_data(
        x = x, use_tensor=False)
    return x

def generate(load_mdl, embed_fn, file_paths, df, transformations, model, out_path):
    mdl = load_mdl()
    for d, file_path in tqdm(file_paths):
        file = os.path.basename(file_path)
        df_new = pd.DataFrame({
                "ds_name": [],
                "file": [],
                "file_path": [],
                "model": [],
                "transf": [],
                "transf_param_name": [],
                "file_embeds": [],
                "file_embeds_path": []
            })
        try:
            y, sr = librosa.load(file_path, sr=SR)
            for transf in transformations:
                transf = transf.lower().replace("_", "")
                tfparams = tf.tf_dict_params[transf] # params set in transformation.py
                for tfparam in tfparams:
                    transf_param_name = str(tfparam).lower().replace("_", "")[:5]                                     
                    file_embeds = f"x_{d}_{file.split(".")[0].replace("_", "").lower() \
                                            }_{model}_{transf}_{transf_param_name.replace(".","p")}.npy"
                    file_embeds_path = os.path.join(out_path, file_embeds)
                    if df.loc[
                        (df["ds_name"]==d) &
                        (df["file"]==file) &
                        (df["model"]==model) &
                        (df["transf"]==transf) &
                        (df["transf_param_name"]==transf_param_name) &
                        (df["file_embeds"]==file_embeds)
                    ].empty:                            
                        x = embed_fn(y, mdl, transf, tfparam)
                        np.save(file_embeds_path, x)
                        new_row = pd.DataFrame([{
                            "ds_name": d,
                            "file": file,
                            "file_path": file_path,
                            "model": model,
                            "transf": transf,
                            "transf_param_name": transf_param_name,
                            "file_embeds": file_embeds,
                            "file_embeds_path": os.path.abspath(file_embeds_path)
                        }])
                        df_new = pd.concat([df_new, new_row], ignore_index=True)
        except Exception:
            del(df_new)
            print(f"Exception in file: {file}")
            continue
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv("metadata.csv", index=False)        
        del(df_new)                
    del(mdl)

def main():
    parser = argparse.ArgumentParser(description="Generate Embeds CLI")
    parser.add_argument("--tracks", type=str, help="Path to file containing the selected tracks")
    parser.add_argument("--datasets", type=str, nargs="+", help="This name will be used to save the embeddings")
    parser.add_argument("--models", type=str, nargs="+", help="Model that will generate the embeddings: CLAP, AudioCLIP, etc...")
    parser.add_argument("--transformations", type=str, nargs="+", help="Transformations to the sounds: pitch, time stretching, etc...")
    parser.add_argument("--output_path", type=str, help="Full path where the embeddings will be saved on")
    args = parser.parse_args()

    tracks = args.tracks
    print(f"tracks: {args.tracks}")

    datasets = args.datasets
    print(f"datasets: {args.datasets}")
    
    models = args.models
    print(f"models: {args.models}")
    
    transformations = args.transformations
    print(f"transformations: {args.transformations}")

    out_path = os.path.abspath(args.output_path)
    print(f"output_path: {args.output_path}")
    os.makedirs(out_path, exist_ok=True)
    
    if Path("metadata.csv").exists():
        df = pd.read_csv("metadata.csv")
    else:
        df = pd.DataFrame({
            "ds_name": [],
            "file": [],
            "file_path": [],
            "model": [],
            "transf": [],
            "transf_param_name": [],
            "file_embeds": [],
            "file_embeds_path": []
        })

    file_paths = []
    with open(tracks, "r") as t:
        tracks = json.load(t)
        for d in datasets:
            if d in tracks.keys():
                for track in tracks[d]:
                    file_paths.append((d, track))

    for model in models:
        model = model.lower().replace("_", "")

        ### CLAP ###
        if model == "clap":
            generate(load_clap, clap_embeds, file_paths, 
                df, transformations, model, out_path)

if __name__ == "__main__":
    main()
    
