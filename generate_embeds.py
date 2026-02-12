import argparse
import transformations as tf
import os    
import librosa
import laion_clap
from pathlib import Path
import numpy as np
import pandas as pd
import uuid
from tqdm import tqdm

CLAP_SR = 48000 # paper requires clap with 48kHz sample rate
CLAP_FUSION = False # fusion or not snipets from the first middle and last third of the sample

def main():
    parser = argparse.ArgumentParser(description="Generate Embeds CLI")
    parser.add_argument("--dataset_path", type=str, help="Full path to load the dataset")
    parser.add_argument("--dataset_name", type=str, help="This name will be used to save the embeddings")
    parser.add_argument("--models", type=str, nargs="+", help="Model that will generate the embeddings: CLAP, AudioCLIP, etc...")
    parser.add_argument("--transformations", type=str, nargs="+", help="Transformations to the sounds: pitch, time stretching, etc...")
    parser.add_argument("--output_path", type=str, help="Full path where the embeddings will be saved on")
    args = parser.parse_args()
    print(f"dataset_path: {args.dataset_path}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"transformations: {args.transformations}")

    models = args.models
    ds_path = args.dataset_path
    files = os.listdir(ds_path)
    transformations = args.transformations
    ds_name = args.dataset_name.lower().replace("_", "")
    out_path = args.output_path

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

    for model in models:
        model = model.lower().replace("_", "")

        ### CLAP ###
        if model == "clap":
            mdl = laion_clap.CLAP_Module(enable_fusion=CLAP_FUSION)
            mdl.load_ckpt() # download the default pretrained checkpoint.
            for file in tqdm(files):
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
                    y, sr = librosa.load(os.path.join(ds_path, file), sr=CLAP_SR)
                    for transf in transformations:
                        transf = transf.lower().replace("_", "")
                        tfparams = tf.tf_dict_params[transf] # params set in transformation.py
                        for tfparam in tfparams:
                            transf_param_name = str(tfparam).lower().replace("_", "")[:5].replace(".","p")                                      
                            file_embeds = f"{ds_name}_{file.split(".")[0].replace("_", "").lower() \
                                                    }_{model}_{transf}_{transf_param_name}.npy"
                            file_write_path = os.path.join(out_path, file_embeds)
                            if df.loc[
                                (df["ds_name"]==ds_name) &
                                (df["file"]==file) &
                                (df["model"]==model) &
                                (df["transf"]==transf) &
                                (df["transf_param_name"]==transf_param_name) &
                                (df["file_embeds"]==file_embeds)
                            ].empty:                            
                                x, sr = tf.tf_dict[transf](y, sr, tfparam)
                                x = x.reshape(1, -1) # (n_channels, embeds_size)
                                x = mdl.get_audio_embedding_from_data(
                                    x = x, use_tensor=False)
                                np.save(file_write_path, x)
                                new_row = pd.DataFrame([{
                                    "ds_name": ds_name,
                                    "file": file,
                                    "file_path": os.path.abspath(os.path.join(ds_path, file)),
                                    "model": model,
                                    "transf": transf,
                                    "transf_param_name": transf_param_name,
                                    "file_embeds": file_embeds,
                                    "file_embeds_path": os.path.abspath(file_write_path)
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

if __name__ == "__main__":
    main()
    
