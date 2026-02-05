import argparse
import transformations as tf
import os    
import librosa
import laion_clap
from pathlib import Path
import numpy as np

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
    ds_name = args.dataset_name
    out_path = args.output_path

    os.makedirs("embeds", exist_ok=True)

    for m in models:
        ### CLAP ###
        if m.lower() == "clap":
            model = laion_clap.CLAP_Module(enable_fusion=CLAP_FUSION)
            model.load_ckpt() # download the default pretrained checkpoint.
            for t in transformations:
                tparams = tf.tf_dict_params[t] # params set in transformation.py
                for tparam in tparams:
                    for f in files[:5]:
                        m = m.lower().replace("_", "")
                        ds_name = ds_name.lower().replace("_", "")
                        t = t.lower().replace("_", "")
                        tparam_name = str(tparam).lower().replace("_", "")
                        f = f.lower().replace("_", "")
                        p = os.path.join(out_path, 
                            f"{m}_{ds_name}_{t}_{tparam_name}_{str(f).split(".")[0]}.npy")
                        path = Path(p)
                        if not path.exists():
                            y, sr = librosa.load(os.path.join(ds_path, f), sr=CLAP_SR)
                            y, sr = tf.tf_dict[t](y, sr, tparam)
                            y = y.reshape(1, -1)
                            y = model.get_audio_embedding_from_data(
                                x = y, use_tensor=False)
                            np.save(p, y)
            del(model)
                    

                    


            

       

if __name__ == "__main__":
    main()
    
