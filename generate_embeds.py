import argparse
import transformations as tf
import os    

CLAP_SR = 48000

def main():
    parser = argparse.ArgumentParser(description="Generate Embeds CLI")
    parser.add_argument("--dataset_path", type=str, help="Full path to load the dataset")
    parser.add_argument("--dataset_name", type=str, help="This name will be used to save the embeddings")
    parser.add_argument("--model", type=str, help="Model that will generate the embeddings: CLAP, AudioCLIP, etc...")
    parser.add_argument("--transformations", nargs="+", type=str, help="Transformations to the sounds: pitch, time stretching, etc...")
    parser.add_argument("--output_path", type=str, help="Full path where the embeddings will be saved on")
    args = parser.parse_args()
    print(f"dataset_path: {args.dataset_path}")
    print(f"dataset_name: {args.dataset_name}")
    print(f"transformations: {args.transformations}")

    model = args.model
    ds_path = args.dataset_path
    files = os.listdir(ds_path)

    ### CLAP ###
    if model.lower() == "clap":
        for f in files:
            

       

if __name__ == "__main__":
    main()
    
