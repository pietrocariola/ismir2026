import argparse
import transformations as tf
import numpy as np
import pandas as pd
import os
from sklearn.manifold import TSNE
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate Projections CLI")
    parser.add_argument("--datasets", type=str, nargs="+", help="Name of the datasets to be included: fmasmall")
    parser.add_argument("--genres", type=str, nargs="+", help="Name of the genres to be included: electronic, experimental, folk, hiphop, instrumental, pop, rock")
    parser.add_argument("--models", type=str, nargs="+", help="Embeddings from the specified models: clap, audioclip, wav2clip")
    parser.add_argument("--transformations", type=str, nargs="+", help="Transformations to the sounds: pitch, time stretching, ...")
    parser.add_argument("--output_folder", type=str, help="Full path of the folder where the projections will be saved on")
    parser.add_argument("--tgt_file", type=str, default=False, help="Choose specific target file")
    parser.add_argument("--samples_per_genre", type=str, default=False, help="Number of samples per genre")
    # parser.add_argument("--output_file", type=str, help="Name of the file")
    args = parser.parse_args()

    datasets = args.datasets
    print(f"datasets: {datasets}")

    genres = args.genres
    print(f"genres: {genres}")

    models = args.models
    print(f"models: {models}")

    transfs = args.transformations
    print(f"transfs: {transfs}")

    output_folder = args.output_folder
    print(f"output_file: {output_folder}")

    tgt_file = args.tgt_file
    print(f"tgt_file: {tgt_file}")

    samples_per_genre = args.samples_per_genre
    print(f"samples_per_genre: {samples_per_genre}")

    # output_file = args.output_file
    # print(f"output_file: {output_file}")

    if tgt_file:
        output_file = ""
        output_file += f"{tgt_file.split('.')[0]}"
        output_file += "_"
        for model in models:
            output_file += f"{model[:2]}{model[-1]}"
        output_file += "_"
        for transf in transfs:
            output_file += f"{transf[:2]}{transf[-1]}"
    else:
        output_file = ""
        for dataset in datasets:
            output_file += f"{dataset[:2]}{dataset[-1]}"
        output_file += "_"
        for genre in genres:
            output_file += f"{genre[:2]}{genre[-1]}"
        output_file += "_"
        for model in models:
            output_file += f"{model[:2]}{model[-1]}"
        output_file += "_"
        for transf in transfs:
            output_file += f"{transf[:2]}{transf[-1]}"
        if samples_per_genre:
            output_file += f"_{samples_per_genre}"
    

    os.makedirs(output_folder, exist_ok=True)

    if not (
        Path(os.path.join(output_folder, "x_"+output_file+".npy")).exists() &
        Path(os.path.join(output_folder, "x2d_"+output_file+".npy")).exists() &
        Path(os.path.join(output_folder, "labels_"+output_file+".json")).exists()
    ): 

        df = pd.read_csv("metadata.csv")

        embeds = []
        labels = []

        if tgt_file:
            for model in models:
                path = df[
                        (df["file"]==tgt_file) &
                        (df["model"]==model) &
                        (df["transf"]=="identity") 
                    ]["file_embeds_path"].iloc[0]
                embeds.append(np.load(path))
                labels.append(model)
                for transf in transfs:
                    params = tf.tf_dict_params[transf]
                    for param in params:
                        path = df[
                                (df["model"]==model) &
                                (df["transf"]==transf) &
                                (df["file"]==tgt_file) &
                                (df["transf_param_name"]==str(param))
                            ]["file_embeds_path"].iloc[0]            
                        embeds.append(np.load(path))
                        labels.append(float(param))
        else:
            for dataset in datasets:
                for genre in genres:
                    for model in models:
                        for transf in transfs:
                            params = tf.tf_dict_params[transf]
                            files = list(set(df[
                                        (df["ds_name"]==dataset) &
                                        (df["genre"]==genre) &
                                        (df["model"]==model) &
                                        (df["transf"]==transf)
                                    ]["file"]))
                            if samples_per_genre:
                                files = files[:int(samples_per_genre)]
                            for i, file in enumerate(files):
                                print(f"dataset:{dataset}, genre:{genre}, model:{model}, transf:{transf}, i:{i}, file:{file}")
                                path = df[
                                    (df["ds_name"]==dataset) &
                                    (df["genre"]==genre) &
                                    (df["model"]==model) &
                                    (df["file"]==file) &
                                    (df["transf"]=="identity") 
                                ]["file_embeds_path"].iloc[0]
                                embeds.append(np.load(path))
                                labels.append(model)
                                for param in params:
                                    if (transf=="timestretch" or transf=="gain"):
                                        param = str(param)
                                        integer, dot, decimal = param.partition('.')
                                        param = integer + '.' + decimal[:3]
                                    path = df[
                                            (df["ds_name"]==dataset) &
                                            (df["genre"]==genre) &
                                            (df["model"]==model) &
                                            (df["transf"]==transf) &
                                            (df["file"]==file) &
                                            (df["transf_param_name"]==str(param))
                                        ]["file_embeds_path"].iloc[0]            
                                    embeds.append(np.load(path))
                                    labels.append(float(param))

        x = np.vstack(embeds)
        tsne = TSNE(n_components=2, random_state=0)
        x2d = tsne.fit_transform(x)

        x_name = f"x_"+output_file+".npy"
        np.save(os.path.join(output_folder, x_name), x)

        x2d_name = f"x2d_"+output_file+".npy"
        np.save(os.path.join(output_folder, x2d_name), x2d)

        labels_name = f"labels_"+output_file+".json"
        with open(os.path.join(output_folder, labels_name), "w") as f:
            json.dump(labels, f)

    print(" ")
    print("---------------------------------")
    print(" ")

if __name__ == "__main__":
    main()