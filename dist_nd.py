import argparse
import transformations as tf
import os
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Calculate Distances in nD CLI")
    parser.add_argument("--input_path", type=str, help="Full path where the embeddings are saved")
    parser.add_argument("--datasets", type=str, nargs="+", help="Calculates distance for each dataset indenpendently")
    parser.add_argument("--models", type=str, nargs="+", help="Calculates distance for each model indenpendently")
    parser.add_argument("--transformations", type=str, nargs="+", help="Calculates distance for each transformation indenpendently")
    parser.add_argument("--normalize", type=int, help="1 to normalize nD embeddings to unit vectors, 0 otherwise.")
    parser.add_argument("--calc_mode", type=str, help="calculation mode: 'dist' for euclidean distance or 'cos' for cosine similarity.")
    parser.add_argument("--output_path", type=str, help="Full path where the distances will be saved on")
    args = parser.parse_args()

    input_path = args.input_path
    print(f"input_path: {args.input_path}")

    datasets = args.datasets
    print(f"datasets: {args.datasets}")
    
    models = args.models
    print(f"models: {args.models}")
    
    transformations = args.transformations
    print(f"transformations: {args.transformations}")

    normalize = args.normalize
    print(f"normalize: {args.normalize}")
    normalize = True if normalize == 1 else False

    calc_mode = args.calc_mode
    print(f"calc_mode: {args.calc_mode}")

    output_path = os.path.abspath(args.output_path)
    print(f"output_path: {args.output_path}")
    os.makedirs(output_path, exist_ok=True)

    for dataset in datasets:
        for model in models:
            for transformation in transformations:
                dir = Path(input_path)
                tracks = list(dir.rglob(f"*{dataset}*{model}*identity*.npy"))
                if tracks == []:
                    print(f"No files were found.")
                    exit()
                params = ['id']
                for param in tf.tf_dict_params[transformation]:
                    params.append(str(param)[:5])
                df = pd.DataFrame(
                    np.zeros((len(params), len(params))),
                    columns=params,
                    index=params
                )
                for track in tracks:
                    track = str(track)
                    track = track.split('_')[2]+'_'+track.split('_')[3]
                    for i, param1 in enumerate(params):
                        if param1 == 'id':
                            f1 = f"x_{dataset}_{track}_{model}_identity_none.npy"
                            f1 = os.path.join(input_path, f1)
                        else:
                            f1 = f"x_{dataset}_{track}_{model}_{transformation}_{str(param1)[:5].replace('.', 'p')}.npy"
                            f1 = os.path.join(input_path, f1)
                        a1 = np.load(f1, 'r')[0,:]
                        a1 = a1 / np.linalg.norm(a1) if normalize else a1
                        for j in range(i, len(params)):
                            param2 = params[j]
                            if param2 == 'id':
                                f2 = f"x_{dataset}_{track}_{model}_identity_none.npy"
                                f2 = os.path.join(input_path, f2)
                            else:
                                f2 = f"x_{dataset}_{track}_{model}_{transformation}_{str(param2)[:5].replace('.', 'p')}.npy"
                                f2 = os.path.join(input_path, f2)
                            a2 = np.load(f2, 'r')[0,:]
                            a2 = a2 / np.linalg.norm(a2) if normalize else a2
                            if calc_mode == 'dist': # euclidean distance
                                d = np.sqrt(np.sum((a2-a1)**2))
                            elif calc_mode == 'cos': # cosine similarity
                                d = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
                            df.iloc[i,j] += d / len(tracks)
                            if i!=j:
                                df.iloc[j,i] += d / len(tracks)
                save_file = f"{calc_mode}_{dataset}_{model}_{transformation}.csv"
                output = os.path.join(output_path, save_file)
                df.to_csv(output, index=True)
                del(df)

if __name__ == "__main__":
    main()
    
