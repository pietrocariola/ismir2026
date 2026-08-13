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
                    params.append(param)
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
                            f1 = f"x_{dataset}_{track}_{model}_{transformation}_{str(param1).replace('.', 'p')}.npy"
                            f1 = os.path.join(input_path, f1)
                        a1 = np.load(f1, 'r')
                        for j in range(i, len(params)):
                            param2 = params[j]
                            if param2 == 'id':
                                f2 = f"x_{dataset}_{track}_{model}_identity_none.npy"
                                f2 = os.path.join(input_path, f2)
                            else:
                                f2 = f"x_{dataset}_{track}_{model}_{transformation}_{str(param2).replace('.', 'p')}.npy"
                                f2 = os.path.join(input_path, f2)
                            a2 = np.load(f2, 'r')
                            d = np.sqrt(np.sum((a2-a1)**2))
                            df.iloc[i,j] += d / len(tracks)
                            df.iloc[j,i] += d / len(tracks)
                output = os.path.join(output_path, f"dist_{dataset}_{model}_{transformation}.csv")
                df.to_csv(output, index=True)
                del(df)

if __name__ == "__main__":
    main()
    
