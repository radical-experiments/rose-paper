import os
import argparse
import pickle
from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import joblib
import tensorflow as tf

NUM_OF_BINS = 502

def preprocess_inputdata(all_data):
    input_data, output, errors, z_data = [], [], [], []
    for key, data in all_data.items():
        paras = key.split("_")
        input_paras = paras[1::2]
        density_profiles = [data['pos'][:,1], data['neg'][:,1]]
        density_errors   = [data['pos'][:,2], data['neg'][:,2]]
        z_values         = [data['pos'][:,0], data['neg'][:,0]]
        input_data.append(input_paras)
        output.append(density_profiles)
        errors.append(density_errors)
        z_data.append(z_values)

    input_arr = np.array(input_data)
    output_arr = np.array(output).reshape(-1, NUM_OF_BINS*2)
    errors_arr = np.array(errors).reshape(-1, NUM_OF_BINS*2)
    z_arr      = np.array(z_data).reshape(-1, NUM_OF_BINS*2)

    print(f"Input data shape: {input_arr.shape}")
    print(f"Output data shape: {output_arr.shape}")
    print(f"Error data shape: {errors_arr.shape}")
    print(f"Bin center data shape: {z_arr.shape}")
    return input_arr, output_arr, errors_arr, z_arr

def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nStarting evaluation at {now}\n")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_root_path", required=True,
        help="Root path to the saved all models, models will be root/pipeline_{}/model/my_model_iter_{}_instance_{}.h5"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory containing data_dump_density_preprocessed_test.pk and scaler_new.pkl"
    )
    args = parser.parse_args()
    pprint(vars(args))

    test_pk = os.path.join(args.data_dir, "data_dump_density_preprocessed_test.pk")
    with open(test_pk, "rb") as f:
        raw_test = pickle.load(f)
    x_test, y_test, _, _ = preprocess_inputdata(raw_test)

    scaler_path = os.path.join(args.data_dir, "scaler_new.pkl")
    scaler = joblib.load(scaler_path)
    x_test_scaled = scaler.transform(x_test)

    rmse_mean_list = []
    rmse_std_list = []
    x_value = [150,50,50,50,50,50,50,50,50,500,500,500,500,500]

    for iter_id in range(len(x_value)):
        rmse_list = []
        for pipeline_id in range(4):
            for instance_id in range(1):
                model_file = os.path.join(args.model_root_path, f"pipeline_{pipeline_id}", "model", f"my_model_iter_{iter_id}_instance_{instance_id}.h5")
                model = tf.keras.models.load_model(model_file, compile=False)
#                print(model.summary())
                y_pred = model.predict(x_test_scaled, verbose=0)
                mse  = np.mean((y_pred - y_test)**2)
                rmse = np.sqrt(mse)
                print(f"\n With {model_file}, Test RMSE: {rmse:.6f}\n")
                rmse_list.append(rmse)
        rmse_mean = np.mean(rmse_list)
        rmse_std = np.std(rmse_list)
        print(f"RMSE for iter {iter_id} = {rmse_mean:.6f} \pm {rmse_std:.6f}")
        rmse_mean_list.append(rmse_mean)
        rmse_std_list.append(rmse_std)

    x    = np.cumsum(x_value)
    y    = np.array(rmse_mean_list)
    yerr = np.array(rmse_std_list)

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x, y, yerr=yerr,
        linestyle='-',        # solid line
        color='C0',           # line in default Matplotlib blue
        linewidth=2,
        marker='o',           # circle markers
        markersize=5,         # slightly smaller dots
        markerfacecolor='C1', # marker fill in default orange
        markeredgecolor='C1', # marker edge in same orange
        capsize=4, 
        elinewidth=1.5
    )
    
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.xlabel(r'Train Dataset Size $N_{\mathrm{train}}$', fontsize=14)
    plt.ylabel(r'RMSE $E$',                 fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig('rmse_vs_num_sample_large.png', dpi=300, bbox_inches='tight')

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Finished evaluation at {now}\n")

if __name__ == "__main__":
    main()

