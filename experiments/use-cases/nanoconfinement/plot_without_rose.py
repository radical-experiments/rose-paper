import os
import argparse
import pickle
from datetime import datetime
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import joblib

def main():
    x_value = [3050, 2550, 2050, 1550, 1050, 550, 400, 350, 300, 250, 200, 150]
    rmse_mean_list = [
        0.013086, 0.014332, 0.015804, 0.018151,
        0.020073, 0.032367, 0.036928, 0.039894,
        0.045379, 0.0458,   0.053402, 0.063953
    ]
    rmse_std_list = [
        0.001051319, 0.000946148, 0.002022343, 0.002401872,
        0.001776407, 0.001849769, 0.00240847,  0.002907493,
        0.004461386, 0.002815039, 0.005491897, 0.004244091
    ]

    x    = np.array(x_value)
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
    plt.savefig('rmse_vs_num_sample_without_rose.png', dpi=300, bbox_inches='tight')

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Finished evaluation at {now}\n")

if __name__ == "__main__":
    main()

