import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="semantic communication systems for speech transmission")
    

    # path of tfrecords files
    parser.add_argument("--path", type=str, default="./results",
                        help="tfrecords path of trainset.")

    
    args = parser.parse_args()
    
    return args

args = parse_args()


if __name__ == "__main__":

    root_path = ''
    res_files = [os.path.join(args.path, r) for r in os.listdir(args.path) if r.endswith(".npy")]
    marker_list = ['bo-','gv-', 'rD--','c*--','mh:','ys:']
    fig, ax = plt.subplots()
    label_list = ['AWGN','Rayleigh', 'Rician']

    j = 0
    for res in res_files: 

        data = np.load(res)
        # x = [i for i in range(20)]
        ax.plot(data,marker_list[j], markersize=4,label=label_list[j])

        j = (j+1) % 6

    plt.ylim([3,5.5])  
    plt.xticks(np.arange(0, 21, 1.0))
    plt.axvline(x=8, color='k', linestyle=':')
    plt.ylabel('PESQ')
    plt.xlabel('SNR (dB)')
    plt.legend()
    # plt.grid()
    plt.show()