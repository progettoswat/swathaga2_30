import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_losses(run_name):

    loss_file = pd.read_csv(f'logs/training-{run_name}.csv')

    train_loss = loss_file['ctc_loss']
    val_loss = loss_file['val_ctc_loss']

    plt.plot(train_loss,  color='red')
    plt.plot(val_loss,  color='blue')

    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    #plt.show()
    plt.savefig(f"plots/{run_name}/losses.jpg")
    plt.close()


def plot_val_metrics(run_name):
    metrics_file = pd.read_csv(f'results/{run_name}/stats.csv')

    wer = metrics_file['Mean WER (Norm)']
    cer = metrics_file['Mean CER (Norm)']

    plt.plot(wer, color='green')
    plt.plot(cer, color='orange')

    min_wer_index = wer.idxmin()
    #min_cer_index = cer.idxmin()
    min_wer_value = wer[min_wer_index]
    min_cer_value = cer[min_wer_index]


    '''legend1 = plt.legend(
        [f'Min WER: {min_wer_value:.2f} (Epoch {min_wer_index})',
         f'Min CER: {min_cer_value:.2f} (Epoch {min_cer_index})'],
        loc='upper right'
    )'''

    legend1 = plt.legend(
        [f'Min WER: {min_wer_value:.2f} (Epoch {min_wer_index})',
         f'CER: {min_cer_value:.2f} (Epoch {min_wer_index})'],
        loc='upper right'
    )



    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['WER', 'CER'], loc='upper left')
    #plt.show()
    plt.gca().add_artist(legend1)
    plt.savefig(f"plots/{run_name}/val_metrics.jpg")
    plt.close()


if __name__ == '__main__':

    # Vanilla Training 16bs
    run_name = "2023_11_04_16_41_41"

    if not os.path.exists("plots"):
        os.mkdir("plots")

    run_folder = os.path.join("plots", run_name)
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)

    plot_losses(run_name)
    plot_val_metrics(run_name)






