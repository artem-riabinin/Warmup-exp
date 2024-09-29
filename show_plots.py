import numpy as np
import matplotlib.pyplot as plt
import utils
import sys
from matplotlib.ticker import (FormatStrFormatter, AutoMinorLocator)

class NNConfiguration: pass 

# Set matplotlib parameters
plt.rcParams.update({
    "lines.markersize": 30,
    "lines.linewidth": 4,
    "font.size": 27,
    "font.family": 'serif',
    "font.serif": 'FreeSerif',
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
    "legend.fontsize": 30,
    "axes.titlesize": 40,
    "axes.labelsize": 40,
    "figure.figsize": [64, 12]
})

def get_subset(ctr, indices):
    return [ctr[ind] for ind in indices]

def main(files):
    fig_four, axs_four = plt.subplots(1, 4)

    K_ar, batch_size_for_worker_ar = [], []

    for fname in files:
        my = utils.deserialize(fname)
        transfered_bits_by_node = my["transfered_bits_by_node"]
        fi_grad_calcs_by_node = my["fi_grad_calcs_by_node"]
        train_loss = my["train_loss"]
        test_loss = my["test_loss"]
        train_acc = my["train_acc"]
        test_acc = my["test_acc"]
        fn_train_loss_grad_norm = my["fn_train_loss_grad_norm"]
        fn_test_loss_grad_norm = my["fn_test_loss_grad_norm"]
        nn_config = my["nn_config"]
        current_data_and_time = my["current_data_and_time"]
        experiment_description = my["experiment_description"]
        K = my["K"]
        algo_name = my["algo_name"]

        K_ar.append(K)

        freq = 50
        train_loss = train_loss[::freq]
        test_loss = test_loss[::freq]
        train_acc = train_acc[::freq]
        test_acc = test_acc[::freq]
        fn_train_loss_grad_norm = fn_train_loss_grad_norm[::freq]
        fn_test_loss_grad_norm = fn_test_loss_grad_norm[::freq]

        # Pass batch_size_for_worker_ar to the print_experiment_info function
        print_experiment_info(fname, current_data_and_time, experiment_description, nn_config, K, batch_size_for_worker_ar)

        # Calculate sums and means
        transfered_bits_sum = np.cumsum(np.sum(transfered_bits_by_node, axis=0))
        fi_grad_calcs_sum = np.cumsum(fi_grad_calcs_by_node, axis=0)

        transfered_bits_mean = transfered_bits_sum / nn_config.kWorkers
        transfered_bits_mean_sampled = transfered_bits_mean[::freq]

        epochs = fi_grad_calcs_sum / nn_config.train_set_full_samples
        iterations = np.arange(len(epochs))
        epochs_sampled = epochs[::freq]

        # Prepare plotting parameters
        g = set_plotting_parameters(algo_name, nn_config)
        markevery = get_markevery(len(transfered_bits_mean_sampled), g, freq)

        plot_results(axs_four, transfered_bits_mean_sampled, fn_train_loss_grad_norm, train_loss, 
                     train_acc, test_acc, algo_name, nn_config, g, markevery)

    finalize_plot(axs_four, experiment_description, K, nn_config, batch_size_for_worker_ar)

def print_experiment_info(fname, current_data_and_time, experiment_description, nn_config, K, batch_size_for_worker_ar):
    print("==========================================================")
    print(f"Informaion about experiment results '{fname}'")
    print(f"  Content has been created at '{current_data_and_time}'")
    print(f"  Experiment description: {experiment_description}")
    print(f"  Dimension of the optimization problem: {nn_config.D}")
    print(f"  Compressor TOP-K K: {K}")
    print(f"  Number of Workers: {nn_config.kWorkers}")
    print(f"  Used step-size: {nn_config.gamma}\n")
    print("Whole config")
    for k in dir(nn_config):
        v = getattr(nn_config, k)
        if isinstance(v, (int, float)):
            print(f"  {k} = {v}")
            if k == "batch_size_for_worker":
                batch_size_for_worker_ar.append(v)
    print("==========================================================")

def set_plotting_parameters(algo_name, nn_config):
    g = 0
    if algo_name.startswith("EF21"):
        g = 0
    if algo_name.startswith("EF21-norm"):
        g = 2
    if algo_name.startswith("EF21-clip"):
        g = 1    
    return g

def get_markevery(length, g, freq):
    mark_mult = 0.2
    return [int(mark_mult * length / 4.0), 
            int(mark_mult * length / 3.5), 
            int(mark_mult * length / 3.0)]

def plot_results(axs_four, transfered_bits_mean_sampled, fn_train_loss_grad_norm, train_loss, 
                 train_acc, test_acc, algo_name, nn_config, g, markevery):
    # Plot configurations
    colors = ['tab:red', 'cornflowerblue', 'darkgreen', 'goldenrod']
    markers = ["o", "*", "v", "^"]

    # Plotting
    axs_four[0].semilogy(transfered_bits_mean_sampled * 1e-9, fn_train_loss_grad_norm,
                          color=colors[g], marker=markers[g],
                          markevery=markevery[g],
                          label=f'{algo_name}; $\\gamma$ = {nn_config.gamma:.1f}')
    axs_four[0].set_xlabel('#Gbits/n')
    axs_four[0].set_ylabel('$||\\nabla f(x)||^2$')
    axs_four[0].grid(True)

    axs_four[1].semilogy(transfered_bits_mean_sampled * 1e-9, train_loss,
                          color=colors[g], marker=markers[g],
                          markevery=markevery[g],
                          label=f'{algo_name}; $\\gamma$ = {nn_config.gamma:.1f}')
    axs_four[1].set_xlabel('#Gbits/n')
    axs_four[1].set_ylabel('Training Loss $f(x)$')
    axs_four[1].grid(True)

    axs_four[2].semilogy(transfered_bits_mean_sampled * 1e-9, train_acc,
                          color=colors[g], marker=markers[g],
                          markevery=markevery[g],
                          label=f'{algo_name}; $\\gamma$ = {nn_config.gamma:.1f}')
    axs_four[2].set_xlabel('#Gbits/n')
    axs_four[2].set_ylabel('Train Accuracy')
    axs_four[2].set_yscale('linear')
    axs_four[2].grid(True)

    axs_four[3].semilogy(transfered_bits_mean_sampled * 1e-9, test_acc,
                          color=colors[g], marker=markers[g],
                          markevery=markevery[g],
                          label=f'{algo_name}; $\\gamma$ = {nn_config.gamma:.1f}')
    axs_four[3].set_xlabel('#Gbits/n')
    axs_four[3].set_ylabel('Test Accuracy')
    axs_four[3].set_yscale('linear')
    axs_four[3].grid(True)

def finalize_plot(axs_four, experiment_description, K, nn_config, batch_size_for_worker_ar):
    for column in axs_four:
        column.legend(loc='upper left')

    plt.suptitle(f'{experiment_description} with $k\\approx${K / nn_config.D:.2f}$D$, $\\tau = {batch_size_for_worker_ar[0]}$')
    plt.tight_layout()

    save_to = f"plot_bsz_{batch_size_for_worker_ar[0]}_K_{(100 * K / nn_config.D):.0f}.pdf"
    plt.savefig(save_to, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
