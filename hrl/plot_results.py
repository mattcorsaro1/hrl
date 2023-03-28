import argparse
import copy
import datetime
import numpy as np
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import sem
import sys
import time

import seaborn as sns
sns.set()

def loadPickleFile(filename, directory):
    values = None
    try:
        with open(directory + '/' + filename, "rb") as f:
            values = pickle.load(f)
    except Exception as e:
        print(f'Exception while reading {filename} in {directory}: {e}')
        raise e
    return values

def generatePlot(y_val_sets_over_seed, plot_title, labels, plot_dir_this_obj, max_x=None, leg_loc="upper left", smooth_over=None):
    print("Now saving", plot_title, "plot in", plot_dir_this_obj)
    plot_filename = plot_dir_this_obj + "/" + plot_title + ".png"
    fig = plt.figure()
    ax = plt.subplot(111)
    #assert(len(y_val_sets_over_seed) == len(labels))
    if smooth_over is not None:
        for method_i in range(len(y_val_sets_over_seed)):
            for seed in range(len(y_val_sets_over_seed[method_i])):
                smoothed_values = []
                for i in range(len(y_val_sets_over_seed[method_i][seed]) - smooth_over + 1):
                    this_window = y_val_sets_over_seed[method_i][seed][i:i+smooth_over]
                    smoothed_values.append(sum(this_window) / float(smooth_over))
                y_val_sets_over_seed[method_i][seed] = smoothed_values

    for method_i in range(len(y_val_sets_over_seed)):
        y_avg = []
        y_err = []
        y_vals_over_seed_this_method = y_val_sets_over_seed[method_i]
        max_x_this_method = max([len(y_vals) for y_vals in y_vals_over_seed_this_method])
        # iterate over x values
        for x in range(max_x_this_method):
            metric_vals_this_x = []
            for seed in range(len(y_vals_over_seed_this_method)):
                if x < len(y_vals_over_seed_this_method[seed]):
                    metric_vals_this_x.append(y_vals_over_seed_this_method[seed][x])
            avg_y = sum(metric_vals_this_x)/len(metric_vals_this_x)
            y_standard_error = sem(metric_vals_this_x)
            y_avg.append(avg_y)
            y_err.append(y_standard_error)
            
        x_vals = list(range(len(y_avg)))
        if smooth_over is not None:
            x_vals = [val + smooth_over for val in x_vals]
        ax.plot(x_vals, y_avg, label=labels[method_i])
        y_err_min = [y_avg[i]-y_err[i] for i in range(len(y_avg))]
        y_err_max = [y_avg[i]+y_err[i] for i in range(len(y_avg))]
        ax.fill_between(x_vals, y_err_min, y_err_max, alpha=0.2)
    if max_x is not None:
        #plt.xlim([0, max_x])
        ax.set_xlim(0, max_x)
    #plt.legend(loc=leg_loc)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=len(y_val_sets_over_seed)/2)

    if plot_title != None:
        if plot_title == "episodic_success_rate_smoothed":
            plot_title = "Episodic Success Rate"
        #plt.title(plot_title)
        ax.set_title(plot_title)
    plt.savefig(plot_filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate plots for thesis experiments.')

    parser.add_argument('--data_dir', type=str, default="/users/mcorsaro/scratch/results/")
    args = parser.parse_args()

    plots_to_make = ["episodic_final_dist", "episodic_score", "episodic_success_rate"]#"evaluation_rewards", "task_success_rate"

    runs_to_plot = {}
    runs_to_plot["door"] = [\
        "door_trial_1_her_clf", \
        "door_trial_1_her_clf_uw", \
        "door_trial_1_her_oracle", \
        "door_trial_1_her_random", \
    ]
    """runs_to_plot["door_HER"] = [\
        "door_gs1_0_HER_3e-3_3e-3", \
        "door_gs1_1_HER_3e-3_3e-4", \
        "door_gs1_2_HER_3e-3_3e-5", \
        "door_gs1_3_HER_3e-4_3e-3", \
        "door_gs1_4_HER_3e-4_3e-4", \
        "door_gs1_5_HER_3e-4_3e-5", \
        "door_gs1_6_HER_3e-5_3e-3", \
        "door_gs1_7_HER_3e-5_3e-4", \
        "door_gs1_8_HER_3e-5_3e-5", \
    ]
    runs_to_plot["door"] = [\
        "door_gs1_9_3e-3_3e-3", \
        "door_gs1_10_3e-3_3e-4", \
        "door_gs1_11_3e-3_3e-5", \
        "door_gs1_12_3e-4_3e-3", \
        "door_gs1_13_3e-4_3e-4", \
        "door_gs1_14_3e-4_3e-5", \
        "door_gs1_15_3e-5_3e-3", \
        "door_gs1_16_3e-5_3e-4", \
        "door_gs1_17_3e-5_3e-5", \
    ]
    titles = [\
        "3e-3_3e-3", \
        "3e-3_3e-4", \
        "3e-3_3e-5", \
        "3e-4_3e-3", \
        "3e-4_3e-4", \
        "3e-4_3e-5", \
        "3e-5_3e-3", \
        "3e-5_3e-4", \
        "3e-5_3e-5", \
    ]"""
    titles = [\
        "CLF", \
        "CLF UW", \
        "Oracle", \
        "Baseline", \
    ]
    '''runs_to_plot["door"] = [\
        "door_baseline_trial_0",\
        #"door_0", \
        #"door_her_0", \
        "door_oracle_trial_0", \
        #"door_oracle_3", \
        #"door_her_oracle_0", \
        "door_clf_trial_0", \
        #"door_clf_3", \
        #"door_clf_her_3", \
        "door_uw_clf_trial_0"
        #"door_clf_uw_3", \
        #"door_clf_uw_her_3", \
    ]
    runs_to_plot["switch_HER"] = [\
        "switch_baseline_her_trial_0",\
        #"switch_0", \
        #"switch_her_0", \
        "switch_oracle_her_trial_0", \
        #"switch_her_oracle_0", \
        #"switch_her_oracle_0", \
        "switch_clf_her_trial_0", \
        #"switch_clf_2", \
        #"switch_clf_her_3", \
        "switch_uw_clf_her_trial_0"
        #"switch_clf_uw_2", \
        #"switch_clf_uw_her_3", \
    ]

    titles = [\
        "Baseline", \
        #"Baseline HER", \
        "Oracle", \
        # (Door)", \
        #"Oracle HER", \
        "CLF", \
        #"CLF HER", \
        "UW CLF", \
        #"UW CLF HER", \
    ]'''

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    plot_dir = args.data_dir + '/plots_' + timestamp
    os.mkdir(plot_dir)
    print("Copy these results with:\n", "scp -r ccv:{} .".format(plot_dir), "\n")#mcorsaro@ssh.ccv.brown.edu
    for obj in runs_to_plot:
        #runs_to_plot_this_obj = [obj + run_title for run_title in runs_to_plot]
        runs_to_plot_this_obj = runs_to_plot[obj]
        plot_dir_this_obj = plot_dir + '/' + obj
        os.mkdir(plot_dir_this_obj)

        obj_plots_to_make = copy.deepcopy(plots_to_make)
        """if obj == "switch":
                                    obj_plots_to_make.append("switch_state")
                                elif obj == "door":
                                    obj_plots_to_make.append("door_hinge_state")
                                    obj_plots_to_make.append("door_latch_state")"""
        for plot in obj_plots_to_make:

            pickle_filenames = {}
            # One entry per method, a list of lists of values, where each inner list corresponds to a random seed
            y_val_sets_over_seed = []
            # Random grasp baseline, oracle grasps, etc.
            for method in runs_to_plot_this_obj:
                y_vals_this_method = []
                result_dir = args.data_dir + '/' + method + '/' + plot
                pickle_files = os.listdir(result_dir)
                pickle_files.sort()
                pickle_filenames[method] = pickle_files
                # For each random seed
                for pickle_file in pickle_files:
                    y_vals_this_method.append(loadPickleFile(pickle_file, result_dir))
                if isinstance(y_vals_this_method[0][0], np.ndarray):
                    for i in range(len(y_vals_this_method)):
                        for j in range(len(y_vals_this_method[i])):
                            assert(len(y_vals_this_method[i][j].shape) == 1 and y_vals_this_method[i][j].shape[0] == 1)
                            y_vals_this_method[i][j] = y_vals_this_method[i][j][0]
                y_val_sets_over_seed.append(y_vals_this_method)

            max_x_val = np.array([[len(vals) for vals in set] for set in y_val_sets_over_seed]).max()
            if isinstance(max_x_val, list):
                max_x_val = max(max_x_val)
            generatePlot(y_val_sets_over_seed, plot, titles, plot_dir_this_obj, max_x=max_x_val, leg_loc="upper left")
            if "episodic" in plot or "state" in plot:
                generatePlot(y_val_sets_over_seed, plot + "_smoothed", titles, plot_dir_this_obj, smooth_over=200, max_x=max_x_val, leg_loc="upper left")
                # for the smoothed plots, also plot each method on individual plot with different line for each seed
                for method_i in range(len(y_val_sets_over_seed)):
                    y_vals_this_method = [[y_vals] for y_vals in y_val_sets_over_seed[method_i]]
                    generatePlot(y_vals_this_method, plot + "_smoothed_" + titles[method_i], pickle_filenames[runs_to_plot_this_obj[method_i]], plot_dir_this_obj, max_x=max_x_val if obj == "door" else None, smooth_over=200, leg_loc="upper left" if obj == "door" else "lower right")

if __name__ == '__main__':
    main()

