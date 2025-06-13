import os
import json
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def draw_dist(x, output_fn):
    # x.sort()
    # print(x[0], x[-1])
    counts, bins = np.histogram(x, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bin_widths = np.diff(bins)
    probs = counts * bin_widths
    
    plt.figure(figsize=(10, 6))

    plt.plot(bin_centers, probs, label='Probability')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.title('Probability Distribution')
    plt.grid(True)
    plt.legend()

    plt.savefig(output_fn)
    plt.close()


def draw_cdf(x, output_fn):    
    # 对数据进行排序
    sorted_data = np.sort(x)
    # 计算累积概率
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    
    plt.figure(figsize=(10, 6))
    # 绘制CDF
    plt.plot(sorted_data, yvals, label='CDF')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.grid(True)
    plt.legend()
    
    # 保存图片
    plt.savefig(output_fn)
    plt.close()


def draw(x, output_fn_prefix):
    draw_dist(x, f"{output_fn_prefix}_dist.png")
    draw_cdf(x, f"{output_fn_prefix}_cdf.png")


def draw_many(data, output_fn_prefix):
    def draw_dist_subplot(ax, x, label):
        counts, bins = np.histogram(x, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_widths = np.diff(bins)
        probs = counts * bin_widths
        ax.plot(bin_centers, probs, label=label)
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability')
        ax.grid(True)
        
    def draw_cdf_subplot(ax, x, label):
        sorted_data = np.sort(x)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        ax.plot(sorted_data, yvals, label=label)
        ax.set_xlabel('Value')
        ax.set_ylabel('Cumulative Probability')
        ax.grid(True)

    # Create a figure with 2 subplots (1 row for dist, 1 row for cdf)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot distributions and CDFs for each key in data
    for key in data.keys():
        draw_dist_subplot(ax1, data[key], key)
        draw_cdf_subplot(ax2, data[key], key)
    
    # Set titles and adjust layout
    ax1.set_title('Probability Distributions')
    ax2.set_title('Cumulative Distribution Functions')
    
    # Set x-axis ticks at intervals of 0.2
    for ax in [ax1, ax2]:
        x_min = min(min(data[key]) for key in data.keys())
        x_max = max(max(data[key]) for key in data.keys())
        # Round to nearest 0.2 for better appearance
        x_min = np.floor(x_min * 5) / 5
        x_max = np.ceil(x_max * 5) / 5
        ax.set_xticks(np.arange(x_min, x_max + 0.2, 0.2))
    
    # Add legends
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{output_fn_prefix}_combined.png", bbox_inches='tight')
    plt.close()


def read_file(merged_args):
    data_path, file_path, keys = merged_args
    data = defaultdict(list)
    with open(os.path.join(data_path, file_path), "r") as f:
        for line in f:
            item = json.loads(line)
            for key in keys:
                try:
                    data[key].append(item['metadata']['repetition'][key])
                except KeyError:
                    continue
    return data


if __name__ == "__main__":
    data_path = "/data1/yyz/projects/data/datatrove_output/rep_test/2_quality_filter/zh/output"
    output_path = "/data1/yyz/projects/data/datatrove_output/rep_test/dist"
    keys = ["dup_line_frac"] + [f"top_{n}_gram" for n in range(2, 5)] + [f"duplicated_{n}_n_grams" for n in range(5, 11)]

    file_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".jsonl"):
                file_list.append(os.path.join(root, file))
    merged_args_list = [(data_path, file, keys) for file in file_list]
    result = defaultdict(list)
    print("reading files...")
    with mp.Pool(processes=32) as pool:
        for file_data in pool.imap_unordered(read_file, merged_args_list):
            for key in keys:
                result[key].extend(file_data[key])
    os.makedirs(output_path, exist_ok=True)
    top_n_gram_keys = [f"top_{n}_gram" for n in range(2, 5)]
    dup_n_gram_keys = [f"duplicated_{n}_n_grams" for n in range(5, 11)]

    top_n_gram_data = {key: result[key] for key in top_n_gram_keys}
    dup_n_gram_data = {key: result[key] for key in dup_n_gram_keys}

    draw_many(top_n_gram_data, os.path.join(output_path, "top_n_gram"))
    draw_many(dup_n_gram_data, os.path.join(output_path, "dup_n_gram"))

    dup_line_frac_data = {key: result[key] for key in ["dup_line_frac"]}
    draw_many(dup_line_frac_data, os.path.join(output_path, "dup_line_frac"))

    # Calculate and output 90th percentile for dup_n_grams to txt file
    percentile_output_file = os.path.join(output_path, "dup_n_grams_90th_percentile.txt")
    with open(percentile_output_file, "w") as f:
        f.write("90th percentile for dup_n_grams:\n")
        for key in dup_n_gram_keys:
            if result[key]:  # Check if there's data
                percentile_90 = np.percentile(result[key], 90)
                f.write(f"{key}: {percentile_90:.6f}\n")
            else:
                f.write(f"{key}: No data available\n")
    print(f"90th percentile results saved to: {percentile_output_file}")

    # Calculate and output 90th percentile for top_n_gram to txt file
    top_n_gram_percentile_file = os.path.join(output_path, "top_n_gram_90th_percentile.txt")
    with open(top_n_gram_percentile_file, "w") as f:
        f.write("90th percentile for top_n_gram:\n")
        for key in top_n_gram_keys:
            if result[key]:  # Check if there's data
                percentile_90 = np.percentile(result[key], 90)
                f.write(f"{key}: {percentile_90:.6f}\n")
            else:
                f.write(f"{key}: No data available\n")
    print(f"90th percentile results saved to: {top_n_gram_percentile_file}")
