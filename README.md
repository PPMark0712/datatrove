# CDF-GC

This repository contains the code for the paper: "Data-Efficient Selection via Grammatical Complexity in Continual Pre-training of Domain-Specific LLMs".

* **Paper Link**: To be updated.
* **Supported Languages**: Currently, this tool only supports Chinese data selection. To extend it to other languages, you will need to modify `src/datatrove/pipeline/cdf_gc/dependency_parser.py` and `part_of_speech_predictor.py`.
* **Note**: This project is built on top of the Hugging Face `datatrove` framework. This README provides information specific to this project and does not cover the `datatrove` framework itself.

## Environment Setup

To run this project, you need to install the `datatrove` library and its dependencies in your Python environment. You can find all the necessary dependencies in the `pyproject.toml` file.

1. Clone the repository

```bash
git clone https://github.com/PPMark0712/CDF-GC.git
```

2. Create and activate the Conda environment

```bash
conda create -n cdf_gc python=3.12
conda activate cdf_gc
```

3. Install the datatrove package

```bash
cd CDF-GC
pip install -e .[cdf_gc]
```
The `.[cdf_gc]` part ensures that the specific dependencies for this project, as listed in the `pyproject.toml` file, are also installed.

Download Pre-trained Models

- LTP Model: Download the [LTP/small](https://huggingface.co/LTP/small) model from the Hugging Face Hub.

- Target LLM: Download the large language model you plan to train, as you'll need its tokenizer for subsequent steps.


## Running the Script

To start the program, run the following `bash` script:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/cdf_gc.py \
	--input_path /path/to/jsonl_folder \
	--glob_pattern *.jsonl \
	--output_path /path/to/output_folder \
	--tasks 64 \
	--workers 32 \
	--workers_per_gpu 4 \
	--sample_rate 0.2 \
	--ltp_model_path /path/to/LTP/small \
	--tokenizer_path /path/to/tokenizer.json
```

## Parameter Descriptions

| Parameter | Type | Description | Default Value | Required |
| :--- | :--- | :--- | :--- | :--- |
| `--input_path` | path | Path to the input JSONL folder. | None | Yes |
| `--glob_pattern` | regex | Glob pattern used to match input files. Defaults to all files in the current path. | None | No |
| `--output_path` | path | Path to the output folder. | None | Yes |
| `--tasks` | integer | The total number of tasks. Input files are evenly distributed among them. | 64 | No |
| `--workers` | integer | Number of CPU workers to process tasks. | 32 | No |
| `--workers_per_gpu` | integer | Number of workers sharing a single GPU during the dependency parsing step. | 4 | No |
| `--sample_rate` | float | Data sampling rate. | 0.2 | No |
| `--ltp_model_path` | path | Path to the LTP Chinese language processing model (used for tokenization, part-of-speech tagging, etc.). | None | Yes |
| `--tokenizer_path` | path | Path to the tokenizer JSON file (e.g., Llama tokenizer). | None | Yes |
| `--limit` | integer | Limits the maximum number of samples each worker reads, useful for quick tests. | None | No |
| `--rerun` | boolean | Whether to rerun the process. If `False`, it will load from the last checkpoint. | `False` | No |

## Output Directory Structure

```
output_path/
├── 1_gc_data/                         # Grammatical Complexity (GC) data
│   ├── 1_dependency_parsing/          # Intermediate results of dependency parsing
│   ├── 2_part_of_speech_predicting/   # Intermediate results of part-of-speech prediction
│   ├── 3_lexical_diversity/           # Lexical diversity metrics
│   ├── 4_syntactic_complexity/        # Syntactic complexity metrics
│   ├── 5_combined_gc/                 # Combined GC results
│   └── 6_normalized_gc/               # Per-dimension min-max normalization results
├── 2_sampling/                        # Sampling results
│   ├── 1_probability/                 # Sampling probability for each data point
│   └── 2_sample_result/               # Final filtered results (the files you need)
└── logs/                              # Log files
    ├── gc/                            # Grammatical complexity calculation logs
    │   ├── dependency_parsing/        # Dependency parsing logs
    │   ├── part_of_speech_predicting/ # Part-of-speech prediction logs
    │   └── gc_calculator/             # GC data integration logs
    └── sampling/                      # Sampling logs
        ├── probability_calculator/    # Normalization and probability calculation logs
        └── sample_result/             # Probability-based sampling logs.
```

Note:
- If you are only interested in the final output, you can directly access all `.jsonl` files located in the `output_path/2_sampling/2_sample_result` directory.
- The `doc_len_tokens` field in the `stats.json` file shows token counts before and after sampling.

## GPU Usage

### Resource Allocation

Only the dependency parsing step requires a GPU; all other parts of the process use only the CPU. To avoid prolonged GPU occupation, dependency parsing is set as the very first step, and GPU resources are released as soon as it's completed. The number of workers for this specific stage is determined by the number of visible GPUs, specified by CUDA_VISIBLE_DEVICES, and the maximum number of processes per GPU, set by `--workers_per_gpu`.

You must set the `CUDA_VISIBLE_DEVICES` environment variable, as the code will automatically utilize all visible GPUs.

The `--workers_per_gpu` parameter allows multiple processes to share a single GPU, with a default value of 1. We haven't conducted detailed VRAM usage tests, so you may need to adjust this value based on your specific VRAM and utilization needs. Through our testing, we found that 4 processes sharing a single 3090 GPU can achieve high utilization.

### Multi-GPU Data Parallelism

The datatrove framework divides multiple files into multiple tasks, which are then processed in parallel by different workers. We've configured different workers to use different GPUs to achieve data parallelism.

Given that the rank in the PipelineStep.run method is a task ID, not a worker ID, we modified line 89 of src/datatrove/executor/local.py to pass the worker's local rank to the pipeline step. This allows us to determine which GPU each worker should use.

### Inference Batching

Since the VRAM usage for dependency parsing of a single sentence is relatively small, we've included `batch_size` and `max_length` parameters in the `ChineseDependencyParser` class within `src/datatrove/pipeline/cdf_gc/dependency_parser.py`. These parameters control the number of sentences in a batch and the maximum sentence length for inference. You shouldn't need to change these parameters, but you can reduce them if you encounter out-of-memory issues.
