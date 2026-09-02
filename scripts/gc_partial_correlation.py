import argparse
import csv
import json
from itertools import zip_longest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_COMPONENTS = ["pos_ent", "con_ent", "dep_ent", "avg_dep_height", "avg_dep_dis"]


def get_args():
    parser = argparse.ArgumentParser(
        description="Collect global GC scores from cdf_gc outputs and compute partial correlations."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Root folder of GC outputs. Supports cdf_gc output root or split lexical/syntactic folders.",
    )
    parser.add_argument(
        "--score_source",
        type=str,
        default="normalized",
        choices=["combined", "normalized"],
        help="Use raw scores from 5_combined_gc or normalized scores from 6_normalized_gc.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=DEFAULT_COMPONENTS,
        help="GC components to include in the correlation analysis.",
    )
    parser.add_argument(
        "--merged_scores_path",
        type=str,
        default=None,
        help="Optional path to save merged global scores as jsonl.",
    )
    parser.add_argument(
        "--json_output_path",
        type=str,
        default=None,
        help="Optional path to save the correlation result as json.",
    )
    parser.add_argument(
        "--csv_output_path",
        type=str,
        default=None,
        help="Optional path to save the partial correlation matrix as csv.",
    )
    parser.add_argument(
        "--pca_csv_output_path",
        type=str,
        default=None,
        help="Optional path to save PCA summary and loadings as csv.",
    )
    parser.add_argument(
        "--correlation_plot_path",
        type=str,
        default=None,
        help="Optional path to save the Pearson/partial correlation heatmaps as png.",
    )
    parser.add_argument(
        "--pca_plot_path",
        type=str,
        default=None,
        help="Optional path to save the PCA scree/loadings plot as png.",
    )
    return parser.parse_args()


def resolve_paths(output_path: str, score_source: str):
    output_root = Path(output_path)
    gc_root = output_root / "1_gc_data"
    split_lexical_folder = output_root / "3_lexical_diversity"
    split_syntactic_folder = output_root / "4_syntactic_complexity"

    if gc_root.exists():
        input_layout = "cdf_gc_output"
        if score_source == "combined":
            score_folder = gc_root / "5_combined_gc"
        else:
            score_folder = gc_root / "6_normalized_gc"
        output_suffix = score_source
    elif split_lexical_folder.exists() and split_syntactic_folder.exists():
        input_layout = "split_gc_components"
        score_folder = output_root
        output_suffix = "aligned_components"
    else:
        raise FileNotFoundError(
            "Unsupported input layout. Expected either "
            "'1_gc_data/5_combined_gc|6_normalized_gc' or "
            "'3_lexical_diversity + 4_syntactic_complexity'."
        )

    analysis_dir = output_root / "analysis"
    merged_scores_path = analysis_dir / f"global_gc_scores_{output_suffix}.jsonl"
    json_output_path = analysis_dir / f"gc_partial_correlation_{output_suffix}.json"
    csv_output_path = analysis_dir / f"gc_partial_correlation_{output_suffix}.csv"
    pca_csv_output_path = analysis_dir / f"gc_pca_{output_suffix}.csv"
    correlation_plot_path = analysis_dir / f"gc_correlation_{output_suffix}.png"
    pca_plot_path = analysis_dir / f"gc_pca_{output_suffix}.png"
    return (
        input_layout,
        score_folder,
        merged_scores_path,
        json_output_path,
        csv_output_path,
        pca_csv_output_path,
        correlation_plot_path,
        pca_plot_path,
    )


def iter_score_rows(score_folder: Path, components: list[str], score_source: str):
    jsonl_files = sorted(score_folder.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No jsonl files found under {score_folder}")

    for jsonl_file in jsonl_files:
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                item = json.loads(line)
                if score_source == "normalized":
                    source_item = item["normalized_gc"]
                else:
                    source_item = item
                try:
                    yield {
                        component: float(source_item[component])
                        for component in components
                    }
                except KeyError as exc:
                    raise KeyError(
                        f"Missing component {exc.args[0]!r} in {jsonl_file}:{line_idx}"
                    ) from exc


def iter_split_score_rows(score_root: Path, components: list[str]):
    lexical_folder = score_root / "3_lexical_diversity"
    syntactic_folder = score_root / "4_syntactic_complexity"
    lexical_files = sorted(lexical_folder.glob("*.jsonl"))
    syntactic_files = sorted(syntactic_folder.glob("*.jsonl"))

    lexical_names = [path.name for path in lexical_files]
    syntactic_names = [path.name for path in syntactic_files]
    if lexical_names != syntactic_names:
        raise ValueError(
            "Lexical and syntactic file names are not aligned: "
            f"{lexical_names[:5]} vs {syntactic_names[:5]}"
        )

    for lexical_file, syntactic_file in zip(lexical_files, syntactic_files):
        with lexical_file.open("r", encoding="utf-8") as lexical_f, syntactic_file.open("r", encoding="utf-8") as syntactic_f:
            for line_idx, (lexical_line, syntactic_line) in enumerate(
                zip_longest(lexical_f, syntactic_f),
                start=1,
            ):
                if lexical_line is None or syntactic_line is None:
                    raise ValueError(
                        f"Line count mismatch for aligned files {lexical_file.name} at line {line_idx}"
                    )

                lexical_item = json.loads(lexical_line)
                syntactic_item = json.loads(syntactic_line)
                merged_item = {
                    **lexical_item,
                    **syntactic_item,
                    "source_file": lexical_file.name,
                    "line_number": line_idx,
                }
                try:
                    yield {
                        key: merged_item[key]
                        for key in ["source_file", "line_number", *components]
                    }
                except KeyError as exc:
                    raise KeyError(
                        f"Missing component {exc.args[0]!r} in aligned files "
                        f"{lexical_file.name}:{line_idx}"
                    ) from exc


def load_score_matrix(score_folder: Path, components: list[str], score_source: str, input_layout: str):
    if input_layout == "split_gc_components":
        rows = list(iter_split_score_rows(score_folder, components))
    else:
        rows = list(iter_score_rows(score_folder, components, score_source))
    if len(rows) < 3:
        raise ValueError("At least 3 samples are required to compute partial correlations.")

    matrix = np.asarray([[row[component] for component in components] for row in rows], dtype=np.float64)
    variances = np.var(matrix, axis=0)
    zero_var_components = [component for component, variance in zip(components, variances) if np.isclose(variance, 0.0)]
    if zero_var_components:
        raise ValueError(f"Found zero-variance components: {zero_var_components}")
    return rows, matrix


def compute_pearson_correlation(matrix: np.ndarray):
    return np.corrcoef(matrix, rowvar=False)


def compute_partial_correlation(matrix: np.ndarray):
    corr = compute_pearson_correlation(matrix)
    precision = np.linalg.pinv(corr)
    diag = np.diag(precision)
    scale = np.sqrt(np.outer(diag, diag))
    partial_corr = -precision / scale
    np.fill_diagonal(partial_corr, 1.0)
    return partial_corr


def compute_pca(matrix: np.ndarray):
    corr = compute_pearson_correlation(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Fix sign ambiguity for readability: make the largest absolute loading positive.
    for idx in range(eigenvectors.shape[1]):
        col = eigenvectors[:, idx]
        max_abs_idx = int(np.argmax(np.abs(col)))
        if col[max_abs_idx] < 0:
            eigenvectors[:, idx] = -col

    explained_ratio = eigenvalues / eigenvalues.sum()
    cumulative_explained_ratio = np.cumsum(explained_ratio)
    loadings = eigenvectors * np.sqrt(eigenvalues)
    return eigenvalues, explained_ratio, cumulative_explained_ratio, eigenvectors, loadings


def matrix_to_dict(matrix: np.ndarray, components: list[str]):
    return {
        row_name: {
            col_name: float(matrix[row_idx, col_idx])
            for col_idx, col_name in enumerate(components)
        }
        for row_idx, row_name in enumerate(components)
    }


def save_merged_scores(rows: list[dict], merged_scores_path: Path):
    merged_scores_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_scores_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_partial_corr_csv(partial_corr: np.ndarray, components: list[str], csv_output_path: Path):
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", *components])
        for row_name, row in zip(components, partial_corr):
            writer.writerow([row_name, *[f"{value:.6f}" for value in row]])


def save_pca_csv(
    eigenvalues: np.ndarray,
    explained_ratio: np.ndarray,
    cumulative_explained_ratio: np.ndarray,
    eigenvectors: np.ndarray,
    loadings: np.ndarray,
    components: list[str],
    pca_csv_output_path: Path,
):
    pca_csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    with pca_csv_output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "principal_component",
                "eigenvalue",
                "explained_ratio",
                "cumulative_explained_ratio",
                *[f"weight_{component}" for component in components],
                *[f"loading_{component}" for component in components],
            ]
        )
        for idx in range(len(eigenvalues)):
            writer.writerow(
                [
                    f"PC{idx + 1}",
                    f"{eigenvalues[idx]:.6f}",
                    f"{explained_ratio[idx]:.6f}",
                    f"{cumulative_explained_ratio[idx]:.6f}",
                    *[f"{value:.6f}" for value in eigenvectors[:, idx]],
                    *[f"{value:.6f}" for value in loadings[:, idx]],
                ]
            )


def print_matrix(title: str, matrix: np.ndarray, components: list[str]):
    col_width = 16
    print(title)
    print("".ljust(col_width) + "".join(name.rjust(col_width) for name in components))
    for row_name, row in zip(components, matrix):
        print(row_name.ljust(col_width) + "".join(f"{value:>{col_width}.6f}" for value in row))


def print_pca_summary(
    eigenvalues: np.ndarray,
    explained_ratio: np.ndarray,
    cumulative_explained_ratio: np.ndarray,
    loadings: np.ndarray,
    components: list[str],
    top_k: int = 3,
):
    print("PCA summary")
    for idx in range(min(top_k, len(eigenvalues))):
        print(
            f"PC{idx + 1}: eigenvalue={eigenvalues[idx]:.6f}, "
            f"explained_ratio={explained_ratio[idx]:.6f}, "
            f"cumulative={cumulative_explained_ratio[idx]:.6f}"
        )
        for component, value in zip(components, loadings[:, idx]):
            print(f"  {component}: {value:.6f}")


def save_correlation_plot(
    pearson_corr: np.ndarray,
    partial_corr: np.ndarray,
    components: list[str],
    correlation_plot_path: Path,
):
    correlation_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    matrices = [
        ("Pearson Correlation", pearson_corr),
        ("Partial Correlation", partial_corr),
    ]
    for ax, (title, matrix) in zip(axes, matrices):
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(len(components)))
        ax.set_yticks(range(len(components)))
        ax.set_xticklabels(components, rotation=45, ha="right")
        ax.set_yticklabels(components)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label="correlation")
    fig.savefig(correlation_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pca_plot(
    eigenvalues: np.ndarray,
    explained_ratio: np.ndarray,
    cumulative_explained_ratio: np.ndarray,
    loadings: np.ndarray,
    components: list[str],
    pca_plot_path: Path,
    top_k: int = 3,
):
    pca_plot_path.parent.mkdir(parents=True, exist_ok=True)
    top_k = min(top_k, len(eigenvalues))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    pcs = np.arange(1, len(eigenvalues) + 1)
    axes[0].bar(pcs, explained_ratio, color="#4C78A8", label="Explained Ratio")
    axes[0].plot(pcs, cumulative_explained_ratio, color="#F58518", marker="o", label="Cumulative Ratio")
    axes[0].set_title("PCA Explained Variance")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Ratio")
    axes[0].set_xticks(pcs)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    loading_matrix = loadings[:, :top_k]
    image = axes[1].imshow(loading_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[1].set_title(f"PCA Loadings (Top {top_k} PCs)")
    axes[1].set_xticks(range(top_k))
    axes[1].set_xticklabels([f"PC{i}" for i in range(1, top_k + 1)])
    axes[1].set_yticks(range(len(components)))
    axes[1].set_yticklabels(components)
    for row_idx in range(loading_matrix.shape[0]):
        for col_idx in range(loading_matrix.shape[1]):
            axes[1].text(col_idx, row_idx, f"{loading_matrix[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes[1], shrink=0.9, label="loading")

    fig.savefig(pca_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    (
        input_layout,
        score_folder,
        default_merged_scores_path,
        default_json_output_path,
        default_csv_output_path,
        default_pca_csv_output_path,
        default_correlation_plot_path,
        default_pca_plot_path,
    ) = resolve_paths(
        args.output_path,
        args.score_source,
    )
    merged_scores_path = Path(args.merged_scores_path) if args.merged_scores_path else default_merged_scores_path
    json_output_path = Path(args.json_output_path) if args.json_output_path else default_json_output_path
    csv_output_path = Path(args.csv_output_path) if args.csv_output_path else default_csv_output_path
    pca_csv_output_path = Path(args.pca_csv_output_path) if args.pca_csv_output_path else default_pca_csv_output_path
    correlation_plot_path = Path(args.correlation_plot_path) if args.correlation_plot_path else default_correlation_plot_path
    pca_plot_path = Path(args.pca_plot_path) if args.pca_plot_path else default_pca_plot_path

    rows, matrix = load_score_matrix(score_folder, args.components, args.score_source, input_layout)
    pearson_corr = compute_pearson_correlation(matrix)
    partial_corr = compute_partial_correlation(matrix)
    eigenvalues, explained_ratio, cumulative_explained_ratio, eigenvectors, loadings = compute_pca(matrix)

    save_merged_scores(rows, merged_scores_path)
    save_partial_corr_csv(partial_corr, args.components, csv_output_path)
    save_pca_csv(
        eigenvalues,
        explained_ratio,
        cumulative_explained_ratio,
        eigenvectors,
        loadings,
        args.components,
        pca_csv_output_path,
    )
    save_correlation_plot(pearson_corr, partial_corr, args.components, correlation_plot_path)
    save_pca_plot(
        eigenvalues,
        explained_ratio,
        cumulative_explained_ratio,
        loadings,
        args.components,
        pca_plot_path,
    )

    result = {
        "input_layout": input_layout,
        "score_folder": str(score_folder),
        "score_source": args.score_source,
        "sample_count": len(rows),
        "components": args.components,
        "pearson_correlation": matrix_to_dict(pearson_corr, args.components),
        "partial_correlation": matrix_to_dict(partial_corr, args.components),
        "pca": [
            {
                "principal_component": f"PC{idx + 1}",
                "eigenvalue": float(eigenvalues[idx]),
                "explained_ratio": float(explained_ratio[idx]),
                "cumulative_explained_ratio": float(cumulative_explained_ratio[idx]),
                "weights": {
                    component: float(eigenvectors[component_idx, idx])
                    for component_idx, component in enumerate(args.components)
                },
                "loadings": {
                    component: float(loadings[component_idx, idx])
                    for component_idx, component in enumerate(args.components)
                },
            }
            for idx in range(len(eigenvalues))
        ],
        "merged_scores_path": str(merged_scores_path),
        "csv_output_path": str(csv_output_path),
        "pca_csv_output_path": str(pca_csv_output_path),
        "correlation_plot_path": str(correlation_plot_path),
        "pca_plot_path": str(pca_plot_path),
    }
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    with json_output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Detected input layout: {input_layout}")
    print(f"Loaded {len(rows)} samples from: {score_folder}")
    print(f"Merged scores saved to: {merged_scores_path}")
    print(f"JSON result saved to: {json_output_path}")
    print(f"CSV result saved to: {csv_output_path}")
    print(f"PCA CSV result saved to: {pca_csv_output_path}")
    print(f"Correlation plot saved to: {correlation_plot_path}")
    print(f"PCA plot saved to: {pca_plot_path}")
    print()
    print_matrix("Pearson correlation matrix", pearson_corr, args.components)
    print()
    print_matrix("Partial correlation matrix", partial_corr, args.components)
    print()
    print_pca_summary(
        eigenvalues,
        explained_ratio,
        cumulative_explained_ratio,
        loadings,
        args.components,
    )


if __name__ == "__main__":
    main()
