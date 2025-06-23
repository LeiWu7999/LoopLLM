import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path
import glob

def plot_metrics(results_df, output_dir):
    """
    Generates and saves plots for each metric from the combined evaluation results.
    """
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # Create a numeric 'step' column for proper sorting and legend ordering
    # baseline is -1, checkpoint-100 is 100, etc.
    results_df['step'] = results_df['checkpoint'].apply(
        lambda x: int(x.split('-')[-1]) if 'checkpoint' in x else -1
    )
    # Sort data to ensure lines are plotted in a meaningful order (baseline, 100, 200, ...)
    results_df = results_df.sort_values(by=['step', 'loop_count'])
    
    # Define the metrics to plot
    metrics_to_plot = {
        'cosine_similarity': {
            'title': 'Cosine Similarity between h_in and h_out vs. Loop Count',
            'ylabel': 'Cosine Similarity (Higher is more similar)'
        },
        'mse': {
            'title': 'Mean Squared Error (MSE) between h_in and h_out vs. Loop Count',
            'ylabel': 'MSE (Lower is more similar)'
        },
        'norm_diff': {
            'title': 'Norm Difference (||h_out|| - ||h_in||) vs. Loop Count',
            'ylabel': 'Average Norm Difference'
        }
    }
    
    for metric, details in metrics_to_plot.items():
        plt.figure(figsize=(12, 8))
        
        plot = sns.lineplot(
            data=results_df,
            x='loop_count',
            y=metric,
            hue='checkpoint',
            style='checkpoint',
            markers=True,
            palette='viridis',
            legend='full',
            hue_order=sorted(results_df['checkpoint'].unique(), key=lambda x: (
                -1 if 'baseline' in x else int(x.split('-')[-1])
            ))
        )
        
        plt.title(details['title'], fontsize=16)
        plt.xlabel('Number of Loops', fontsize=12)
        plt.ylabel(details['ylabel'], fontsize=12)
        plt.xticks(results_df['loop_count'].unique())
        plt.legend(title='Checkpoint', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        if metric == 'mse':
            plot.set_yscale('log')
            plt.ylabel(f"{details['ylabel']} (log scale)")

        output_path = Path(output_dir) / f"{metric}_vs_loop_count.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot model degeneration evaluation results from multiple CSVs.")
    parser.add_argument("--results_dir", type=str, default="./degeneration_results", help="Directory containing the CSV metric files.")
    parser.add_argument("--output_dir", type=str, default="./degeneration_results", help="Directory to save the plots.")
    args = parser.parse_args()

    # Find all metric files in the directory
    search_pattern = str(Path(args.results_dir) / "*_metrics.csv")
    metric_files = glob.glob(search_pattern)

    if not metric_files:
        print(f"Error: No '*_metrics.csv' files found in '{args.results_dir}'")
        print("Please run the evaluation scripts first.")
        return
    
    print(f"Found {len(metric_files)} result files to merge: {metric_files}")

    # Load and concatenate all dataframes
    df_list = [pd.read_csv(f) for f in metric_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate and save plots
    plot_metrics(combined_df, args.output_dir)

if __name__ == "__main__":
    main() 