import os
import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
    "grid.alpha": 0.4
})

COLORS = {
    "SimpleRNN": "#1f77b4",
    "BiRNN": "#ff7f0e",
    "LSTM": "#2ca02c",
    "GRU": "#d62728"
}

def plot_top20_tokens(top20_data, save_path="results/top20_words.png"):
    """
    Plots the top 20 most frequent tokens in the dataset.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokens, counts = zip(*top20_data)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(tokens)), counts, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right", fontsize=11)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Top 20 Most Frequent Tokens in 'The Adventures of Sherlock Holmes'", fontsize=13, fontweight="bold")
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + max(counts) * 0.01,
                 f"{height}", ha="center", va="bottom", fontsize=8, rotation=0)
                 
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved top 20 tokens plot to {save_path}")

def plot_loss_curves(histories, save_path="results/loss_curves.png"):
    """
    Plots train and validation loss curves across training epochs for all models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    for model_name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        color = COLORS.get(model_name, "#333333")
        axes[0].plot(epochs, hist["train_loss"], label=model_name, color=color, marker="o", markersize=4)
        axes[1].plot(epochs, hist["val_loss"], label=model_name, color=color, marker="s", markersize=4)
        
    axes[0].set_title("Training Loss vs. Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(frameon=True)
    
    axes[1].set_title("Validation Loss vs. Epochs", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(frameon=True)
    
    plt.suptitle("Loss Trajectories across Models (Sherlock Holmes Corpus)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss curves plot to {save_path}")

def plot_perplexity_bar(metrics_df, save_path="results/perplexity_bar.png"):
    """
    Bar chart comparing test perplexity across models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    
    models = metrics_df["Model"].tolist()
    ppls = metrics_df["Perplexity"].tolist()
    colors = [COLORS.get(m, "#4a5568") for m in models]
    
    bars = plt.bar(models, ppls, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    plt.ylabel("Test Perplexity (Lower is Better)", fontsize=12)
    plt.title("Model Comparison: Test Perplexity (exp(Loss))", fontsize=13, fontweight="bold")
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + max(ppls) * 0.015,
                 f"{height:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
                 
    plt.ylim(0, max(ppls) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved perplexity bar chart to {save_path}")

def plot_accuracy_bar(metrics_df, save_path="results/accuracy_bar.png"):
    """
    Grouped bar chart for Top-1 and Top-5 test accuracy across models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 5))
    
    models = metrics_df["Model"].tolist()
    top1 = (metrics_df["Top1_Acc"] * 100).tolist()
    top5 = (metrics_df["Top5_Acc"] * 100).tolist()
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = plt.bar(x - width/2, top1, width, label="Top-1 Accuracy", color="#2563eb", edgecolor="black", alpha=0.85)
    rects2 = plt.bar(x + width/2, top5, width, label="Top-5 Accuracy", color="#10b981", edgecolor="black", alpha=0.85)
    
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Model Comparison: Top-1 vs. Top-5 Test Accuracy", fontsize=13, fontweight="bold")
    plt.xticks(x, models, fontsize=11)
    plt.legend(frameon=True)
    
    for bar in rects1:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9)
    for bar in rects2:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=9)
        
    plt.ylim(0, max(top5) * 1.18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved accuracy bar chart to {save_path}")

def plot_train_time_bar(metrics_df, save_path="results/train_time_bar.png"):
    """
    Bar chart comparing total training time across models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    
    models = metrics_df["Model"].tolist()
    times = metrics_df["Total_Train_Time_s"].tolist()
    colors = [COLORS.get(m, "#4a5568") for m in models]
    
    bars = plt.bar(models, times, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    plt.ylabel("Total Training Time (seconds)", fontsize=12)
    plt.title("Model Comparison: Total GPU Training Time", fontsize=13, fontweight="bold")
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + max(times) * 0.015,
                 f"{height:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
                 
    plt.ylim(0, max(times) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training time bar chart to {save_path}")

def plot_temperature_metrics(text_metrics_df, save_path="results/temperature_metrics.png"):
    """
    Plots text quality metrics (Distinct-1, Distinct-2, Repeated Trigram, Unique Word Ratio)
    against temperature per model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ("Distinct_1", "Distinct-1 Ratio (Lexical Diversity)", axes[0, 0]),
        ("Distinct_2", "Distinct-2 Ratio (Bigram Diversity)", axes[0, 1]),
        ("Repeated_Trigram_Ratio", "Repeated Trigram Ratio (Repetition)", axes[1, 0]),
        ("Unique_Word_Ratio", "Unique Word Ratio", axes[1, 1])
    ]
    
    models = text_metrics_df["Model"].unique()
    
    for col_name, title, ax in metrics:
        for model in models:
            sub = text_metrics_df[text_metrics_df["Model"] == model].sort_values("Temperature")
            ax.plot(
                sub["Temperature"],
                sub[col_name],
                marker="o",
                label=model,
                color=COLORS.get(model, "#333333"),
                linewidth=2
            )
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Sampling Temperature")
        ax.set_ylabel("Score")
        ax.set_xticks([0.5, 1.0, 1.5])
        ax.legend(frameon=True)
        
    plt.suptitle("Impact of Sampling Temperature on Generated Text Quality", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved temperature metrics plot to {save_path}")

    # Also save individual plots into results/temperature_plots/
    indiv_dir = os.path.join(os.path.dirname(save_path), "temperature_plots")
    os.makedirs(indiv_dir, exist_ok=True)
    metric_filenames = {
        "Distinct_1": "distinct_1.png",
        "Distinct_2": "distinct_2.png",
        "Repeated_Trigram_Ratio": "repeated_trigram_ratio.png",
        "Unique_Word_Ratio": "unique_word_ratio.png"
    }
    for col_name, title, _ in metrics:
        plt.figure(figsize=(7, 5))
        for model in models:
            sub = text_metrics_df[text_metrics_df["Model"] == model].sort_values("Temperature")
            plt.plot(
                sub["Temperature"],
                sub[col_name],
                marker="o",
                label=model,
                color=COLORS.get(model, "#333333"),
                linewidth=2
            )
        plt.title(title, fontweight="bold", fontsize=12)
        plt.xlabel("Sampling Temperature")
        plt.ylabel("Score")
        plt.xticks([0.5, 1.0, 1.5])
        plt.legend(frameon=True)
        plt.tight_layout()
        single_path = os.path.join(indiv_dir, metric_filenames[col_name])
        plt.savefig(single_path, dpi=300)
        plt.close()
    print(f"Saved individual temperature plots to {indiv_dir}")
