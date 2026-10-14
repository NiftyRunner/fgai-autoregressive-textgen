import os
import sys
# Ensure project root is in python path
sys.path.insert(0, os.path.abspath("."))
import argparse
import json
import pandas as pd
import torch
import torch.nn as nn

from src.data_utils import (
    download_and_clean_data,
    tokenize_text,
    compute_corpus_statistics,
    prepare_dataloaders
)
from src.models import (
    get_model_by_name,
    count_parameters
)
from src.train_utils import (
    set_seed,
    get_device,
    train_model,
    evaluate
)
from src.generate import (
    FIXED_SEEDS,
    generate_text,
    compute_text_quality_metrics
)
from src.plot_utils import (
    plot_top20_tokens,
    plot_loss_curves,
    plot_perplexity_bar,
    plot_accuracy_bar,
    plot_train_time_bar,
    plot_temperature_metrics
)

def run_all(fast_mode=False, max_epochs=15, batch_size=128, lr=1e-3, seed=42):
    print("=" * 80)
    print(f"STARTING EXPERIMENTS | FAST_MODE={fast_mode} | MAX_EPOCHS={max_epochs}")
    print("=" * 80)
    
    set_seed(seed)
    device = get_device()
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. Download & Data Preprocessing
    raw_text = download_and_clean_data()
    all_tokens = tokenize_text(raw_text)
    stats = compute_corpus_statistics(raw_text, all_tokens)
    print(f"Dataset stats: {stats['char_count']} chars, {stats['token_count']} tokens, {stats['unique_tokens']} unique tokens")
    print(f"Sentence stats: {stats['sentence_count']} sentences, Mean length: {stats['sentence_len_mean']:.1f} tokens")
    
    # Save top 20 tokens plot
    plot_top20_tokens(stats["top_20_tokens"])
    
    # Prepare DataLoaders
    max_vocab = 1000 if fast_mode else 8000
    epochs = 2 if fast_mode else max_epochs
    patience = 2 if fast_mode else 3
    
    data_bundle = prepare_dataloaders(
        raw_text=raw_text,
        seq_len=25,
        max_vocab_size=max_vocab,
        batch_size=batch_size,
        fast_mode=fast_mode,
        seed=seed
    )
    vocab = data_bundle["vocab"]
    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]
    
    vocab_size = len(vocab)
    print(f"Vocabulary Size: {vocab_size} (Special: <pad>=0, <unk>=1)")
    print(f"Dataset Windows -> Train: {len(data_bundle['train_dataset'])}, Val: {len(data_bundle['val_dataset'])}, Test: {len(data_bundle['test_dataset'])}")
    
    model_names = ["SimpleRNN", "BiRNN", "LSTM", "GRU"]
    trained_models = {}
    histories = {}
    metrics_records = []
    
    criterion = nn.CrossEntropyLoss()
    
    # 2. Train and Evaluate each model
    for name in model_names:
        print("\n" + "-" * 60)
        print(f"TRAINING MODEL: {name}")
        print("-" * 60)
        
        # Reset seed before creating each model for fair comparative conditions
        set_seed(seed)
        model = get_model_by_name(name, vocab_size=vocab_size, embed_dim=128, hidden_size=256, dropout=0.3)
        param_info = count_parameters(model)
        print(f"{name} Trainable Parameters: {param_info['trainable']:,} | Total: {param_info['total']:,}")
        print(f"Parameter Breakdown: {param_info['breakdown']}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_res = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_epochs=epochs,
            patience=patience,
            max_grad_norm=1.0,
            lr=lr,
            verbose=True
        )
        
        histories[name] = train_res["history"]
        trained_models[name] = model
        
        # Evaluate on Test Set
        print(f"Evaluating {name} on Test Set...")
        test_res = evaluate(model, test_loader, criterion, device)
        
        rec = {
            "Model": name,
            "Parameters": param_info["trainable"],
            "Train_Loss": train_res["history"]["train_loss"][-1],
            "Val_Loss": train_res["best_val_loss"],
            "Test_Loss": test_res["loss"],
            "Perplexity": test_res["perplexity"],
            "Top1_Acc": test_res["top1_acc"],
            "Top5_Acc": test_res["top5_acc"],
            "Epochs_Trained": train_res["epochs_trained"],
            "Avg_Epoch_Time_s": train_res["avg_epoch_time"],
            "Total_Train_Time_s": train_res["total_time"]
        }
        metrics_records.append(rec)
        print(f"--> {name} Test Loss: {test_res['loss']:.4f} | Perplexity: {test_res['perplexity']:.2f} | Top-1: {test_res['top1_acc']*100:.2f}% | Top-5: {test_res['top5_acc']*100:.2f}% | Time: {train_res['total_time']:.1f}s")
        
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = "results/metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved metrics to {metrics_csv_path}:")
    print(metrics_df.to_string(index=False))
    
    # 3. Generate Comparative Plots
    plot_loss_curves(histories)
    plot_perplexity_bar(metrics_df)
    plot_accuracy_bar(metrics_df)
    plot_train_time_bar(metrics_df)
    
    # 4. Text Generation across Models, Seeds, and Temperatures
    print("\n" + "=" * 80)
    print("STARTING AUTOREGRESSIVE TEXT GENERATION")
    print("=" * 80)
    
    temperatures = [0.5, 1.0, 1.5]
    samples_records = []
    text_metrics_records = []
    
    md_lines = [
        "# Autoregressive Text Generation Samples",
        "",
        "This file records generated samples for all four models (**SimpleRNN**, **BiRNN**, **LSTM**, **GRU**) evaluated on 4 fixed seed prompts from *The Adventures of Sherlock Holmes* across three sampling temperatures ($T = 0.5, 1.0, 1.5$).",
        "",
        "Each continuation was generated autoregressively for exactly 60 tokens.",
        ""
    ]
    
    for model_name in model_names:
        model = trained_models[model_name]
        md_lines.append(f"## Model: {model_name}\n")
        
        for temp in temperatures:
            md_lines.append(f"### Temperature = {temp}\n")
            temp_metrics_list = []
            
            for seed_idx, seed_text in enumerate(FIXED_SEEDS, 1):
                # Autoregressive generation
                gen_out = generate_text(
                    model=model,
                    vocab=vocab,
                    seed_text=seed_text,
                    num_words_to_generate=60,
                    temperature=temp,
                    seq_len=25,
                    device=device,
                    seed=seed + seed_idx * 100
                )
                
                # Compute lexical diversity & repetition metrics on generated continuation
                q_metrics = compute_text_quality_metrics(gen_out["generated_tokens"])
                temp_metrics_list.append(q_metrics)
                
                samples_records.append({
                    "Model": model_name,
                    "Temperature": temp,
                    "Seed_ID": seed_idx,
                    "Seed_Text": seed_text,
                    "Generated_Continuation": gen_out["generated_text"],
                    "Full_Text": gen_out["full_text"],
                    "Distinct_1": q_metrics["distinct_1"],
                    "Distinct_2": q_metrics["distinct_2"],
                    "Repeated_Trigram_Ratio": q_metrics["repeated_trigram_ratio"],
                    "Unique_Word_Ratio": q_metrics["unique_word_ratio"]
                })
                
                md_lines.append(f"**Seed {seed_idx}:** *\"{seed_text}\"*  ")
                md_lines.append(f"**Generated Continuation:** {gen_out['generated_text']}  ")
                md_lines.append(f"**Metrics:** Distinct-1: {q_metrics['distinct_1']:.3f} | Distinct-2: {q_metrics['distinct_2']:.3f} | Rep-Trigram: {q_metrics['repeated_trigram_ratio']:.3f} | Unique-Word: {q_metrics['unique_word_ratio']:.3f}\n")
                
            # Average quality metrics across seeds for this (model, temp)
            avg_d1 = float(pd.Series([m["distinct_1"] for m in temp_metrics_list]).mean())
            avg_d2 = float(pd.Series([m["distinct_2"] for m in temp_metrics_list]).mean())
            avg_trig = float(pd.Series([m["repeated_trigram_ratio"] for m in temp_metrics_list]).mean())
            avg_uw = float(pd.Series([m["unique_word_ratio"] for m in temp_metrics_list]).mean())
            
            text_metrics_records.append({
                "Model": model_name,
                "Temperature": temp,
                "Distinct_1": avg_d1,
                "Distinct_2": avg_d2,
                "Repeated_Trigram_Ratio": avg_trig,
                "Unique_Word_Ratio": avg_uw
            })
            
    # Save samples
    samples_df = pd.DataFrame(samples_records)
    samples_df.to_csv("generated_samples/samples.csv", index=False)
    with open("generated_samples/samples.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Saved generated samples to generated_samples/samples.csv and generated_samples/samples.md")
    
    # Save text metrics and plot
    text_metrics_df = pd.DataFrame(text_metrics_records)
    text_metrics_df.to_csv("results/text_metrics.csv", index=False)
    plot_temperature_metrics(text_metrics_df)
    print("Saved text metrics to results/text_metrics.csv and results/temperature_metrics.png")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT RUN COMPLETE!")
    print("=" * 80)
    return {
        "metrics_df": metrics_df,
        "text_metrics_df": text_metrics_df,
        "samples_df": samples_df,
        "histories": histories,
        "stats": stats,
        "vocab": vocab
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run quick smoke test")
    parser.add_argument("--epochs", type=int, default=15, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()
    
    run_all(fast_mode=args.fast, max_epochs=args.epochs, batch_size=args.batch_size)
