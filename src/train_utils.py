import time
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader, criterion, device):
    """
    Computes average loss, top-1 accuracy, and top-5 accuracy on a DataLoader.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Top-1 accuracy
            _, preds = logits.max(dim=-1)
            correct_top1 += (preds == y).sum().item()
            
            # Top-5 accuracy
            _, top5_preds = logits.topk(k=min(5, logits.size(-1)), dim=-1)
            correct_top5 += (top5_preds == y.unsqueeze(-1)).any(dim=-1).sum().item()
            
    avg_loss = total_loss / max(1, total_samples)
    top1_acc = correct_top1 / max(1, total_samples)
    top5_acc = correct_top5 / max(1, total_samples)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc
    }

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer=None,
    criterion=None,
    device=None,
    max_epochs=15,
    patience=3,
    max_grad_norm=1.0,
    lr=1e-3,
    verbose=True
):
    """
    Trains a model with early stopping on validation loss and gradient clipping.
    Measures GPU runtime accurately with torch.cuda.synchronize().
    """
    if device is None:
        device = get_device()
    model.to(device)
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "epoch_times": []
    }
    
    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0
    best_epoch = 0
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_start_time = time.perf_counter()
    
    for epoch in range(1, max_epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()
        
        # Training Phase
        model.train()
        running_train_loss = 0.0
        train_samples = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                
            optimizer.step()
            
            batch_sz = y.size(0)
            running_train_loss += loss.item() * batch_sz
            train_samples += batch_sz
            
        epoch_train_loss = running_train_loss / max(1, train_samples)
        
        # Validation Phase
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_val_loss = val_metrics["loss"]
        epoch_val_ppl = val_metrics["perplexity"]
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_ppl"].append(epoch_val_ppl)
        history["epoch_times"].append(epoch_duration)
        
        if verbose:
            print(
                f"Epoch {epoch:02d}/{max_epochs:02d} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"Val PPL: {epoch_val_ppl:.2f} | "
                f"Time: {epoch_duration:.2f}s"
            )
            
        # Early Stopping Logic
        if epoch_val_loss < best_val_loss - 1e-4:
            best_val_loss = epoch_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}. Restoring weights from epoch {best_epoch}.")
                break
                
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_end_time = time.perf_counter()
    total_duration = total_end_time - total_start_time
    
    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return {
        "history": history,
        "total_time": total_duration,
        "avg_epoch_time": float(np.mean(history["epoch_times"])),
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }
