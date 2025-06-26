# Preferences Time Series GPT

* https://github.com/rcalix1/UCIdataTS_GPT_preferencesDPO/tree/main/2025

## SPO

<pre lang="python"><code>```python import torch import torch.nn.functional as F # 🎯 Ground truth and two predicted sequences y_target = torch.tensor([1.0, 2.0, 3.0]) y_pref = torch.tensor([1.1, 2.1, 2.9]) # Preferred prediction y_rej = torch.tensor([0.5, 1.5, 2.0]) # Rejected prediction # 📉 Score function: negative MSE (lower MSE → higher score) def score(y_pred, y_target): return -F.mse_loss(y_pred, y_target, reduction='mean') # 🧮 Compute scores for both predictions score_pref = score(y_pref, y_target) score_rej = score(y_rej, y_target) # 🔥 Temperature for softmax scaling T = 1.0 # 🧠 Compute logits and softmax probabilities logits = torch.tensor([score_pref / T, score_rej / T]) probs = F.softmax(logits, dim=0) # ✅ Manual cross-entropy loss (preferred = index 0) label_index = 0 manual_loss = -torch.log(probs[label_index]) # 📊 Print everything print(f"Score (pref): {score_pref.item():.4f}") print(f"Score (rej): {score_rej.item():.4f}") print(f"Softmax probs: {probs.tolist()}") print(f"Manual loss: {manual_loss.item():.4f}") ``` </code></pre>
