# Preferences Time Series GPT

* https://github.com/rcalix1/UCIdataTS_GPT_preferencesDPO/tree/main/2025

## SPO


<pre lang="python"><code>
import torch
import torch.nn.functional as F

# 🎯 Ground truth and two predictions
y_target = torch.tensor([1.0, 2.0, 3.0])
y_pref   = torch.tensor([1.1, 2.1, 2.9])  # Preferred output
y_rej    = torch.tensor([0.5, 1.5, 2.0])  # Rejected output

# 📉 Scoring function: negative MSE (higher is better)
def score(y_pred, y_target):
    return -F.mse_loss(y_pred, y_target, reduction='mean')

# 🧮 Compute preference scores
score_pref = score(y_pref, y_target)
score_rej  = score(y_rej, y_target)

# 🔥 Softmax temperature
T = 1.0

# 🧠 Convert scores to logits and compute softmax
logits = torch.tensor([score_pref / T, score_rej / T])
probs  = F.softmax(logits, dim=0)

# ✅ Manual cross-entropy: assume preferred = index 0
label_index = 0
manual_loss = -torch.log(probs[label_index])

# 📊 Output results
print(f"Score (preferred): {score_pref.item():.4f}")
print(f"Score (rejected):  {score_rej.item():.4f}")
print(f"Softmax probs:     {probs.tolist()}")
print(f"Preference loss:   {manual_loss.item():.4f}")

 </code></pre>
 

