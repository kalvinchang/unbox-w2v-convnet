# utils.py is from https://github.com/juice500ml/unbox-w2v-convnet/blob/main/utils.py

from transformers import AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_signal, get_feature
import torch
import numpy as np
from tqdm import tqdm
import librosa
import librosa.display
import scipy

# Goal: Identify the "perceptual" boundary in SSL feature space for 2 vowels
# Method: Optimize input signal to reproduce SSL features for the signals for 2 vowels
# Hypothesis: The proximity of the cluster will reproduce [a] and [i].



# WavLM normalizes the magnitude and bias of the signal
    # average magnitude following a Gaussian
net = AutoModel.from_pretrained("microsoft/wavlm-large").to("cpu")
for p in net.parameters():
    p.requires_grad_(False)

# we're synthesizing a vowel (F0 100 Hz)
gt_signal = get_signal(100)[:1600]
gt_feats = net.feature_extractor(gt_signal[None, :])
gt_feats = gt_feats.transpose(1, 2)
_, gt_feats = net.feature_projection(gt_feats)

signal = torch.nn.Parameter(torch.FloatTensor([0.0] * 1600), requires_grad=True)
optimizer = torch.optim.Adam([signal])

acc_loss = []
loop = tqdm(range(1000))
for it in loop:
    optimizer.zero_grad()

    feats = net.feature_extractor(signal[None, :])
    feats = feats.transpose(1, 2)
    _, feats = net.feature_projection(feats)

    # L2 loss
    loss = torch.square(feats - gt_feats).mean()
    acc_loss.append(loss.item())
    loss.backward()
    optimizer.step()
    loop.set_description(f"loss: {loss.item():.4f}")

# visual verification
plt.plot(signal.detach().numpy())
