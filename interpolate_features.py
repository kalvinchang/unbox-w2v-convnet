# utils.py is from https://github.com/juice500ml/unbox-w2v-convnet/blob/main/utils.py

from transformers import AutoModelForPreTraining
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_signal, get_feature
import torch
import numpy as np
from tqdm import tqdm
import librosa
import librosa.display
import scipy


# Goal: Identify the boundary in SSL feature space for [a] and [i]
# Method: Optimize input signal to reproduce SSL features
# Hypothesis: The proximity of the cluster will reproduce [a] and [i].

net = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large").to("cpu")
signal = torch.nn.Parameter(torch.FloatTensor([0.0] * 1600), requires_grad=True)
# signal = get_signal(100)[:1600]
# signal = torch.nn.Parameter(torch.FloatTensor(get_signal(100)[:1600]), requires_grad=True)

for p in net.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.Adam([signal])

acc_loss = []
for it in tqdm(range(10000)):
    optimizer.zero_grad()

    sig = signal
    feats = net.wav2vec2.feature_extractor(sig[None, :])
    feats = feats.transpose(1, 2)
    _, feats = net.wav2vec2.feature_projection(feats)

    batch_size, sequence_length, hidden_size = feats.shape
    feats = net.quantizer.weight_proj(feats)
    feats = feats.view(net.quantizer.num_groups * sequence_length, -1)
    probs = feats.softmax(-1).reshape(sequence_length, net.quantizer.num_groups, -1)

    freq_100 = (292, 290)
    loss = -torch.log(probs[:, 0, freq_100[0]]).mean() - torch.log(probs[:, 1, freq_100[1]]).mean()
    acc_loss.append(loss.item())
    loss.backward()
    optimizer.step()
