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

def reconstruct_signal(net, gt_signal, file_name):
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

    # visual verification - did we reconstruct the signal?
    plt.plot(signal.detach().numpy())
    plt.savefig(f'plots/{file_name}.png')
    plt.clf()


if __name__ == "__main__":
    # WavLM normalizes the magnitude and bias of the signal
    # average magnitude following a Gaussian
    net = AutoModel.from_pretrained("microsoft/wavlm-large").to("cpu")
    for p in net.parameters():
        p.requires_grad_(False)

    # given summation of 3 sine signals, can we reconstruct the sine signal?
    vowels = zip(
        'ieɛaɑʌɤɯyøœɶɒɔou',
        [240, 390, 610, 850, 750, 600, 460, 300, 235, 370, 585, 820, 700, 500, 360, 250],
        [2400, 2300, 1900, 1610, 940, 1170, 1310, 1390, 2100, 1900, 1710, 1530, 750, 700, 640, 595],
    )
    for vowel, f1, f2 in vowels:
        # TODO: magnitude - empirically determine
        # we're synthesizing a vowel (F0 100 Hz)
        vowel_signal = get_signal([f1, f2])[:1600]
        # TODO: F3
        reconstruct_signal(net, vowel_signal, 'single_vowel_' + vowel)

    # TODO: check the intermediate representations
        # linear interpolation of the features of [i] and [u]
