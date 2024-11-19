# utils.py is from https://github.com/juice500ml/unbox-w2v-convnet/blob/main/utils.py

from transformers import AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_signal, get_feature
from configs import exp_configs
import torch
import numpy as np
from tqdm import tqdm
import librosa
import librosa.display
import scipy

# Goal: Identify the "perceptual" boundary in SSL feature space for 2 vowels
# Method: Optimize input signal to reproduce SSL features for the signals for 2 vowels
# Hypothesis: The proximity of the cluster will reproduce [a] and [i].

def extract_s3m_features(net, signal):
    feats = net.feature_extractor(signal[None, :])
    feats = feats.transpose(1, 2)
    _, feats = net.feature_projection(feats)
    return feats


# reconstruct signal given features using optimization
def reconstruct_signal_from_feats(net, gold_feats, file_name):
    signal = torch.nn.Parameter(torch.FloatTensor([0.0] * 1600), requires_grad=True)
    optimizer = torch.optim.Adam([signal])

    acc_loss = []
    loop = tqdm(range(1000))
    feats = None
    for it in loop:
        optimizer.zero_grad()

        feats = extract_s3m_features(net, signal)

        # L2 loss
        loss = torch.square(feats - gold_feats).mean()
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
    # source: JC Catford et al., A practical introduction to phonetics
    vowels = zip(
        'ieɛaɑʌɤɯyøœɶɒɔou',
        [240, 390, 610, 850, 750, 600, 460, 300, 235, 370, 585, 820, 700, 500, 360, 250],
        [2400, 2300, 1900, 1610, 940, 1170, 1310, 1390, 2100, 1900, 1710, 1530, 750, 700, 640, 595],
    )
    vowel_feats = {}
    for vowel, f1, f2 in vowels:
        # TODO: add F0

        # TODO: magnitude - empirically determine
        # we're synthesizing a vowel from the sum of 3 sine signals
        vowel_signal = get_signal(freq=[f1, f2], mag=exp_configs["f0f1f2"]["mag"][:2])[:1600]
        # TODO: F3
        vowel_feat = extract_s3m_features(net, vowel_signal)
        vowel_feats[vowel] = vowel_feat
        feats = reconstruct_signal_from_feats(net, vowel_feat, 'single_vowel_' + vowel)
        # TODO: plot the spectrogram
        # TODO: listen

    # linear interpolation of the S3M features of [i] and [u]
    # can we reconstruct the signal?
    for vowel1, vowel1_feats in vowel_feats.items():
        for vowel2, vowel2_feats in vowel_feats.items():
            if vowel1 != vowel2:
                for coeff in np.linspace(0, 1, 11):
                    # interpolate the S3M features, not the signal
                    interpolated_features = coeff * vowel1_feats + (1 - coeff) * vowel2_feats
                    reconstruct_signal(net, interpolated_features, f'interpolated_{vowel1}_{vowel2}_{coeff}')

    # TODO: check the intermediate representations
    # TODO: listen
