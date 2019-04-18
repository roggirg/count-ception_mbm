import matplotlib.pyplot as plt
import numpy as np


def save_samples(output, target, idx):
    output_arr = output[0].cpu().numpy()
    target_arr = target[0].cpu().numpy()

    fig = plt.figure(figsize=(16, 6))
    ax0 = fig.add_subplot(121)
    ax0.set_title("Target Count Map")
    ax0.imshow(np.concatenate(target_arr, axis=1))
    ax1 = fig.add_subplot(122)
    ax1.set_title("Output Count Map")
    ax1.imshow(np.concatenate(output_arr, axis=1))
    plt.tight_layout()
    plt.savefig('test_outputs/samples_{0}'.format(idx))


