import argparse
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import MBM
from model import ModelCountception
from utils.save_utils import save_samples
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Sealion count training')
parser.add_argument('--pkl-file', default="utils/MBM-dataset.pkl", type=str, help='path to pickle file.')
parser.add_argument('--batch-size', default=1, type=int, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--ckpt', default='checkpoints/after_600_epochs.model', type=str, help='Path to checkpoint file.')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    test_dataset = MBM(pkl_file=args.pkl_file, transform=transforms.Compose([transforms.ToTensor()]), mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    criterion = nn.L1Loss()
    model = ModelCountception().to(device)
    model.eval()

    print("Loading weights...")
    from_before = torch.load(args.ckpt)
    model_weights = from_before['model_weights']
    model.load_state_dict(model_weights)

    test_loss = []
    count_loss = []
    with torch.no_grad():
        for idx, (input, target, target_count) in enumerate(test_dataloader):
            input = input.to(device)
            target = target.to(device)
            output = model.forward(input)
            test_loss.append(criterion(output, target).data.cpu().numpy())

            patch_size = 32
            ef = ((patch_size / 1) ** 2.0)
            output_count = (output.cpu().numpy() / ef).sum(axis=(2, 3))
            target_count = target_count.data.cpu().numpy()
            count_loss.append(abs(output_count - target_count))

            save_samples(output, target, idx)
        print('MAE of Test Set: ', np.mean(test_loss))
        print('Mean Difference in Counts', np.mean(count_loss))


if __name__ == '__main__':
    main()
