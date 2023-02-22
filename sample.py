import os
import argparse
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from med_seg_diff_pytorch import Unet, MedSegDiff
from med_seg_diff_pytorch.dataset import ISICDataset, GenericNpyDataset
from accelerate import Accelerator
import skimage.io as io



## Parse CLI arguments ##
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-od', '--output_dir', type=str, default="output", help="Output dir.")
    parser.add_argument('-ld', '--logging_dir', type=str, default="logs", help="Logging dir.")
    parser.add_argument('-mp', '--mixed_precision', type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Whether to do mixed precision")
    parser.add_argument('-img', '--img_folder', type=str, default='ISBI2016_ISIC_Part3B_Training_Data',
                        help='The image file path from data_path')
    parser.add_argument('-csv', '--csv_file', type=str, default='ISBI2016_ISIC_Part3B_Training_GroundTruth.csv',
                        help='The csv file to load in from data_path')
    parser.add_argument('-sc', '--self_condition', action='store_true', help='Whether to do self condition')
    parser.add_argument('-ic', '--mask_channels', type=int, default=1, help='input channels for training (default: 3)')
    parser.add_argument('-c', '--input_img_channels', type=int, default=3,
                        help='output channels for training (default: 3)')
    parser.add_argument('-is', '--image_size', type=int, default=128, help='input image size (default: 128)')
    parser.add_argument('-dd', '--data_path', default='./data', help='directory of input image')
    parser.add_argument('-d', '--dim', type=int, default=64, help='dim (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs (default: 10000)')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='batch size to train on (default: 8)')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of timesteps (default: 1000)')
    parser.add_argument('-ds', '--dataset', default='generic', help='Dataset to use')
    parser.add_argument('--save_every', type=int, default=100, help='save_every n epochs (default: 100)')
    parser.add_argument('--num_ens', type=int, default=5,
                        help='number of times to sample to make an ensable of predictions like in the paper (default: 5)')
    parser.add_argument('--load_model_from', default=None, help='path to pt file to load from')
    parser.add_argument('--save_uncertainty', action='store_true',
                        help='Whether to store the uncertainty in predictions (only works for ensablmes)')

    return parser.parse_args()


def load_data(args):
    # Load dataset
    if args.dataset == 'ISIC':
        transform_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), ]
        transform_train = transforms.Compose(transform_list)
        dataset = ISICDataset(args.data_path, args.csv_file, args.img_folder, transform=transform_train, training=False,
                              flip_p=0.5)
    elif args.dataset == 'generic':
        transform_list = [transforms.ToPILImage(), transforms.Resize(args.image_size), transforms.ToTensor()]
        transform_train = transforms.Compose(transform_list)
        dataset = GenericNpyDataset(args.data_path, transform=transform_train, test_flag=True)
    else:
        raise NotImplementedError(f"Your dataset {args.dataset} hasn't been implemented yet.")

    ## Define PyTorch data generator
    training_generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)

    return training_generator


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    inference_dir = os.path.join(args.output_dir, 'inference')
    os.makedirs(inference_dir, exist_ok=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("med-seg-diff", config=vars(args))

    ## DEFINE MODEL ##
    model = Unet(
        dim=args.dim,
        image_size=args.image_size,
        dim_mults=(1, 2, 4, 8),
        mask_channels=args.mask_channels,
        input_img_channels=args.input_img_channels,
        self_condition=args.self_condition
    )

    ## LOAD DATA ##
    data_loader = load_data(args)

    diffusion = MedSegDiff(
        model,
        timesteps=args.timesteps
    ).to(accelerator.device)

    if args.load_model_from is not None:
        save_dict = torch.load(args.load_model_from)
        diffusion.model.load_state_dict(save_dict['model_state_dict'])

    for (imgs, masks, fnames) in tqdm(data_loader):
        # pre allocate preds
        preds = torch.zeros((imgs.shape[0], args.num_ens, imgs.shape[2], imgs.shape[3]))
        for i in range(args.num_ens):
            preds[:, i:i+1, :, :] = diffusion.sample(imgs).cpu().detach()
        preds_mean = preds.mean(dim=1)
        preds_std = preds.std(dim=1)

        for idx in range(preds.shape[0]):
            io.imsave(os.path.join(inference_dir, fnames[idx].replace('.npy', '.png')), preds_mean[idx, :, :])
            if args.save_uncertainty:
                io.imsave(os.path.join(inference_dir, fnames[idx].replace('.npy', '_std.png')), preds_std[idx, :, :])



if __name__ == '__main__':
    main()
