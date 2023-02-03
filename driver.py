import argparse
import wandb
import torch
from train import Unet, MedSegDiff
from loader_isic import ISICDataset
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import PIL



## Initialize weights and biases project ##
wandb.init(project="med-seg-diff")

## Parse CLI arguments ##
parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.95, help='The beta1 parameter for the Adam optimizer.')
parser.add_argument('-ab2', '--adam_beta2', type=float, default=0.999, help='The beta2 parameter for the Adam optimizer.')
parser.add_argument('-aw', '--adam_weight_decay', type=float, default=1e-6, help='Weight decay magnitude for the Adam optimizer.')
parser.add_argument('-ae', '--adam_epsilon', type=float, default=1e-08, help='Epsilon value for the Adam optimizer.')
parser.add_argument('-ic', '--input-channels', type=int, default=1, help='input channels for training (default: 3)')
parser.add_argument('-c', '--channels', type=int, default=3, help='output channels for training (default: 3)')
parser.add_argument('-is', '--image-size', type=int, default=128, help='input image size (default: 128)')
parser.add_argument('-dd', '--data-dir', default='./data', help='directory of input image')
parser.add_argument('-d', '--dim', type=int, default=64, help='dim (deaault: 64)')
parser.add_argument('-e', '--epochs', type=int, default=1, help='number of epochs (default: 128)')
parser.add_argument('-bs', '--batch-size', type=int, default=8, help='batch size to train on (default: 8)')
parser.add_argument('-ds', '--dataset', default='ISIC', help='Dataset to use')
args = parser.parse_args()
wandb.config.update(args) # adds all of the arguments as config variables






def load_data(args):
    # Create transforms for data
    transform_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
    transform_train = transforms.Compose(transform_list)

    # Load dataset
    if args.dataset == 'ISIC':
        dataset = ISICDataset(args, args.data_dir, transform_train)

    ## Define PyTorch data generator
    training_generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True)

    return training_generator



def main():
    ## DEFINE MODEL ##
    model = Unet(
        dim = args.dim, #64,
        image_size = args.image_size,
        dim_mults = (1, 2, 4, 8),
        input_channels = args.input_channels,
        channels = args.channels
    )

    diffusion = MedSegDiff(
        model,
        timesteps = args.epochs
    ).cuda()


    ## LOAD DATA ##
    data_loader = load_data(args)
    #training_generator = tqdm(data_loader, total=int(len(data_loader)))

    ## Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    ## TRAIN MODEL ##

    running_loss = 0.0
    counter = 0

    ## Iterate across training loop
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        for ii, (img, mask) in tqdm(enumerate(data_loader), total=int(len(data_loader))):
        #for bi, sample in enumerate(training_generator):
            #img, mask = sample
            loss = diffusion(mask, img)
            wandb.log({'loss': loss}) # Log loss to wandb
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item() * img.size(0)
        counter += 1
        #training_generator.set_postfix(loss=(running_loss / (counter * data_loader.batch_size)))
        epoch_loss = running_loss / len(data_loader)
        print('Training Loss : {:.4f}'.format(epoch_loss))


    ## INFERENCE ##
    pred = diffusion.sample(img).cpu()#.detach().numpy()
    wandb.log({'pred': wandb.Image(pred)})
    wandb.log({'img': wandb.Image(img)})
    wandb.log({'mask': wandb.Image(mask)})
'''
    for bi_eval, sample_eval in enumerate(training_generator):
        img_sample, mask_sample = sample_eval
        pred_eval = diffusion.sample(img_sample).cpu().detach().numpy()
        wandb.log({'pred_eval': wandb.Image(numpy_to_pil(pred_eval[0]))})
        wandb.log({'img_eval': wandb.Image(img_sample)})
        wandb.log({'mask_eval': wandb.Image(mask_sample)})
'''



if __name__ == '__main__':
    main()
