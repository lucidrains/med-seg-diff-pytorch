import argparse
import wandb
import torch
from med_seg_diff_pytorch import Unet, MedSegDiff
from loader_isic import ISICDataset
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import PIL
def numpy_to_pil(img):
    img = img*255
    print(img.shape)
    img = np.transpose(img, (1, 2, 0))[:, :, 0]
    print(img.shape)
    img = img.astype(np.uint8)
    img = PIL.Image.fromarray(img)
    return img

def main():
    wandb.init(project="med-seg-diff")
    parser = argparse.ArgumentParser()
    parser.add_argument('-ic', '--input-channels', type=int, default=1, help='input channels for training (default: 3)')
    parser.add_argument('-c', '--channels', type=int, default=3, help='output channels for training (default: 3)')
    parser.add_argument('-is', '--image-size', type=int, default=128, help='input image size (default: 128)')
    parser.add_argument('-dd', '--data-dir', default='./data', help='directory of input image')
    parser.add_argument('-d', '--dim', type=int, default=64, help='dim (deaault: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs (default: 128)')
    parser.add_argument('-bs', '--batch-size', type=int, default=8, help='batch size to train on (default: 8)')

    args = parser.parse_args()
    wandb.config.update(args) # adds all of the arguments as config variables


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
        timesteps = args.epochs # 1000
    ).cuda()


    ## LOAD DATA ##
    tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
    transform_train = transforms.Compose(tran_list)
    ds = ISICDataset(args, args.data_dir, transform_train)
    #args.in_ch = 4


    datal = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)

    for i in tqdm(range(len(data))):
        img, mask = next(data)


        ### SETUP DATA ##
        #segmented_imgs = torch.rand(8, 3, 128, 128)  # inputs are normalized from 0 to 1
        #input_imgs = torch.rand(8, 3, 128, 128)

        #print("Seg: {}".format(segmented_imgs.shape))
        #print("Inp: {}".format(input_imgs.shape))

        #print("Img: {}".format(img.shape))
        #print("Mask: {}".format(mask.shape))



        ## TRAIN MODEL ##
        #loss = diffusion(segmented_imgs, input_imgs)
        loss = diffusion(mask, img)
        wandb.log({'loss': loss}) # Log loss to wanbd
        loss.backward()


    # after a lot of training


    ## INFERENCE ##
    pred = diffusion.sample(img).cpu().detach().numpy()
    pil_img = numpy_to_pil(pred[0])
    pil_img.save("output.png")
    #pred = diffusion.sample(input_imgs)     # pass in your unsegmented images
    print("Pred: {}".format(pred.shape))



if __name__ == '__main__':
    main()
