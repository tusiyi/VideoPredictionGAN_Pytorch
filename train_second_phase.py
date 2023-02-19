import torch.optim as optim

from pathlib import Path
from tqdm import tqdm
import wandb
import argparse

from model.GAN import Generator, Discriminator
from utils.dataset import *
from utils.loss import *
from utils.misc import load_ckpt


def train(net_G, net_D, args, dec):
    # data loader
    train_loader, val_loader = get_dataloader('KITTI',
                                              data_dir=args.data_dir,
                                              batch_size=args.batch_size,
                                              train_val_ratio=args.ratio)
    n_train = len(train_loader) * args.batch_size

    # optimizers
    optimizer_D = optim.Adam(net_D.parameters(), lr=args.lr_D, betas=(0.0, 0.9))
    # initialize wandb
    experiment = wandb.init(project='project', resume='allow', anonymous='must')

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        net_G.eval()
        net_D.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch} / {args.epochs}', unit='seq') as pbar:
            for batch in train_loader:
                past_frames, future_frames = batch
                # to device
                past_frames = past_frames.to(device=dec, dtype=torch.float32)
                future_frames = future_frames.to(device=dec, dtype=torch.float32)
                # random noise: d5
                d5 = torch.randn((past_frames.shape[0], args.num_G_channels[-1],
                                  args.img_size[0] // 16, args.img_size[1] // 16))
                d5 = d5.to(device=dec, dtype=torch.float32)
                pred_next_frame = net_G(past_frames, d5)  # output shape: (N, 3, 128, 160)

                pred_images = torch.cat([past_frames, pred_next_frame.unsqueeze(1)], dim=1)
                true_images = torch.cat([past_frames, future_frames], dim=1)

                # GAN Loss for Discriminator
                pred_D = net_D(pred_images)
                true_D = net_D(true_images)
                gan1 = gan_loss(pred_D, False)
                gan2 = gan_loss(true_D, True)
                loss_D = gan1 + gan2
                # back propagate
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                # update pbar and steps
                pbar.update(batch[0].shape[0])
                global_step += 1
                # record metrics
                experiment.log({
                    'epoch': epoch,
                    'step': global_step,
                    'lr': args.lr_D,
                    'GAN Loss(F)': gan1.item(),
                    'GAN Loss(T)': gan2.item(),
                    'total loss D': loss_D.item(),
                })
        # save checkpoints
        if not os.path.exists(args.ckpt_save_dir):
            os.mkdir(args.ckpt_save_dir)
        ckpt_name = os.path.join(args.ckpt_save_dir, args.ckpt_name + '.tar')
        torch.save({
            'D': net_D.state_dict(),
        }, Path(ckpt_name).absolute().as_posix())
        print(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser('Settings')
    parser.add_argument('--phase', type=int, default=2, help='Training phase.')
    parser.add_argument('--dataset', type=str, default='KITTI', help='Dataset name.')
    parser.add_argument('--seq_len', type=int, default=9, help='Sequence length.')
    parser.add_argument('--img_channel', type=int, default=3, help='Image channel.')
    parser.add_argument('--img_size', type=int, default=[128, 160], nargs='+',
                        help='Image size.')
    parser.add_argument('--num_G_channels', type=int, default=[64, 128, 256, 512], nargs='+',
                        help='Number of channels of each module (Generator).')
    parser.add_argument('--num_D_channels', type=int, default=[64, 128, 512, 1024, 2048], nargs='+',
                        help='Number of channels of each module (Discriminator).')
    parser.add_argument('--num_blocks', type=int, default=5, help='Number of blocks in Discriminator.')
    parser.add_argument('--num_rbds', type=int, default=[1, 2, 2, 2, 2], nargs='+',
                        help='Number of RBD layers in each block(Discriminator)')
    parser.add_argument('--avg_size', type=int, default=16, help='Number of spatial average size in Discriminator.')
    parser.add_argument('--data_dir', type=str, default='./data/kitti',
                        help='Data directory.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--ratio', type=float, default=1.0, help='Ratio of train set and val set.')
    parser.add_argument('--lr_D', type=float, default=1e-4, help='Learning rate of discriminator.')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs.')
    parser.add_argument('--ckpt_save_dir', type=str, default='./ckpts/phase2',
                        help='Directory to save checkpoints.')
    parser.add_argument('--ckpt_name', type=str, default='checkpoint_D_only',
                        help='Name for checkpoint to save')
    parser.add_argument('--resume_ckpt', type=str, default='',
                        help='Resume checkpoints(Both G and D).')
    parser.add_argument('--resume_ckpt_G', type=str, default='./ckpts/phase1/checkpoint_G_only.tar',
                        help='Resume checkpoints(required, Generator only).')

    return parser.parse_args()


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # hyper-parameters
    args = get_args()
    # networks
    generator = Generator(seq_len=args.seq_len,
                          img_channel=args.img_channel,
                          img_size=args.img_size,
                          num_channels=args.num_G_channels)
    discriminator = Discriminator(num_blocks=args.num_blocks,
                                  num_rbds=args.num_rbds,
                                  num_channels=args.num_D_channels,
                                  img_channel=args.img_channel,
                                  seq_len=args.seq_len,
                                  img_size=args.img_size,
                                  avg_size=args.avg_size)
    # load trained generator
    ckpt = load_ckpt(args.resume_ckpt_G)
    generator.load_state_dict(ckpt['G'])
    for _, param in generator.named_parameters():
        param.requires_grad = False

    # if pretrained discriminator, load it
    if args.resume_ckpt:
        ckpt = load_ckpt(args.resume_ckpt)
        discriminator.load_state_dict(ckpt['D'])
    # to device
    generator.to(device=device)
    discriminator.to(device=device)
    # loss definition
    gan_loss = GANLoss(gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0)
    mae_loss = nn.L1Loss()
    vgg_loss = PerceptualLoss()

    train(net_G=generator, net_D=discriminator, args=args, dec=device)
