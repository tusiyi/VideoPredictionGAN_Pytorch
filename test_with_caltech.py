from tqdm import tqdm
import wandb
import argparse

from model.GAN import Generator
from utils.dataset import *
from utils.loss import *
from utils.misc import load_ckpt


def test(net_G, args, dec):
    # data loader
    test_loader = get_dataloader(dataset=args.dataset,
                                 seq_len=args.seq_len,
                                 data_dir=args.data_dir,
                                 inter=args.inter,
                                 image_size=args.img_size,
                                 )
    n_test = len(test_loader)

    experiment = wandb.init(project='project', resume='allow', anonymous='must')

    global_step = 0
    net_G.eval()
    with tqdm(total=n_test, unit='seq') as pbar:
        for batch in test_loader:
            past_frames, future_frames = batch
            past_frames = past_frames.to(device=dec, dtype=torch.float32)
            future_frames = future_frames.to(device=dec, dtype=torch.float32)

            orginal_past = past_frames.clone()  # 备份past frames
            num_future = future_frames.shape[1]
            pred_frames = []
            # predict multiple future frames
            for i in range(num_future):
                # random noise: d5
                d5 = torch.randn((past_frames.shape[0], args.num_G_channels[-1],
                                  args.img_size[0] // 16, args.img_size[1] // 16))
                d5 = d5.to(device=dec, dtype=torch.float32)
                pred_next_frame = net_G(past_frames, d5)  # output shape: (N, 3, 128, 160)
                pred_frames.append(pred_next_frame)
                # # update generator inputs
                past_frames = torch.cat([past_frames[:, 1:, ...], pred_next_frame.unsqueeze(1)], dim=1)

            # concate predictions
            if num_future > 1:
                pred_images = torch.cat([orginal_past, torch.stack(pred_frames).permute(1, 0, 2, 3, 4)], dim=1)
            else:
                pred_images = torch.cat([orginal_past, pred_next_frame.unsqueeze(1)], dim=1)
            true_images = torch.cat([orginal_past, future_frames], dim=1)
            # use wandb to record predictions
            experiment.log({
                'step': global_step,
                'true images': wandb.Image(true_images[0].cpu()),
                'pred images': wandb.Image(pred_images[0].cpu()),
            })

            pbar.update(batch[0].shape[0])
            global_step += 1


def get_args():
    parser = argparse.ArgumentParser('Settings')
    # parser.add_argument('--phase', type=int, default=1, help='Training phase.')
    parser.add_argument('--dataset', type=str, default='Caltech', help='Dataset name.')
    parser.add_argument('--seq_len', type=int, default=9, help='Sequence length.')
    parser.add_argument('--img_channel', type=int, default=3, help='Image channel.')
    parser.add_argument('--img_size', type=int, default=[128, 160], nargs='+',
                        help='Image size.')
    parser.add_argument('--num_G_channels', type=int, default=[64, 128, 256, 512], nargs='+',
                        help='Number of channels of each module (Generator).')
    parser.add_argument('--data_dir', type=str, default='./test_data',
                        help='Data directory.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--resume_ckpt', type=str, default='./ckpts/phase3/checkpoint.tar',
                        help='Resume checkpoints.')
    parser.add_argument('--inter', type=int, default=1, help='Interval of video frames.')

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
    # load checkpoints
    ckpt = load_ckpt(args.resume_ckpt)
    generator.load_state_dict(ckpt['G'])
    # to device
    generator.to(device=device)

    test(net_G=generator, args=args, dec=device)
