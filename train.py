import os
import math
import argparse
import warnings

import torch
import wandb

from torch.optim import Adam

from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel as DP

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as vutils

from datetime import datetime

from sysbinder import SysBinderImageAutoEncoder
from data import GlobDataset, DummyDataset
from utils import linear_warmup, cosine_anneal

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--image_channels', type=int, default=3)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--data_path', required=False)
parser.add_argument('--log_path', default='logs/')

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_enc', type=float, default=1e-4)
parser.add_argument('--lr_dec', type=float, default=3e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_half_life', type=int, default=250000)
parser.add_argument('--clip', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=4)
parser.add_argument('--num_blocks', type=int, default=8)
parser.add_argument('--cnn_hidden_size', type=int, default=512)
parser.add_argument('--slot_size', type=int, default=2048)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--num_prototypes', type=int, default=64)

parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_decoder_layers', type=int, default=8)
parser.add_argument('--num_decoder_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--dropout', type=int, default=0.1)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000)

parser.add_argument('--use_dp', default=True, action='store_true')
parser.add_argument('--use_broadcast_decoder', action=argparse.BooleanOptionalAction)

parser.add_argument('--wandb_project', type=str, default='Test project')
parser.add_argument('--wandb_group', type=str, default='Test group')
parser.add_argument('--wandb_run_name', type=str, default='Test run')
parser.add_argument('--wandb_resume_run_id', type=str, required=False)

parser.add_argument('--topdis_rtd_loss_coef', type=float, default=0.1)
parser.add_argument('--topdis_lp', type=int, default=2)
parser.add_argument('--topdis_q_normalize', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.set_float32_matmul_precision('medium')

os.makedirs(args.log_path, exist_ok=False)
run_id = args.wandb_resume_run_id
resume = None if run_id is None else 'must'
wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.wandb_run_name, sync_tensorboard=True,
           dir=args.log_path, resume=resume, id=run_id)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)


def visualize(image, recon_dvae, recon_tf, attns, N=8):

    # tile
    tiles = torch.cat((
        image[:N, None, :, :, :],
        recon_dvae[:N, None, :, :, :],
        recon_tf[:N, None, :, :, :],
        attns[:N, :, :, :, :]
    ), dim=1).flatten(end_dim=1)

    # grid
    grid = vutils.make_grid(tiles, nrow=(1 + 1 + 1 + args.num_slots), pad_value=0.8)

    return grid

if args.data_path is None:
    warnings.warn('DummyDataset is used!')
    train_dataset = DummyDataset(img_size=args.image_size, img_channels=args.image_channels)
    val_dataset = DummyDataset(img_size=args.image_size, img_channels=args.image_channels)
else:
    train_dataset = GlobDataset(root=args.data_path, phase='train', img_size=args.image_size)
    val_dataset = GlobDataset(root=args.data_path, phase='val', img_size=args.image_size)

train_sampler = None
val_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size = len(val_loader)

log_interval = train_epoch_size // 5

model = SysBinderImageAutoEncoder(args)

if os.path.isfile(args.checkpoint_path):
    print(f'From checkpoint: {args.checkpoint_path}')
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0

model = model.cuda()
if args.use_dp:
    model = DP(model)

optimizer = Adam([
    {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
    {'params': (x[1] for x in model.named_parameters() if 'image_encoder' in x[0]), 'lr': 0.0},
    {'params': (x[1] for x in model.named_parameters() if 'image_decoder' in x[0]), 'lr': 0.0},
])
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

for epoch in range(start_epoch, args.epochs):
    model.train()
    
    for batch, image in enumerate(train_loader):
        global_step = epoch * train_epoch_size + batch

        tau = cosine_anneal(
            global_step,
            args.tau_start,
            args.tau_final,
            0,
            args.tau_steps)

        lr_warmup_factor_enc = linear_warmup(
            global_step,
            0.,
            1.0,
            0.,
            args.lr_warmup_steps)

        lr_warmup_factor_dec = linear_warmup(
            global_step,
            0.,
            1.0,
            0,
            args.lr_warmup_steps)

        lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))

        optimizer.param_groups[0]['lr'] = args.lr_dvae
        optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor_enc * args.lr_enc
        optimizer.param_groups[2]['lr'] = lr_decay_factor * lr_warmup_factor_dec * args.lr_dec

        image = image.cuda()

        optimizer.zero_grad()
        
        (recon_dvae, cross_entropy, mse, rtd_loss, attns) = model(image, tau)

        if args.use_dp:
            mse = mse.mean()
            cross_entropy = cross_entropy.mean()
            rtd_loss = rtd_loss.mean()

        loss = mse + cross_entropy + args.topdis_rtd_loss_coef * rtd_loss
        
        loss.backward()

        clip_grad_norm_(model.parameters(), args.clip, 'inf')

        optimizer.step()
        
        with torch.no_grad():
            if batch % log_interval == 0:
                print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                      epoch+1, batch, train_epoch_size, loss.item(), mse.item()))
                
                writer.add_scalar('TRAIN/loss', loss.item(), global_step)
                writer.add_scalar('TRAIN/cross_entropy', cross_entropy.item(), global_step)
                writer.add_scalar('TRAIN/mse', mse.item(), global_step)
                writer.add_scalar('TRAIN/rtd_loss', rtd_loss.item(), global_step)

                writer.add_scalar('TRAIN/tau', tau, global_step)
                writer.add_scalar('TRAIN/lr_dvae', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_enc', optimizer.param_groups[1]['lr'], global_step)
                writer.add_scalar('TRAIN/lr_dec', optimizer.param_groups[2]['lr'], global_step)

    with torch.no_grad():
        recon_tf = (model.module if args.use_dp else model).reconstruct_autoregressive(image[:8])
        grid = visualize(image, recon_dvae, recon_tf, attns, N=8)
        writer.add_image('TRAIN_recons/epoch={:03}'.format(epoch+1), grid)
    
    with torch.no_grad():
        model.eval()

        val_cross_entropy = 0.
        val_mse = 0.
        val_rtd_loss = 0.

        for batch, image in enumerate(val_loader):
            image = image.cuda()

            (recon_dvae, cross_entropy, mse, rtd_loss, attns) = model(image, tau)

            if args.use_dp:
                mse = mse.mean()
                cross_entropy = cross_entropy.mean()
                rtd_loss = rtd_loss.mean()

            val_cross_entropy += cross_entropy.item()
            val_mse += mse.item()
            val_rtd_loss += rtd_loss.item()

        val_cross_entropy /= (val_epoch_size)
        val_mse /= (val_epoch_size)
        val_rtd_loss /= (val_epoch_size)

        val_loss = val_mse + val_cross_entropy + args.topdis_rtd_loss_coef * val_rtd_loss

        writer.add_scalar('VAL/loss', val_loss, epoch+1)
        writer.add_scalar('VAL/cross_entropy', val_cross_entropy, epoch + 1)
        writer.add_scalar('VAL/mse', val_mse, epoch+1)
        writer.add_scalar('VAL/rtd_loss', val_rtd_loss, epoch + 1)

        print('====> Epoch: {:3} \t Loss = {:F}'.format(epoch+1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(model.module.state_dict() if args.use_dp else model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

            if 50 <= epoch:
                recon_tf = (model.module if args.use_dp else model).reconstruct_autoregressive(image[:8])
                grid = visualize(image, recon_dvae, recon_tf, attns, N=8)
                writer.add_image('VAL_recons/epoch={:03}'.format(epoch + 1), grid)

        writer.add_scalar('VAL/best_loss', best_val_loss, epoch+1)

        checkpoint = {
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'model': model.module.state_dict() if args.use_dp else model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

        print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

writer.close()
wandb.finish()
