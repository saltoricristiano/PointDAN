import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from PointNetv2 import PointNet_AutoEncoder
from trial_pyg import Shapenet_data
from torch.autograd import Variable
import time
import numpy as np
import os
import argparse

import warnings

import wandb
from pytorch3d.loss import chamfer_distance
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

warnings.filterwarnings("ignore")

# Command setting
parser = argparse.ArgumentParser(description='Main')
parser.add_argument('--source', '-s', type=str, help='source dataset', default='shapenet')
parser.add_argument('--batchsize', '-b', type=int, help='batch size', default=8)
parser.add_argument('--gpu', '-g', type=str, help='cuda id', default='0')
parser.add_argument('--epochs', '-e', type=int, help='training epoch', default=2000)
parser.add_argument('--models', '-m', type=str, help='alignment model', default='MDA')
parser.add_argument('--lr',type=float, help='learning rate', default=0.0001)
parser.add_argument('--datadir',type=str, help='directory of data', default='./dataset/')
# parser.add_argument('-tb_log_dir', type=str, help='directory of tb', default='./logs')
parser.add_argument('--wandb_name', type=str, help='name of the wandb experiment', default='Pointnet_auto-graph')
parser.add_argument('--num_points', '-n', type=int, help='training epoch', default=2048)
args = parser.parse_args()

output_dir = os.path.join("output", args.wandb_name)

# if not os.path.exists(os.path.join(os.getcwd(), args.tb_log_dir)):
#     os.makedirs(os.path.join(os.getcwd(), args.tb_log_dir))
# writer = SummaryWriter(log_dir=args.tb_log_dir)

device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

BATCH_SIZE = args.batchsize * len(args.gpu.split(','))
LR = args.lr
weight_decay = 5e-4
momentum = 0.9
max_epoch = args.epochs
dir_root = os.path.join(args.datadir, 'PointDA_data',args.source)


def show_pcl(pc_velo):

    
    d = {'x':pc_velo[:,0], 'y':pc_velo[:, 1], 'z':pc_velo[:, 2]}
    data = pd.DataFrame(data=d)

    fig1 = go.Scatter3d(x=data['x'],
                    y=data['y'],
                    z=data['z'],
                    marker=dict(color='black',
                                opacity=1.,
                                reversescale=True,
                                colorscale='rainbow',
                                size=1),
                    line=dict (width=0.3),
                    mode='markers',
                    name='LiDAR-data')

    
    plot_list = [fig1]
    return plot_list


# print(dir_root)
def main():
    print ('Start Training\nInitiliazing\n')
    print('src:', args.source)

    # Data loading

    source_train_dataset = Shapenet_data(pc_input_num=args.num_points, status='train', aug=False, data_root=dir_root)
    source_test_dataset = Shapenet_data(pc_input_num=args.num_points, status='test', aug=False, data_root=dir_root)

    num_source_train = len(source_train_dataset)
    num_source_test = len(source_test_dataset)

    source_train_dataloader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    source_test_dataloader = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)

    print('num_source_train: {:d}, num_source_test: {:d} '.format(num_source_train, num_source_test))
    print('batch_size:', BATCH_SIZE)

    # Model

    model = PointNet_AutoEncoder(num_points=args.num_points, bs_size=args.batchsize)
    model = model.to(device=device)

    criterion = chamfer_distance

    remain_epoch=50

    # Optimizer

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler= optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs+remain_epoch)


    wandb.init(project="pcl2pcl", entity="unitn-mhug-csalto", name=args.wandb_name)
    wandb.config.update({"source":args.source,
                         "epochs":max_epoch,
                         "batch_size":BATCH_SIZE, 
                         "lr":LR,
                         "decay":weight_decay,
                         "momentum":momentum,
                         "dir_root":dir_root})

    wandb.watch(model)


    title_frame="%s"%(args.source)

    layout = go.Layout(scene=dict(
                             bgcolor='rgba(1,1,1,1)',
                             xaxis = dict(title="X",
                                         backgroundcolor="rgb(0, 0, 0)",
                                         gridcolor="black",
                                         showbackground=True,
                                         zerolinecolor="black"),
                            yaxis = dict(title="Y",
                                        backgroundcolor="rgb(0, 0,0)",
                                        gridcolor="black",
                                        showbackground=True,
                                        zerolinecolor="black"),
                            zaxis = dict(title="Z",
                                        backgroundcolor="rgb(0, 0,0)",
                                        gridcolor="black",
                                        showbackground=True,
                                        zerolinecolor="black")),
                        scene_aspectmode='data',
                        title=title_frame)

    best_target_test_acc = 0

    for epoch in range(max_epoch):
        since_e = time.time()
            
        scheduler.step(epoch=epoch)

        wandb.log({"lr":scheduler.get_lr()[0]})

        model.train()

        loss_total = 0
        data_total = 0
        data_t_total = 0

        # Training

        for batch_idx, data in enumerate(source_train_dataloader):

            # data, label = batch_s

            # fig_dict = {"layout": layout}
            # lidar_plot = show_pcl(data)
            # fig_dict['data'] = lidar_plot
            # fig = go.Figure(lidar_plot)
            # os.makedirs('lidar', exist_ok=True)

            # fig.write_html(file=os.path.join('lidar', str(epoch)+'_'+str(batch_idx)+'.html'))

            

            in_pos = data.pos.view([args.batchsize, args.num_points, 3])[:, :1024]
            in_pos = in_pos.contiguous().view([-1, 3])
            in_batch = data.batch.view([args.batchsize, args.num_points])[:, :1024]
            in_batch = in_batch.contiguous().view(-1)

            data = data.to(device=device)
            pred = model(pos=in_pos.type(torch.cuda.FloatTensor), batch=in_batch.to(device=device))
            print(pred.shape)
            loss_chamfer, _ = criterion(pred, data.pos.type(torch.cuda.FloatTensor).view(args.batchsize, -1, 3))

            loss_chamfer.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_total += loss_chamfer.item() * args.batchsize
            data_total += args.batchsize

            if batch_idx % 100 == 0:
                print('[It {}: chamfer loss {:.4f}]'.format(batch_idx, loss_total/data_total))


        pred_log = pred.cpu().view(args.batchsize, -1, 3)
        pred_log = pred_log[-1]
        pred_log = pred_log.detach().numpy()

        data_log = data.pos.cpu().view(args.batchsize, -1, 3)
        data_log = data_log[-1]
        data_log = data_log.numpy()

        wandb.log({"train_original_%d"%batch_idx: wandb.Object3D(data_log)})
        wandb.log({"train_recon_%d"%batch_idx: wandb.Object3D(pred_log)})

        # fig_dict = {"layout": layout}
        # lidar_plot = show_pcl(data_log)
        # fig_dict['data'] = lidar_plot
        # fig = go.Figure(lidar_plot)
        # os.makedirs('lidar', exist_ok=True)

        # fig.write_html(file=os.path.join('lidar', str(epoch)+'_'+str(batch_idx)+'.html'))

        # fig_dict = {"layout": layout}
        # lidar_plot = show_pcl(pred_log)
        # fig_dict['data'] = lidar_plot
        # fig = go.Figure(lidar_plot)
        # os.makedirs('lidar', exist_ok=True)

        # fig.write_html(file=os.path.join('lidar', str(epoch)+'_'+str(batch_idx)+'_recon.html'))

        print('[Epoch {}: Avg chamfer loss {:.4f}]'.format(epoch, loss_total/data_total))
        wandb.log({"train_chamfer":loss_total/data_total})

        # Testing

        with torch.no_grad():
            model.eval()
            loss_total = 0
            data_total = 0

            for batch_idx, data in enumerate(source_test_dataloader):
                
                in_pos = data.pos.view([args.batchsize, args.num_points, 3])[:, :1024]
                in_pos = in_pos.view([-1, 3])
                in_batch = data.batch.view([args.batchsize, args.num_points])[:, :1024]
                in_batch = in_batch.view(-1)

                data = data.to(device=device)
                pred = model(pos=in_pos, batch=in_batch)
                loss, _ = criterion(pred, data.pos.type(torch.cuda.FloatTensor).view(args.batchsize, -1, 3))

                loss_total += loss.item() * args.batchsize
                data_total += args.batchsize

                # to_log = np.random.choice([x for x in range(BATCH_SIZE)], 10)
                pred_log = pred.view(args.batchsize, -1, 3).cpu()
                pred_log = pred_log[0]
                pred_log = pred_log.numpy()
                data_log = data.pos.view(args.batchsize, -1, 3).cpu()
                data_log = data_log[0]
                data_log = data_log.numpy()


            wandb.log({"test_original_%d"%batch_idx: wandb.Object3D(data_log)})
            wandb.log({"test_recon_%d"%batch_idx: wandb.Object3D(pred_log)})

            # fig_dict = {"layout": layout}
            # lidar_plot = show_pcl(data_log)
            # fig_dict['data'] = lidar_plot
            # fig = go.Figure(lidar_plot)
            # os.makedirs('lidar', exist_ok=True)
            # fig.write_html(file=os.path.join('lidar', str(epoch)+'_'+str(batch_idx)+'.html'))

            pred_loss = loss_total/data_total


            print ('TEST - [Epoch: {} \t loss: {:.4f}]'.format(
                   epoch, pred_loss))
            # writer.add_scalar('accs/target_test_acc', pred_acc, epoch)
            wandb.log({"test_chamfer":pred_loss})

            # to_log = np.random.choice([x for x in range(BATCH_SIZE)], 10)
            # pred_log = pred[to_log].cpu().numpy()
            # data_log = data[to_log].cpu().numpy()
            # for i in range(10):
            #     wandb.log({"original_%d"%i: wandb.Object3D(data[i])})
            #     wandb.log({"recon_%d"%i: wandb.Object3D(pred_log[i])})

        time_pass_e = time.time() - since_e
        print('The {} epoch takes {:.0f}m {:.0f}s'.format(epoch, time_pass_e // 60, time_pass_e % 60))
        print(args)
        print(' ')


if __name__ == '__main__':
    since = time.time()
    main()
    time_pass = since - time.time()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))

