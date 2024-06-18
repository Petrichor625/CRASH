from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import os, time
import argparse
import shutil
import cv2

from torch.utils.data import DataLoader
from src.Models import CRASH
from src.eval_tools import evaluation_P_R80, print_results, vis_results
import ipdb
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import average_precision_score

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
ROOT_PATH = os.path.dirname(__file__)


def average_losses(losses_all):
    total_loss, cross_entropy, aux_loss = 0, 0, 0
    losses_mean = {}
    for losses in losses_all:
        total_loss += losses['total_loss']
        cross_entropy += losses['cross_entropy']
        aux_loss += losses['auxloss']
    losses_mean['total_loss'] = total_loss / len(losses_all)
    losses_mean['cross_entropy'] = cross_entropy / len(losses_all)
    losses_mean['auxloss'] = aux_loss / len(losses_all)
    return losses_mean


def test_all(testdata_loader, model):

    all_pred = []
    all_labels = []
    all_toas = []
    losses_all = []
    with torch.no_grad():
        for i, (batch_xs, batch_ys, batch_toas) in enumerate(testdata_loader):
            losses, all_outputs, hiddens = model(batch_xs, batch_ys, batch_toas,
                    hidden_in=None, nbatch=len(testdata_loader), testing=False)
            losses['total_loss'] =  p.loss_u1 / 2 * losses['cross_entropy']
            losses['total_loss'] += p.loss_u2 / 2 * losses['auxloss']
            losses['total_loss'] += losses['log'].mean()
            losses_all.append(losses)

            num_frames = batch_xs.size()[1]
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            for t in range(num_frames):
                pred = all_outputs[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
            all_pred.append(pred_frames)
            label_onehot = batch_ys.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)
            toas = np.squeeze(batch_toas.cpu().numpy()).astype(np.int)
            all_toas.append(toas)

    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))

    return all_pred, all_labels, all_toas, losses_all

def preprocess_results(pred_score, cumsum=False):
    from scipy.interpolate import make_interp_spline

    xvals = np.linspace(0,len(pred_score)-1,10)
    pred_mean_reduce = pred_score[xvals.astype(np.int)]

    xvals_new = np.linspace(1,len(pred_score)+1, 100)
    pred_score = make_interp_spline(xvals, pred_mean_reduce)(xvals_new)

    pred_score[pred_score >= 1.0] = 1.0-1e-3
    xvals = np.copy(xvals_new)
    xvals = np.insert(xvals_new, 0, 0)
    pred_score = np.insert(pred_score, 0, pred_score[0])

    if cumsum:
        pred_score = np.cumsum(pred_score)
        pred_score = pred_score / np.max(pred_score)
    return xvals, pred_score

def draw_curve(xvals, pred_score):
    plt.plot(xvals, pred_score, linewidth=3.0)
    plt.axhline(y=0.5, xmin=0, xmax=max(xvals)/(100 + 2), linewidth=3.0, color='g', linestyle='--')
    plt.tight_layout()

def get_video_frames(video_file, n_frames=100):
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    video_data = []
    counter = 0
    while (ret):
        video_data.append(frame)
        ret, frame = cap.read()
        counter += 1
    assert len(video_data) >= n_frames, video_file
    video_data = video_data[:n_frames]
    return video_data

def draw_anchors(video_file,used_detections,video_id):
    video_path = video_file
    cap = cv2.VideoCapture(video_path)

    fps = 20.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    output_video_path = f'anchors/{video_id}_anchors.mp4'  
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


    anchor_boxes = used_detections

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            for box in anchor_boxes[frame_idx]:
                x1, y1, x2, y2 = box[:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 


            out.write(frame)


            frame_idx += 1
            if frame_idx >= frame_count:
                break  
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def test_all_vis(testdata_loader, model, vis=True, multiGPU=False, device=torch.device('cuda')):

    if multiGPU:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.eval()

    all_pred = []
    all_labels = []
    all_toas = []
    vis_data = []

    with torch.no_grad():
        for i, (batch_xs, batch_ys, batch_toas, detections, video_ids) in tqdm(enumerate(testdata_loader), desc="batch progress", total=len(testdata_loader)):
            losses, all_outputs, hiddens = model(batch_xs, batch_ys, batch_toas,
                    hidden_in=None,  nbatch=len(testdata_loader), testing=False)

            num_frames = batch_xs.size()[1]
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)

            for t in range(num_frames):2
                pred = all_outputs[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)


            all_pred.append(pred_frames)
            label_onehot = batch_ys.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)
            toas = np.squeeze(batch_toas.cpu().numpy()).astype(np.int)
            all_toas.append(toas)

            if vis:
                vis_data.append({'pred_frames': pred_frames, 'label': label,
                                'toa': toas, 'detections': detections, 'video_ids': video_ids})
                for j,video_id in enumerate(video_ids):
                    used_alphas = []
                    used_detections = detections[j]
                    video_file = f'videos/positive/{video_id}.mp4' if vis_data[i]['label'][j]==1 else f'videos/negative/{video_id}.mp4'
                    draw_anchors(video_file,used_detections,video_id)
                    video_data = get_video_frames(video_file, n_frames=100)
                    xvals,pred_score = preprocess_results(vis_data[i]['pred_frames'][j])
                    fig, ax = plt.subplots(1, figsize=(24, 3.5))
                    fontsize = 25
                    plt.ylim(0, 1.1)
                    plt.xlim(0, len(xvals)+1)
                    plt.ylabel('Probability', fontsize=fontsize)
                    plt.xlabel('Frame (FPS=2)', fontsize=fontsize)
                    plt.xticks(range(0, len(xvals)+1, 10), fontsize=fontsize)
                    plt.yticks(fontsize=fontsize)
                    from matplotlib.animation import FFMpegWriter
                    curve_writer = FFMpegWriter(fps=2, metadata=dict(title='Movie Test', artist='Matplotlib',comment='Movie support!'))
                    curve_save = 'cruves/' + video_id + '_curve_video.mp4'
                    with curve_writer.saving(fig, curve_save, 100):
                        for t in range(len(xvals)):
                            draw_curve(xvals[:(t+1)], pred_score[:(t+1)])
                            curve_writer.grab_frame()
                    curve_frames = get_video_frames(curve_save, n_frames=100)
                    video_writer = cv2.VideoWriter(f'twotypes/{video_id}_vis.avi', cv2.VideoWriter_fourcc(*'DIVX'), 2.0, (video_data[0].shape[1], video_data[0].shape[0]))
                

                    for t, frame in enumerate(video_data):
                        attention_frame = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.uint8)
                        now_weight = used_alphas[t]
                        now_weight = now_weight.cpu()
                        now_weight = now_weight
                        det_boxes = used_detections[t]  
                        index = np.argsort(now_weight)

                        for num_box in index:
                            attention_frame[int(det_boxes[num_box,1]):int(det_boxes[num_box,3]),int(det_boxes[num_box,0]):int(det_boxes[num_box,2])] = now_weight[num_box]*1000

                        img = curve_frames[t]
                        attention_frame_resized = cv2.resize(attention_frame,(frame.shape[1], frame.shape[0]))
                        attention_frame = cv2.applyColorMap(attention_frame_resized, cv2.COLORMAP_BONE)
                        dst = cv2.addWeighted(frame,0.9,attention_frame,0.4,0)
                        width = frame.shape[1]
                        height = int(img.shape[0] * (width / img.shape[1]))
                        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

                        h1,w1 = dst.shape[:2]
                        h2,w2 = img.shape[:2]
                        visss = np.zeros((h1+h2, max(w1, w2),3),np.uint8)

                        visss[:h1, :w1,:3] = dst
                        visss[h1:h1+h2, :w2, :3] = img
                        dst = cv2.resize(visss,(1280,720))

                        video_writer.write(dst)

    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))

    return all_pred, all_labels, all_toas, vis_data


def write_scalars(logger, cur_epoch, cur_iter, losses, lr):
    total_loss = losses['total_loss'].mean().item()
    cross_entropy = losses['cross_entropy'].mean()
    aux_loss = losses['auxloss'].mean().item()


    logger.add_scalars("train/losses/total_loss", {'total_loss': total_loss}, cur_iter)
    logger.add_scalars("train/losses/cross_entropy", {'cross_entropy': cross_entropy}, cur_iter)
    logger.add_scalars("train/losses/aux_loss", {'aux_loss': aux_loss}, cur_iter)
    logger.add_scalars("train/learning_rate/lr", {'lr': lr}, cur_iter)


def write_test_scalars(logger, cur_epoch, cur_iter, losses, metrics):
    total_loss = losses['total_loss'].mean().item()
    cross_entropy = losses['cross_entropy'].mean()
    loss_info = {'total_loss': total_loss, 'cross_entropy': cross_entropy}
    aux_loss = losses['auxloss'].mean().item()
    loss_info.update({'aux_loss': aux_loss})
    logger.add_scalars("test/losses/total_loss", loss_info, cur_iter)
    logger.add_scalars("test/accuracy/AP", {'AP': metrics['AP'],'P_R80': metrics['P_R80']}, cur_iter)
    logger.add_scalars("test/accuracy/time-to-accident", {'mTTA': metrics['mTTA'],
                                                          'TTA_R80': metrics['TTA_R80']}, cur_iter)



def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', isTraining=True):
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        if isTraining:
            optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, start_epoch


def train_eval():
    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    model_dir = os.path.join(p.output_dir, p.dataset, 'snapshot')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logs_dir = os.path.join(p.output_dir, p.dataset, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger = SummaryWriter(logs_dir)

    gpu_ids = [int(id) for id in p.gpus.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if p.dataset == 'dad':
        from src.DataLoader import DADDataset
        train_data = DADDataset(data_path, p.feature_name, 'training', toTensor=True, device=device)
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device)
    elif p.dataset == 'a3d':
        from src.DataLoader import A3DDataset
        train_data = A3DDataset(data_path, p.feature_name, 'train', toTensor=True, device=device)
        test_data = A3DDataset(data_path, p.feature_name, 'test', toTensor=True, device=device)
    elif p.dataset == 'crash':
        from src.DataLoader import CrashDataset
        train_data = CrashDataset(data_path, p.feature_name, 'train', toTensor=True, device=device)
        test_data = CrashDataset(data_path, p.feature_name, 'test', toTensor=True, device=device)
    else:
        raise NotImplementedError
    traindata_loader = DataLoader(dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)

    model = CRASH(train_data.dim_feature, p.hidden_dim, p.latent_dim,
                       n_layers=p.num_rnn, n_obj=train_data.n_obj, n_frames=train_data.n_frames, fps=train_data.fps,
                       with_saa=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=p.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)
    model.train()
    start_epoch = -1
    if p.resume:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer=optimizer, filename=p.model_file)

    iter_cur = 0
    best_metric = 0
    best_P_R80 = 0.5
    metrics = {}
    metrics['AP']= 0
    for k in range(p.epoch):
        loop = tqdm(enumerate(traindata_loader),total=len(traindata_loader))
        if k <= start_epoch:
            iter_cur += len(traindata_loader)
            continue
        for i, (batch_xs, batch_ys, batch_toas) in loop:
            optimizer.zero_grad()
            losses, all_outputs, hidden_st = model(batch_xs, batch_ys, batch_toas, nbatch=len(traindata_loader))
            losses['total_loss'] =  p.loss_u1 / 2 * losses['cross_entropy']
            losses['total_loss'] += p.loss_u2 / 2 * losses['auxloss']
            losses['total_loss'] += losses['logdet'].mean()

            losses['total_loss'].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            loop.set_description(f"Epoch  [{k}/{p.epoch}]")
            loop.set_postfix(loss= losses['total_loss'].item() )
            lr = optimizer.param_groups[0]['lr']
            write_scalars(logger, k, iter_cur, losses, lr)

            iter_cur += 1
            if iter_cur % p.test_iter == 0:
                model.eval()
                all_pred, all_labels, all_toas, losses_all = test_all(testdata_loader, model)
                model.train()
                loss_val = average_losses(losses_all)
                metrics = {}
                metrics['AP'], metrics['mTTA'], metrics['TTA_R80'], metrics['P_R80'] = evaluation_P_R80(all_pred, all_labels, all_toas, fps=test_data.fps)
                write_test_scalars(logger, k, iter_cur, loss_val, metrics)

        model_file = os.path.join(model_dir, 'model_%02d.pth'%(k))
        torch.save({'epoch': k,
                    'model': model.module.state_dict() if len(gpu_ids)>1 else model.state_dict(),
                    'optimizer': optimizer.state_dict()}, model_file)
        if metrics['AP'] > best_metric:
            best_metric = metrics['AP']
            update_final_model(model_file, os.path.join(model_dir, 'final_model.pth'))

        scheduler.step(losses['total_loss'])
    logger.close()


def update_final_model(src_file, dest_file):
    assert os.path.exists(src_file), "src file does not exist!"
    if os.path.exists(dest_file):
        if not os.path.samefile(src_file, dest_file):
            os.remove(dest_file)
    shutil.copyfile(src_file, dest_file)


def test_eval():
    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    result_dir = os.path.join(p.output_dir, p.dataset, 'test')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    p.visualize = False if p.evaluate_all else p.visualize
    vis_dir = None
    if p.visualize:
        vis_dir = os.path.join(result_dir, 'vis')
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    gpu_ids = [int(id) for id in p.gpus.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = p.gpus
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if p.dataset == 'dad':
        from src.DataLoader import DADDataset
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device, vis=True)
    elif p.dataset == 'a3d':
        from src.DataLoader import A3DDataset
        train_data = A3DDataset(data_path, p.feature_name, 'train', toTensor=True, device=device)
        test_data = A3DDataset(data_path, p.feature_name, 'test', toTensor=True, device=device)
    elif p.dataset == 'crash':
        from src.DataLoader import CrashDataset
        test_data = CrashDataset(data_path, p.feature_name, 'test', toTensor=True, device=device, vis=True)
    else:
        raise NotImplementedError
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)
    num_samples = len(test_data)

    model = CRASH(test_data.dim_feature, p.hidden_dim, p.latent_dim,
                       n_layers=p.num_rnn, n_obj=test_data.n_obj, n_frames=test_data.n_frames, fps=test_data.fps,
                       with_saa=True)

    if p.evaluate_all:
        model_dir = os.path.join(p.output_dir, p.dataset, 'snapshot')
        assert os.path.exists(model_dir)
        Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all = [], [], [], [], []
        modelfiles = sorted(os.listdir(model_dir))
        for filename in modelfiles:
            epoch_str = filename.split("_")[-1].split(".pth")[0]
            model_file = os.path.join(model_dir, filename)
            model, _, _ = load_checkpoint(model, filename=model_file, isTraining=False)
            all_pred, all_labels, all_toas, _ = test_all_vis(testdata_loader, model, vis=False, device=device)
            AP, mTTA, TTA_R80, P_R80 = evaluation_P_R80(all_pred, all_labels, all_toas, fps=test_data.fps)
            all_vid_scores = [max(pred[:int(toa)]) for toa, pred in zip(all_toas, all_pred)]
            AP_video = average_precision_score(all_labels, all_vid_scores)
            APvid_all.append(AP_video)
            Epochs.append(epoch_str)
            AP_all.append(AP)
            mTTA_all.append(mTTA)
            TTA_R80_all.append(TTA_R80)

        print_results(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, result_dir)
    else:
        result_file = os.path.join(vis_dir, "..", "pred_res.npz")
        model, _, _ = load_checkpoint(model, filename=p.model_file, isTraining=False)
        all_pred, all_labels, all_toas, vis_data = test_all_vis(testdata_loader, model, vis=True, device=device)
        np.savez(result_file[:-4], pred=all_pred, label=all_labels, toas=all_toas, vis_data=vis_data)

        all_vid_scores = [max(pred[:int(toa)]) for toa, pred in zip(all_toas, all_pred)]
        AP_video = average_precision_score(all_labels, all_vid_scores)
        AP, mTTA, TTA_R80, P_R80 = evaluation_P_R80(all_pred, all_labels, all_toas, fps=test_data.fps)
        vis_results(vis_data, p.batch_size, vis_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data',
                        help='The relative path of dataset.')
    parser.add_argument('--dataset', type=str, default='dad', choices=['dad', 'crash','a3d'],
                        help='The name of dataset. Default: dad')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='The base learning rate. Default: 1e-3')
    parser.add_argument('--epoch', type=int, default=80,
                        help='The number of training epoches. Default: 80')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size in training process. Default: 10')
    parser.add_argument('--num_rnn', type=int, default=2,
                        help='The number of RNN cells for each timestamp. Default: 2')
    parser.add_argument('--feature_name', type=str, default='vgg16', choices=['vgg16', 'res101'],
                        help='The name of feature embedding methods. Default: vgg16')
    parser.add_argument('--test_iter', type=int, default=64,
                        help='The number of iteration to perform a evaluation process. Default: 64')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='The dimension of hidden states in RNN. Default: 512')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='The dimension of latent space. Default: 256')
    parser.add_argument('--loss_u1', type=float, default=1,
                        help='The weighting factor of auxiliary loss. Default: 1')
    parser.add_argument('--loss_u2', type=float, default=15,
                        help='The weighting factor of auxiliary loss. Default: 15')
    parser.add_argument('--gpus', type=str, default="0",
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    parser.add_argument('--phase', type=str, choices=['train', 'test'],
                        help='The state of running the model. Default: train')
    parser.add_argument('--evaluate_all', action='store_true',
                        help='Whether to evaluate models of all epoches. Default: False')
    parser.add_argument('--visualize', action='store_true',
                        help='The visualization flag. Default: False')
    parser.add_argument('--resume', action='store_true',
                        help='If to resume the training. Default: False')
    parser.add_argument('--model_file', type=str, default='./output/CRASH/vgg16/dad/snapshot_attention_v18/model_23.pth',
                        help='The trained model file for demo test only.')

    p = parser.parse_args()
    if p.phase == 'test':
        test_eval()
    else:
        train_eval()