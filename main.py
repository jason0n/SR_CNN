import os
import datetime
from math import log10
from tqdm import tqdm
import pandas as pd

import torch

import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn

from options import parse_opts

from baseline import basemodel
from data_util import DatasetFromFolder
from torchvision.transforms import Compose, ToTensor

def transform():
    return Compose([
        ToTensor(),
    ])


def display_config(args):

    print('-------YOUR SETTINGS_________')
    for arg in vars(args):
        print("%15s: %s" %(str(arg), str(getattr(args, arg))))
    print('')


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class MSE_loss(nn.Module):
    def __init__(self):
        super(MSE_loss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, img1, img2):
        loss = self.mse(img1, img2)

        return loss


def train(args, epoch, train_loader, optimizer, model, criterion):

    model.train()

    epoch_loss = 0
    epoch_psnr = 0
    running_results = {'batch_sizes': 0, 'loss': 0, 'psnrs': 0,'psnr': 0}

    n_iterations = len(train_loader)
    train_bar = tqdm(train_loader)

    for hr_frame, lr_frames in train_bar:
    #循环这个 DataLoader 对象，将一个batch的img, label进行预处理后加载到模型中进行训练
        batch_size = lr_frames.size(0)
        running_results['batch_sizes'] += batch_size

        # Warp with torch Variable
        if torch.cuda.is_available():
            hr_frame = Variable(hr_frame).cuda()
            lr_frames = Variable(lr_frames).cuda()

        # Zero the grad
        optimizer.zero_grad()

        # Forward
        sr_frame= model(lr_frames)
        # Compute Loss
        loss = criterion(sr_frame, hr_frame)

        epoch_loss += loss.item()
        running_results['loss'] += loss.item() * batch_size

        # Calculate PSNR
        psnr = 10 * log10(1 / loss.item())
        epoch_psnr += psnr

        batch_mse = ((sr_frame - hr_frame) ** 2).data.mean()
        batch_psnr = 10 * log10(1 / batch_mse)
        running_results['psnrs'] += batch_psnr * batch_size
        running_results['psnr'] = running_results['psnrs'] / running_results['batch_sizes']

        # Backward + update
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        train_bar.set_description(
            desc='[%d/%d] loss: %.8f psnr: %.4f' % (epoch, args.nEpochs,
                                         running_results['loss'] / running_results['batch_sizes'],
                                         running_results['psnr']))

    epoch_loss /= n_iterations
    epoch_psnr /= n_iterations

    print('===> Epoch {} complete: Avg. Loss: {:.8f}, '
          'Avg. PSNR: {:.4f}'.format(epoch, epoch_loss, epoch_psnr))

    return epoch_loss, epoch_psnr

def test(test_loader, model, criterion):

    model.eval()


    avg_loss = 0
    avg_psnr = 0


    with torch.no_grad():
        for test_hr_frame, test_lr_frames in test_loader:

            # Warp with torch Variable
            if torch.cuda.is_available():
                test_hr_frame = Variable(test_hr_frame).cuda()
                test_lr_frames = Variable(test_lr_frames).cuda()

            sr_frame = model(test_lr_frames)

            # Compute Loss
            loss = criterion(sr_frame, test_hr_frame)
            avg_loss += loss.item()
            # Calculate PSNR
            psnr = 10 * log10(1 / loss.item())
            avg_psnr += psnr


    num_batches = len(test_loader)
    avg_loss /= num_batches
    avg_psnr /= num_batches

    print('===> Avg. Loss on test set: {:.4f}, Avg. PSNR on test set: {:.4f} dB'.format(avg_loss, avg_psnr))

    return avg_loss, avg_psnr



def main():
    # Configs
    args = parse_opts()
    display_config(args)

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    #创建一个Dataset对象，相当于创建了一个数据集，里面包含导入图像数据以及数据预处理的过程
    print('===> Loading datasets')
    train_set = DatasetFromFolder(args.hr_data, args.upscale_factor,args.file_list, True,args.patch_size_w, args.patch_size_h,transform=transform())
    test_set = DatasetFromFolder(args.test_hr_data, args.upscale_factor,args.test_file_list, False,0,0,transform=transform())
    # 封装成可迭代对象用于后面训练测试以batch为单位加载数据到模型中进行训练
    train_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=1, shuffle=False)

    # Load model
    print('===> Building model')
    #定义超分网络
    model = basemodel()

    print('---------- Networks architecture -------------')
    print_network(model)
    print('----------------------------------------------')

    if args.pretrained:
    #加载之前训练过的模型
        model_name = os.path.join(args.save_path + args.pretrained_sr_model)
        if os.path.exists(model_name):
            model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
            print('Pre-trained SR model is loaded.')
        else:
            print('This Pre-trained SR model does not exist!')
            raise ValueError

    # Load loss function

    criterion = MSE_loss()

    # Setup device for running
    # if torch.cuda.is_available():
    #     model.cuda()
    #     criterion.cuda()

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print('===> Current learning rate : lr = {}\n'.format(optimizer.param_groups[0]['lr']))


    if args.pretrained:
        index_label_count = 0
    epoch_train_losses = []
    epoch_train_psnr = []
    epoch_test_losses = []
    epoch_test_psnr = []
    epoch_lr = []
    epochss = []
    for epoch in range(args.start_epoch, args.nEpochs + 1):


        # Training
        train_loss, train_psnr = train(args, epoch, train_loader, optimizer, model, criterion)

        # test
        test_loss, test_psnr= test(test_loader, model, criterion)

        epochss.append(epoch)
        epoch_train_losses.append(train_loss)
        epoch_train_psnr.append(train_psnr)
        epoch_test_losses.append(test_loss)
        epoch_test_psnr.append(test_psnr)
        epoch_lr.append(optimizer.param_groups[0]['lr'])

        # Save training data：将训练结果写入excel文件中
        out_path = 'statistics/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        if args.pretrained:
            index_label_count += 1

            data_frame = pd.DataFrame(
                data={'Epoch':epochss, 'train_Loss': epoch_train_losses, 'train_psnr': epoch_train_psnr,
                      'test_Loss': epoch_test_losses, 'test_psnr': epoch_test_psnr,'lr':epoch_lr}, index=range(args.start_epoch,args.start_epoch+index_label_count))

            data_frame.to_csv(out_path + '{}_x'.format(args.model_name) + str(args.upscale_factor) + '_continue_train.csv',
                              index_label='Epoch')
        else:
            data_frame = pd.DataFrame(
                data={'train_Loss': epoch_train_losses, 'train_psnr': epoch_train_psnr,
                      'test_Loss': epoch_test_losses, 'test_psnr': epoch_test_psnr,'lr':epoch_lr}, index=range(1, epoch + 1))

            data_frame.to_csv(out_path + '{}_x'.format(args.model_name) + str(args.upscale_factor) + '_train.csv', index_label='Epoch')

        # 保存模型
        torch.save(model.state_dict(), 'epochs/%s_epoch_%d_%d.pth' % (args.model_name, args.upscale_factor, epoch))


if __name__ == '__main__':
    main()