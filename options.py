
import argparse


def parse_opts():

    description = 'Image Super-Resolution pytorch implementation'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--model_name',
                        default='base',
                        type=str,
                        help='Model name')

    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help='Batch size')

    parser.add_argument('--learning_rate',
                        default=1e-4,
                        type=float,
                        help='learning rate')
    parser.add_argument('--upscale_factor',
                        default=4,
                        type=int,
                        help='Upsacle factor')

    parser.add_argument('--patch_size_w',
                        default=64,
                        type=int,
                        help='Patch size')
    parser.add_argument('--patch_size_h',
                        default=64,
                        type=int,
                        help='Patch size')

    parser.add_argument('--hr_data',
                        default='./train',
                        type=str,
                        help='Dataset directory')

    parser.add_argument('--test_hr_data',
                        default='./test',
                        type=str,
                        help='Dataset directory')

    parser.add_argument('--file_list',
                        default='data.txt',
                        type=str,
                        help='Train dataset split code')


    parser.add_argument('--test_file_list',
                        default='test.txt',
                        type=str,
                        help='Val dataset split code')

    parser.add_argument('--threads',
                        default=1,
                        type=int,
                        help='Number of threads for data loader to use')

    parser.add_argument('--seed',
                        default=123,
                        type=int,
                        help='random seed to use. Default=123')

    parser.add_argument('--start_epoch',
                        default=1,
                        type=int,
                        help='Starting epoch for continuing training')

    parser.add_argument('--nEpochs',
                        default=100,
                        type=int,
                        help='Number os total epochs to run')

    parser.add_argument('--checkpoint',
                        default=10,
                        type=int,
                        help='Trained model is saved at every this epoch')

    parser.add_argument('--save_path',
                        default='./epochs/',
                        type=str,
                        help='Save data (.pth) of previous training')

    parser.add_argument('--pretrained',
                        default=False,
                        type=bool,
                        help='Whether to load pretrained models')

    parser.add_argument('--pretrained_sr_model',
                        default='base_epoch_2_59.pth',
                        type=str,
                        help='SR pretrained base model')


    args = parser.parse_args()

    return args