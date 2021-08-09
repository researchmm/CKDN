"""Sample PyTorch Inference script
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import logging
import numpy as np
import torch

from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging
from timm.utils import *

import ckdn

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=288, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')


def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    # create model
    model = ckdn.model()
    #model = vgg.vgg16()
    logging.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    state_dicts = torch.load(args.checkpoint)['state_dict']
    new_state_dict = {}
    for k in state_dicts.keys():
        new_state_dict[k.split('module.')[1]] = state_dicts[k]

    model.load_state_dict(new_state_dict,strict=False) 

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = model, False #apply_test_time_pool(model, config, args)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    loader = create_loader(
        Dataset(args.data,'images'),
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=0.875)

    model.eval()

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.cuda()
            output = model(input)
            outputs.append(output[:,0].cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                logging.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

    outputs = np.concatenate(outputs, axis=0).squeeze()
    if outputs.size==1:
        outputs = [outputs]
    with open(os.path.join(args.output_dir, './output.txt'), 'w') as out_file:
        filenames = loader.dataset.filenames()
        for filename, output in zip(filenames, [str(o) for o in outputs]):
            filename = os.path.basename(filename)
            out_file.write('{0},{1}\n'.format(
                filename, output))

    
if __name__ == '__main__':
    main()
