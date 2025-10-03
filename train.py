#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts.

This script supports knowledge distillation with various methods including OFA, CRD, RKD, and more.

Original work by Ross Wightman (https://github.com/rwightman)
Modified by Zhiwei Hao (haozhw@bit.edu.cn)
"""
import argparse
import logging
import os
import time
from collections import OrderedDict, defaultdict
from contextlib import suppress
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.utils
import yaml
from timm.data import AugMixDataset, create_dataset, create_loader, FastCollateMixup, Mixup, \
    resolve_data_config
from timm.loss import *
from timm.models import convert_splitbn_model, create_model, model_parameters, safe_model_name, load_checkpoint
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import *
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from custom_forward import register_new_forward
from distillers import get_distiller
from utils import CIFAR100InstanceSample, ImageNetInstanceSample, TimePredictor
from custom_model import *

# Import optional dependencies
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

# Global settings
torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


def parse_args():
    """Parse command line arguments and optional YAML config file."""
    # First parse just the config file argument
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    
    # Main parser for all other arguments
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    
    # ------------------------------------- My params ---------------------------------------
    # Basic parameters
    parser.add_argument('--model', default='resnet18', type=str)
    parser.add_argument('--teacher', default='deit_tiny_patch16_224', type=str)
    parser.add_argument('--teacher-pretrained', default='', type=str)
    parser.add_argument('--use-ema-teacher', action='store_true')

    parser.add_argument('--distiller', default='ofa', type=str)
    parser.add_argument('--gt-loss-weight', default=1., type=float)
    parser.add_argument('--kd-loss-weight', default=1., type=float)

    # KD parameters
    parser.add_argument('--kd-temperature', default=1, type=float)

    # OFA parameters
    parser.add_argument('--ofa-eps', default=[1], nargs='+', type=float)
    parser.add_argument('--ofa-stage', default=[1, 2, 3, 4], nargs='+', type=int)
    parser.add_argument('--ofa-loss-weight', default=1, type=float)
    parser.add_argument('--ofa-temperature', default=1, type=float)

    # DIST parameters
    parser.add_argument('--dist-beta', default=1, type=float)
    parser.add_argument('--dist-gamma', default=1, type=float)
    parser.add_argument('--dist-tau', default=1, type=float)

    # DKD parameters
    parser.add_argument('--dkd-alpha', default=1, type=float)
    parser.add_argument('--dkd-beta', default=2, type=float)
    parser.add_argument('--dkd-temperature', default=1, type=float)

    # Correlation parameters
    parser.add_argument('--correlation-scale', default=0.02, type=float)
    parser.add_argument('--correlation-feat-dim', default=128, type=int)

    # CRD parameters
    parser.add_argument('--crd-feat-dim', default=128, type=int)
    parser.add_argument('--crd-k', default=16384, type=int)
    parser.add_argument('--crd-momentum', default=0.5, type=float)
    parser.add_argument('--crd-temperature', default=0.07, type=float)

    # RKD parameters
    parser.add_argument('--rkd-distance-weight', default=25, type=float)
    parser.add_argument('--rkd-angle-weight', default=50, type=float)
    parser.add_argument('--rkd-eps', default=1e-12, type=float)
    parser.add_argument('--rkd-squared', action='store_true', default=False)

    # FitNet parameters
    parser.add_argument('--fitnet-stage', default=[1, 2, 3, 4], nargs='+', type=int)
    parser.add_argument('--fitnet-loss-weight', default=1, type=float)

    # Misc
    parser.add_argument('--speedtest', action='store_true')

    parser.add_argument('--eval-interval', default=1, type=int)  # eval every 1 epochs before epochs * eval_interval_end
    parser.add_argument('--eval-interval-end', default=0.75, type=float)
    # ---------------------------------------------------------------------------------------

    # Dataset parameters
    parser.add_argument('data_dir', metavar='DIR',
                        help='path to dataset')

    parser.add_argument('--dataset', '-d', metavar='NAME', default='cifar100',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')

    parser.add_argument('--train-split', metavar='NAME', default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', metavar='NAME', default='validation',
                        help='dataset validation split (default: validation)')

    parser.add_argument('--dataset-download', action='store_true', default=False,
                        help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                        help='path to class to idx mapping file (default: "")')

    # Model parameters
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--img-size', type=int, default=None, metavar='N',
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int, metavar='N N N',
                        help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    parser.add_argument('--crop-pct', default=None, type=float, metavar='N',
                        help='Input image center crop percent (for validation only)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                        help='Input batch size for training (default: 128)')
    parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                        help='Validation batch size override (default: None)')

    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=2e-5,
                        help='weight decay (default: 2e-5)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--layer-decay', type=float, default=None,
                        help='layer-wise learning rate decay (default: None)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    parser.add_argument('--aug-repeats', type=float, default=0,
                        help='Number of augmentation repetitions (distributed training only) (default: 0)')
    parser.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    parser.add_argument('--jsd-loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--bce-loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce-target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    # Batch norm parameters
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=[0.9998], nargs='+',
                        help='decay factor for model weights moving average (default: 0.9998)')

    # Misc
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--worker-seeding', type=str, default='all',
                        help='worker seed mode (default: all)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                        help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                        help='number of checkpoints to keep (default: 10)')
    parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='save images of input bathes every log interval for debugging')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                        help='name of train experiment, name of sub-folder for output')
    parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1"')
    parser.add_argument('--tta', type=int, default=0, metavar='N',
                        help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                        help='use the multi-epochs-loader to save time at the beginning of every epoch')
                        
    # Parse arguments
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def configure_environment(args):
    """Configure distributed settings, device, seed and AMP selection.
    
    Sets up environment variables including prefetcher, distributed training,
    device selection, and automatic mixed precision.
    
    Args:
        args: Command line arguments
        
    Returns:
        use_amp (str|None): 'apex' or 'native' if AMP is requested and available, else None
    """
    # Set data prefetcher mode (disabled for CRD distillation)
    args.prefetcher = (not args.no_prefetcher) and (args.distiller != 'crd')
    
    # Configure distributed training
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    
    # Set device and process rank
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    
    if args.distributed:
        assert 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.device = int(os.environ['LOCAL_RANK'])

        torch.cuda.set_device(args.device)
        torch.distributed.init_process_group(
            backend='nccl', 
            init_method=args.dist_url,
            world_size=args.world_size, 
            rank=args.rank
        )
        torch.distributed.barrier()

        _logger.info(f'Training in distributed mode with multiple processes. '
                     f'Process {args.rank}, total {args.world_size}.')
    else:
        _logger.info('Training with a single process on 1 GPU.')
        
    assert args.rank >= 0

    # Setup Automatic Mixed Precision (AMP)
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
            
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                      "Install NVIDIA apex or upgrade to PyTorch 1.6+")

    # Set random seeds for reproducibility
    random_seed(args.seed, args.rank)
    
    return use_amp


def build_model_and_teacher(args, Distiller):
    """Create student and teacher models with proper initialization.
    
    Builds the student model with specified parameters and optionally loads
    a teacher model for knowledge distillation.
    
    Args:
        args: Command line arguments
        Distiller: Distillation class to use
        
    Returns:
        tuple: (student_model, teacher_model)
    """
    # Create student model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        checkpoint_path=args.initial_checkpoint)
    
    # Register custom forward hooks for feature extraction if needed
    if Distiller.requires_feat:
        register_new_forward(model)

    # Create teacher model if specified
    teacher = None
    if args.teacher:
        teacher = create_model(
            args.teacher,
            num_classes=args.num_classes)
        load_checkpoint(teacher, args.teacher_pretrained, use_ema=args.use_ema_teacher)
        
        if Distiller.requires_feat:
            register_new_forward(teacher)

        # Set teacher to eval mode and freeze parameters
        teacher.requires_grad_(False)
        teacher.eval()

    # Set num_classes from model if not specified in args
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    # Log model info
    if args.rank == 0:
        _logger.info(f'Model {safe_model_name(args.model)} created, '
                     f'param count: {sum([m.numel() for m in model.parameters()]):,}')

    return model, teacher


def create_datasets_and_loaders(args, model, Distiller):
    """Create datasets and data loaders for training and validation.
    
    Handles dataset creation, data augmentation setup, and data loader configuration.
    
    Args:
        args: Command line arguments
        model: Student model
        Distiller: Distillation class
        
    Returns:
        tuple: (data_config, loader_train, loader_eval, num_aug_splits, mixup_fn)
    """
    # Resolve data configuration
    data_config = resolve_data_config(vars(args), model=model, verbose=args.rank == 0)

    # Setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # Enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # Setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and args.apex_amp:
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.rank == 0:
            _logger.info('Converted model to use Synchronized BatchNorm.')

    # Create the train and eval datasets
    if args.dataset == 'cifar100':
        if args.distiller == 'crd':
            dataset_train = CIFAR100InstanceSample(root=args.data_dir, train=True, is_sample=True, k=args.crd_k)
        else:
            dataset_train = torchvision.datasets.CIFAR100(args.data_dir, train=True)
        dataset_eval = torchvision.datasets.CIFAR100(args.data_dir, train=False)
        data_config['mean'] = (0.5071, 0.4865, 0.4409)
        data_config['std'] = (0.2673, 0.2564, 0.2762)
    else:
        if args.distiller == 'crd':
            dataset_train = ImageNetInstanceSample(root=f'{args.data_dir}/train', name=args.dataset,
                                                class_map=args.class_map, load_bytes=False, is_sample=True,
                                                k=args.crd_k)
        else:
            dataset_train = create_dataset(
                args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
                class_map=args.class_map, download=args.dataset_download,
                batch_size=args.batch_size, repeats=args.epoch_repeats)

        dataset_eval = create_dataset(
            args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
            class_map=args.class_map, download=args.dataset_download,
            batch_size=args.batch_size)

    # Setup mixup / cutmix
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    collate_fn = None
    mixup_fn = None
    
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
            
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # Wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # Create data loaders with augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
        
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )
    
    return data_config, dataset_train, dataset_eval, loader_train, loader_eval, num_aug_splits, mixup_fn


def create_train_loss_fn(args, num_aug_splits):
    """Create appropriate training loss function based on configuration.
    
    Args:
        args: Command line arguments
        num_aug_splits: Number of augmentation splits
        
    Returns:
        nn.Module: Loss function
    """
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    
    if args.jsd_loss:
        assert num_aug_splits > 1, "JSD loss requires aug_splits > 1"
        return JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # Smoothing handled by mixup target transform
        if args.bce_loss:
            return BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            return SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            return BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            return LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        return nn.CrossEntropyLoss()


def setup_amp(use_amp, model, optimizer, args):
    """Configure Automatic Mixed Precision (AMP) for training.
    
    Args:
        use_amp: AMP type ('apex', 'native', or None)
        model: Model to convert
        optimizer: Optimizer to convert
        args: Command line arguments
        
    Returns:
        tuple: (amp_autocast, loss_scaler)
    """
    amp_autocast = suppress  # Default: do nothing (no mixed precision)
    loss_scaler = None
    
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.rank == 0:
            _logger.info('AMP not enabled. Training in float32.')
            
    return amp_autocast, loss_scaler


def setup_model_ema(model, args):
    """Setup Exponential Moving Average (EMA) for model weights.
    
    Args:
        model: Model to track with EMA
        args: Command line arguments
        
    Returns:
        list: List of (ema_model, decay) tuples
    """
    model_emas = None
    
    if args.model_ema:
        # Create EMA models after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_emas = []
        for decay in args.model_ema_decay:
            ema_device = 'cpu' if args.model_ema_force_cpu else None
            model_ema = ModelEmaV2(model, decay=decay, device=ema_device)
            model_emas.append((model_ema, decay))
    
    return model_emas


def setup_distributed_training(model, use_amp, args):
    """Configure model for distributed training.
    
    Args:
        model: Model to wrap for distributed training
        use_amp: AMP type ('apex', 'native', or None)
        args: Command line arguments
        
    Returns:
        torch.nn.Module: Wrapped model for distributed training
    """
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], 
                              broadcast_buffers=not args.no_ddp_bb)
    
    return model


def setup_output_and_saver(args, model, optimizer, loss_scaler, model_emas, data_config, args_text):
    """Setup output directory and checkpoint saver.
    
    Args:
        args: Command line arguments
        model: Student model
        optimizer: Optimizer
        loss_scaler: AMP loss scaler
        model_emas: List of EMA models
        data_config: Data configuration
        args_text: Text representation of arguments
        
    Returns:
        tuple: (output_dir, saver, ema_savers)
    """
    output_dir = None
    saver = None
    ema_savers = None
    
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if args.eval_metric == 'loss' else False
        saver_dir = os.path.join(output_dir, 'checkpoint')
        os.makedirs(saver_dir, exist_ok=True)
        
        # Modified to only keep the best checkpoint (max_history=1)
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, amp_scaler=loss_scaler,
            checkpoint_dir=saver_dir, recovery_dir=saver_dir, decreasing=decreasing,
            max_history=1)  # Only keep the best checkpoint

        if model_emas is not None:
            ema_savers = []
            for ema, decay in model_emas:
                ema_saver_dir = os.path.join(output_dir, f'ema{decay}_checkpoint')
                os.makedirs(ema_saver_dir, exist_ok=True)
                # Also set max_history=1 for EMA models
                ema_saver = CheckpointSaver(
                    model=model, optimizer=optimizer, args=args, model_ema=ema, amp_scaler=loss_scaler,
                    checkpoint_dir=ema_saver_dir, recovery_dir=ema_saver_dir, decreasing=decreasing,
                    max_history=1)  # Only keep the best checkpoint
                ema_savers.append(ema_saver)

        # Save training arguments
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
            
    return output_dir, saver, ema_savers


def train_one_epoch(
        epoch, distiller, loader, optimizer, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_emas=None, mixup_fn=None):
    """Train for one epoch with progress tracking and logging.
    
    Args:
        epoch: Current epoch number
        distiller: Distiller model
        loader: Data loader
        optimizer: Optimizer
        args: Command line arguments
        lr_scheduler: Learning rate scheduler
        saver: Checkpoint saver
        output_dir: Output directory
        amp_autocast: AMP autocasting context
        loss_scaler: AMP loss scaler
        model_emas: List of EMA models
        mixup_fn: Mixup function
        
    Returns:
        dict: Training metrics
    """
    # Disable mixup after mixup_off_epoch
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    # Setup for second-order optimization if enabled
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    
    # Metrics tracking
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses_m_dict = defaultdict(AverageMeter)  # Track individual loss components

    # Set model to training mode
    distiller.train()

    # Start batch processing timer
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    
    # Process each batch
    for batch_idx, (input, target, *additional_input) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        # Move data to device and apply mixup if enabled
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            additional_input = [i.cuda() for i in additional_input]
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        # Forward pass with automatic mixed precision if enabled
        with amp_autocast():
            output, losses_dict = distiller(input, target, *additional_input, epoch=epoch)
            loss = sum(losses_dict.values())

        # Update loss metrics (non-distributed mode)
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            for k in losses_dict:
                losses_m_dict[k].update(losses_dict[k].item(), input.size(0))

        # Backward pass and optimization
        optimizer.zero_grad()
        if loss_scaler is not None:
            # With AMP loss scaling
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(distiller, exclude_head='agc' in args.clip_mode),
                create_graph=second_order
            )
        else:
            # Standard backward pass
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(distiller, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode
                )
            optimizer.step()

        # Update EMA models if enabled
        if model_emas is not None:
            for ema, _ in model_emas:
                if hasattr(distiller, 'module'):
                    ema.update(distiller.module.student)
                else:
                    ema.update(distiller.student)

        # Wait for CUDA operations to complete
        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        
        # Log progress at intervals or on last batch
        if last_batch or batch_idx % args.log_interval == 0:
            # Calculate learning rate
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            
            # Reduce metrics in distributed mode
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                reduced_loss_dict = {}
                for k in losses_dict:
                    reduced_loss_dict[k] = reduce_tensor(losses_dict[k].data, args.world_size)
                
                losses_m.update(reduced_loss.item(), input.size(0))
                for k in reduced_loss_dict:
                    losses_m_dict[k].update(reduced_loss_dict[k].item(), input.size(0))
            
            # Format individual loss components for logging
            if args.rank == 0:
                losses_infos = []
                for k, v in losses_m_dict.items():
                    info = f'{k.capitalize()}: {v.val:#.4g} ({v.avg:#.3g})'
                    losses_infos.append(info)
                losses_info = '  '.join(losses_infos)
                
                # Log batch progress
                _logger.info(
                    f'Train: {epoch} [{batch_idx:>4d}/{len(loader)} ({100. * batch_idx / last_idx:>3.0f}%)]  '
                    f'Loss: {losses_m.val:#.4g} ({losses_m.avg:#.3g})  '
                    f'{losses_info}  '
                    f'LR: {lr:.3e}  '
                    f'Time: {batch_time_m.val:.3f}s ({batch_time_m.avg:.3f}s)  '
                    f'Data: {data_time_m.val:.3f}s'
                )
                
                # Save input images for debugging if requested
                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, f'train-batch-{batch_idx}.jpg'),
                        padding=0,
                        normalize=True
                    )

        # Save recovery checkpoint at intervals if specified
        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        # Step LR scheduler based on iterations if specified
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        # Update timer for next batch
        end = time.time()

    # Synchronize lookahead optimizer if used
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    """Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        loader: Data loader
        loss_fn: Loss function
        args: Command line arguments
        amp_autocast: AMP autocasting context
        log_suffix: Log suffix for identifying different evaluations
        
    Returns:
        dict: Validation metrics
    """
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            # Forward pass with AMP autocasting if enabled
            with amp_autocast():
                output = model(input)
                
            # Handle tuple/list output
            if isinstance(output, (tuple, list)):
                output = output[0]

            # Handle test-time augmentation
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            # Calculate loss and accuracy
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # Reduce metrics in distributed training
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            # Wait for CUDA operations to complete
            torch.cuda.synchronize()

            # Update metrics
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            batch_time_m.update(time.time() - end)
            
            # Log progress at intervals
            if args.rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f}s ({batch_time_m.avg:.3f}s)  '
                    f'Loss: {losses_m.val:>7.4f} ({losses_m.avg:>6.4f})  '
                    f'Acc@1: {top1_m.val:>7.4f} ({top1_m.avg:>7.4f})  '
                    f'Acc@5: {top5_m.val:>7.4f} ({top5_m.avg:>7.4f})'
                )
            
            end = time.time()

    # Return metrics as an OrderedDict
    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    return metrics


def evaluate_ema_models(model_emas, ema_savers, loader_eval, validate_loss_fn, args, amp_autocast, eval_metric, epoch):
    """Evaluate EMA models if available.
    
    Args:
        model_emas: List of EMA models
        ema_savers: List of EMA checkpoint savers
        loader_eval: Validation data loader
        validate_loss_fn: Validation loss function
        args: Command line arguments
        amp_autocast: AMP autocasting context
        eval_metric: Evaluation metric name
        epoch: Current epoch number
    """
    if model_emas is None or args.model_ema_force_cpu or not args.rank == 0:
        return
        
    for j, ((ema, decay), ema_saver) in enumerate(zip(model_emas, ema_savers)):
        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            distribute_bn(ema, args.world_size, args.dist_bn == 'reduce')

        ema_eval_metrics = validate(
            ema.module, loader_eval, validate_loss_fn, args,
            amp_autocast=amp_autocast, log_suffix=f' (EMA {decay:.5f})'
        )

        if ema_saver is not None:
            save_metric = ema_eval_metrics[eval_metric]
            ema_saver.save_checkpoint(epoch, metric=save_metric)


def get_state_dict(model, use_ema=False):
    """Get the state dictionary from a model, handling DistributedDataParallel and other wrappers."""
    if hasattr(model, 'module'):
        # Handle DistributedDataParallel wrapper
        return model.module.state_dict()
    else:
        return model.state_dict()


def main():
    """Main function for training and evaluating models with knowledge distillation."""
    # Setup logging and parse arguments
    setup_default_logging(log_path='train.log')
    args, args_text = parse_args()
    
    # Configure environment (distributed, device, AMP selection, seed)
    use_amp = configure_environment(args)

    # Get distiller class from the distillation method
    Distiller = get_distiller(args.distiller)

    # Build student and teacher models
    model, teacher = build_model_and_teacher(args, Distiller)

    # Create datasets and loaders
    data_config, dataset_train, dataset_eval, loader_train, loader_eval, num_aug_splits, mixup_fn = create_datasets_and_loaders(args, model, Distiller)

    # Create loss function for training
    train_loss_fn = create_train_loss_fn(args, num_aug_splits)
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # Setup distiller with models and loss function
    distiller = Distiller(
        model, teacher=teacher, criterion=train_loss_fn, 
        args=args, num_data=len(dataset_train)
    )
    
    # Log parameter counts
    if args.rank == 0:
        student_params, extra_params = distiller.get_learnable_parameters()
        _logger.info(f'\n-------------------------------'
                     f'\nLearnable parameters'
                     f'\nStudent: {student_params / 1e6:.2f}M'
                     f'\nExtra: {extra_params / 1e6:.2f}M'
                     f'\n-------------------------------')

    # Move distiller to CUDA
    distiller = distiller.cuda()

    # Setup optimizer
    optimizer = create_optimizer_v2(distiller, **optimizer_kwargs(cfg=args))

    # Setup AMP
    amp_autocast, loss_scaler = setup_amp(use_amp, distiller, optimizer, args)

    # Setup model EMA
    model_emas = setup_model_ema(model, args)

    # Setup distributed training
    distiller = setup_distributed_training(distiller, use_amp, args)

    # Setup learning rate schedule
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0 if args.start_epoch is None else args.start_epoch
    
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)
        if args.rank == 0:
            _logger.info(f'LR scheduler stepped to epoch {start_epoch}')

    if args.rank == 0:
        _logger.info(f'Scheduled epochs: {num_epochs}')

    # Setup output directory and checkpoint saver
    output_dir, saver, ema_savers = setup_output_and_saver(
        args, model, optimizer, loss_scaler, model_emas, data_config, args_text)

    # Main training loop
    best_metric = None
    best_epoch = None
    
    try:
        # Initialize time predictor for ETA calculation
        tp = TimePredictor(num_epochs - start_epoch)
        
        for epoch in range(start_epoch, num_epochs):
            # Set epoch for distributed sampler
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)
            
            # Train one epoch
            train_metrics = train_one_epoch(
                epoch, distiller, loader_train, optimizer, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler,
                model_emas=model_emas, mixup_fn=mixup_fn
            )
            
            # Distribute batch norm stats for distributed training
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(distiller, args.world_size, args.dist_bn == 'reduce')
            
            # Determine if evaluation should be performed
            should_evaluate = (
                not args.speedtest and
                (epoch > int(args.eval_interval_end * args.epochs) or 
                 epoch % args.eval_interval == 0)
            )
            
            if should_evaluate:
                # Evaluate student model
                eval_metrics = validate(
                    model, loader_eval, validate_loss_fn, args, 
                    amp_autocast=amp_autocast
                )
                
                # Save checkpoint if improved
                if saver is not None:
                    save_metric = eval_metrics[args.eval_metric]
                    best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                
                # Evaluate EMA models if available
                evaluate_ema_models(
                    model_emas, ema_savers, loader_eval, validate_loss_fn,
                    args, amp_autocast, args.eval_metric, epoch
                )
                
                # Update summary
                if output_dir is not None and args.rank == 0:
                    update_summary(
                        epoch, train_metrics, eval_metrics, 
                        os.path.join(output_dir, 'summary.csv'),
                        write_header=best_metric is None
                    )
            
            # Save the latest model checkpoint (only for rank 0 in distributed training)
            if saver is not None and args.rank == 0:
                # Save latest checkpoint separately
                checkpoint_state = {
                    'epoch': epoch,
                    'arch': args.model,
                    'state_dict': get_state_dict(model, args.model_ema),
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                }
                if loss_scaler is not None:
                    checkpoint_state['amp_scaler'] = loss_scaler.state_dict()
                
                latest_path = os.path.join(saver.checkpoint_dir, 'latest.pth.tar')
                torch.save(checkpoint_state, latest_path)
                if args.rank == 0:
                    _logger.info(f'Latest checkpoint saved to {latest_path}')
                
                # Also save latest EMA models if available
                if model_emas is not None and not args.model_ema_force_cpu:
                    for j, ((ema, decay), ema_saver) in enumerate(zip(model_emas, ema_savers)):
                        ema_state = {
                            'epoch': epoch,
                            'arch': args.model,
                            'state_dict': get_state_dict(ema.module, False),
                            'optimizer': optimizer.state_dict(),
                            'args': args,
                        }
                        if loss_scaler is not None:
                            ema_state['amp_scaler'] = loss_scaler.state_dict()
                            
                        latest_ema_path = os.path.join(ema_saver.checkpoint_dir, 'latest.pth.tar')
                        torch.save(ema_state, latest_ema_path)
                        if args.rank == 0:
                            _logger.info(f'Latest EMA checkpoint (decay={decay:.5f}) saved to {latest_ema_path}')
            
            # Step LR scheduler
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, eval_metrics[args.eval_metric] if should_evaluate else None)
            
            # Update and display time prediction
            tp.update()
            if args.rank == 0:
                _logger.info(f'ETA: {tp.get_pred_text()}')
                _logger.info(f'Average epoch time: {np.mean(tp.time_list):.2f}s/epoch')

    except KeyboardInterrupt:
        _logger.info('Training interrupted by user')

    # Final reporting
    if best_metric is not None and args.rank == 0:
        _logger.info(f'*** Best metric: {best_metric:.4f} (epoch {best_epoch})')
    
    # Move log file to output directory
    if args.rank == 0 and output_dir:
        os.system(f'mv train.log {output_dir}')


if __name__ == '__main__':
    main()
