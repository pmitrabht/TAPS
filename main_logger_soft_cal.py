import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import shutil
import multiprocessing
import csv

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torchvision
import numpy as np

# local imports
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from ops.dataset import TSNDataSet
from ops.models_gate import TSN_Gate
from ops.transforms import *
from ops import dataset_config
from ops.utils import (
    AverageMeter, accuracy, cal_map, Recorder,
    init_gflops_table, compute_gflops_by_mask, compute_gflops_dynstc,
    adjust_learning_rate, ExpAnnealing
)
from opts import parser
from ops.my_logger import Logger
import common
from os.path import join as ospj

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from shutil import copyfile
from regularization import Loss
import pickle

# calibration loss
from calibration_loss import SoftBinnedECE


# ----------------------------
# Helpers: CSV logger + keep/skip summarizer
# ----------------------------
class MetricsLogger:
    def __init__(self, out_csv_path):
        self.out_csv_path = out_csv_path
        if not os.path.exists(out_csv_path):
            os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
            with open(out_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch","split",
                    "loss_cls","loss_prune","loss_bshape","loss_calib","loss_total",
                    "acc_top1","acc_top5",
                    "avg_keep_overall","avg_skip_overall",
                    "gflops_upb","gflops_real"
                ])

    def write_row(self, **kw):
        def _num(x):
            try: return float(x)
            except: return x
        with open(self.out_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                kw.get("epoch"), kw.get("split"),
                _num(kw.get("loss_cls")), _num(kw.get("loss_prune")), _num(kw.get("loss_bshape")), _num(kw.get("loss_calib")), _num(kw.get("loss_total")),
                _num(kw.get("acc_top1")), _num(kw.get("acc_top5")),
                _num(kw.get("avg_keep_overall")), _num(kw.get("avg_skip_overall")),
                _num(kw.get("gflops_upb")), _num(kw.get("gflops_real")),
            ])


def summarize_keep_soft(soft_mask_stack_list, pn_num_outputs):
    """
    Summarize soft keep/skip across layers from policy outputs.

    soft_mask_stack_list: list[layer] -> Tensor[..., K] where K=#policy outputs
      If pn_num_outputs==1: [KEEP]
      If pn_num_outputs==2: [SKIP, KEEP]
      If pn_num_outputs>=3: [SKIP, REUSE, KEEP, (maybe MORE)]
    Returns:
      per_layer_keep (list of floats), per_layer_skip (list of floats),
      overall_keep (float), overall_skip (float)
    """
    per_layer_keep, per_layer_skip = [], []
    keeps, skips, cnt = 0.0, 0.0, 0

    for m in soft_mask_stack_list:
        if pn_num_outputs == 1:
            keep_prob = m[..., 0].mean().item()
            skip_prob = 1.0 - keep_prob
        elif pn_num_outputs == 2:
            skip_prob = m[..., 0].mean().item()
            keep_prob = m[..., 1].mean().item()
        else:  # >=3
            skip_prob = m[..., 0].mean().item()
            keep_prob = m[..., 2].mean().item()
        per_layer_keep.append(keep_prob)
        per_layer_skip.append(skip_prob)
        keeps += keep_prob
        skips += skip_prob
        cnt += 1

    overall_keep = keeps / max(1, cnt)
    overall_skip = skips / max(1, cnt)
    return per_layer_keep, per_layer_skip, overall_keep, overall_skip


# ----------------------------
# Core
# ----------------------------
def inner_main(argv):
    args = parser.parse_args()
    common.set_manual_data_path(args.data_path, args.exps_path)

    test_mode = (args.test_from != "")

    set_random_seed(args.random_seed, args)

    (
        args.num_class, args.train_list, args.val_list, args.root_path,
        prefix, args.train_folder_suffix, args.val_folder_suffix
    ) = dataset_config.return_dataset(args.dataset, args.data_path)

    if args.gpus is not None:
        print("Use GPU: {} for training".format(args.gpus))

    logger = Logger()
    sys.stdout = logger

    exp_header_list_full = np.asarray([f for f in os.listdir(common.EXPS_PATH) if f.startswith("g")])
    exp_header_list = np.asarray([f[f.find("_") + 1:] for f in os.listdir(common.EXPS_PATH) if f.startswith("g")])
    resume_recorders_pkl = ''
    if args.exp_header in exp_header_list:
        earlier_exp_folders = exp_header_list_full[np.where(exp_header_list == args.exp_header)]
        earlier_exp_folders = np.sort(earlier_exp_folders)
        num_of_eralier_folders = len(earlier_exp_folders)
        for early_folder_ind in range(num_of_eralier_folders - 1, -1, -1):
            exp_full_name = earlier_exp_folders[early_folder_ind]
            exp_path = os.path.join(common.EXPS_PATH, exp_full_name)

            if len(os.listdir(os.path.join(exp_path, "models"))) == 0:
                shutil.rmtree(exp_path)
            else:
                args.base_pretrained_from = f"{exp_full_name}/models/ckpt.latest.pth.tar"
                log_fn = [f for f in os.listdir(exp_path) if "log" in f][0]
                with open(os.path.join(exp_path, log_fn), "r") as last_exp:
                    lines_string = last_exp.read()
                    last_epoch_start_ind = lines_string.rfind("Epoch:[") + len("Epoch:[")
                    last_epoch_end_ind = last_epoch_start_ind + lines_string[last_epoch_start_ind:].find("]")
                    last_epoch = int(lines_string[last_epoch_start_ind:last_epoch_end_ind])
                args.start_epoch = last_epoch
                last_saved_recorders_pkl = os.path.join(exp_path, 'val_records.pkl')
                if os.path.exists(last_saved_recorders_pkl):
                    resume_recorders_pkl = last_saved_recorders_pkl
                break

    model = TSN_Gate(args=args)

    use_DCP = not args.disable_channelwise_masking
    base_model_gflops, gflops_list, g_meta = init_gflops_table(model, args)
    if test_mode:
        args.warmup_hard_gates_disable = False
        args.no_soft_mask_until_epoch = -1

    policies = model.get_optim_policies()
    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if torch.cuda.is_available() and args.gpus is not None and len(args.gpus) > 0:
        print(f" Using GPUs: {args.gpus}")
        model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    else:
        print("⚠️ CUDA not available or --gpus not specified. Using CPU mode.")

    if test_mode or args.base_pretrained_from != "":
        the_model_path = args.base_pretrained_from
        if test_mode:
            if "pth.tar" not in args.test_from:
                the_model_path = ospj(args.test_from, "models", "ckpt.best.pth.tar")
            else:
                the_model_path = args.test_from
        the_model_path = common.EXPS_PATH + "/" + the_model_path
        sd = torch.load(the_model_path, map_location=torch.device('cpu'))['state_dict']
        model_dict = model.state_dict()
        model_dict.update(sd)
        model.load_state_dict(model_dict, strict=False)

    cudnn.benchmark = True

    train_loader, val_loader = get_data_loaders(model, prefix, args)
    criterion = torch.nn.CrossEntropyLoss().cuda() if args.gpus is not None else torch.nn.CrossEntropyLoss().cpu()

    loss_device = torch.device('cpu') if args.gpus is None else torch.device('cuda:0')
    my_criterion = Loss(args.bs_pdf_alpha, args.bs_pdf_beta, args.bs_beta_cdf_res, loss_device)

    # Calibration loss (optional)
    calib_loss_fn = None
    if getattr(args, "calib_enable", False) and getattr(args, "calib_beta", 0.0) > 0.0:
        calib_loss_fn = SoftBinnedECE(
        num_bins=args.calib_bins,
        temperature=args.calib_kernel_width,  
        use_max_conf=getattr(args, "calib_use_max_conf", True),
        p_norm=args.calib_pnorm
    )

        if args.gpus is not None:
            calib_loss_fn = calib_loss_fn.cuda()

    exp_full_path = setup_log_directory(args.exp_header, test_mode, args, logger)

    # CSV metrics
    metrics_csv_path = os.path.join(exp_full_path, "epoch_metrics.csv")
    metrics_logger = MetricsLogger(metrics_csv_path)

    if not test_mode:
        with open(os.path.join(exp_full_path, 'args.txt'), 'w') as f:
            f.write(str(args))

    init_empty_recorders = True
    if len(resume_recorders_pkl) > 0:
        try:
            with open(resume_recorders_pkl, 'rb') as f:
                [map_record, mmap_record, prec_record, prec5_record, gflops_record] = pickle.load(f)
            init_empty_recorders = False
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            print('Could not load recorders data for resumed training')
    if init_empty_recorders:
        map_record, mmap_record, prec_record, prec5_record, gflops_record = get_recorders(5)

    best_train_usage_str = None
    best_val_usage_str = None

    for epoch in range(args.start_epoch, args.epochs):
        if use_DCP and epoch < args.eff_loss_after:
            args.disable_channelwise_masking = True
        elif use_DCP and epoch >= args.eff_loss_after:
            args.disable_channelwise_masking = False

        # train
        if not args.skip_training and not test_mode:
            set_random_seed(args.train_random_seed + epoch, args)
            adjust_learning_rate(optimizer, epoch, -1, -1, args.lr_type, args.lr_steps, args)
            train_usage_str = train(
                train_loader, model, criterion, optimizer, epoch,
                base_model_gflops, gflops_list, g_meta, my_criterion,
                args, metrics_logger=metrics_logger, calib_loss_fn=calib_loss_fn
            )
        else:
            train_usage_str = "(Eval mode)"

        torch.cuda.empty_cache()

        # validation
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            set_random_seed(args.random_seed, args)
            mAP, mmAP, prec1, prec5, val_usage_str, gflops_per_clip = validate(
                val_loader, model, criterion, epoch, base_model_gflops, gflops_list, g_meta, exp_full_path,
                my_criterion, args, metrics_logger=metrics_logger, calib_loss_fn=calib_loss_fn
            )

            map_record.update(mAP)
            mmap_record.update(mmAP)
            prec_record.update(prec1)
            prec5_record.update(prec5)
            gflops_record.update(float(gflops_per_clip.cpu().numpy()))
            with open(os.path.join(exp_full_path, 'val_records.pkl'), 'wb') as f:
                pickle.dump([map_record, mmap_record, prec_record, prec5_record, gflops_record], f)

            if prec_record.is_current_best():
                best_train_usage_str = train_usage_str if not args.skip_training else "(Eval Mode)"
                best_val_usage_str = val_usage_str

            print('Best Prec@1: %.3f (epoch=%d) w. Prec@5: %.3f' % (
                prec_record.best_val, prec_record.best_at,
                prec5_record.at(prec_record.best_at)))

            if test_mode or args.skip_training:
                break
            else:
                saved_things = {'state_dict': model.state_dict()}
                save_checkpoint(saved_things, prec_record.is_current_best(), False, exp_full_path, "ckpt.best")
                save_checkpoint(saved_things, True, False, exp_full_path, "ckpt.latest")

                if epoch in args.backup_epoch_list:
                    save_checkpoint(None, False, True, exp_full_path, str(epoch))
                torch.cuda.empty_cache()

    if test_mode:
        if args.skip_log == False:
            os.rename(logger._log_path, ospj(logger._log_dir_name, logger._log_file_name[:-4] +
                                             "_mm_%.2f_a_%.2f_f.txt" % (mmap_record.best_val, prec_record.best_val)))
    else:
        print("Best train usage:%s\nBest val usage:%s" % (best_train_usage_str, best_val_usage_str))


def build_dataflow(dataset, is_train, batch_size, workers, not_pin_memory):
    workers = min(workers, multiprocessing.cpu_count())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                                              num_workers=workers, pin_memory=not not_pin_memory, sampler=None,
                                              drop_last=is_train)
    return data_loader


def get_data_loaders(model, prefix, args):
    # Access augmentation function safely
    get_aug_flip = model.get_augmentation(flip=True) if not isinstance(model, torch.nn.DataParallel) else model.module.get_augmentation(flip=True)
    get_aug_nofl = model.get_augmentation(flip=False) if not isinstance(model, torch.nn.DataParallel) else model.module.get_augmentation(flip=False)

    input_mean = model.input_mean if not isinstance(model, torch.nn.DataParallel) else model.module.input_mean
    input_std  = model.input_std  if not isinstance(model, torch.nn.DataParallel) else model.module.input_std

    train_transform_flip = torchvision.transforms.Compose([
        get_aug_flip,
        Stack(roll=("BNInc" in args.arch)),
        ToTorchFormatTensor(div=("BNInc" not in args.arch)),
        GroupNormalize(input_mean, input_std),
    ])

    train_transform_nofl = torchvision.transforms.Compose([
        get_aug_nofl,
        Stack(roll=("BNInc" in args.arch)),
        ToTorchFormatTensor(div=("BNInc" not in args.arch)),
        GroupNormalize(input_mean, input_std),
    ])

    scale_size = int(model.scale_size) if not isinstance(model, torch.nn.DataParallel) else int(model.module.scale_size)
    crop_size = model.crop_size if not isinstance(model, torch.nn.DataParallel) else model.module.crop_size
    input_mean = model.input_mean if not isinstance(model, torch.nn.DataParallel) else model.module.input_mean
    input_std = model.input_std if not isinstance(model, torch.nn.DataParallel) else model.module.input_std

    val_transform = torchvision.transforms.Compose([
        GroupScale(scale_size),
        GroupCenterCrop(crop_size),
        Stack(roll=("BNInc" in args.arch)),
        ToTorchFormatTensor(div=("BNInc" not in args.arch)),
        GroupNormalize(input_mean, input_std),
    ])

    train_dataset = TSNDataSet(args.root_path, args.train_list,
                               num_segments=args.num_segments,
                               image_tmpl=prefix,
                               transform=(train_transform_flip, train_transform_nofl),
                               dense_sample=args.dense_sample,
                               dataset=args.dataset,
                               filelist_suffix=args.filelist_suffix,
                               folder_suffix=args.train_folder_suffix,
                               save_meta=args.save_meta,
                               always_flip=args.always_flip,
                               conditional_flip=args.conditional_flip,
                               adaptive_flip=args.adaptive_flip)

    val_dataset = TSNDataSet(args.root_path, args.val_list,
                             num_segments=args.num_segments,
                             image_tmpl=prefix,
                             random_shift=False,
                             transform=(val_transform, val_transform),
                             dense_sample=args.dense_sample,
                             dataset=args.dataset,
                             filelist_suffix=args.filelist_suffix,
                             folder_suffix=args.val_folder_suffix,
                             save_meta=args.save_meta)

    train_loader = build_dataflow(train_dataset, True, args.batch_size, args.workers, args.not_pin_memory)
    val_loader = build_dataflow(val_dataset, False, args.batch_size, args.workers, args.not_pin_memory)

    return train_loader, val_loader


def _normalize_cal_loss(calib_loss_fn, output, target):
    """
    Defensive: normalize whatever the calibration fn returns into a scalar tensor on correct device/dtype.
    """
    cal_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)
    if calib_loss_fn is None:
        return cal_loss
    cal_out = calib_loss_fn(output, target)  # expects logits and class indices
    if isinstance(cal_out, dict):
        # try common keys
        for k in ("loss", "ece", "sb_ece", "value"):
            if k in cal_out:
                cal_out = cal_out[k]
                break
        else:
            cal_out = 0.0
    if isinstance(cal_out, (tuple, list)) and len(cal_out) > 0:
        cal_out = cal_out[0]
    if not torch.is_tensor(cal_out):
        cal_out = torch.tensor(float(cal_out), device=output.device, dtype=output.dtype)
    return cal_out.mean()

def save_uncertainty_outputs(all_results, all_targets, exp_full_path, temperature=1.0):
    import torch.nn.functional as F
    import pandas as pd
    import os

    all_results_tensor = torch.cat(all_results, dim=0)  # (N, C)
    all_targets_tensor = torch.cat(all_targets, dim=0)[:, 0]  # (N,)

    # Apply temperature scaling
    logits_scaled = all_results_tensor / temperature
    probs_scaled = F.softmax(logits_scaled, dim=1)

    # Extract confidence, p_true, and predicted labels
    confidence, pred_label = probs_scaled.max(dim=1)
    p_true = probs_scaled[torch.arange(len(probs_scaled)), all_targets_tensor]
    calib_gap = (confidence - p_true).abs()

    # Convert logits to string (for each sample)
    logits_str = [",".join([f"{x:.4f}" for x in row.cpu().numpy()]) for row in all_results_tensor]

    # Build DataFrame
    df = pd.DataFrame({
        "pred_label": pred_label.cpu().numpy(),
        "true_label": all_targets_tensor.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
        "p_true": p_true.cpu().numpy(),
        "calib_gap": calib_gap.cpu().numpy(),
        "temperature": [temperature] * len(pred_label),
        "logits": logits_str
    })

    # Save to CSV
    output_path = os.path.join(exp_full_path, "uncertainty_outputs.csv")
    df.to_csv(output_path, index=False)
    print(f"[Uncertainty] Saved prediction/confidence/logits to {output_path}")

    return df





def train(train_loader, model, criterion, optimizer, epoch,
          base_model_gflops, gflops_list, g_meta, my_criterion,
          args, metrics_logger=None, calib_loss_fn=None):
    # meters: add calibration (9 total)
    batch_time, data_time, closses, rlosses, bslosses, callosses, losses, top1, top5 = get_average_meters(9)

    mask_stack_list_list = [0 for _ in gflops_list]
    upb_batch_gflops_list = []
    real_batch_gflops_list = []

    # accumulators for soft keep/skip
    soft_keep_sums, soft_skip_sums = [], []
    num_layers_seen = None

    tau = get_current_temperature(epoch, args.exp_decay, args.init_tau, args.exp_decay_factor)

    # train mode
    if isinstance(model, torch.nn.DataParallel):
        model.module.partialBN(not args.no_partialbn)
    else:
        model.partialBN(not args.no_partialbn)
    model.train()

    end = time.time()
    print("#%s# lr:%.6f\ttau:%.4f" % (args.exp_header, optimizer.param_groups[-1]['lr'] * 0.1, tau))

    for i, input_tuple in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.warmup_epochs > 0:
            adjust_learning_rate(optimizer, epoch, len(train_loader), i, "linear", None, args)

        # input and target
        batchsize = input_tuple[0].size(0)
        if args.gpus is None:
            input_var_list = [input_item.cpu() for input_item in input_tuple[:-1]]
            target = input_tuple[-1].cpu()
        else:
            input_var_list = [input_item.cuda(non_blocking=True) for input_item in input_tuple[:-1]]
            target = input_tuple[-1].cuda(non_blocking=True)

        # forward
        output, mask_stack_list, _, _, dyn_outputs, soft_mask_stack_list = model(
            input=input_var_list, tau=tau, is_training=True,
            curr_step=epoch * len(train_loader) + i,
            targets=target, epoch=epoch, first_batch=(i == 0)
        )

        upb_gflops_dynstc, real_gflops_dynstc = compute_gflops_dynstc(
            dyn_outputs, args.num_segments, args.batch_size, base_model_gflops
        )

        # base losses
        closs, rloss, bs_loss = my_criterion(
            output, target[:, 0],
            dyn_outputs['flops_real'], dyn_outputs['flops_ori'][0],
            args.batch_size, args.den_target, args.sparsity_lambda,
            soft_mask_stack_list, epoch, args
        )

        # calibration loss (defensive normalize)
        cal_loss = _normalize_cal_loss(calib_loss_fn, output, target[:, 0])
        beta = getattr(args, "calib_beta", 0.0)

        # total loss (keep calibration!)
        dynstc_loss = closs.mean() + rloss.mean() + bs_loss.mean() + beta * cal_loss

        # record FLOPs
        upb_batch_gflops_list.append(upb_gflops_dynstc.detach())
        real_batch_gflops_list.append(real_gflops_dynstc.detach())

        # acc
        prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

        # meters
        closses.update(closs.mean().item(), batchsize)
        rlosses.update(rloss.mean().item(), batchsize)
        bslosses.update(bs_loss.mean().item(), batchsize)
        callosses.update(cal_loss.item(), batchsize)
        losses.update(dynstc_loss.item(), batchsize)

        top1.update(prec1.item(), batchsize)
        top5.update(prec5.item(), batchsize)

        # backward
        dynstc_loss.backward()
        if args.clip_gradient is not None:
            clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()
        optimizer.zero_grad()

        # gather soft keep/skip
        if num_layers_seen is None:
            num_layers_seen = len(soft_mask_stack_list)
            soft_keep_sums = [0.0 for _ in range(num_layers_seen)]
            soft_skip_sums = [0.0 for _ in range(num_layers_seen)]
        pl_keep, pl_skip, _, _ = summarize_keep_soft(soft_mask_stack_list, args.pn_num_outputs)
        for li in range(num_layers_seen):
            soft_keep_sums[li] += pl_keep[li]
            soft_skip_sums[li] += pl_skip[li]

        # gather masks
        for layer_i, mask_stack in enumerate(mask_stack_list):
            mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)

        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        if i == 0:  # one-time debug per epoch
            print(f"[calib] epoch={epoch} cal_loss={cal_loss.item():.6f} beta={beta}")

        if i % args.print_freq == 0:
            print_output = ('Epoch:[{0:02d}][{1:03d}/{2:03d}] lr {3:.6f} '
                            'Time {batch_time.val:.3f}({batch_time.avg:.3f}) '
                            '{data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), optimizer.param_groups[-1]['lr'] * 0.1,
                batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5))
            print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Cls',  loss=closses)
            print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Spar', loss=rlosses)
            print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Bshp', loss=bslosses)
            print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Cal',  loss=callosses)
            print(print_output)

    upb_batch_gflops = torch.mean(torch.stack(upb_batch_gflops_list))
    real_batch_gflops = torch.mean(torch.stack(real_batch_gflops_list))

    # epoch-level averages for keep/skip
    if num_layers_seen is not None and len(train_loader) > 0:
        per_layer_keep_epoch = [v / len(train_loader) for v in soft_keep_sums]
        per_layer_skip_epoch = [v / len(train_loader) for v in soft_skip_sums]
        avg_keep_overall = float(np.mean(per_layer_keep_epoch))
        avg_skip_overall = float(np.mean(per_layer_skip_epoch))
    else:
        per_layer_keep_epoch, per_layer_skip_epoch = [], []
        avg_keep_overall, avg_skip_overall = None, None

    print("Train keep/skip (soft, per layer):",
          ["L{}:{:.3f}/{:.3f}".format(i, k, s) for i, (k, s) in enumerate(zip(per_layer_keep_epoch, per_layer_skip_epoch))])

    # CSV logging
    if metrics_logger is not None:
        metrics_logger.write_row(
            epoch=epoch, split="train",
            loss_cls=closses.avg, loss_prune=rlosses.avg, loss_bshape=bslosses.avg,
            loss_calib=callosses.avg, loss_total=losses.avg,
            acc_top1=top1.avg, acc_top5=top5.avg,
            avg_keep_overall=avg_keep_overall, avg_skip_overall=avg_skip_overall,
            gflops_upb=upb_batch_gflops.item(), gflops_real=real_batch_gflops.item()
        )

    usage_str = get_policy_usage_str(upb_batch_gflops, real_batch_gflops)
    print(usage_str)
    return usage_str


def validate(val_loader, model, criterion, epoch, base_model_gflops, gflops_list, g_meta, exp_full_path, my_criterion,
             args, metrics_logger=None, calib_loss_fn=None):
    # meters: add calibration (8 total)
    batch_time, closses, rlosses, bslosses, callosses, losses, top1, top5 = get_average_meters(8)
    all_results = []
    all_targets = []

    tau = get_current_temperature(epoch, args.exp_decay, args.init_tau, args.exp_decay_factor)

    mask_stack_list_list = [0 for _ in gflops_list]
    mask_stack_list_skipping = [0 for _ in gflops_list]
    upb_batch_gflops_list = []
    real_batch_gflops_list = []

    model.eval()

    end = time.time()
    with torch.no_grad():
        mask_t1_sum_S, mask_t1_sum_R, mask_t1_sum_K, mask_t1_sum_M, mask_s_sum = [], [], [], [], []
        mask_t1_S_bins, mask_t1_R_bins, mask_t1_K_bins = [], [], []
        for _ in range(len(gflops_list)):
            mask_t1_S_bins.append([]); mask_t1_K_bins.append([]); mask_t1_R_bins.append([])
            mask_t1_sum_S.append([]);  mask_t1_sum_R.append([]);  mask_t1_sum_K.append([]); mask_t1_sum_M.append([])
            mask_s_sum.append([])

        for i, input_tuple in enumerate(val_loader):
            batchsize = input_tuple[0].size(0)

            if args.gpus is None:
                input_data = input_tuple[0].cpu(); target = input_tuple[-1].cpu()
            else:
                input_data = input_tuple[0].cuda(non_blocking=True); target = input_tuple[-1].cuda(non_blocking=True)

            if 'dynstc' in args.arch:
                output, mask_stack_list, mask2_stack_list, gate_meta, dyn_outputs, soft_mask_stack_list = \
                    model(input=[input_data], tau=tau, is_training=False, curr_step=0, targets=target, epoch=epoch, first_batch=i == 0)
            else:
                output, mask_stack_list, mask2_stack_list, gate_meta = \
                    model(input=[input_data], tau=tau, is_training=False, curr_step=0)

            # masks stats
            for num_layer in range(len(mask_t1_sum_S)):
                if args.pn_num_outputs == 1:
                    mask_t1_sum_S[num_layer].append(1 - torch.mean(mask_stack_list[num_layer][:, :, :, 0]).item())
                else:
                    mask_t1_sum_S[num_layer].append(torch.mean(mask_stack_list[num_layer][:, :, :, 0]).item())
                if args.pn_num_outputs == 2:
                    mask_t1_sum_K[num_layer].append(torch.mean(mask_stack_list[num_layer][:, :, :, 1]).item())
                elif args.pn_num_outputs == 1:
                    mask_t1_sum_K[num_layer].append(torch.mean(mask_stack_list[num_layer][:, :, :, 0]).item())
                if args.pn_num_outputs > 2:
                    mask_t1_sum_R[num_layer].append(torch.mean(mask_stack_list[num_layer][:, :, :, 1]).item())
                    mask_t1_sum_K[num_layer].append(torch.mean(mask_stack_list[num_layer][:, :, :, 2]).item())
                if args.pn_num_outputs == 4:
                    mask_t1_sum_M[num_layer].append(torch.mean(mask_stack_list[num_layer][:, :, :, 3]).item())

            # FLOPs
            upb_gflops_dynstc, real_gflops_dynstc = compute_gflops_dynstc(
                dyn_outputs, args.num_segments, args.batch_size, base_model_gflops
            )

            # losses
            closs, rloss, bs_loss = my_criterion(
                output, target[:, 0],
                dyn_outputs['flops_real'], dyn_outputs['flops_ori'][0],
                args.batch_size, args.den_target, args.sparsity_lambda,
                soft_mask_stack_list, epoch, args
            )

            # calibration loss (defensive normalize)
            cal_loss = _normalize_cal_loss(calib_loss_fn, output, target[:, 0])
            beta = getattr(args, "calib_beta", 0.0)

            dynstc_loss = closs.mean() + rloss.mean() + bs_loss.mean() + beta * cal_loss

            upb_batch_gflops_list.append(upb_gflops_dynstc)
            real_batch_gflops_list.append(real_gflops_dynstc)

            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

            all_results.append(output); all_targets.append(target)

            # meters
            closses.update(closs.mean().item(), batchsize)
            rlosses.update(rloss.mean().item(), batchsize)
            bslosses.update(bs_loss.mean().item(), batchsize)
            callosses.update(cal_loss.item(), batchsize)
            losses.update(dynstc_loss.item(), batchsize)
            top1.update(prec1.item(), batchsize); top5.update(prec5.item(), batchsize)

            # gather masks
            for layer_i, mask_stack in enumerate(mask_stack_list):
                mask_stack_list_list[layer_i] += torch.sum(mask_stack.detach(), dim=0)
                if args.pn_num_outputs == 1:
                    mask_stack_list_skipping[layer_i] += torch.sum(1 - mask_stack.detach(), dim=0)

            # timing
            batch_time.update(time.time() - end); end = time.time()

            if i % args.print_freq == 0:
                print_output = ('Test: [{0:03d}/{1:03d}] '
                                'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                                'Loss{loss.val:.4f}({loss.avg:.4f})'
                                'Prec@1 {top1.val:.3f}({top1.avg:.3f}) '
                                'Prec@5 {top5.val:.3f}({top5.avg:.3f})\t'
                                .format(i, len(val_loader), batch_time=batch_time,
                                        loss=losses, top1=top1, top5=top5))
                print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Cls',  loss=closses)
                print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Spar', loss=rlosses)
                print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Bshp', loss=bslosses)
                print_output += ' {header:s} ({loss.avg:.3f})'.format(header='Cal',  loss=callosses)
                print(print_output)

    upb_batch_gflops = torch.mean(torch.stack(upb_batch_gflops_list))
    real_batch_gflops = torch.mean(torch.stack(real_batch_gflops_list))

    mAP, _ = cal_map(torch.cat(all_results, 0).cpu(),
                     torch.cat(all_targets, 0)[:, 0:1].cpu())  # single-label mAP
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu())  # multi-label mAP

    print('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses))
    print('==========================================================================================')
    print('policy networks stats:')
    for num_layer in range(len(mask_t1_sum_S)):
        if args.pn_num_outputs == 2 or args.pn_num_outputs == 1:
            print(f"{num_layer}: Temporal Skip - {np.around(np.mean(mask_t1_sum_S[num_layer]), 3)}, "
                  f"Temporal Keep - {np.around(np.mean(mask_t1_sum_K[num_layer]), 3)}")
        elif args.pn_num_outputs == 3:
            print(f"{num_layer}: Temporal Skip - {np.around(np.mean(mask_t1_sum_S[num_layer]), 3)}, "
                  f"Temporal Reuse - {np.around(np.mean(mask_t1_sum_R[num_layer]), 3)}, "
                  f"Temporal Keep - {np.around(np.mean(mask_t1_sum_K[num_layer]), 3)}")
    print('==========================================================================================')

    # per-epoch keep/skip summary + CSV
    per_layer_keep_epoch, per_layer_skip_epoch = [], []
    for num_layer in range(len(mask_t1_sum_S)):
        k = float(np.mean(mask_t1_sum_K[num_layer])) if len(mask_t1_sum_K[num_layer]) else 0.0
        s = float(np.mean(mask_t1_sum_S[num_layer])) if len(mask_t1_sum_S[num_layer]) else 0.0
        per_layer_keep_epoch.append(k); per_layer_skip_epoch.append(s)
    avg_keep_overall = float(np.mean(per_layer_keep_epoch)) if len(per_layer_keep_epoch) > 0 else None
    avg_skip_overall = float(np.mean(per_layer_skip_epoch)) if len(per_layer_skip_epoch) > 0 else None

    print("Val keep/skip (soft, per layer):",
          ["L{}:{:.3f}/{:.3f}".format(i, k, s) for i, (k, s) in enumerate(zip(per_layer_keep_epoch, per_layer_skip_epoch))])

    if metrics_logger is not None:
        metrics_logger.write_row(
            epoch=epoch, split="val",
            loss_cls=closses.avg, loss_prune=rlosses.avg, loss_bshape=bslosses.avg, loss_calib=callosses.avg,
            loss_total=losses.avg,
            acc_top1=top1.avg, acc_top5=top5.avg,
            avg_keep_overall=avg_keep_overall, avg_skip_overall=avg_skip_overall,
            gflops_upb=torch.mean(torch.stack(upb_batch_gflops_list)).item() if len(upb_batch_gflops_list) else None,
            gflops_real=torch.mean(torch.stack(real_batch_gflops_list)).item() if len(real_batch_gflops_list) else None
        )

    usage_str = get_policy_usage_str(upb_batch_gflops, real_batch_gflops)
    print(usage_str)

    flops_per_clip = real_batch_gflops * args.num_segments
    
    temperature = getattr(args, "calib_kernel_width", 1.0)
    save_uncertainty_outputs(all_results, all_targets, exp_full_path, temperature=temperature)

    
    return mAP, mmAP, top1.avg, top5.avg, usage_str, flops_per_clip


# ----------------------------
# Misc utils
# ----------------------------
def set_random_seed(the_seed, args):
    np.random.seed(the_seed)
    torch.manual_seed(the_seed)


def compute_exp_decay_tau(epoch, init_tau, exp_decay_factor):
    return init_tau * np.exp(exp_decay_factor * epoch)


def get_current_temperature(num_epoch, exp_decay, init_tau, exp_decay_factor):
    return compute_exp_decay_tau(num_epoch, init_tau, exp_decay_factor) if exp_decay else init_tau


def get_policy_usage_str(upb_gflops, real_gflops):
    return "Equivalent GFLOPS: upb: %.4f   real: %.4f" % (upb_gflops.item(), real_gflops.item())


def get_recorders(number):
    return [Recorder() for _ in range(number)]


def get_average_meters(number):
    return [AverageMeter() for _ in range(number)]


def save_checkpoint(state, is_best, shall_backup, exp_full_path, decorator):
    if is_best:
        torch.save(state, '%s/models/%s.pth.tar' % (exp_full_path, decorator))
    if shall_backup:
        copyfile("%s/models/ckpt.best.pth.tar" % exp_full_path,
                 "%s/models/oldbest.%s.pth.tar" % (exp_full_path, decorator))


def setup_log_directory(exp_header, test_mode, args, logger):
    exp_full_name = "g%s_%s" % (logger._timestr, exp_header)
    exp_full_path = ospj(common.EXPS_PATH, exp_full_name)
    os.makedirs(exp_full_path)
    os.makedirs(ospj(exp_full_path, "models"))
    if not args.skip_log:
        logger.create_log(exp_full_path, test_mode, args.num_segments, args.batch_size, args.top_k)
    return exp_full_path


def main(argv):
    t0 = time.time()
    inner_main(argv)
    print("Finished in %.4f seconds\n" % (time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1:])
