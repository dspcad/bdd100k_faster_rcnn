import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils, copy


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    cnt = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #targets = [{k: v for k, v in t.items()} for t in targets]

        print(f"debug: img shape: {len(images)}   {images[0].shape}")

        #for t in targets:
        #    print(f"debug:      {t}")
        #targets = [{k: v.to(device) for k, v in t} for list_target in targets]

        loss_dict = model(images, targets)

        #if cnt==10:
        #    break;
        # print("======== beg ========")
        # print(loss_dict)
        # print(images)
        # print(targets)
        # print("======== end ========")

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        cnt+=1


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)

    for i in range(6,16):
        coco_evaluator = CocoEvaluator(coco, iou_types)
        coco_evaluator.coco_eval["bbox"].params.catIds = [i]

        cnt = 0
        for image, targets in metric_logger.log_every(data_loader, 100, header):
            image = list(img.to(device) for img in image)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

            if cnt == 10:
                break
            cnt +=1

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print(f"Averaged stats {i}:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        #cat_id2name = {1: 'pedestrian', 2: 'rider', 3: 'car', 4: 'truck', 5 : 'bus', 6 : 'motorcycle', 7 : 'bicycle', 8 : 'traffic light', 9 : 'traffic sign', 10 : 'green', 11 : 'yellow', 12 : 'red', 13 : 'do not enter', 14 : 'stop sign', 15 : 'speed limit'}
        #for catId in coco_evaluator.coco_gt.getCatIds():
        #    print(f"catId: {catId}")
        #    print(f"catId: {cat_id2name[catId]} ({catId})")
        #    cocoEval = copy.deepcopy(coco_evaluator)
        #    cocoEval.coco_eval["bbox"].params.catIds = [catId]
        #    cocoEval.evaluate()
        #    cocoEval.accumulate()
        #    cocoEval.summarize()
    
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
    return coco_evaluator
