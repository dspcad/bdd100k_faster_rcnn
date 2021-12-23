import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch, utils
from engine import train_one_epoch, evaluate
#import torchvision.transforms as transforms
import transforms as transforms
from PIL import Image
import cv2


# from torchvision.datasets.coco import CocoDetection
from coco_utils import CocoDetection
from coco_utils import ConvertCocoPolysToMask
from torch.utils.data import DataLoader
from coco_utils import resize, resizeVal


# distributed train
from torch.utils.data.distributed import DistributedSampler
import argparse
import os
import random


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_model_instance_detection(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model



###############################################################################################################
# decouple the imgs and labels to avoid "RuntimeError: each element in list of batch should be of equal size" #
###############################################################################################################
def collate_fn(batch):
    return zip(*batch)
    #return tuple(zip(*batch))


def run():
    num_epochs_default = 300
    batch_size_default = 32 # 1024
    learning_rate_default = 0.001
    random_seed_default = 42
    model_dir_default = "saved_models"
    model_filename_default = "faster_rcnn_distributed.pth"

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=batch_size_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    argv = parser.parse_args()


    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume


    model_filepath = os.path.join(model_dir, model_filename)
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda:{}".format(local_rank))


    # train on the GPU or on the CPU, if a GPU is not available
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')


    # train_data_path = "/nfs/home/data/Euclid/Dataset/ObjectDetection/BDD100K/bdd100k/images/100k/train/"
    train_data_path = "/home/hhwu/datasets/bdd100k/bdd100k/images/100k/train/"

    transform_train = transforms.Compose([ConvertCocoPolysToMask(),
                                          transforms.ToTensor(),
                                          resize(480,640)])
                                          #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                          #resize(480,640)])
    transform_val   = transforms.Compose([ConvertCocoPolysToMask(),
                                          transforms.ToTensor(),
                                          resizeVal(480,640)])


    # our dataset has background and the other 15 classes of the target objects (1+15)
    num_classes = 16

    # define training and validation data loaders
    bdd100k_train         = CocoDetection(train_data_path, "./det_train_coco_gyr_dss.json", transforms=transform_train)
    bdd100k_train_sampler = DistributedSampler(dataset=bdd100k_train)
    train_dataloader      = DataLoader(bdd100k_train, batch_size=batch_size, shuffle=False, sampler=bdd100k_train_sampler, num_workers=0, collate_fn=collate_fn)

    bdd100k_val      = CocoDetection(val_data_path, "./det_val_coco_gyr_dss.json", transforms=transform_val)
    val_dataloader   = DataLoader(bdd100k_val, batch_size=8, shuffle=False, num_workers=8, collate_fn=collate_fn)

    # get the model using our helper function
    # model = torch.nn.DataParallel(get_model_instance_detection(num_classes)).to(device)
    model = get_model_instance_detection(num_classes).to(device)
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    if resume == True:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))


    # construct an optimizer
    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


    for epoch in range(1,num_epochs+1):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(ddp_model, optimizer, train_dataloader, device, epoch, print_freq=10, local_rank=local_rank)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)
        #torch.save(model, f"faster_rcnn_50_epoch_{epoch}.pt" )
        if local_rank == 0 and epoch % 10==0:
            evaluate(ddp_model, val_dataloader, device=device)
            saved_model_filepath = os.path.join(model_dir, f"faster_rcnn_50_epoch_{epoch}.pt")
            torch.save(ddp_model.state_dict(), saved_model_filepath)
    


            
    
    print("Done training!")


if __name__ == "__main__":
    run()
