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
from coco_utils import resize


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


def main(training=False):
    num_epochs_default = 10000
    batch_size_default = 256 # 1024
    learning_rate_default = 0.1
    random_seed_default = 0
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


    train_data_path = "/NFS/share/Euclid/Dataset/ObjectDetection/BDD100K/bdd100k/images/100k/train/"
    val_data_path   = "/NFS/share/Euclid/Dataset/ObjectDetection/BDD100K/bdd100k/images/100k/val/"

    transform_train = transforms.Compose([ConvertCocoPolysToMask(),
                                          transforms.ToTensor(),
                                          resize(480,640)])
                                          #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                          #resize(480,640)])

    # our dataset has background and the other 15 classes of the target objects (1+15)
    num_classes = 16

    if training:
        # define training and validation data loaders
        bdd100k_train    = CocoDetection(train_data_path, "./det_train_coco.json", transforms=transform_train)
        #bdd100k_train    = CocoDetection("/home/hhwu/datasets/bdd100k/bdd100k/images/100k/train", "/home/hhwu/datasets/bdd100k/bdd100k/COCOFormat/classes_15/det_train_coco_gyr_dss.json", transforms=None)
        bdd100k_train_sampler = DistributedSampler(dataset=bdd100k_train)
        train_dataloader = DataLoader(bdd100k_train, batch_size=4, shuffle=False, sampler=bdd100k_train_sampler, num_workers=4, collate_fn=collate_fn)


        # get the model using our helper function
        # model = torch.nn.DataParallel(get_model_instance_detection(num_classes)).to(device)
        model = get_model_instance_detection(num_classes).to(device)
        model = model.to(device)
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        # move model to the right device
        # model.to(device)

        # construct an optimizer
        params = [p for p in ddp_model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # let's train it for 10 epochs
        num_epochs = 100

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(ddp_model, optimizer, train_dataloader, device, epoch, print_freq=10, local_rank=local_rank)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            # evaluate(model, data_loader_test, device=device)
            #torch.save(model, f"faster_rcnn_50_epoch_{epoch}.pt" )
            if local_rank == 0 and epoch > 0 and epoch % 10==0:
                model_filepath = os.path.join(model_dir, f"faster_rcnn_50_epoch_{epoch}.pt")
                torch.save(model, model_filepath)
    


    bdd100k_val      = CocoDetection(val_data_path, "./det_val_coco.json", transforms=transform_train)
    val_dataloader   = DataLoader(bdd100k_val, batch_size=128, shuffle=False, num_workers=0)

    model = torch.load("faster_rcnn_50_epoch_50.pt")
    # pick one image from the test set
    img, _ = bdd100k_val[0]
    print(img)
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

        
        print(prediction[0]['boxes'])
        img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        img1.save("target.png")
        
        img1 = torchvision.transforms.ToTensor()(img1)
        img1 = torchvision.transforms.ConvertImageDtype(dtype=torch.uint8) (img1)
        colors=["yellow" for i in prediction[0]['boxes']]
        img1 = torchvision.utils.draw_bounding_boxes(img1, prediction[0]['boxes'], colors=colors ,width=3,fill=True)
        target = Image.fromarray(img1.permute(1,2,0).byte().numpy())
        target.save("target1.png")


        #img2 = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
        #img2.save("result.png")
        

    print("That's it!")


if __name__ == "__main__":
    main(True)
    #run(False)
    # run(True)
