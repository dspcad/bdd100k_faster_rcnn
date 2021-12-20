import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
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

import utils

def get_model_instance_detection(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model



###############################################################################################################
# decouple the imgs and labels to avoid "RuntimeError: each element in list of batch should be of equal size" #
###############################################################################################################
def collate_fn(batch):
    return zip(*batch)
    #return tuple(zip(*batch))


def main(training=False):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')


    train_data_path = "/home/hhwu/datasets/bdd100k/bdd100k/images/100k/train/"
    val_data_path   = "/home/hhwu/datasets/bdd100k/bdd100k/images/100k/val/"

    transform_train = transforms.Compose([ConvertCocoPolysToMask(),
                                          transforms.ToTensor()])
                                          #resize(480,640)])
                                          #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # our dataset has background and the other 15 classes of the target objects (1+15)
    num_classes = 16

    if training:
        # define training and validation data loaders
        bdd100k_train    = CocoDetection(train_data_path, "./det_train_coco_gyr_dss.json", transforms=transform_train)
        #bdd100k_train    = CocoDetection("/home/hhwu/datasets/bdd100k/bdd100k/images/100k/train", "/home/hhwu/datasets/bdd100k/bdd100k/COCOFormat/classes_15/det_train_coco_gyr_dss.json", transforms=None)
        train_dataloader = DataLoader(bdd100k_train, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)


        # get the model using our helper function
        # model = torch.nn.DataParallel(get_model_instance_detection(num_classes)).to(device)
        model = get_model_instance_detection(num_classes).to(device)


        # move model to the right device
        # model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # let's train it for 10 epochs
        num_epochs = 10

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            # evaluate(model, data_loader_test, device=device)
            torch.save(model, f"faster_rcnn_50_epoch_{epoch}.pt" )
        
    bdd100k_val      = CocoDetection(val_data_path, "./det_val_coco_gyr_dss.json", transforms=transform_train)
    val_dataloader   = DataLoader(bdd100k_val, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_fn)


    #val_dataloader = build_detection_test_loader(DatasetRegistry.get("BDD100K_val_dataset"))
    model = torch.load("faster_rcnn_50_epoch_9.pt")
    # model = get_model_instance_detection(81).to(device)
    # pick one image from the test set
    img, target = bdd100k_val[0]
    print(target)
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        evaluate(model, val_dataloader, device=device)
        prediction = model([img.to(device)])

        
        # print(prediction)
        img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        img1.save("target.png")
        
        img1 = torchvision.transforms.ToTensor()(img1)
        img1 = torchvision.transforms.ConvertImageDtype(dtype=torch.uint8) (img1)
        colors=["yellow" for i in prediction[0]['boxes']]
        #img1 = torchvision.utils.draw_bounding_boxes(img1, prediction[0]['boxes'], colors=colors ,width=2,fill=True)
        img1 = torchvision.utils.draw_bounding_boxes(img1, target['boxes'], colors=colors ,width=2,fill=True)
        target = Image.fromarray(img1.permute(1,2,0).byte().numpy())
        target.save("target1.png")


        #img2 = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
        #img2.save("result.png")
        

    print("That's it!")


if __name__ == "__main__":
    main(False)
    #run(False)
    # run(True)
