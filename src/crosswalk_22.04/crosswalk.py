import torch
import os
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize

# 데이터셋 불러오기
train_img_dir = "data/train"
train_annot_file = "data/train/_annotations.coco.json"

valid_img_dir = "data/valid"
valid_annot_file = "data/valid/_annotations.coco.json"

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)

transform = Compose([Resize((224, 224)), ToTensor()])
print(transform)

train_dataset = CocoDetection(root=train_img_dir,
                              annFile=train_annot_file,
                              transform=transform)

valid_dataset = CocoDetection(root=valid_img_dir,
                              annFile=valid_annot_file,
                              transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

 ## 데이터셋 확인
for imgs, targets in train_loader:
    print(f"tensor size of an Image: {imgs[0].shape}")
    print(f"annot of the first Image: {targets[0]}")
    break


# 모델 학습
model = fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# GPU 연결
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)

# 옵티마이저
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 10

for t in targets:
    print(t[0])
    for k, v in t[0].items():
        print("k:", k)
        if type(v) is not torch.Tensor:
            v = torch.tensor(v)
        print("v:", v)
        print('=====')

# 학습
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for imgs, targets in train_loader:
        imgs = [img.to(device) for img in imgs]
        for t in targets:
            for k, v in t[0].items():
                if type(v) is not torch.Tensor:
                    v = torch.tensor(v)
                targets = [{k: v.to(device)}]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    print(f"{epoch+1}번째 시도 Loss: {epoch_loss}")

model.eval()
torch.save(model.state_dict(), "crosswalk_best_weight.pth")