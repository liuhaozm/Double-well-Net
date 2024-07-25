import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time
import torch.nn as nn
import numpy as np
import torch.optim as optim
from DN-I import DNI # DN-I model
from DN-II import DNII # DN-II model

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


learning_rate=1e-4
device="cuda" if torch.cuda.is_available() else "cpu"
batch_size=8
num_epochs=400
num_workers=4
image_height=192 
image_width=256 
trainL=-1 # training data size, use -1 for all data
pin_memory=False
load_model=False
train_dir="../data/train"
train_maskdir="../data/train_mask"
val_dir="../data/test400"
val_maskdir="../data/test400_mask"

def train_fn(loader,model,optimizer,loss_fn):
#     loop=tqdm(loader)
    start=time.time()
    losses=[]
    nums=[]
    for batch_idx, (data,targets) in enumerate(loader):
        data=data.to(device=device)
        targets=targets.float().unsqueeze(1).to(device=device)
        

        predictions=model(data)
        loss=loss_fn(predictions,targets)
        losses.append(loss.item())
        nums.append(len(data))
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    end=time.time()
    tt=end-start
    total=np.sum(nums)
    avg_loss=np.sum(np.multiply(losses,nums))/total
    return tt, avg_loss
        
def main():
    lossAll=[]
    accAll=[]
    diceAll=[]
    traintime=[]
    train_transform =A.Compose(
    [
        A.Resize(height=image_height, width=image_width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    val_transform=A.Compose(
    [
        A.Resize(height=image_height, width=image_width),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    # choose one model
    # DN-I model
    model=DNI(features=[128,128,128,128,256],num_blocks=10).to(device)
    
#     # DN-II model
#     model=DNII(features=[64,64,64,128,128],num_blocks=3).to(device)
    
    
    loss_fn=nn.BCELoss()
    optimizer=optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loader, val_loader=get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        trainL,
        num_workers,
        pin_memory,
    )
    
    if load_model:
        checkpoint_old=torch.load("checkpoint.pth.tar")
        model.load_state_dict(checkpoint_old["state_dict"])
        optimizer.load_state_dict(checkpoint_old['optimizer'])
        accAll=checkpoint_old['accAll']
        lossAll=checkpoint_old['lossAll']
        diceAll=checkpoint_old['diceAll']
        traintime=checkpoint_old['traintime']
    for epoch in range(num_epochs):
        tt,loss=train_fn(train_loader, model, optimizer, loss_fn)
        
        lossAll.append(loss)
        
        # check accuracy
        acc,dice=check_accuracy(val_loader, model,device=device)
        print(
        f"Epoch {epoch+1}/{num_epochs}, loss: {loss:.4f}, dice: {dice:.4f}, acc: {acc:.2f}, time used: {tt:.2f}s")
        
        accAll.append(acc)
        diceAll.append(dice)
        traintime.append(tt)
        
        # save model
        checkpoint={
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lossAll": lossAll,
            "accAll": accAll,
            "diceAll": diceAll,
            "traintime": traintime,
        }
        torch.save(checkpoint,"checkpoint.pth.tar")
        
        
        
    
    # print some examples
#     save_predictions_as_imgs(
#         val_loader,model,folder="saved_images/",device=device,
#     )
    
if __name__ == "__main__":
    main()