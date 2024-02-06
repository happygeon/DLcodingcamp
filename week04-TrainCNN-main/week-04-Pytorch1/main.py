import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms as T
from tqdm import tqdm

from dataset import FoodDataset
from model import vanillaCNN, vanillaCNN2, VGG19

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['CNN1', 'CNN2', 'VGG'], required=True, help='model architecture to train')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='the number of train epochs')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs('./save', exist_ok=True)
    os.makedirs(f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}', exist_ok=True)
    
    transforms = T.Compose([
        T.Resize((227,227), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomVerticalFlip(0.5),
        T.RandomHorizontalFlip(0.5),
    ])

    train_dataset = FoodDataset("./data", "train", transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dataset = FoodDataset("./data", "val", transforms=transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    if args.model == 'CNN1':
        model = vanillaCNN()
    elif args.model == 'CNN2':
        model = vanillaCNN2()
    elif args.model == 'VGG': 
        model = VGG19()
    else:
        raise ValueError("model not supported")
        
    ##########################   fill here   ###########################
        
    # TODO : Training Loop을 작성해주세요
    # 1. logger, optimizer, criterion(loss function)을 정의합니다.
    # train loader는 training에 val loader는 epoch 성능 측정에 사용됩니다.
    # torch.save()를 이용해 epoch마다 model이 저장되도록 해 주세요
            
    ######################################################################
    logger = logging.getLogger()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    for epoch in range(args.epoch):
        model.train()
        train_loss = 0
        logger.info(f"training epoch {epoch+1}")
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        logger.info(f"Epoch {epoch + 1} Loss: {train_loss}")
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        logger.info(f"Validating epoch {epoch+1}")
        with torch.no_grad():
            for batch in tqdm(val_loader):
                image = batch['image'].to(device)
                label = batch['label'].to(device)
                output = model(image)
                loss = criterion(output, label)
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        logger.info(f"Epoch {epoch + 1} accuracy: {accuracy}")
        
        torch.save(model.state_dict(), f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}/epoch_{epoch}.pth')