import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

# train the model
def train(trainLoader, optimizer, resnet, head, cost, EPOCH, epoch, device):
    resnet.train()
    head.train()
    train_loss = 0
    train_correct = 0
    total = 0
    for i, (images, labels) in enumerate(trainLoader):
        # train_loss = 0
        # for images, labels in trainLoader:
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()

        embeddings = resnet(images)
        thetas = head(embeddings, labels)

        loss = cost(thetas, labels)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        if ( i +1) % 100 == 0 :
            print ('Epoch [%d/%d], Loss. %.6f' % (epoch + 1, EPOCH, loss.data[0]))

    # compute epoch loss
    train_epoch_loss = train_loss / len(trainLoader)

    # compute epoch acc
    resnet.eval()
    head.eval()
    for i, (images, labels) in enumerate(trainLoader):
        # extract feature
        # images = Variable(images).cuda()
        # labels = Variable(labels).cuda()
        embeddings = resnet(images)
        labels = Variable(labels)

        thetas = head(embeddings, labels)
        # calculate train_acc
        _, predicted = torch.max(thetas.data, 1)
        total += labels.size(0)
        train_correct += (predicted.cpu() == labels.cpu()).sum()

    print('Train_acc: %.6f' % (train_correct/total))

    return train_epoch_loss, train_correct/total
