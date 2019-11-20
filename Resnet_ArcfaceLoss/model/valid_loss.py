import torch
from torchvision import transforms
from torch.autograd import Variable


# test the model
def test(testLoader, resnet, head, cost, device):
    # Test the model
    resnet.eval()
    head.eval()
    test_correct = 0
    total = 0
    test_loss = 0

    for i, (images, labels) in enumerate(testLoader):
        # extract feature
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        embeddings = resnet(images)
        labels = Variable(labels)

        thetas = head(embeddings, labels)
        # calculate test_loss
        loss = cost(thetas, labels)
        test_loss += loss.item()
        # calculate test_acc
        _, predicted = torch.max(thetas.data, 1)
        total += labels.size(0)
        test_correct += (predicted.cpu() == labels.cpu()).sum()

    print('Test Accuracy of the model on the  test images: %d %%' % (100 * test_correct / total))

    return test_loss, test_correct / total

