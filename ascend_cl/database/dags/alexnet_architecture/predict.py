import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

def getActivations(img, model):
    if not os.path.exists('activations'):
        os.makedirs('activations')

    # print(len(model.features))

    x = img
    for modelId in range(len(model.features)):
        layer = model.features[modelId]
        x = layer(x)
        fileName = os.path.join('activations', '{}.txt'.format(modelId))
        if os.path.exists(fileName):
            os.remove(fileName)
        result = np.array(x.cpu().detach().numpy())
        result = np.transpose(result, (0, 2, 3, 1))
        for l in range(len(result)):
            with open(fileName, 'ab') as f:
                np.savetxt(f, result[l].flatten())

    x = x.view(x.size(0), 256 * 2 * 2)
    fileName = os.path.join('activations', '{}.txt'.format('flatten'))
    if os.path.exists(fileName):
        os.remove(fileName)
    result = np.array(x.cpu().detach().numpy())
    for l in range(len(result)):
        with open(fileName, 'ab') as f:
            np.savetxt(f, result[l].flatten())

    x = model.classifier[1](x)
    fileName = os.path.join('activations', '{}.txt'.format(15))
    if os.path.exists(fileName):
        os.remove(fileName)
    result = np.array(x.cpu().detach().numpy())
    for l in range(len(result)):
        with open(fileName, 'ab') as f:
            np.savetxt(f, result[l].flatten())

    x = model.classifier[2](x)
    fileName = os.path.join('activations', '{}.txt'.format(16))
    if os.path.exists(fileName):
        os.remove(fileName)
    result = np.array(x.cpu().detach().numpy())
    for l in range(len(result)):
        with open(fileName, 'ab') as f:
            np.savetxt(f, result[l].flatten())

    x = model.classifier[4](x)
    fileName = os.path.join('activations', '{}.txt'.format(17))
    if os.path.exists(fileName):
        os.remove(fileName)
    result = np.array(x.cpu().detach().numpy())
    for l in range(len(result)):
        with open(fileName, 'ab') as f:
            np.savetxt(f, result[l].flatten())

    x = model.classifier[5](x)
    fileName = os.path.join('activations', '{}.txt'.format(18))
    if os.path.exists(fileName):
        os.remove(fileName)
    result = np.array(x.cpu().detach().numpy())
    for l in range(len(result)):
        with open(fileName, 'ab') as f:
            np.savetxt(f, result[l].flatten())

    x = model.classifier[6](x)
    fileName = os.path.join('activations', '{}.txt'.format(19))
    if os.path.exists(fileName):
        os.remove(fileName)
    result = np.array(x.cpu().detach().numpy())
    for l in range(len(result)):
        with open(fileName, 'ab') as f:
            np.savetxt(f, result[l].flatten())

def predict():

    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    images, labels = next(iter(testloader))
    images = images.to(torch.device('cuda'))
    labels = labels.to(torch.device('cuda'))
    img = np.array(images[0].cpu().detach().numpy())
    # imshow(img)

    img = np.transpose(img, (1, 2, 0))
    img = img.flatten()

    if os.path.exists('img.txt'):
        os.remove('img.txt')

    fileName = open('img.txt', 'w')

    for i in range(images.shape[2]):
        for j in range(images.shape[3]):
            for k in range(images.shape[1]):
                fileName.write("{}\n".format(images[0][k][i][j]))

    fileName.close()

    print(img, classes[labels[0]])

    PATH = './model.pth'
    model = torch.load(PATH)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(1)))

    getActivations(images, model)
    return


if __name__ == "__main__":
    predict()
