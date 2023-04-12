import torch
import numpy as np
import os

def clean():
    if os.path.exists('conv0.txt'):
        os.remove('conv0.txt')
    if os.path.exists('conv1.txt'):
        os.remove('conv1.txt')
    if os.path.exists('conv2.txt'):
        os.remove('conv2.txt')
    if os.path.exists('conv3.txt'):
        os.remove('conv3.txt')
    if os.path.exists('conv4.txt'):
        os.remove('conv4.txt')

    if os.path.exists('bias0.txt'):
        os.remove('bias0.txt')
    if os.path.exists('bias1.txt'):
        os.remove('bias1.txt')
    if os.path.exists('bias2.txt'):
        os.remove('bias2.txt')
    if os.path.exists('bias3.txt'):
        os.remove('bias3.txt')
    if os.path.exists('bias4.txt'):
        os.remove('bias4.txt')

    if os.path.exists('img.txt'):
        os.remove('img.txt')

    if os.path.exists('fc0_weight.txt'):
        os.remove('fc0_weight.txt')
    if os.path.exists('fc0_bias.txt'):
        os.remove('fc0_bias.txt')
    if os.path.exists('fc1_weight.txt'):
        os.remove('fc1_weight.txt')
    if os.path.exists('fc1_bias.txt'):
        os.remove('fc1_bias.txt')
    if os.path.exists('fc2_weight.txt'):
        os.remove('fc2_weight.txt')
    if os.path.exists('fc2_bias.txt'):
        os.remove('fc2_bias.txt')

def roundOff(x):
    return round(float(x), 12)

def recalc():

    if not os.path.exists('weights'):
        os.makedirs('weights')

    PATH = './model.pth'
    model = torch.load(PATH)

    conv0 = model.features[0]
    wt_fileName = open(os.path.join('weights', 'conv0.txt'), 'w')
    bias_fileName = open(os.path.join('weights', 'bias0.txt'), 'w')

    for i in range(conv0.weight.shape[0]):
        for j in range(conv0.weight.shape[2]):
            for k in range(conv0.weight.shape[3]):
                for l in range(conv0.weight.shape[1]):
                    wt_fileName.write("{}\n".format(roundOff(conv0.weight.data[i][l][j][k])))
        bias_fileName.write("{}\n".format(roundOff(conv0.bias.data[i])))

    wt_fileName.close()
    bias_fileName.close()

    conv1 = model.features[4]
    wt_fileName = open(os.path.join('weights', 'conv1.txt'), 'w')
    bias_fileName = open(os.path.join('weights', 'bias1.txt'), 'w')

    for i in range(conv1.weight.shape[0]):
        for j in range(conv1.weight.shape[2]):
            for k in range(conv1.weight.shape[3]):
                for l in range(conv1.weight.shape[1]):
                    wt_fileName.write("{}\n".format(roundOff(conv1.weight.data[i][l][j][k])))
        bias_fileName.write("{}\n".format(roundOff(conv1.bias.data[i])))

    wt_fileName.close()
    bias_fileName.close()

    conv2 = model.features[8]
    wt_fileName = open(os.path.join('weights', 'conv2.txt'), 'w')
    bias_fileName = open(os.path.join('weights', 'bias2.txt'), 'w')

    for i in range(conv2.weight.shape[0]):
        for j in range(conv2.weight.shape[2]):
            for k in range(conv2.weight.shape[3]):
                for l in range(conv2.weight.shape[1]):
                    wt_fileName.write("{}\n".format(roundOff(conv2.weight.data[i][l][j][k])))
        bias_fileName.write("{}\n".format(roundOff(conv2.bias.data[i])))

    wt_fileName.close()
    bias_fileName.close()

    conv3 = model.features[10]
    wt_fileName = open(os.path.join('weights', 'conv3.txt'), 'w')
    bias_fileName = open(os.path.join('weights', 'bias3.txt'), 'w')

    for i in range(conv3.weight.shape[0]):
        for j in range(conv3.weight.shape[2]):
            for k in range(conv3.weight.shape[3]):
                for l in range(conv3.weight.shape[1]):
                    wt_fileName.write("{}\n".format(roundOff(conv3.weight.data[i][l][j][k])))
        bias_fileName.write("{}\n".format(roundOff(conv3.bias.data[i])))

    wt_fileName.close()
    bias_fileName.close()

    conv4 = model.features[12]
    wt_fileName = open(os.path.join('weights', 'conv4.txt'), 'w')
    bias_fileName = open(os.path.join('weights', 'bias4.txt'), 'w')

    for i in range(conv4.weight.shape[0]):
        for j in range(conv4.weight.shape[2]):
            for k in range(conv4.weight.shape[3]):
                for l in range(conv4.weight.shape[1]):
                    wt_fileName.write("{}\n".format(roundOff(conv4.weight.data[i][l][j][k])))
        bias_fileName.write("{}\n".format(roundOff(conv4.bias.data[i])))

    wt_fileName.close()
    bias_fileName.close()

    fc0 = model.classifier[1]
    wt_fileName = open(os.path.join('weights', 'fc0_weight.txt'), 'w')
    bias_fileName = open(os.path.join('weights', 'fc0_bias.txt'), 'w')

    cntr = 0
    for j in range(fc0.weight.shape[0]):
        res = []
        for i in range(fc0.weight.shape[1]):
            # wt_fileName.write("{}\n".format(roundOff(fc0.weight.data[j][i])))
            res.append(roundOff(fc0.weight.data[j][i]))
            cntr += 1
        res = np.array(res)
        res = res.reshape(256, 2, 2)
        res = np.transpose(res, (1, 2, 0))
        res = res.flatten()
        for i in range(fc0.weight.shape[1]):
            wt_fileName.write("{}\n".format(res[i]))
        bias_fileName.write("{}\n".format(roundOff(fc0.bias.data[j])))
    print(cntr)

    wt_fileName.close()
    bias_fileName.close()

    fc1 = model.classifier[4]
    wt_fileName = open(os.path.join('weights', 'fc1_weight.txt'), 'w')
    bias_fileName = open(os.path.join('weights', 'fc1_bias.txt'), 'w')

    cntr = 0
    for j in range(fc1.weight.shape[0]):
        for i in range(fc1.weight.shape[1]):
            wt_fileName.write("{}\n".format(roundOff(fc1.weight.data[j][i])))
            cntr += 1
        bias_fileName.write("{}\n".format(roundOff(fc1.bias.data[j])))
    print(cntr)

    wt_fileName.close()
    bias_fileName.close()

    fc2 = model.classifier[6]
    wt_fileName = open(os.path.join('weights', 'fc2_weight.txt'), 'w')
    bias_fileName = open(os.path.join('weights', 'fc2_bias.txt'), 'w')

    cntr = 0
    for j in range(fc2.weight.shape[0]):
        for i in range(fc2.weight.shape[1]):
            wt_fileName.write("{}\n".format(roundOff(fc2.weight.data[j][i])))
            cntr += 1
        bias_fileName.write("{}\n".format(roundOff(fc2.bias.data[j])))
    print(cntr)

    wt_fileName.close()
    bias_fileName.close()

def get_weights():

    clean()

    PATH = './model.pth'
    model = torch.load(PATH)
    conv0 = model.features[0].weight

    for i in range(len(conv0)):
        res = np.array(conv0[i].cpu().detach().numpy())
        res = np.transpose(res, (1, 2, 0))
        res = res.flatten()
        with open('conv0.txt', 'ab') as f:
            np.savetxt(f, res)

    bias0 = model.features[0].bias

    for i in range(len(bias0)):
        res = np.array(bias0[i].cpu().detach().numpy())
        res = res.flatten()
        with open('bias0.txt', 'ab') as f:
            np.savetxt(f, res)

    conv1 = model.features[4].weight

    for i in range(len(conv1)):
        res = np.array(conv1[i].cpu().detach().numpy())
        res = np.transpose(res, (1, 2, 0))
        res = res.flatten()
        with open('conv1.txt', 'ab') as f:
            np.savetxt(f, res)

    bias1 = model.features[4].bias

    for i in range(len(bias1)):
        res = np.array(bias1[i].cpu().detach().numpy())
        res = res.flatten()
        with open('bias1.txt', 'ab') as f:
            np.savetxt(f, res)

    conv2 = model.features[8].weight

    for i in range(len(conv2)):
        res = np.array(conv2[i].cpu().detach().numpy())
        res = np.transpose(res, (1, 2, 0))
        res = res.flatten()
        with open('conv2.txt', 'ab') as f:
            np.savetxt(f, res)

    bias2 = model.features[8].bias

    for i in range(len(bias2)):
        res = np.array(bias2[i].cpu().detach().numpy())
        res = res.flatten()
        with open('bias2.txt', 'ab') as f:
            np.savetxt(f, res)

    conv3 = model.features[10].weight

    for i in range(len(conv3)):
        res = np.array(conv3[i].cpu().detach().numpy())
        res = np.transpose(res, (1, 2, 0))
        res = res.flatten()
        with open('conv3.txt', 'ab') as f:
            np.savetxt(f, res)

    bias3 = model.features[10].bias

    for i in range(len(bias3)):
        res = np.array(bias3[i].cpu().detach().numpy())
        res = res.flatten()
        with open('bias3.txt', 'ab') as f:
            np.savetxt(f, res)

    conv4 = model.features[12].weight

    for i in range(len(conv4)):
        res = np.array(conv4[i].cpu().detach().numpy())
        res = np.transpose(res, (1, 2, 0))
        res = res.flatten()
        with open('conv4.txt', 'ab') as f:
            np.savetxt(f, res)

    bias4 = model.features[12].bias

    for i in range(len(bias4)):
        res = np.array(bias4[i].cpu().detach().numpy())
        res = res.flatten()
        with open('bias4.txt', 'ab') as f:
            np.savetxt(f, res)

    fc0_weight = model.classifier[1].weight
    res = np.array(fc0_weight.cpu().detach().numpy())
    res = res.flatten()
    with open('fc0_weight.txt', 'ab') as f:
        np.savetxt(f, res)

    fc0_bias = model.classifier[1].bias
    res = np.array(fc0_bias.cpu().detach().numpy())
    res = res.flatten()
    with open('fc0_bias.txt', 'ab') as f:
        np.savetxt(f, res)

    fc1_weight = model.classifier[4].weight
    res = np.array(fc1_weight.cpu().detach().numpy())
    res = res.flatten()
    with open('fc1_weight.txt', 'ab') as f:
        np.savetxt(f, res)

    fc1_bias = model.classifier[4].bias
    res = np.array(fc1_bias.cpu().detach().numpy())
    res = res.flatten()
    with open('fc1_bias.txt', 'ab') as f:
        np.savetxt(f, res)

    fc2_weight = model.classifier[6].weight
    res = np.array(fc2_weight.cpu().detach().numpy())
    res = res.flatten()
    with open('fc2_weight.txt', 'ab') as f:
        np.savetxt(f, res)

    fc2_bias = model.classifier[6].bias
    res = np.array(fc2_bias.cpu().detach().numpy())
    res = res.flatten()
    with open('fc2_bias.txt', 'ab') as f:
        np.savetxt(f, res)


if __name__ == "__main__":
    # get_weights()
    recalc()