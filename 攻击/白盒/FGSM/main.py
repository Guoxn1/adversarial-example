import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets,transforms
import numpy as np 
import matplotlib.pyplot as plt 



# 定义lenet
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def fgsm_attack(image,epsilon,data_grad):
    sign_data_grad = data_grad.sign()
    
    # 构造图像
    perturbed_image = image+epsilon*sign_data_grad

    # 去除超过范围的值
    perturbed_image = torch.clamp(perturbed_image,0,1)

    return perturbed_image

def test(model,device,test_loader,epsilon):
    correct = 0
    adv_examples = []

    for data,target in test_loader:
        data,target = data.to(device),target.to(device)

        # 设置数据是需要梯度的.
        data.requires_grad = True
        output = model(data)
        #这个是真实的输出，一般是正确的   这里面的output是10个值  还没有传入softmax

        init_pred = output.max(1,keepdim=True)[1]
        # 如果这种最大值输出和目标不一样那就不需要攻击了
        if init_pred.item() != target.item():
            continue
        # 否则 就需要计算损失函数对数据的梯度 来生成对抗样本
        loss = F.nll_loss(output,target)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data

        perturbed_image = fgsm_attack(data,epsilon,data_grad)
        output = model(perturbed_image)
        # 这里计算出对抗样本的预测情况
        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def draw_acc(epsilons,accuracies):
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

def draw_picture(epsilons,examples):
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()


def main():
    epsilons = [0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]
    pretrained_model = "data/lenet_mnist_model.pth"
    use_cuda=True

    test_loader = torch.utils.data.DataLoader(datasets.MNIST("./data",train=False,download=True,transform=transforms.Compose([
        transforms.ToTensor(),
    ])),batch_size=1,shuffle=True)

    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    model = Net().to(device)
    model.load_state_dict(torch.load(pretrained_model,map_location="cpu"))
    model.eval()

    accuracies = []
    examples = []
    for eps in epsilons:
        acc,ex = test(model,device,test_loader,eps)
        accuracies.append(acc)
        examples.append(ex)
    draw_acc(epsilons,accuracies)
    draw_picture(epsilons,examples)


main()