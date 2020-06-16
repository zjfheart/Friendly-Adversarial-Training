from __future__  import print_function
import argparse
import torch.nn as nn
import torchvision
from torchvision import transforms
import pickle
from models import *
import attack_generator as attack
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--net', type=str, default="WRN", help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--tau', type=int,default=0, help='slippery step tau')
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width-factor', type=int, default=10,help='WRN width factor')
parser.add_argument('--drop-rate', type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--method', type=str,default="fat", help = "choose form: fat, fat_for_trades, fat_for_mart")
parser.add_argument('--model-path', default='./FAT_results/checkpoint.pth.tar', help='model for white-box attack evaluation')
parser.add_argument('--acc-path', default='./FAT_results/acc.pkl', help='path of pickle file recording the test result')

args = parser.parse_args()

# settings
seed = args.seed
depth = args.depth
width_factor = args.width_factor
drop_rate = args.drop_rate
method = args.method
model_path = args.model_path
acc_path = args.acc_path

torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

print('==> Load Model')
if args.net == "smallcnn":
    model = SmallCNN().cuda()
    net = "smallcnn"
if args.net == "resnet18":
    model = ResNet18().cuda()
    net = "resnet18"
if args.net == "WRN":
    model = Wide_ResNet(depth=depth, num_classes=10, widen_factor=width_factor, dropRate=drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(depth,width_factor,drop_rate)
if args.net == 'WRN_madry':
    model = Wide_ResNet_Madry(depth=depth, num_classes=10, widen_factor=width_factor, dropRate=drop_rate).cuda()
    net = "WRN_madry{}-{}-dropout{}".format(depth, width_factor, drop_rate)
model = torch.nn.DataParallel(model)
print(net)

model.load_state_dict(torch.load(model_path)['state_dict'])

print('==> Evaluating Performance under White-box Adversarial Attack')

state = dict()
loss, test_nat_acc = attack.eval_clean(model, test_loader)
state.update({"natural_acc": test_nat_acc})

if method == "fat":
    # Evalutions the same as DAT.
    loss, fgsm_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=0.031, step_size=0.031,loss_fn="cent", category="Madry",rand_init=True)
    state.update({"fgsm_acc": fgsm_acc})
    loss, pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", category="Madry", rand_init=True)
    state.update({"pgd20_acc": pgd20_acc})
    loss, cw_acc = attack.eval_robust(model, test_loader, perturb_steps=30, epsilon=0.031, step_size=0.031 / 4,loss_fn="cw", category="Madry", rand_init=True)
    state.update({"cw_acc": cw_acc})

    print(
        'Natural Test Acc %.2f | FGSM Test Acc %.2f | PGD20 Test Acc %.2f | CW Test Acc %.2f |\n' % (
            test_nat_acc,
            fgsm_acc,
            pgd20_acc,
            cw_acc)
    )
else:
    # Evalutions the same as TRADES.
    # wri : with random init, wori : without random init
    loss, fgsm_wori_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=0.031, step_size=0.031,loss_fn="cent", category="Madry",rand_init=False)
    state.update({"fgsm_wori_acc": fgsm_wori_acc})
    loss, pgd20_wori_acc = attack.eval_robust(model,test_loader, perturb_steps=20, epsilon=0.031, step_size=0.003,loss_fn="cent",category="Madry",rand_init=False)
    state.update({"pgd20_wori_acc": pgd20_wori_acc})
    loss, cw_wori_acc = attack.eval_robust(model,test_loader, perturb_steps=30, epsilon=0.031, step_size=0.003,loss_fn="cw",category="Madry",rand_init=False)
    state.update({"cw_wori_acc": cw_wori_acc})
    loss, fgsm_wri_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=0.031, step_size=0.031,loss_fn="cent", category="Madry",rand_init=True)
    state.update({"fgsm_wri_acc": fgsm_wri_acc})
    loss, pgd20_wri_acc = attack.eval_robust(model,test_loader, perturb_steps=20, epsilon=0.031, step_size=0.003,loss_fn="cent",category="Madry",rand_init=True)
    state.update({"pgd20_wri_acc": pgd20_wri_acc})
    loss, cw_wri_acc = attack.eval_robust(model,test_loader, perturb_steps=30, epsilon=0.031, step_size=0.003,loss_fn="cw",category="Madry",rand_init=True)
    state.update({"cw_wri_acc": cw_wri_acc})

    print(
        'Natural Test Acc %.2f | FGSM-WORI Test Acc %.2f | PGD20-WORI Test Acc %.2f | CW-WORI Test Acc %.2f | FGSM-WRI Test Acc %.2f | PGD20-WRI Test Acc %.2f | CW-WRI Test Acc %.2f |\n ' % (
            test_nat_acc,
            fgsm_wori_acc,
            pgd20_wori_acc,
            cw_wori_acc,
            fgsm_wri_acc,
            pgd20_wri_acc,
            cw_wri_acc
        )
    )

with open(acc_path, 'wb') as f:
    pickle.dump(state, f)
    f.close()


