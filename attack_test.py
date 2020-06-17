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
parser.add_argument('--net', type=str, default="WRN", help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10,help='WRN width factor')
parser.add_argument('--drop_rate', type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--attack_method', type=str,default="dat", help = "choose form: dat and trades")
parser.add_argument('--model_path', default='./FAT_models/fat_for_trades_wrn34-10_eps0.031_beta1.0.pt', help='model for white-box attack evaluation')
parser.add_argument('--acc_path', default='./attck_results/acc.pkl', help='path of pickle file store the test result')

args = parser.parse_args()

# settings
depth = args.depth
width_factor = args.width_factor
drop_rate = args.drop_rate
method = args.method
model_path = args.model_path
acc_path = args.acc_path

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
if args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "svhn":
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

print('==> Load Model')
if args.net == "smallcnn":
    model = SmallCNN().cuda()
    net = "smallcnn"
if args.net == "resnet18":
    model = ResNet18().cuda()
    net = "resnet18"
if args.net == "WRN":
    ## WRN-34-10
    model = Wide_ResNet(depth=depth, num_classes=10, widen_factor=width_factor, dropRate=drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(depth,width_factor,drop_rate)
if args.net == 'WRN_madry':
    ## WRN-32-10
    model = Wide_ResNet_Madry(depth=depth, num_classes=10, widen_factor=width_factor, dropRate=drop_rate).cuda()
    net = "WRN_madry{}-{}-dropout{}".format(depth, width_factor, drop_rate)
model = torch.nn.DataParallel(model)
print(net)
print (torch.load(model_path).keys())
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
if method == 'trades'
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


