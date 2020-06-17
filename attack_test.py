import argparse
import torch.nn as nn
import torchvision
from torchvision import transforms
from models import *
import attack_generator as attack

parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--net', type=str, default="WRN", help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10,help='WRN width factor')
parser.add_argument('--drop_rate', type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--attack_method', type=str,default="dat", help = "choose form: dat and trades")
parser.add_argument('--model_path', default='./FAT_models/fat_for_trades_wrn34-10_eps0.031_beta1.0.pth.tar', help='model for white-box attack evaluation')
parser.add_argument('--method',type=str,default='dat',help='select attack setting following DAT or TRADES')

args = parser.parse_args()

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
    model = Wide_ResNet(depth=args.depth, num_classes=10, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(args.depth,args.width_factor,args.drop_rate)
if args.net == 'WRN_madry':
    ## WRN-32-10
    model = Wide_ResNet_Madry(depth=args.depth, num_classes=10, widen_factor=args.width_factor, dropRate=args.drop_rate).cuda()
    net = "WRN_madry{}-{}-dropout{}".format(args.depth, args.width_factor, args.drop_rate)
model = torch.nn.DataParallel(model)
print(net)

model.load_state_dict(torch.load(args.model_path)['state_dict'])

print('==> Evaluating Performance under White-box Adversarial Attack')

loss, test_nat_acc = attack.eval_clean(model, test_loader)
print('Natural Test Accuracy: {:.2f}%'.format(100. * test_nat_acc))
if args.method == "dat":
    # Evalutions the same as DAT.
    loss, fgsm_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=0.031, step_size=0.031,loss_fn="cent", category="Madry",rand_init=True)
    print('FGSM Test Accuracy: {:.2f}%'.format(100. * fgsm_acc))
    loss, pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", category="Madry", rand_init=True)
    print('PGD20 Test Accuracy: {:.2f}%'.format(100. * pgd20_acc))
    loss, cw_acc = attack.eval_robust(model, test_loader, perturb_steps=30, epsilon=0.031, step_size=0.031 / 4,loss_fn="cw", category="Madry", rand_init=True)
    print('CW Test Accuracy: {:.2f}%'.format(100. * cw_acc))
if args.method == 'trades':
    # Evalutions the same as TRADES.
    # wri : with random init, wori : without random init
    loss, fgsm_wori_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=0.031, step_size=0.031,loss_fn="cent", category="Madry",rand_init=False)
    print('FGSM without Random Start Test Accuracy: {:.2f}%'.format(100. * fgsm_wori_acc))
    loss, pgd20_wori_acc = attack.eval_robust(model,test_loader, perturb_steps=20, epsilon=0.031, step_size=0.003,loss_fn="cent",category="Madry",rand_init=False)
    print('PGD20 without Random Start Test Accuracy: {:.2f}%'.format(100. * pgd20_wori_acc))
    loss, cw_wori_acc = attack.eval_robust(model,test_loader, perturb_steps=30, epsilon=0.031, step_size=0.003,loss_fn="cw",category="Madry",rand_init=False)
    print('CW without Random Start Test Accuracy: {:.2f}%'.format(100. * cw_wori_acc))
    loss, fgsm_wri_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=0.031, step_size=0.031,loss_fn="cent", category="Madry",rand_init=True)
    print('FGSM with Random Start Test Accuracy: {:.2f}%'.format(100. * fgsm_wri_acc))
    loss, pgd20_wri_acc = attack.eval_robust(model,test_loader, perturb_steps=20, epsilon=0.031, step_size=0.003,loss_fn="cent",category="Madry",rand_init=True)
    print('PGD20 with Random Start Test Accuracy: {:.2f}%'.format(100. * pgd20_wri_acc))
    loss, cw_wri_acc = attack.eval_robust(model,test_loader, perturb_steps=30, epsilon=0.031, step_size=0.003,loss_fn="cw",category="Madry",rand_init=True)
    print('CW with Random Start Test Accuracy: {:.2f}%'.format(100. * cw_wri_acc))
