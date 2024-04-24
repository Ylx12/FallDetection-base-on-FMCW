import shutil
import time
import os
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Network import RDTNet
from Dataset_reader import Fall_Dataset
from Function import BCEFocalLoss, seed_setting

seed = 11
seed_setting(seed)

# state = 'train'
state = 'verify'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

device = torch.device("cuda")


def train(model, trainloader, train_batch_size, epoch, device, optimizer, criterion):
    model.train()
    start_time = time.time()
    train_loss = 0
    ll = 0
    sum_acc = 0

    for batch_idx, (inputs, labels, _) in enumerate(trainloader):
        bs = len(inputs)
        ll = ll + bs
        inputs = torch.reshape(inputs, ((len(inputs)) * 16, 1, 62, 50))
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, (torch.reshape(labels, outputs.shape)).float())
        predict = (outputs > 0.5).data.squeeze()
        loss.backward()
        optimizer.step()

        sum_acc += (predict == labels).sum().item()
        train_loss += loss.item()

        if batch_idx % int(len(trainloader) / 5) == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}".format(
                    epoch,
                    ll,
                    len(trainloader.dataset),
                    100.0 * batch_idx / len(trainloader),
                    loss.data.item(),
                    optimizer.param_groups[0]["lr"],
                )
            )

    average_train_loss = train_loss / (len(trainloader.dataset) / train_batch_size)
    end_time = time.time()
    elapse_time = end_time - start_time
    accuracy = 100 * sum_acc / ll
    print('average_train_loss', average_train_loss, 'elapse_time', elapse_time, 'train_acc:', accuracy)

    return model, average_train_loss


def test(model, testloader, test_batch_size, criterion, device, best_acc, best_fall_acc, save_path):
    model.eval()

    ll = 0
    sum_acc = 0
    test_loss = 0
    classes = ['non-fall', 'fall']

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    for i, (inputs, labels, _) in enumerate(testloader):
        ll = ll + len(inputs)
        inputs = torch.reshape(inputs, (len(inputs) * 16, 1, 62, 50))
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, torch.reshape(labels, outputs.shape).float()).data.item()

        predict = (outputs > 0.5).data.squeeze()
        sum_acc += (predict == labels).sum().item()

        for label, pred in zip(labels, predict):
            if label == pred:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    average_test_loss = test_loss / (len(testloader.dataset) / test_batch_size)

    avg_acc = 0
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            accur = 0
        else:
            accur = 100 * float(correct_count) / total_pred[classname]
        avg_acc += accur
        print("Accuracy for class {:5s} is: {:.2f}%({}/{})".format(classname, accur, correct_count,
                                                                   total_pred[classname]))
    print("Average accuracy is: {:.2f}%".format(avg_acc / len(classes)))
    print('sum_acc:', sum_acc)
    test_acc = 100.0 * sum_acc / ll
    fall_acc = accur
    print('Test accuracy: %.4f ' % (test_acc))
    print("# ---------------------------------------------------- #")
    if best_acc < 99:
        if test_acc > best_acc:
            best_acc = test_acc
            best_fall_acc = fall_acc
            print('<<<<<<saving model(Acc rise)')
            torch.save(model.state_dict(), save_path)
        elif (abs(test_acc - best_acc) < 1e-5) & (fall_acc > best_fall_acc):
            print('<<<<<<saving model(Recall rates rise)')
            best_fall_acc = fall_acc
            torch.save(model.state_dict(), save_path)
        print('The best test accuracy is %.4f' % (best_acc))
        print('Its fall accuracy is %.4f' % (best_fall_acc))
        print(' ')
    else:
        if ((best_acc >= 99) & (fall_acc > best_fall_acc) & (test_acc >= 99)):
            print('<<<<<<saving model(Recall rates rise)')
            best_acc = test_acc
            best_fall_acc = fall_acc
            torch.save(model.state_dict(), save_path)
        elif (abs(fall_acc - best_fall_acc) < 1e-5) & (test_acc > best_acc):
            print('<<<<<<saving model(Acc rates rise)')
            best_acc = test_acc
            best_fall_acc = fall_acc
            torch.save(model.state_dict(), save_path)
        print('Test accuracy is %.4f' % (best_acc))
        print('The best recall rate is %.4f' % (best_fall_acc))
        print(' ')

    return best_acc, best_fall_acc, average_test_loss


def inference(modelin, testloader, device):
    model = modelin
    model.eval()
    model.to(device=device)
    sum_acc = 0
    classes = ['non-fall', 'fall']

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    n = 0
    for i, (inputs, labels, _) in enumerate(testloader):
        n = n + len(inputs)
        inputs = torch.reshape(inputs, (len(inputs) * 16, 1, 62, 50))
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predict = (outputs > 0.5).data.squeeze()
        sum_acc += (predict == labels).sum().item()

        for label, pred in zip(labels, predict):
            if label == pred:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    avg_acc = 0
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            accur = 0
        else:
            accur = 100 * float(correct_count) / total_pred[classname]
        avg_acc += accur
        print("Accuracy for class {:5s} is: {:.2f}%({}/{})".format(classname, accur, correct_count,
                                                                   total_pred[classname]))
    print('sum_acc:', sum_acc, n)
    test_acc = 100.0 * sum_acc / n

    print('Test accuracy: %.4f ' % (test_acc))
    print("# ---------------------------------------------------- #")

    return test_acc

if state == 'verify':
    model = RDTNet([[1, 16, 32, 64, 128, 256], [256, 256, 256]])
    pth_path = 'model/RDTNet.pth'
    model_pth = torch.load(pth_path)
    model.load_state_dict(model_pth)

    valid_dir = '../list/test2.txt'
    val_dataset = Fall_Dataset(valid_dir)
    test_batch_size = 128
    test_loader = DataLoader(val_dataset, test_batch_size, shuffle=True)
    inference(model, test_loader, device)

elif state == 'train':
    bianhao = 0
    log_dir = 'runs/model' + str(bianhao)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    train_dir = '../list/train2.txt'
    valid_dir = '../list/test2.txt'

    train_dataset = Fall_Dataset(train_dir)
    val_dataset = Fall_Dataset(valid_dir)
    train_batch_size = 32
    test_batch_size = 32

    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, test_batch_size, shuffle=True)

    channel_list = [[1, 16, 32, 64, 128, 256], [256, 256, 256]]

    model = RDTNet(channel_list)  # 603

    model.eval()
    model.to(device=device)

    # 优化器设置
    optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-2)
    criterion = BCEFocalLoss(alpha=0.33)

    # best_acc = full_inference(model, test_loader, device)
    best_acc = 0
    fall_acc = 0

    saving_dir = 'model/'
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    scheduler = ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(200):
        model, train_loss = train(model, train_loader, train_batch_size, epoch, device, optimizer, criterion)

        if best_acc < 99:
            filename = saving_dir + 'RDTNet_model' + str(bianhao) + '.pth'
        else:
            filename = saving_dir + 'RDTNet_model' + str(bianhao) + '_epoch' + str(epoch) + '.pth'

        best_acc, fall_acc, average_test_loss = test(model, test_loader, test_batch_size, criterion, device, best_acc,
                                                     fall_acc, filename)
        scheduler.step(average_test_loss)

        writer.add_scalar(tag="accuracy",  # 可以暂时理解为图像的名字
                          scalar_value=best_acc,  # 纵坐标的值
                          global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                          )
        writer.add_scalar(tag="train_loss",  # 可以暂时理解为图像的名字
                          scalar_value=train_loss,  # 纵坐标的值
                          global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                          )
        writer.add_scalar(tag="test_loss",  # 可以暂时理解为图像的名字
                          scalar_value=average_test_loss,  # 纵坐标的值
                          global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                          )

    writer.close()
