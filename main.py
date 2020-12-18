import os
import numpy as np
import torch
from torch import optim

from config import Config
from data import MyDataLoader
from model import CRCNN, PairwiseRankingLoss

def print_result(predict_label, id2rel, start_idx=8001):
    with open('predicted_result.txt', 'w', encoding='utf-8') as fw:
        for i in range(0, predict_label.shape[0]):
            fw.write('{}\t{}\n'.format(
                start_idx+i, id2rel[int(predict_label[i])]))


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


    return X_train, y_train, X_valid, y_valid


def traink(model, X_train, y_train, X_val, y_val, BATCH_SIZE, learning_rate, TOTAL_EPOCHS):
    train_loader = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        correct = 0  # 记录正确的个数，每个epoch训练完成之后打印accuracy
        for i, (images, labels) in enumerate(train_loader):
            images = images.float()
            labels = torch.squeeze(labels.type(torch.LongTensor))
            optimizer.zero_grad()  # 清零
            outputs = model(images)
            # 计算损失函数
            loss = criterion(outputs, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())
            # 计算正确率
            y_hat = model(images)
            pred = y_hat.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

            if (i + 1) % 10 == 0:
                # 每10个batches打印一次loss
                print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                                                                    i + 1, len(X_train) // BATCH_SIZE,
                                                                    loss.item()))
        accuracy = 100. * correct / len(X_train)
        print('Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'.format(
            epoch + 1, loss.item(), correct, len(X_train), accuracy))
        train_acc.append(accuracy)

        # 每个epoch计算测试集accuracy
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.float()
                labels = torch.squeeze(labels.type(torch.LongTensor))
                optimizer.zero_grad()
                y_hat = model(images)
                loss = criterion(y_hat, labels).item()  # batch average loss
                val_loss += loss * len(labels)  # sum up batch loss
                pred = y_hat.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()

        val_losses.append(val_loss / len(X_val))
        accuracy = 100. * correct / len(X_val)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            val_loss, correct, len(X_val), accuracy))
        val_acc.append(accuracy)

    return losses, val_losses, train_acc, val_acc


def train(model, criterion, loader, config):
    train_loader, dev_loader, _ = loader
    optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=1e-3)

    print(model)
    print('traning model parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('%s :  %s' % (name, str(param.data.shape)))
    print('--------------------------------------')
    print('start to train the model ...')

    eval_tool = Eval(config)
    max_f1 = -float('inf')
    current_lr = config.lr
    for epoch in range(1, config.epoch+1):
        if epoch > 5:
            current_lr *= 0.95
            change_lr(optimizer, current_lr)

        for step, (data, label) in enumerate(train_loader):
            model.train()
            data = data.to(config.device)
            label = label.to(config.device)

            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        _, train_loss, _ = eval_tool.evaluate(model, criterion, train_loader)
        f1, dev_loss, _ = eval_tool.evaluate(model, criterion, dev_loader)

        print('[%03d] train_loss: %.3f | dev_loss: %.3f | micro f1 on dev: %.4f'
              % (epoch, train_loss, dev_loss, f1), end=' ')
        if f1 > max_f1:
            max_f1 = f1
            torch.save(model.state_dict(), os.path.join(config.model_dir, 'model.pkl'))
            print('>>> save models!')
        else:
            print()

    def k_fold_train(model, criterion, X_train, y_train, config):

        train_loss_sum, valid_loss_sum = 0, 0
        train_acc_sum, valid_acc_sum = 0, 0

        for i in range(config.fold_k):
            print('*' * 25, '第', i + 1, '折', '*' * 25)
            data = get_kfold_data(config.fold_k, i, X_train, y_train)  # 获取k折交叉验证的训练和验证数据
            # 每份数据进行训练
            train_loss, val_loss, train_acc, val_acc = traink(model, criterion, *data, config)

            print('train_loss:{:.5f}, train_acc:{:.3f}%'.format(train_loss[-1], train_acc[-1]))
            print('valid loss:{:.5f}, valid_acc:{:.3f}%\n'.format(val_loss[-1], val_acc[-1]))

            train_loss_sum += train_loss[-1]
            valid_loss_sum += val_loss[-1]
            train_acc_sum += train_acc[-1]
            valid_acc_sum += val_acc[-1]

        print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)

        print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum / k, train_acc_sum / k))
        print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum / k, valid_acc_sum / k))

        return


if __name__ == '__main__':
    config = Config()
    print('--------------------------------------')
    print('some config:')
    config.print_config()

    print('--------------------------------------')
    print('start to load data ...')
    loader = MyDataLoader(config)

    word2id = loader.vocab.word2id
    word_vec = loader.word_emb
    rel2id = loader.rel2id
    id2rel = loader.id2rel
    class_num = loader.class_num

    train_dev_data, train_dev_labels = None, None
    if config.mode == 1:  # train mode
        train_dev_data, train_dev_labels = loader.train_data, loader.train_labels
    test_data, test_labels = loader.test_data, loader.test_labels

    print('load data finish!')

    print('--------------------------------------')



    if config.mode == 1:





