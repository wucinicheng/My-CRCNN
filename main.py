import os
import numpy as np
import torch
from torch import optim, nn
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange, tqdm

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


def traink(train_dev_data, train_index, dev_index, model, criterion, optimizer, config):
    # 从StratifiedKFold提取对应折数的数据集
    train_data   = torch.from_numpy(np.array(train_dev_data)[train_index]).squeeze(dim=1).to(config.device)
    dev_data     = torch.from_numpy(np.array(train_dev_data)[dev_index]).squeeze(dim=1).to(config.device)
    train_labels = torch.from_numpy(np.array(train_dev_labels)[train_index]).to(config.device)
    dev_labels   = torch.from_numpy(np.array(train_dev_labels)[dev_index]).to(config.device)
    # 使用TensorDataset封装数据集
    train_dataset = TensorDataset(train_data, train_labels)
    dev_dataset   = TensorDataset(dev_data, dev_labels)
    # DataLoader转化
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
    val_loader = DataLoader(dev_dataset, config.batch_size, shuffle=True)

    # print(model)
    # print('traning model parameters:')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print('%s :  %s' % (name, str(param.data.shape)))
    #
    # print('--------------------------------------')
    # print('start to train the model ...')
    print('--------------------------------------')
    print('start to train the model ...')

    current_lr = config.lr
    best_f1_score = -9999.
    for epoch in trange(1, config.epoch+1):
        if epoch > 50:
            current_lr *= 0.95
            change_lr(optimizer, current_lr)

        tbar = tqdm(train_loader)
        for step, (data, label) in enumerate(tbar):
            model.train()
            data = data.to(config.device)
            label = label.to(config.device)
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            desc = 'Training: Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % \
                   (epoch, config.epoch, step + 1, len(train_loader), loss.item())
            tbar.set_description(desc)
            tbar.update()


        model.eval()
        predict_labels = []
        true_labels    = []
        val_loss = 0.
        with torch.no_grad():
            tbar = tqdm(val_loader)
            for i, (data, labels) in enumerate(tbar):
                data = data.to(config.device)
                labels = labels.to(config.device)

                optimizer.zero_grad()
                scores = model(data)
                loss = criterion(scores, labels)

                val_loss += loss.item() * scores.shape[0]

                pred = scores.max(1, keepdim=True)[1].squeeze(dim=-1).cpu().detach().numpy().tolist()
                labels = labels.cpu().detach().numpy().tolist()

                predict_labels.extend(pred)
                true_labels.extend(labels)

                desc = 'Validation: Epoch %d, Average loss: %.4f' % (epoch, loss)
                tbar.set_description(desc)
                tbar.update()

        avg_val_loss = val_loss / len(dev_data)
        val_f1_score = f1_score(true_labels, predict_labels, average="micro")
        accuracy = accuracy_score(true_labels, predict_labels)
        print('Validation set: Average loss: {:.4f}, F1 score: {:.4f}, Accuracy:{:.3f}%\n'.format(avg_val_loss, val_f1_score, accuracy))

        if best_f1_score < val_f1_score:
            best_f1_score = val_f1_score
            torch.save(model.state_dict(), os.path.join(config.model_dir,
                                                        'model_epoch%s_%s.pkl' % (str(epoch), str(val_f1_score))))

    return best_f1_score


def k_fold_train(train_dev_data, kfolder, config):
    f1_score = 0.0

    for train_index, dev_index in kfolder: # 获取k折交叉验证的训练和验证数据

        model = CRCNN(word_vec=word_vec, class_num=class_num, config=config)
        model.to(config.device)
        criterion = PairwiseRankingLoss(config=config)
        optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=1e-3)
        # 每份数据进行训练
        best_f1_score = traink(train_dev_data, train_index, dev_index, model, criterion, optimizer, config)
        f1_score += best_f1_score

    print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)
    print('average F1 score:{:.4f}'.format(f1_score / config.fold_k))

def test(data, lables, config):
    print('--------------------------------------')
    print('start test ...')
    model = CRCNN(word_vec=word_vec, class_num=class_num, config=config)
    model.to(config.device)
    model.load_state_dict(torch.load(os.path.join(config.model_dir, 'model.pkl')))

    model.eval()
    predict_labels = []
    true_labels = []
    val_loss = 0.
    with torch.no_grad():
        tbar = tqdm(val_loader)
        for i, (data, labels) in enumerate(tbar):
            data = data.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, labels)

            val_loss += loss.item() * scores.shape[0]

            pred = scores.max(1, keepdim=True)[1].squeeze(dim=-1).cpu().detach().numpy().tolist()
            labels = labels.cpu().detach().numpy().tolist()

            predict_labels.extend(pred)
            true_labels.extend(labels)

            desc = 'Validation: Epoch %d, Average loss: %.4f' % (epoch, loss)
            tbar.set_description(desc)
            tbar.update()

    avg_val_loss = val_loss / len(dev_data)
    val_f1_score = f1_score(true_labels, predict_labels, average="micro")
    accuracy = accuracy_score(true_labels, predict_labels)
    print('Test set: Average loss: {:.4f}, F1 score: {:.4f}, Accuracy:{:.3f}%\n'.format(avg_val_loss, val_f1_score,
                                                                                        accuracy))

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
        skf = StratifiedKFold(n_splits=config.fold_k, shuffle=False)
        kfolder = skf.split(train_dev_data, train_dev_labels)
        k_fold_train(train_dev_data, kfolder, config)


    test(test_data, test_labels, config)






