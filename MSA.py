import torch
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from matplotlib import pyplot as plt
class MSA():
    def __init__(self, opt, train_loader, dev_loader, test_loader, model, loss_function):
        self.opt = opt
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model = model
        self.loss_function = loss_function
        print("create new MAA success......")
    def get_param(self, data, input_text=True, input_image=True):
        texts, bert_attention_mask, images, labels = data
        if not input_text:
            texts, bert_attention_mask = torch.zeros_like(texts), torch.zeros_like(bert_attention_mask)
        if not input_image:
            images = torch.zeros_like(images)
        return texts.cuda(), bert_attention_mask.cuda(), images.cuda(), labels.cuda()
    def train(self):
        print("start to train.......")
        accuracy_list, precision_list, recall_list, F1_list = [], [], [], []
        optimizer = AdamW(self.model.parameters(), lr=self.opt.lr)
        best_F1, best_model = 0, self.model
        for epoch in tqdm(range(self.opt.epoch)):
            y_true, y_pred = [], []
            self.model.train()
            self.model.zero_grad()

            for data in tqdm(self.train_loader):
                texts, bert_attention_mask, images, labels = self.get_param(data)
                output = self.model([texts, bert_attention_mask, images])
                loss = self.loss_function(output, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                y_true.extend(labels.cpu())
                y_pred.extend(torch.max(output, 1)[1].cpu())
            print('Epoch: %d:\nTrain: F1(macro): %.6f' % (epoch, f1_score(np.array(y_true), np.array(y_pred), average='macro')))
            if self.dev_loader is not None:
                best_F1, best_model, dev_accuracy, dev_precision, dev_R, dev_F1 = self.dev(best_F1, best_model)
                accuracy_list.append(dev_accuracy)
                precision_list.append(dev_precision)
                recall_list.append(dev_R)
                F1_list.append(dev_F1)
        if self.dev_loader is not None:
            print("best Dev F1(macro): %.6f" % best_F1)
            print("save model to best-model.pth")
            torch.save(best_model.state_dict(), self.opt.save_model_path + '/best-model.pth')
            index_list = list(range(1, self.opt.epoch + 1))
            plt.plot(index_list, accuracy_list, label="accuarcy")
            plt.plot(index_list, precision_list, label="precision")
            plt.plot(index_list, recall_list, label="recall")
            plt.plot(index_list, F1_list, label="F1(macro)")
            plt.legend()
            plt.savefig('./image/dev.png')
            plt.show()
            
        else:
            print("save model to best_model_with_all_train.pth")
            torch.save(self.model.state_dict(), self.opt.save_model_path + '/best_model_with_all_train.txt')
    def dev(self, best_F1=0, best_model = None, input_text=True, input_image=True):
        y_true, y_pred = [], []
        with torch.no_grad():
            self.model.eval()
            for data in tqdm(self.dev_loader):
                texts, bert_attention_mask, images, labels = self.get_param(data, input_text, input_image)
                output = self.model([texts, bert_attention_mask, images])
                y_true.extend(labels.cpu())
                y_pred.extend(torch.max(output, 1)[1].cpu())

            dev_accuracy = accuracy_score(y_true, y_pred)
            dev_F1 = f1_score(y_true, y_pred, average='macro')
            dev_R = recall_score(y_true, y_pred, average='macro')
            dev_precision = precision_score(y_true, y_pred, average='macro')
            print(classification_report(y_true, y_pred, target_names=["positive", "neutral", "negative"]))
            if dev_F1 > best_F1:
                best_F1, best_model = dev_F1, self.model
            return best_F1, best_model, dev_accuracy, dev_precision, dev_R, dev_F1
    def predict(self, error_guid):
        y_pred = []
        print("error guid:{}".format(error_guid))
        with torch.no_grad():
            self.model.eval()
            for data in tqdm(self.test_loader):
                texts, bert_attention_mask, images, _ = self.get_param(data)
                output = self.model([texts, bert_attention_mask, images])
                y_pred.extend(torch.max(output, 1)[1].cpu())
        with open('./data/test_without_label.txt', 'r', encoding="utf-8") as f:
            data = f.read().split("\n")[1:]
            f.close()
        with open('./data/result.txt', 'w', encoding="utf-8") as f:
            f.write("guid,tag\n")
            label_dic = {0: "positive", 1: "neutral", 2: "negative"}
            pred_index, data_index = 0, 0
            while pred_index < len(y_pred) and data_index < len(data):
                guid = data[data_index].split(",")[0]
                if guid in error_guid:
                    f.write(guid + ',' + 'neutral' + '\n')
                    data_index += 1
                    continue
                f.write(guid + ',' + label_dic[int(y_pred[pred_index])] + '\n')
                data_index += 1
                pred_index += 1
            f.close()