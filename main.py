import os
import argparse
import data_loader
import torch
import torch.nn.modules as nn
import model
from transformers import BertTokenizer
from MSA import MSA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#划分训练集、验证集的函数，仅在实验过程中第一次运行时调用，后续使用第一次运行后得到的结果作为训练集、验证集
def get_dev(folder_path, filename, dev_size=0.1):
    dev_size = dev_size if dev_size < 1 else int(dev_size)
    file_path = folder_path + filename
    with open(file_path, 'r', encoding="utf-8") as f:
        train_data = f.read().split("\n")[1:]
        while (True):
            try:
                train_data.remove("")
            except:
                break
        label = [0 for _ in range(len(train_data))]
    train_data, dev_data, train_label, dev_label = train_test_split(train_data, label, test_size=dev_size)
    with open(folder_path + '/train_with_label.txt', 'w', encoding='utf-8') as f:
        f.write("guid,tag\n")
        for data in train_data:
            f.write(data)
            f.write("\n")
        f.close()
    with open(folder_path + '/dev_with_label.txt', 'w', encoding='utf-8') as f:
        f.write("guid,tag\n")
        for data in dev_data:
            f.write(data)
            f.write("\n")
        f.close()
    print("split train data with: train {}, dev {}".format(len(train_data), len(dev_data)))  
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-run_type', type=str, default="train", help='train, test or predict')
    parse.add_argument('-train_type', type=str, default="dev", help='dev: origin train data -> train data + dev data; all: train on origin train data')
    parse.add_argument('-dev_size', type=float, default="0.1", help='the number(>= 1) or percent(< 1) of dev')
    parse.add_argument('-save_model_path', type=str, default='model', help='save the good model.pth path')
    parse.add_argument('-text', action='store_true', default=False, help='input text when run dev')
    parse.add_argument('-image', action='store_true', default=False, help='input image when run dev')
    #以下保留部分原代码中使用的命令行参数，主要内容为模型和训练过程中使用的参数
    parse.add_argument('-epoch', type=int, default=30, help='train epoch num')
    parse.add_argument('-batch_size', type=int, default=8, help='batch size number')
    parse.add_argument('-lr', type=float, default=2e-5, help='learning rate')
    opt = parse.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    assert torch.cuda.is_available()
    loss_function = nn.CrossEntropyLoss().cuda()
    fuse_model = model.FuseModel().cuda()

    print('Init Data Process:')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased/vocab.txt')

  
    if opt.run_type == "train":
        #使用全部训练集训练模型，用于预测最后的结果文件
        if opt.train_type == 'all':
            train_loader, _ = data_loader.get_data(opt, "./data/", 'train.txt', tokenizer, data_type='train')
            dev_loader = None
        #使用训练集训练模型，使用验证集查看模型效果，用于实验过程
        elif opt.train_type == 'dev':
            get_dev("./data/", filename='/train.txt', dev_size=opt.dev_size)
            train_loader, _ = data_loader.get_data(opt, "./data/", 'train_with_label.txt', tokenizer, data_type='train')
            dev_loader, _ = data_loader.get_data(opt, "./data/", 'dev_with_label.txt', tokenizer, data_type='dev')
        msa = MSA(opt, train_loader, dev_loader, None, fuse_model, loss_function)
        msa.train()
    elif opt.run_type == 'test':
        # 通过训练出来的模型(不是最后用来预测结果的模型,是有划分验证集的模型)分别在验证集上查看模型输入双模态、单模态的效果
        model_path = opt.save_model_path + "/best-model.pth"
        fuse_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        dev_loader, _ = data_loader.get_data(opt, "./data/", 'dev_with_label.txt', tokenizer, data_type='dev')
        msa = MSA(opt, None, dev_loader, None, fuse_model, loss_function)
        msa.dev(input_text=opt.text, input_image=opt.image)
    elif opt.run_type == "predict":
        #通过训练出来的模型(全部训练集训练的模型)预测最后的结果文件
        model_path = opt.save_model_path + "/best_model_with_all_train.pth"
        fuse_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        test_loader, error_guid = data_loader.get_data(opt, "./data/", 'test_without_label.txt', tokenizer, data_type='test')
        msa = MSA(opt, None, None, test_loader, fuse_model, loss_function)
        # print(error_guid_and_guid_list[0])
        # print(error_guid_and_guid_list[1])
        msa.predict(error_guid)
    else:
        print("no run type named {}".format(opt.run_type))