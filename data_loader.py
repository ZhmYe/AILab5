from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import torch.nn.utils.rnn as run_utils
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
label_dic = {'positive': 0, 'neutral': 1, 'negative': 2, 'null': 3}
class MyDataset(Dataset):
    def __init__(self, folder_path, filename, text_tokenizer, image_transforms, data_type):
        self.folder_path = folder_path
        self.image_transforms = image_transforms
        self.data_type = data_type
        self.error_guid = []
        with open(folder_path + filename, "r", encoding='utf-8') as f: 
            guid_tags = f.read().split("\n")[1:]
            while True:
                try:
                    guid_tags.remove("")
                except:
                    break
            f.close()
        self.guid_list, self.text_list, self.label_list= [], [], []
        for guid_tag in guid_tags:
            guid, label = guid_tag.split(",")
            label = label_dic[label]    
            try:
                with open(folder_path + '/data/{}.txt'.format(guid), "r", encoding='utf-8') as f:
                    txt = f.read()
                    f.close()
            except:
                try:
                    with open(folder_path + '/data/{}.txt'.format(guid), "r", encoding="ISO-8859-1") as f:
                        txt = f.read().encode("iso-8859-1").decode('gbk')
                        f.close()
                except:
                    print("/data/{}.txt read error!".format(guid))
                    self.error_guid.append(guid)
                    continue
            self.guid_list.append(guid)
            self.label_list.append(label)
            self.text_list.append(txt)
        self.text_to_id = [text_tokenizer.convert_tokens_to_ids(text_token) for text_token in tqdm([text_tokenizer.tokenize('[CLS]' + text + '[SEP]') for text in tqdm(self.text_list)])]
    def __len__(self):
        return len(self.text_to_id)

    def __getitem__(self, index):
        image_path = self.folder_path + '/data/' + str(self.guid_list[index]) + '.jpg'
        image_read = Image.open(image_path)
        image_read.load()
        return self.text_to_id[index], self.image_transforms(image_read), self.label_list[index]
    def get_error_guid(self):
        return self.error_guid

def collate(batch_data):
    text_to_id = [torch.LongTensor(data[0]) for data in batch_data]
    image = torch.FloatTensor([np.array(data[1]) for data in batch_data])
    label = torch.LongTensor([data[2] for data in batch_data])
    max_length = max([text.size(0) for text in text_to_id])
    bert_attention_mask = []
    for text in text_to_id:
        text_mask_cell = [1] * text.size(0)
        text_mask_cell.extend([0] * (max_length - text.size(0)))
        bert_attention_mask.append(text_mask_cell[:])
    text_to_id = run_utils.pad_sequence(text_to_id, batch_first=True, padding_value=0)
    return text_to_id, torch.LongTensor(bert_attention_mask), image, label


def get_data(opt, folder_path, filename, text_tokenizer, data_type):
    transform = transforms.Compose([
                        transforms.Resize(256),                    
                        transforms.CenterCrop(224),                
                        transforms.ToTensor(),                     
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]                  
                        )])
    dataset = MyDataset(folder_path, filename, text_tokenizer, transform, data_type)
    error_guid = dataset.get_error_guid()
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True if data_type == 'train' else False, collate_fn=collate, pin_memory=True)
    return data_loader, error_guid
