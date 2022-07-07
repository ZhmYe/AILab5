import torch.nn.modules as nn
import torchvision.models
import torch
from transformers import BertConfig, BertForPreTraining
class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.config = BertConfig.from_pretrained('bert-base-uncased/')
        self.model = BertForPreTraining.from_pretrained('bert-base-uncased/', config=self.config)
        self.model = self.model.bert
        for param in self.model.parameters():
            param.requires_grad = True
    def forward(self, input, attention_mask):
        return self.model(input, attention_mask=attention_mask)
class FuseModel(nn.Module):
    def __init__(self):
        super(FuseModel, self).__init__()
        self.text_model = Bert()
        image_model = torchvision.models.resnet50(pretrained=True)
        image_model.eval()
        self.image_shape = image_model.fc.in_features
        layers = list(image_model.children())[:-1] #不需要最后一层线性层
        self.image_model = nn.Sequential(*layers)
        self.image_linear = nn.Sequential(
            nn.Linear(in_features=self.image_shape, out_features=32),
            nn.ReLU(inplace=True)
        )
        self.txt_linear = nn.Sequential(
            nn.Linear(in_features=self.text_model.config.hidden_size, out_features=32),
            nn.ReLU(inplace=True)
        )
        self.fuse = torch.nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=False, dropout=0.4)
        self.linear = nn.Sequential(
                            nn.Dropout(),
                            nn.Linear(in_features=64, out_features=128),
                            nn.ReLU(inplace=True),
                            nn.Dropout(),
                            nn.Linear(in_features=128, out_features=3)
                        )

    def forward(self, input):
        txt, bert_attention_mask, image = input
        txt = self.txt_linear(self.text_model(txt, attention_mask=bert_attention_mask).last_hidden_state[:, 0, :]).unsqueeze(dim=0)
        image = self.image_linear(self.image_model(image).flatten(1)).unsqueeze(dim=0)
        fusion = self.fuse(torch.cat((image, txt), dim=2)).squeeze()
        return self.linear(fusion)
