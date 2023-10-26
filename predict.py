import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

def read_data(path, num=None):
    with open(path, 'r', encoding='utf-8') as f:
        all_data = f.read().split('\n\n')
        if num == None:
            return all_data[:-1]
        elif num > 0:
            return all_data[:-1][:num]
        elif num < 0:
            return all_data[:-1][num:]

def get_word_2_index(file):
    with open(file) as f:
        index2word = f.read().split('\n')
    word2index = {w:i for i,w in enumerate(index2word)}

    return word2index, index2word

class MyDataset(Dataset):
    def __init__(self, train_data, word2index):
        self.train_data = train_data
        self.word2index = word2index
    def __getitem__(self, index):
        # 需要转换成index,同时进行填充和裁剪等处理
        text_data = self.train_data[index]
        text_data = text_data.split('\n')
        text_idx = []
        for data in text_data:
            text_idx.extend([self.word2index.get(i, 1) for i in data])
            text_idx.append(2)
        input_idx = text_idx[:-1]
        label_idx = text_idx[1:]

        assert len(input_idx) == len(label_idx), "input和label长度不一致"

        return input_idx, label_idx, len(label_idx)
    
    def __len__(self):
        return len(self.train_data)
    
    def pro_data(self, batch_data):
        # 对一个batch进行填充和裁剪等处理
        batch_text_idx, batch_label_idx, batch_len = zip(*batch_data)
        batch_max_len = max(batch_len)
        batch_text_new = []
        batch_label_new = []
        for text_idx, label_idx in zip(batch_text_idx, batch_label_idx):
            text_idx = text_idx + [0]*(batch_max_len - len(text_idx))
            batch_text_new.append(text_idx)
            label_idx = label_idx + [0]*(batch_max_len - len(label_idx))
            batch_label_new.append(label_idx)
        
        return torch.tensor(batch_text_new), torch.tensor(batch_label_new)

# embeding有一个位置编码也有一个字符编码        
class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Embedding(seq_max_len, emb_size)
        self.token_embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        seq_len = x.shape[1]
        position = torch.arange(0, seq_len, device=x.device)
        position = position.reshape(1,-1)
        position = position.expand_as(x)
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(position)
        emb = token_emb + pos_emb
        return emb

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_num = 4
        self.Q = nn.Linear(emb_size, emb_size)
        self.K = nn.Linear(emb_size, emb_size)
        self.V = nn.Linear(emb_size, emb_size)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x, attention_mask):
        batch, seq_len, emb_dim = x.shape
        copy_x = x
        q = self.Q.forward(x)
        k = self.K.forward(x)
        v = self.V.forward(x)
        q = q.reshape(batch, seq_len, self.head_num, -1).transpose(1,2)
        k = k.reshape(batch, seq_len, self.head_num, -1).transpose(1,2)
        v = v.reshape(batch, seq_len, self.head_num, -1).transpose(1,2)
        weight = q @ k.transpose(-2,-1) / int(k.shape[-1] ** 0.5)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(dim=1)
            attention_mask = attention_mask.repeat(1, self.head_num, 1, weight.shape[-1])
            look_head_mask = torch.triu(torch.ones_like(attention_mask), 1).to(x.device)
            attention_mask = attention_mask | look_head_mask
            weight.masked_fill_(attention_mask, 1e-9)
        weight = self.softmax(weight)
        x = weight @ v
        x = x.transpose(1, 2).reshape(batch, seq_len, -1)
        x = self.norm(x + copy_x)
        return x

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(emb_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, emb_size)
        self.norm = nn.LayerNorm(emb_size)
    def forward(self, x):
        copy_x = x
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.norm(x + copy_x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_block1 = MultiHeadAttention()
        self.attention_block2 = MultiHeadAttention()
        self.feed_forward = FeedForward()
    def forward(self, x, attention_mask):
        x = self.attention_block1(x, attention_mask)
        x = self.attention_block2(x, attention_mask)
        x = self.feed_forward(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = Embedding()
        n = 3
        # *号代表把这个列表解开，另一种写法使用ModuleList但是调用时需要添加循环,但是使用nn.sequntial参数不好控制
        # self.layers = nn.Sequential(*[DecoderLayer() for i in range(n)])
        self.layers = nn.ModuleList([DecoderLayer() for i in range(n)])

    def forward(self, x):
        attention_mask = get_attention_mask(x)
        emb = self.emb(x)
        for layer in self.layers:
            output = layer.forward(emb, attention_mask)
        return output
    
class GPT_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.cls = nn.Linear(emb_size, vocab_size)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        decode_out = self.decoder.forward(x)
        pre = self.cls(decode_out)
        # loss直接接受二维的输入 label.reshape(-1)小细节
        if label is not None:
            loss = self.loss_fun(pre.reshape(-1,pre.shape[-1]), label.reshape(-1))
            return loss
        else:
            pre = torch.argmax(pre, dim=-1)
            return pre
    
    def answer(self, input_text):
        input_idx = [word2index.get(i, 1) if i != '\n' else 2 for i in input_text]
        input_idx = torch.tensor([input_idx], device=device)
        pre_max_len = 100
        while True:
            pre = int(self.forward(input_idx)[0][-1])
            if len(input_idx) > pre_max_len or pre == 2:
                break
            # word = index2word(pre)
            input_idx = torch.cat((input_idx, torch.tensor([[pre]], dtype=input_idx.dtype, device=input_idx.device)), -1)
        input_idx = input_idx[0].tolist()
        output_text_list = [index2word[i] for i in input_idx]
        output_text = "".join(output_text_list)
        return output_text
         
def get_attention_mask(x):
    # mask是屏蔽操作，只要为true就进行填充,可以把注意力矩阵进行mask
    padding_position = x == 0
    padding_position = torch.unsqueeze(padding_position, dim=-1)
    # padding_position = padding_position.expand(*x.shape, emb_size)
    return padding_position

if __name__ == "__main__":
    train_data = read_data(os.path.join('data', 'train.txt'), num=10000)
    test_data = read_data(os.path.join('data', 'train.txt'), num=200)
    word2index, index2word = get_word_2_index(os.path.join('wangmenghao','data', 'vocab.txt'))
    batch_size = 10
    epoch = 100
    # embedding 把22，46的数据变为，22，46，768
    emb_size = 768
    hidden_size = 1024
    seq_max_len = 512
    vocab_size = len(index2word)
    mode = 'train'
    lr = 0.0001

    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    train_dataset = MyDataset(train_data, word2index)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False, collate_fn=train_dataset.pro_data)
    test_dataset = MyDataset(test_data, word2index)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=test_dataset.pro_data)

    model = GPT_Model().to(device)
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    # if mode == 'train':
        # best_loss = 1e7
    for e in range(epoch):
        for batch_text_idx, batch_label_idx in tqdm(train_dataloader):
            batch_text_idx = batch_text_idx.to(device)
            batch_label_idx = batch_label_idx.to(device)
            loss = model.forward(batch_text_idx, batch_label_idx)
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(f'loss={loss:.3f}')
        # if loss < best_loss:
        #     best_loss = loss
        #     torch.save(model, 'best_model.pt')
    # if mode == 'eval':
    # model = torch.load('best_model.pt')
    while True:
        input_text = input("请输入：") + '\n'
        if input_text == '\n':
            break
        answer = model.answer(input_text)
        print(answer)
        # num = 0
        # for batch_text_idx, batch_label_idx in tqdm(test_dataloader):
        #     pre = model.forward(batch_text_idx, batch_label_idx)
        #     num += torch.sum(pre == batch_label_idx)
        # acc = num / len(train_dataset)
        # print(f'acc={acc:.3f}')