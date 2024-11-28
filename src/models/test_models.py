import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexNN(nn.Module):
    """用于测试优化器的复杂神经网络"""
    def __init__(self, input_dim=784, hidden_dims=[512, 384, 256, 128], num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_dropout = nn.Dropout(0.1)  # 轻微的输入dropout
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # 隐藏层
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.Dropout(0.2 + 0.1 * i)  # 逐渐增加dropout率
            ))
            
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # 输入预处理
        x = self.input_bn(x)
        x = self.input_dropout(x)
        x = F.relu(self.layers[0](x))
        
        # 隐藏层
        for layer in self.layers[1:]:
            x = layer(x)
        
        # 输出层
        x = self.output(x)
        return x  # 移除log_softmax，使用普通的CrossEntropyLoss

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ComplexCNN(nn.Module):
    """复杂CNN模型"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class LSTM_Attention(nn.Module):
    """带注意力机制的LSTM模型"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                           bidirectional=True)
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        # LSTM输出
        lstm_out, _ = self.lstm(embedded)  # (batch_size, sequence_length, hidden_dim*2)
        
        # 注意力权重
        attention_weights = self.attention(lstm_out)  # (batch_size, sequence_length, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 应用注意力
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_dim*2)
        
        # 分类
        output = self.fc(context)
        return output
