import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        self.cnn = nn.Sequential(
            # Conv1: Conv(1->64, 3x3) + ReLU + MaxPool(2x2)
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # H -> H/2, W -> W/2
            
            # Conv2: Conv(64->128, 3x3) + ReLU + MaxPool(2x2)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # H/2 -> H/4, W/2 -> W/4
            
            # Conv3: Conv(128->256, 3x3) + BN + ReLU
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Conv4: Conv(256->256, 3x3) + ReLU + MaxPool(2x1, pad 0x1)
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1), (0, 0)), # H/4 -> H/8, W/4 -> W/4
            
            # Conv5: Conv(256->512, 3x3) + BN + ReLU
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Conv6: Conv(512->512, 3x3) + ReLU + MaxPool(2x1, pad 0x1)
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1), (0, 0)), # H/8 -> H/16, W/4 -> W/4
            
            # Conv7: Conv(512->512, 2x2) + BN + ReLU
            nn.Conv2d(512, 512, 2, padding=0), # H/16 -> H/16-1, W/4 -> W/4-1
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, batch_first=True),
            # Dropout will be applied manually or added in forward, let's keep it simple
        )
        self.dropout = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x: (B, 1, 64, 1024)
        conv = self.cnn(x) # (B, 512, H', W')
        
        # Collapse height
        conv = self.pool(conv) # (B, 512, 1, W')
        conv = conv.squeeze(2) # (B, 512, W')
        
        # RNN needs (B, T, C)
        conv = conv.permute(0, 2, 1) # (B, W', 512)
        
        # BiLSTM
        out, _ = self.rnn[0](conv)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        
        # FC
        out = self.fc(out) # (B, W', num_classes)
        return out
