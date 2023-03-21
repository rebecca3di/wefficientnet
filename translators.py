import torch.nn as nn
import torch

class combine_translator(nn.Module):
    def __init__(self):
        super(combine_translator, self).__init__()
        self.amp_head = nn.Sequential(
            nn.Linear(1170, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 800),
        )

        self.pha_head = nn.Sequential(
            nn.Linear(1170, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 800),
        )

        head = [
            self.upconvblock(1, 3, 3, 2),
            # self.upconvblock(3, 3, 3, 2),
            self.upconvblock(3, 3, 5, 3),
            self.upconvblock(3, 3, 3, 2),
            self.upconvblock(3, 3, 3, 2),
            nn.Conv2d(3, 3, 1, 1, 0),
            nn.BatchNorm2d(3),
            # nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((184 * 2, 184 * 2)),
        ]
        self.head = nn.Sequential(*head)

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    @staticmethod
    def upconvblock(in_c, out_c, k, s):
        assert k % 2 == 1
        layers = [
            nn.ConvTranspose2d(in_c, out_c, k, s, k // 2, k // 2),
            nn.BatchNorm2d(out_c),
            # nn.Softplus(),
            nn.ReLU(),
            # nn.ConvTranspose2d(out_c, out_c, k, s, k // 2),
            # nn.BatchNorm2d(out_c),
            # # nn.Softplus(),
            # nn.ReLU(),
        ]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        amp_ = self.amp_head(x[0].reshape(x[0].shape[0], -1))
        pha_ = self.pha_head(x[1].reshape(x[1].shape[0], -1))
        x = torch.cat([amp_, pha_], -1).view([x[0].shape[0], 40, 40]).unsqueeze(1)
        # x = self.pool(x).unsqueeze(1)
        x = self.head(x)

        # x = F.sigmoid(x.float()).half() if (x.dtype == torch.float16) else F.sigmoid(x)
        # x = torch.clip(x, 0, 1)
        x = torch.sigmoid(x)

        x = torch.cat([(x[:, [_], ...] - self.mean[_]) / self.std[_] for _ in range(3)], dim=1)
        
        return x
