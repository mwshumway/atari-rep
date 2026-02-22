import torch.nn as nn

class Model(nn.Module):
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def encode(self, x):
        b, b_info = self.backbone(x)
        n, n_info = self.neck(b)

        info = {
            "backbone": b_info,
            "neck": n_info,
        }
        return n, info
    
    def forward(self, x):
        """
        :param x (torch.Tensor): (n, t, f, c, h, w)
        :return x (torch.Tensor): (n, t, d)
        
        (n, t, f, c, h, w) -> (n, t, c, h, w) -> (n, t, d) -> (n, t, d)
        """
        b, b_info = self.backbone(x)
        n, n_info = self.neck(x)
        h, h_info = self.head(x)

        info = {
            "backbone": b_info,
            "neck": n_info,
            "head": h_info,
        }

        return h, info