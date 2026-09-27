"""Primary nowcast (D1): U-Net, 13 VIL frames in -> 24 frames out (0-2 h).
Inputs also get IR + lightning so the model can see initiation before radar echo."""
import torch, torch.nn as nn, torch.nn.functional as F

N_IN, N_OUT = 13, 24


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, c_in=N_IN * 3, c_out=N_OUT, w=32):
        super().__init__()
        self.e1, self.e2, self.e3 = block(c_in, w), block(w, w * 2), block(w * 2, w * 4)
        self.mid = block(w * 4, w * 8)
        self.u3, self.d3 = nn.ConvTranspose2d(w * 8, w * 4, 2, 2), block(w * 8, w * 4)
        self.u2, self.d2 = nn.ConvTranspose2d(w * 4, w * 2, 2, 2), block(w * 4, w * 2)
        self.u1, self.d1 = nn.ConvTranspose2d(w * 2, w, 2, 2), block(w * 2, w)
        self.out = nn.Conv2d(w, c_out, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d = self.d3(torch.cat([self.u3(m), e3], 1))
        d = self.d2(torch.cat([self.u2(d), e2], 1))
        d = self.d1(torch.cat([self.u1(d), e1], 1))
        return torch.sigmoid(self.out(d))   # ReLU here let whole lead-time channels die (constant 0)


def make_input(vil, ir, lght):
    """Normalise a (N_IN,H,W) stack of each modality into model channels."""
    return torch.cat([torch.as_tensor(vil) / 255.0,
                      (300.0 - torch.as_tensor(ir)) / 100.0,
                      torch.log1p(torch.as_tensor(lght))], 0).float()


@torch.no_grad()
def predict(model, vil, ir, lght):
    model.eval()
    y = model(make_input(vil, ir, lght)[None])[0] * 255.0
    return y.clamp(0, 255).numpy()
