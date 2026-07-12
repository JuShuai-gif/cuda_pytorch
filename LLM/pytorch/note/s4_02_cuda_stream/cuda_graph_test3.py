import gc
import time

import torch

from torch import nn

class PaddedGraphModel(nn.Module):
    SUPPORTED_BS = [1,2,4,8,16]
    
    def __init__(self,dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,dim),            
        ).cuda()
        
        self.graphs: dict[int,torch.cuda.CUDAGraph] = {}
        self._capture()
        
    def _capture(self):
        for bs in self.SUPPORTED_BS:
            x = torch.randn(bs,256,device="cuda")
            # warmup
            for _ in range(3):
                self.net(x)
            torch.cuda.synchronize()
            
            
            g = torch.cuda.CUDAGraph()
            
            # 
            buf = torch.randn(bs,256,device="cuda")
            with torch.cuda.graph(g):
                out = self.net(buf)
                
            self.graphs[bs] = (g,buf,out)
            
    def forward(self, x: torch.Tensor):
        bs = x.size(0)
        padded_bs = min(b for b in self.SUPPORTED_BS if b >= bs)
        g,buf,out = self.graphs[padded_bs]
        
        # pad if needed
        if bs < padded_bs:
            x_padded = torch.cat([x,x[:padded_bs - bs]])
        else:
            x_padded = x
        
        buf.copy_(x_padded)
        g.replay()
        return out[:bs]
    
if __name__ == "__main__":
    model = PaddedGraphModel()
    
    for bs in [1,2,4,1,8,3,16,5]:
        x = torch.randn(bs,256,device="cuda")
        y = model(x)
        print(f"bs={bs:>2d}  output shape = {list(y.shape)}")






