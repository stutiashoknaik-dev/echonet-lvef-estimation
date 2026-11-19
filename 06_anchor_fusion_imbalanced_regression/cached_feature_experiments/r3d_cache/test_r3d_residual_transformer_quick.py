
# ============================================================
# FINAL TEST: R3D-18 CACHE + RESIDUAL TRANSFORMER
# ============================================================

import os, math, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE=os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")
CACHE=os.path.join(BASE,"outputs","stage2_3dcnn_cache")
SAVE=os.path.join(BASE,"outputs","r3d_residual_transformer_quick")
BEST=os.path.join(SAVE,"best_model.pth")
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH=64; D_MODEL=256; NHEAD=8; LAYERS=2; FF=512; DROP=.20; GLOBAL_DIM=16; ATTN_H=128

def load_cache(split):
    d=torch.load(os.path.join(CACHE,f"{split}_fused.pt"),map_location="cpu")
    return d["fused"].float(),d["global_delta"].float(),d["ef"].float(),d["video_ids"]

class AttnPool(nn.Module):
    def __init__(self,d,h=128):
        super().__init__(); self.score=nn.Sequential(nn.Linear(d,h),nn.Tanh(),nn.Linear(h,1))
    def forward(self,x):
        w=torch.softmax(self.score(x),1); return (x*w).sum(1)

class ResidualTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        layer=nn.TransformerEncoderLayer(D_MODEL,NHEAD,FF,DROP,activation="gelu",batch_first=True,norm_first=True)
        self.tr=nn.TransformerEncoder(layer,LAYERS)
        self.norm=nn.LayerNorm(D_MODEL)
    def forward(self,x):
        return self.norm(x+self.tr(x))

class Model(nn.Module):
    def __init__(self,inp,seq):
        super().__init__()
        self.proj=nn.Sequential(nn.LayerNorm(inp),nn.Linear(inp,D_MODEL))
        self.pos=nn.Parameter(torch.randn(1,seq,D_MODEL)*.02)
        self.res_tr=ResidualTransformerBlock()
        self.pool=AttnPool(D_MODEL,ATTN_H)
        self.g=nn.Sequential(nn.LayerNorm(4),nn.Linear(4,GLOBAL_DIM),nn.ReLU())
        self.reg=nn.Sequential(nn.Linear(D_MODEL+GLOBAL_DIM,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.25),
                               nn.Linear(256,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.15),nn.Linear(64,1))
    def forward(self,x,g):
        x=self.proj(x); x=x+self.pos[:,:x.size(1)]; x=self.res_tr(x); x=self.pool(x); g=self.g(g)
        return self.reg(torch.cat([x,g],1)).squeeze(1)

def metrics(y,p):
    c=pearsonr(y,p)[0] if np.std(y)>0 and np.std(p)>0 else np.nan
    return {"MAE":float(mean_absolute_error(y,p)),"RMSE":float(math.sqrt(mean_squared_error(y,p))),
            "R2":float(r2_score(y,p)),"Pearson_r":float(c),"Bias":float(np.mean(p-y))}

def buckets(y,p):
    d=pd.DataFrame({"GT_EF":y,"Pred_EF":p}); d["Abs_Error"]=(d["Pred_EF"]-d["GT_EF"]).abs()
    d["Bucket"]=pd.cut(d["GT_EF"],[0,30,40,50,60,70,100],labels=["<30","30-40","40-50","50-60","60-70","70+"])
    return d.groupby("Bucket",observed=True).agg(n=("Abs_Error","size"),MAE=("Abs_Error","mean")).reset_index()

@torch.no_grad()
def main():
    print("\n===== FINAL TEST: R3D + RESIDUAL TRANSFORMER =====")
    print("Device:",DEVICE)
    if DEVICE.type=="cuda": print("GPU:",torch.cuda.get_device_name(0))

    x,g,y,ids=load_cache("test")
    model=Model(x.shape[-1],x.shape[1]).to(DEVICE)
    ck=torch.load(BEST,map_location=DEVICE)
    model.load_state_dict(ck["model_state_dict"],strict=True); model.eval()

    print("Checkpoint loaded. Best epoch:",ck.get("epoch","?"))
    print("Best Val MAE:",ck.get("best_val_mae","?"))

    dl=DataLoader(TensorDataset(x,g),BATCH,shuffle=False)
    pred=[]
    for xb,gb in dl:
        pred.append(model(xb.to(DEVICE),gb.to(DEVICE)).cpu())

    p=np.clip(torch.cat(pred).numpy(),0,100); yy=y.numpy()
    m=metrics(yy,p); b=buckets(yy,p)

    print("\nTEST:",m)
    print(b.to_string(index=False))

    pd.DataFrame({"Video":ids,"GT_EF":yy,"Pred_EF":p}).to_csv(os.path.join(SAVE,"test_predictions.csv"),index=False)
    pd.DataFrame([m]).to_csv(os.path.join(SAVE,"test_metrics.csv"),index=False)
    b.to_csv(os.path.join(SAVE,"test_bucket_metrics.csv"),index=False)

if __name__=="__main__": main()
