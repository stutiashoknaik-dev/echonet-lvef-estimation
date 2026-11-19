
# ============================================================
# QUICK SCREEN: R3D-18 CACHE + RESIDUAL TRANSFORMER
# ============================================================
# Reuses the existing cache:
#   [N, 8, 339]
#
# No new video extraction.
# Only the temporal head is changed.
#
# Architecture:
#   cached R3D multimodal features
#        -> projection
#        -> Transformer
#        -> residual skip
#        -> temporal attention
#        -> global-delta
#        -> EF regression
# ============================================================

import os, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE=os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")
CACHE=os.path.join(BASE,"outputs","stage2_3dcnn_cache")
SAVE=os.path.join(BASE,"outputs","r3d_residual_transformer_quick")
os.makedirs(SAVE,exist_ok=True)

BEST=os.path.join(SAVE,"best_model.pth")
HIST=os.path.join(SAVE,"training_history.csv")

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH=64
EPOCHS=70
PATIENCE=12
LR=3e-4
WD=1e-4
D_MODEL=256
NHEAD=8
LAYERS=2
FF=512
DROP=.20
GLOBAL_DIM=16
ATTN_H=128
SEED=42

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_cache(split):
    d=torch.load(os.path.join(CACHE,f"{split}_fused.pt"),map_location="cpu")
    return d["fused"].float(),d["global_delta"].float(),d["ef"].float(),d["video_ids"]

class AttnPool(nn.Module):
    def __init__(self,d,h=128):
        super().__init__()
        self.score=nn.Sequential(nn.Linear(d,h),nn.Tanh(),nn.Linear(h,1))
    def forward(self,x):
        w=torch.softmax(self.score(x),1)
        return (x*w).sum(1)

class ResidualTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        layer=nn.TransformerEncoderLayer(
            d_model=D_MODEL,nhead=NHEAD,dim_feedforward=FF,
            dropout=DROP,activation="gelu",
            batch_first=True,norm_first=True
        )
        self.tr=nn.TransformerEncoder(layer,num_layers=LAYERS)
        self.norm=nn.LayerNorm(D_MODEL)
    def forward(self,x):
        return self.norm(x + self.tr(x))

class Model(nn.Module):
    def __init__(self,inp,seq):
        super().__init__()
        self.proj=nn.Sequential(nn.LayerNorm(inp),nn.Linear(inp,D_MODEL))
        self.pos=nn.Parameter(torch.randn(1,seq,D_MODEL)*.02)
        self.res_tr=ResidualTransformerBlock()
        self.pool=AttnPool(D_MODEL,ATTN_H)
        self.g=nn.Sequential(nn.LayerNorm(4),nn.Linear(4,GLOBAL_DIM),nn.ReLU())
        self.reg=nn.Sequential(
            nn.Linear(D_MODEL+GLOBAL_DIM,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.25),
            nn.Linear(256,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.15),
            nn.Linear(64,1)
        )
    def forward(self,x,g):
        x=self.proj(x)
        x=x+self.pos[:,:x.size(1)]
        x=self.res_tr(x)
        x=self.pool(x)
        g=self.g(g)
        return self.reg(torch.cat([x,g],1)).squeeze(1)

def metrics(y,p):
    c=pearsonr(y,p)[0] if np.std(y)>0 and np.std(p)>0 else np.nan
    return {"MAE":float(mean_absolute_error(y,p)),
            "RMSE":float(math.sqrt(mean_squared_error(y,p))),
            "R2":float(r2_score(y,p)),
            "Pearson_r":float(c),
            "Bias":float(np.mean(p-y))}

def buckets(y,p):
    d=pd.DataFrame({"GT_EF":y,"Pred_EF":p})
    d["Abs_Error"]=(d["Pred_EF"]-d["GT_EF"]).abs()
    d["Bucket"]=pd.cut(d["GT_EF"],[0,30,40,50,60,70,100],
                       labels=["<30","30-40","40-50","50-60","60-70","70+"])
    return d.groupby("Bucket",observed=True).agg(n=("Abs_Error","size"),
                                                  MAE=("Abs_Error","mean")).reset_index()

@torch.no_grad()
def predict(model,loader):
    model.eval(); out=[]
    for x,g,_ in loader:
        out.append(model(x.to(DEVICE),g.to(DEVICE)).cpu())
    return torch.cat(out).numpy()

def main():
    seed_all(SEED)
    print("\n===== R3D + RESIDUAL TRANSFORMER QUICK =====")
    print("Device:",DEVICE)
    if DEVICE.type=="cuda": print("GPU:",torch.cuda.get_device_name(0))

    tx,tg,ty,_=load_cache("train")
    vx,vg,vy,_=load_cache("val")

    print("Train:",tx.shape)
    print("Val:",vx.shape)

    tl=DataLoader(TensorDataset(tx,tg,ty),BATCH,shuffle=True)
    vl=DataLoader(TensorDataset(vx,vg,vy),BATCH,shuffle=False)

    model=Model(tx.shape[-1],tx.shape[1]).to(DEVICE)
    print("Trainable parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))

    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    crit=nn.SmoothL1Loss(beta=1.0)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,"min",factor=.5,patience=4)

    best=float("inf"); bad=0; history=[]

    for ep in range(1,EPOCHS+1):
        model.train()
        for x,g,y in tl:
            x=x.to(DEVICE); g=g.to(DEVICE); y=y.to(DEVICE)
            p=model(x,g); loss=crit(p,y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()

        vp=predict(model,vl)
        m=metrics(vy.numpy(),vp)
        b=buckets(vy.numpy(),vp)
        sch.step(m["MAE"])

        print(f"Epoch {ep:02d}/{EPOCHS} | Val MAE: {m['MAE']:.4f} | R2: {m['R2']:.4f} | Pearson: {m['Pearson_r']:.4f} | "
              + " | ".join(f"{r['Bucket']}:{r['MAE']:.2f}" for _,r in b.iterrows()))

        history.append({"Epoch":ep,**m})
        pd.DataFrame(history).to_csv(HIST,index=False)

        if m["MAE"]<best:
            best=m["MAE"]; bad=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":ep,
                        "best_val_mae":best,"input_dim":tx.shape[-1],
                        "seq_len":tx.shape[1]},BEST)
            print(f"  New best! Val MAE = {best:.4f}")
        else:
            bad+=1
        if bad>=PATIENCE:
            print("Early stopping triggered."); break

    print("Best Val MAE:",best)
    print("Checkpoint:",BEST)

if __name__=="__main__":
    main()
