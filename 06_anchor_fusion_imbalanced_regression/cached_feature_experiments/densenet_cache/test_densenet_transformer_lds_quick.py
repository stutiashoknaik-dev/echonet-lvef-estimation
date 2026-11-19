# FINAL TEST ONLY: DenseNet cached features + Transformer + exact v2 LDS
import os, math, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE=os.environ.get("ECHONET_ROOT", "/content/drive/MyDrive/EchoNet-Dynamic")
CACHE=os.path.join(BASE,"outputs/stage2_fused_cache")
V2=os.path.join(BASE,"outputs/anchor_fusion_densenet_bilstm_v2/best_model.pth")
TR=os.path.join(BASE,"outputs/densenet_transformer_lds_quick/best_model.pth")
OUT=os.path.join(BASE,"outputs/densenet_transformer_lds_quick")
TEST=os.path.join(CACHE,"test_fused.pt")
os.makedirs(OUT,exist_ok=True)

SEQ_LEN=32; INPUT_DIM=1107; D_MODEL=256; NHEAD=8; NUM_LAYERS=2; FF_DIM=512
DROPOUT=.20; GLOBAL_DIM=16; BATCH=64
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttnPool(nn.Module):
    def __init__(self,d,h=128):
        super().__init__()
        self.score=nn.Sequential(
            nn.Linear(d,h),
            nn.Tanh(),
            nn.Linear(h,1)
        )
    def forward(self,x):
        w=torch.softmax(self.score(x),dim=1)
        return (x*w).sum(1)

class V2Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(1107,256,2,batch_first=True,bidirectional=True,dropout=.3)
        self.pool=AttnPool(512)
        self.gd=nn.Sequential(nn.LayerNorm(4),nn.Linear(4,16),nn.ReLU())
        self.reg=nn.Sequential(nn.Linear(528,256),nn.LayerNorm(256),nn.ReLU(),nn.Dropout(.3),
                               nn.Linear(256,64),nn.LayerNorm(64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
    def forward(self,x,g):
        x,_=self.lstm(x); x=self.pool(x); g=self.gd(g); return self.reg(torch.cat([x,g],1)).squeeze(1)

class TransformerHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.LayerNorm(INPUT_DIM),
            nn.Linear(INPUT_DIM, D_MODEL),
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, SEQ_LEN, D_MODEL)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS,
        )

        self.temporal_pool = AttnPool(D_MODEL)

        self.global_delta_encoder = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, GLOBAL_DIM),
            nn.ReLU(inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Linear(D_MODEL + GLOBAL_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, x, g):
        x = self.input_projection(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.temporal_pool(x)
        g = self.global_delta_encoder(g)
        return self.regressor(torch.cat([x, g], dim=1)).squeeze(1)

def load_test():
    d=torch.load(TEST,map_location="cpu")
    return d["fused"].float(),d["global_delta"].float(),d["ef"].float(),d["video_ids"]

@torch.no_grad()
def pred(model,x,g):
    model.eval(); out=[]
    dl=DataLoader(TensorDataset(x,g),batch_size=BATCH,shuffle=False)
    for xb,gb in dl: out.append(model(xb.to(DEVICE),gb.to(DEVICE)).cpu())
    return np.clip(torch.cat(out).numpy(),0,100)

def metrics(y,p):
    return {"MAE":float(mean_absolute_error(y,p)),
            "RMSE":float(math.sqrt(mean_squared_error(y,p))),
            "R2":float(r2_score(y,p)),
            "Pearson_r":float(pearsonr(y,p)[0]),
            "Bias":float(np.mean(p-y))}

def buckets(y,p,name):
    df=pd.DataFrame({"GT":y,"P":p}); df["E"]=(df.P-df.GT).abs()
    bins=[0,30,40,50,60,70,100]; labs=["<30","30-40","40-50","50-60","60-70","70+"]
    df["B"]=pd.cut(df.GT,bins=bins,labels=labs)
    q=df.groupby("B",observed=True).agg(n=("E","size"),MAE=("E","mean")).reset_index()
    q["Model"]=name; return q[["Model","B","n","MAE"]]

def load_v2():
    m=V2Head().to(DEVICE); ck=torch.load(V2,map_location=DEVICE); s=ck.get("model_state_dict",ck)
    own=m.state_dict(); comp={k:v for k,v in s.items() if k in own and own[k].shape==v.shape}
    m.load_state_dict(comp,strict=False); return m

def main():
    print("\n===== FINAL TEST: v2 vs Transformer + LDS =====")
    print("Device:",DEVICE)
    x,g,ef,ids=load_test()
    print("TEST cache:",x.shape,g.shape,ef.shape,len(ids))
    v2=load_v2()
    tr=TransformerHead().to(DEVICE)
    ck=torch.load(TR,map_location=DEVICE)
    state = ck.get("model_state_dict", ck)
    own = tr.state_dict()
    compatible = {
        k: v for k, v in state.items()
        if k in own and own[k].shape == v.shape
    }
    if len(compatible) != len(own):
        missing = [k for k in own if k not in compatible]
        raise RuntimeError(
            f"Transformer checkpoint/head mismatch: loaded "
            f"{len(compatible)}/{len(own)} tensors; missing={missing}"
        )
    tr.load_state_dict(compatible, strict=False)
    tr.eval()
    print("Transformer+LDS checkpoint loaded successfully.")
    print("Transformer+LDS best epoch:",ck.get("epoch","?"))
    print("Transformer+LDS best Val MAE:",ck.get("best_val_mae","?"))
    y=ef.numpy(); p1=pred(v2,x,g); p2=pred(tr,x,g)
    m1=metrics(y,p1); m2=metrics(y,p2)
    print("\nOVERALL TEST")
    print(pd.DataFrame([{"Model":"v2_BiLSTM_LDS",**m1},{"Model":"DenseNet_Transformer_LDS",**m2}]).to_string(index=False))
    print(f"\nMAE improvement (positive=Transformer+LDS better): {m1['MAE']-m2['MAE']:.4f}")
    print(f"Relative improvement: {(m1['MAE']-m2['MAE'])/m1['MAE']*100:.2f}%")
    b1=buckets(y,p1,"v2_BiLSTM_LDS"); b2=buckets(y,p2,"DenseNet_Transformer_LDS")
    allb=pd.concat([b1,b2],ignore_index=True); piv=allb.pivot(index="B",columns="Model",values="MAE").reset_index()
    piv["Transformer_minus_v2"]=piv["DenseNet_Transformer_LDS"]-piv["v2_BiLSTM_LDS"]
    piv["Improvement_positive"]=piv["v2_BiLSTM_LDS"]-piv["DenseNet_Transformer_LDS"]
    print("\nBUCKET COMPARISON"); print(piv.to_string(index=False))
    df=pd.DataFrame({"Video":ids,"GT_EF":y,"V2_EF":p1,"Transformer_LDS_EF":p2})
    df["V2_Abs_Error"]=(df.V2_EF-df.GT_EF).abs(); df["Transformer_LDS_Abs_Error"]=(df.Transformer_LDS_EF-df.GT_EF).abs()
    df["Error_Change"]=df.V2_Abs_Error-df.Transformer_LDS_Abs_Error
    df["Outcome"]=np.where(df.Error_Change>0,"Improved",np.where(df.Error_Change<0,"Worsened","Same"))
    df.to_csv(os.path.join(OUT,"test_v2_vs_transformer_lds_predictions.csv"),index=False)
    pd.DataFrame([{"Model":"v2_BiLSTM_LDS",**m1},{"Model":"DenseNet_Transformer_LDS",**m2}]).to_csv(os.path.join(OUT,"test_v2_vs_transformer_lds_metrics.csv"),index=False)
    allb.to_csv(os.path.join(OUT,"test_v2_vs_transformer_lds_buckets.csv"),index=False)
    print("\nImproved:",(df.Outcome=="Improved").sum()," Worsened:",(df.Outcome=="Worsened").sum()," Same:",(df.Outcome=="Same").sum())
    print("Saved to:",OUT)
    print("TEST used only for final evaluation.")

if __name__=="__main__": main()
