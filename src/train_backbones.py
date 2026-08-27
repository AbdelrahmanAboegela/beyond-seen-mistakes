import argparse,json,random,time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import f1_score
from alexgym_data import load_exercise, CRITERIA
from loco_split import make_loco_split, DEFAULT_MIN_TRAIN_STATE, DEFAULT_MIN_VAL_STATE

torch.set_num_threads(2)

class DS(Dataset):
    def __init__(self,X,Y,idx): self.X=torch.tensor(X[idx],dtype=torch.float32); self.Y=torch.tensor(Y[idx],dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.Y[i]

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def metrics(y,p):
    q=(p>=.5).astype(int); t=y.astype(int)
    return {'macro_f1':float(f1_score(t,q,average='macro',zero_division=0)),'micro_f1':float(f1_score(t,q,average='micro',zero_division=0)),
            'bit_accuracy':float((q==t).mean()),'exact_match':float((q==t).all(1).mean()),
            'per_criterion_accuracy':[float((q[:,i]==t[:,i]).mean()) for i in range(t.shape[1])]}

def pred(m,dl):
    m.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in dl: ys.append(y.numpy()); ps.append(torch.sigmoid(m(x)).numpy())
    return np.concatenate(ys),np.concatenate(ps)

class TCN(nn.Module):
    def __init__(self,din,nout,d=96):
        super().__init__(); self.frame=nn.Sequential(nn.LayerNorm(din),nn.Linear(din,d),nn.GELU(),nn.Dropout(.12));
        self.conv=nn.Sequential(nn.Conv1d(d,d,5,padding=2),nn.GELU(),nn.Conv1d(d,d,3,padding=1),nn.GELU()); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.12),nn.Linear(d,nout))
    def forward(self,x): h=self.conv(self.frame(x).transpose(1,2)); return self.head(torch.cat([h.mean(-1),h.amax(-1)],1))

class GRUModel(nn.Module):
    def __init__(self,din,nout,d=72):
        super().__init__(); self.inp=nn.Sequential(nn.LayerNorm(din),nn.Linear(din,d),nn.GELU()); self.rnn=nn.GRU(d,d,batch_first=True,bidirectional=True); self.head=nn.Sequential(nn.Linear(4*d,d),nn.GELU(),nn.Dropout(.12),nn.Linear(d,nout))
    def forward(self,x): h,_=self.rnn(self.inp(x)); return self.head(torch.cat([h.mean(1),h.amax(1)],1))

class TransformerModel(nn.Module):
    def __init__(self,din,nout,d=96,heads=4,layers=2):
        super().__init__(); self.inp=nn.Sequential(nn.LayerNorm(din),nn.Linear(din,d)); self.pos=nn.Parameter(torch.randn(1,16,d)*.02)
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=heads,dim_feedforward=2*d,dropout=.12,activation='gelu',batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(enc,num_layers=layers); self.norm=nn.LayerNorm(d); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.12),nn.Linear(d,nout))
    def forward(self,x):
        h=self.enc(self.inp(x)+self.pos[:,:x.shape[1]]); h=self.norm(h); return self.head(torch.cat([h.mean(1),h.amax(1)],1))

class SelectiveSSMBlock(nn.Module):
    """Small Mamba-inspired selective recurrent state-space block (not the official Mamba implementation)."""
    def __init__(self,d):
        super().__init__(); self.norm=nn.LayerNorm(d); self.decay=nn.Linear(d,d); self.cand=nn.Linear(d,d); self.gate=nn.Linear(d,d); self.out=nn.Linear(d,d)
    def scan(self,x,reverse=False):
        B,T,D=x.shape; s=x.new_zeros(B,D); ys=[]; rng=range(T-1,-1,-1) if reverse else range(T)
        for t in rng:
            u=x[:,t]; a=torch.sigmoid(self.decay(u)); c=torch.tanh(self.cand(u)); s=a*s+(1-a)*c; y=torch.sigmoid(self.gate(u))*s; ys.append(y)
        if reverse: ys=ys[::-1]
        return torch.stack(ys,1)
    def forward(self,x):
        u=self.norm(x); y=.5*(self.scan(u,False)+self.scan(u,True)); return x+self.out(y)
class SSMModel(nn.Module):
    def __init__(self,din,nout,d=32,layers=1):
        super().__init__(); self.inp=nn.Sequential(nn.LayerNorm(din),nn.Linear(din,d),nn.GELU()); self.blocks=nn.ModuleList([SelectiveSSMBlock(d) for _ in range(layers)]); self.norm=nn.LayerNorm(d); self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(.12),nn.Linear(d,nout))
    def forward(self,x):
        h=self.inp(x)
        for b in self.blocks: h=b(h)
        h=self.norm(h); return self.head(torch.cat([h.mean(1),h.amax(1)],1))

# MediaPipe pose edges. Face/finger points retain self loops and are not privileged.
EDGES=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(24,26),(26,28),(28,30),(30,32),(27,31),(28,32)]
def adjacency(V=33):
    A=np.eye(V,dtype=np.float32)
    for i,j in EDGES: A[i,j]=A[j,i]=1
    deg=A.sum(1); D=np.diag(1/np.sqrt(np.maximum(deg,1e-6))); return torch.tensor(D@A@D)
class GraphBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__(); self.lin=nn.Linear(cin,cout); self.temp=nn.Conv2d(cout,cout,kernel_size=(5,1),padding=(2,0)); self.norm=nn.BatchNorm2d(cout); self.act=nn.GELU()
    def forward(self,x,A): # B,T,V,C
        x=torch.matmul(x.permute(0,1,3,2),A.t()).permute(0,1,3,2).contiguous(); x=self.lin(x); x=x.permute(0,3,1,2); x=self.act(self.norm(self.temp(x))); return x.permute(0,2,3,1)
class STGCNModel(nn.Module):
    def __init__(self,nout,d=48):
        super().__init__(); self.register_buffer('A',adjacency()); self.b1=GraphBlock(3,d); self.b2=GraphBlock(d,d); self.head=nn.Sequential(nn.Linear(4*d,d),nn.GELU(),nn.Dropout(.12),nn.Linear(d,nout))
    def one(self,z):
        h=self.b2(self.b1(z,self.A),self.A); # B,T,V,D
        return torch.cat([h.mean((1,2)),h.amax(1).mean(1)],1)
    def forward(self,x):
        B,T,D=x.shape; v=D//2; f=x[:,:,:v].reshape(B,T,33,3); l=x[:,:,v:].reshape(B,T,33,3); return self.head(torch.cat([self.one(f),self.one(l)],1))

def build(kind,din,nout):
    if kind=='tcn': return TCN(din,nout)
    if kind=='gru': return GRUModel(din,nout)
    if kind=='transformer': return TransformerModel(din,nout)
    if kind=='ssm': return SSMModel(din,nout)
    if kind=='stgcn': return STGCNModel(nout)
    raise ValueError(kind)

def run(ex,target,seed,kind,data,epochs=100,checkpoint=None,min_train_state=DEFAULT_MIN_TRAIN_STATE,min_val_state=DEFAULT_MIN_VAL_STATE):
    seed_all(seed); X,Y,co,g,df=load_exercise(data,ex,T=16)
    tr,va,te,split_audit=make_loco_split(Y,co,g,target,seed,min_train_state=min_train_state,min_val_state=min_val_state)
    dltr=DataLoader(DS(X,Y,tr),32,shuffle=True); dlv=DataLoader(DS(X,Y,va),64); dlt=DataLoader(DS(X,Y,te),64)
    m=build(kind,X.shape[-1],Y.shape[1]); pos=Y[tr].sum(0); neg=len(tr)-pos; pw=torch.tensor(np.clip(neg/np.maximum(pos,1),.25,8),dtype=torch.float32)
    lf=nn.BCEWithLogitsLoss(pos_weight=pw); op=torch.optim.AdamW(m.parameters(),lr=1.5e-3,weight_decay=2e-4)
    best=-1; state=None; stale=0; t0=time.time(); e=-1  # epochs=0 would leave e unbound below
    for e in range(epochs):
        m.train()
        for x,y in dltr: op.zero_grad(); loss=lf(m(x),y); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5); op.step()
        yv,pv=pred(m,dlv); s=metrics(yv,pv)['macro_f1']
        if s>best+1e-4: best=s; state={k:v.detach().clone() for k,v in m.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=15: break
    # epochs=0 never records a best state, so there is nothing to restore
    if state is not None: m.load_state_dict(state)
    yv,pv=pred(m,dlv); yt,pt=pred(m,dlt)
    if checkpoint:
        cp=Path(checkpoint);cp.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'model':m.state_dict(),'model_kind':kind,'exercise':ex,'target':target,
                    'seed':seed,'input_dim':int(X.shape[-1]),'n_outputs':int(Y.shape[1]),
                    'split_audit':split_audit},cp)
    return {'exercise':ex,'target':target,'seed':seed,'model':kind,'criteria':CRITERIA[ex],'n_train':len(tr),'n_val':len(va),'n_test':len(te),'n_params':sum(p.numel() for p in m.parameters()),'split_audit':split_audit,'val':metrics(yv,pv),'test':metrics(yt,pt),'epochs':e+1,'train_seconds':time.time()-t0}

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data'); ap.add_argument('--exercise',default='squat'); ap.add_argument('--target',required=True); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--model',choices=['tcn','gru','transformer','ssm','stgcn'],required=True); ap.add_argument('--out',required=True); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--checkpoint')
    ap.add_argument('--min-train-state',type=int,default=DEFAULT_MIN_TRAIN_STATE); ap.add_argument('--min-val-state',type=int,default=DEFAULT_MIN_VAL_STATE)
    a=ap.parse_args()
    d=run(a.exercise,a.target,a.seed,a.model,a.data,a.epochs,a.checkpoint,a.min_train_state,a.min_val_state); Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(d,indent=2)); print(json.dumps(d,indent=2)); import sys,os; sys.stdout.flush(); os._exit(0)
