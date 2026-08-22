import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
ROOT_DIR = SRC_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse,json,random,time
from collections import Counter
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import f1_score
try:
    from .alexgym_data import load_exercise
    from .modern_models import pose_features, ANAT_MAP
    from .loco_split import make_loco_split, DEFAULT_MIN_TRAIN_STATE, DEFAULT_MIN_VAL_STATE
    from .lop_loss import LOSS_TYPES, compute_lop_weights, elementwise_loss
except ImportError:
    from alexgym_data import load_exercise
    from modern_models import pose_features, ANAT_MAP
    from loco_split import make_loco_split, DEFAULT_MIN_TRAIN_STATE, DEFAULT_MIN_VAL_STATE
    from lop_loss import LOSS_TYPES, compute_lop_weights, elementwise_loss

torch.set_num_threads(2)

class DS(Dataset):
    def __init__(self,X,Y,idx):
        self.X=torch.tensor(X[idx],dtype=torch.float32);self.Y=torch.tensor(Y[idx],dtype=torch.float32)
    def __len__(self):return len(self.X)
    def __getitem__(self,i):return self.X[i],self.Y[i]

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def metrics(y,p):
    q=(p>=.5).astype(int);t=y.astype(int)
    return {'macro_f1':float(f1_score(t,q,average='macro',zero_division=0)),
            'micro_f1':float(f1_score(t,q,average='micro',zero_division=0)),
            'bit_accuracy':float((q==t).mean()),'exact_match':float((q==t).all(1).mean()),
            'per_criterion_accuracy':[float((q[:,i]==t[:,i]).mean()) for i in range(t.shape[1])]}

class ViewEncoder(nn.Module):
    def __init__(self,din=297,d=56):
        super().__init__()
        self.frame=nn.Sequential(nn.LayerNorm(din),nn.Linear(din,d),nn.GELU(),nn.Dropout(.08))
        self.conv=nn.Sequential(nn.Conv1d(d,d,5,padding=2),nn.GELU(),nn.Conv1d(d,d,3,padding=1),nn.GELU())
    def forward(self,x):
        z=self.conv(self.frame(x).transpose(1,2));return torch.cat([z.mean(-1),z.amax(-1)],1)

class FACT(nn.Module):
    """Factorized Anatomical Criterion Tokens.

    Each criterion receives only its annotation-aligned view and a soft anatomical
    subgraph mask.  Its token is classified both by a local linear decision and
    criterion/state prototypes.  No other-label context is available at inference.
    """
    def __init__(self,exercise,d=56,ed=32,features='j',soft_mask=True,use_proto=True,global_mask=False,shared_adapter=False,map_control='anatomy'):
        super().__init__();self.ex=exercise;self.C=len(ANAT_MAP[exercise]);self.features=features;self.soft_mask=soft_mask;self.use_proto=use_proto;self.global_mask=global_mask;self.shared_adapter=shared_adapter;self.map_control=map_control
        cin=33*3*len(features);self.front=ViewEncoder(cin,d);self.lat=ViewEncoder(cin,d)
        self.adapter=nn.ModuleList([nn.Sequential(nn.Linear(2*d,ed),nn.GELU(),nn.LayerNorm(ed)) for _ in range(1 if shared_adapter else self.C)])
        self.local_w=nn.Parameter(torch.randn(self.C,ed)*.04);self.local_b=nn.Parameter(torch.zeros(self.C))
        self.proto=nn.Parameter(torch.randn(self.C,2,ed)*.08);self.log_temp=nn.Parameter(torch.tensor(-.3));self.proto_mix=nn.Parameter(torch.tensor(-.8))
        masks=torch.zeros(self.C,33);views=[]
        for c,(v,joints) in enumerate(ANAT_MAP[exercise]):
            masks[c,joints]=1.;views.append(0 if v=='F' else 1)
        if map_control=='permuted':
            masks=torch.roll(masks,1,0);views=list(np.roll(np.asarray(views),1))
        elif map_control=='random':
            gen=torch.Generator().manual_seed({'squat':3101,'deadlift':3102,'lunges':3103}[exercise])
            random_masks=torch.zeros_like(masks)
            for c in range(self.C):random_masks[c,torch.randperm(33,generator=gen)[:int(masks[c].sum())]]=1.
            masks=random_masks
        elif map_control!='anatomy':raise ValueError(map_control)
        self.register_buffer('base_masks',masks);self.register_buffer('views',torch.tensor(views,dtype=torch.long))
        # learned residual can softly expand the anatomy prior without making the model global
        self.mask_resid=nn.Parameter(torch.zeros(self.C,33))
    def tokens(self,x):
        p=pose_features(x,self.features) # B,2,T,33,F
        B,W,T,V,D=p.shape;outs=[]
        for c in range(self.C):
            if self.global_mask:
                m=torch.ones(33,device=p.device)
            elif self.soft_mask:
                # Prior: annotated joints start near 1, others near .05; residual learns controlled expansion.
                base=self.base_masks[c]*3.0+(1-self.base_masks[c])*(-3.0)
                m=torch.sigmoid(base+self.mask_resid[c])
            else:m=self.base_masks[c]
            z=p[:,self.views[c]]*m[None,None,:,None]
            z=z.reshape(B,T,V*D)
            h=self.front(z) if int(self.views[c])==0 else self.lat(z)
            outs.append(self.adapter[0 if self.shared_adapter else c](h))
        return torch.stack(outs,1)
    def decode(self,e):
        lin=(e*self.local_w[None]).sum(-1)+self.local_b
        dist=((e[:,:,None]-self.proto[None])**2).mean(-1);tau=F.softplus(self.log_temp)+.15
        pl=(dist[:,:,0]-dist[:,:,1])/tau
        return lin+torch.sigmoid(self.proto_mix)*pl if self.use_proto else lin
    def forward(self,x,return_tokens=False):
        e=self.tokens(x);log=self.decode(e);return (log,e) if return_tokens else log

def proto_pull(e,y,proto):
    B,C,D=e.shape;ci=torch.arange(C,device=e.device)[None].expand(B,-1);chosen=proto[ci,y.long()]
    return ((e-chosen)**2).mean()

def criterion_supcon(e,y,temp=.18):
    """Criterion-wise supervised contrastive objective.
    Positives share y_c irrespective of all other labels, directly discouraging
    composition-specific clustering.
    """
    B,C,D=e.shape
    if B<3:return e.sum()*0
    z=F.normalize(e,-1);losses=[]
    eye=torch.eye(B,dtype=torch.bool,device=e.device)
    for c in range(C):
        sim=(z[:,c]@z[:,c].T)/temp
        sim=sim-sim.max(1,keepdim=True).values.detach()
        pos=(y[:,c,None]==y[None,:,c]) & ~eye
        valid=pos.any(1)
        if not valid.any():continue
        den=torch.logsumexp(sim.masked_fill(eye,-1e9),1)
        # average log-prob over positives for each anchor
        lp=sim-den[:,None]
        v=-(lp.masked_fill(~pos,0).sum(1)/pos.sum(1).clamp_min(1))[valid].mean()
        losses.append(v)
    return torch.stack(losses).mean() if losses else e.sum()*0

def h1_logit_consistency(log,y,margin=.6,max_pairs=128):
    B,C=log.shape
    if B<2:return log.sum()*0
    diff=(y[:,None,:]!=y[None,:,:]);ham=diff.sum(-1);pairs=(ham==1).nonzero(as_tuple=False);pairs=pairs[pairs[:,0]<pairs[:,1]]
    if len(pairs)==0:return log.sum()*0
    if len(pairs)>max_pairs:pairs=pairs[torch.randperm(len(pairs),device=log.device)[:max_pairs]]
    i,j=pairs[:,0],pairs[:,1];changed=diff[i,j].float().argmax(-1);ar=torch.arange(len(pairs),device=log.device)
    # unchanged outputs should be stable; changed output must separate in signed direction
    sq=(log[i]-log[j]).pow(2);mask=torch.ones_like(sq);mask[ar,changed]=0;inv=(sq*mask).sum()/mask.sum().clamp_min(1)
    yi=y[i,changed];yj=y[j,changed];signed_i=(2*yi-1)*log[i,changed];signed_j=(2*yj-1)*log[j,changed]
    sep=F.relu(margin-signed_i).mean()+F.relu(margin-signed_j).mean()
    return .5*inv+.5*sep

def predict(m,dl):
    m.eval();ys=[];ps=[]
    with torch.no_grad():
        for x,y in dl:ys.append(y.numpy());ps.append(torch.sigmoid(m(x)).numpy())
    return np.concatenate(ys),np.concatenate(ps)

def variant_name(map_control='anatomy',loss_type='bce'):
    """Result-file name for a FACT configuration.

    Only the reported configuration is called plain ``fact``; every ablation
    carries a suffix, so an ablation run can never be pooled with the reported
    panel by ``aggregate_runs.py`` or accepted by ``verify_release.py``.
    """
    name='fact'
    if map_control!='anatomy':name+=f'_{map_control}_map'
    if loss_type!='bce':name+=f'_{loss_type}'
    return name

def context_counters(Y,tr):
    """Per-criterion Counter over leave-one-criterion-out training label contexts."""
    return [Counter(tuple(np.delete(Y[i].astype(np.int8),c).tolist()) for i in tr) for c in range(Y.shape[1])]

def context_weights(y,counters,alpha):
    """Down-weight criterion decisions whose label context is common in training.

    Counts come from the training pool only, never from the held-out diagnosis.
    An unseen context has count 0 and is clamped to 1 so its weight stays finite.
    """
    rows=y.detach().cpu().numpy().astype(np.int8)
    w=np.empty((len(rows),len(counters)),np.float32)
    for c,cnt in enumerate(counters):
        w[:,c]=[max(cnt[tuple(np.delete(r,c).tolist())],1)**(-alpha) for r in rows]
    wb=torch.as_tensor(w,dtype=y.dtype,device=y.device)
    return wb/(wb.mean(0,keepdim=True)+1e-8)

def run(ex,target,seed,data='data',epochs=80,patience=12,lr=1.5e-3,lam_proto=0.0,lam_supcon=0.0,lam_h1=0.0,features='j',soft_mask=True,eval_test=True,context_alpha=0.0,use_proto=True,global_mask=False,shared_adapter=False,map_control='anatomy',checkpoint=None,min_train_state=DEFAULT_MIN_TRAIN_STATE,min_val_state=DEFAULT_MIN_VAL_STATE,
        loss_type='bce',lop_alpha=0.5,focal_gamma=2.0):
    seed_all(seed);X,Y,co,g,df=load_exercise(data,ex,T=16);tr,va,te,split_audit=make_loco_split(Y,co,g,target,seed,min_train_state=min_train_state,min_val_state=min_val_state)
    dl=DataLoader(DS(X,Y,tr),32,shuffle=True);dv=DataLoader(DS(X,Y,va),128);dt=DataLoader(DS(X,Y,te),128)
    m=FACT(ex,features=features,soft_mask=soft_mask,use_proto=use_proto,global_mask=global_mask,shared_adapter=shared_adapter,map_control=map_control);pos=Y[tr].sum(0);neg=len(tr)-pos;pw=torch.tensor(np.clip(neg/np.maximum(pos,1),.25,8),dtype=torch.float32)
    # LOP weights are read from the training fold only, but they are indexed by
    # the held-out target, so this ablation knows which diagnosis was withheld.
    lop_weights=compute_lop_weights(Y[tr],target,alpha=lop_alpha) if loss_type=='lop_weighted' else None
    # Criterion-context importance weights: for criterion c, a context is y_-c.
    # The counters depend only on the training pool, so they are built once.
    counters=context_counters(Y,tr) if context_alpha>0 else None
    op=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=2e-4);best=-1;state=None;stale=0;t0=time.time();ep=-1
    for ep in range(epochs):
        m.train()
        for x,y in dl:
            op.zero_grad();log,e=m(x,True)
            wbatch=context_weights(y,counters,context_alpha) if counters is not None else torch.ones_like(log)
            raw=elementwise_loss(loss_type,log,y,pos_weight=pw,lop_weights=lop_weights,gamma=focal_gamma)
            loss=(raw*wbatch).mean()+lam_proto*proto_pull(e,y,m.proto)+lam_supcon*criterion_supcon(e,y)+lam_h1*h1_logit_consistency(log,y)
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);op.step()
        yv,pv=predict(m,dv);sc=metrics(yv,pv)['macro_f1']
        if sc>best+1e-4:best=sc;state={k:v.detach().clone() for k,v in m.state_dict().items()};stale=0
        else:stale+=1
        if stale>=patience:break
    if state is not None: m.load_state_dict(state)
    yv,pv=predict(m,dv);test=None
    if eval_test:yt,pt=predict(m,dt);test=metrics(yt,pt)
    if checkpoint:
        cp=Path(checkpoint);cp.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'model':m.state_dict(),'model_kind':'fact','exercise':ex,'target':target,
                    'seed':seed,'features':features,'soft_mask':soft_mask,
                    'use_proto':use_proto,'global_mask':global_mask,
                    'shared_adapter':shared_adapter,'map_control':map_control,'split_audit':split_audit},cp)
    return {'method':'FACT','model':variant_name(map_control,loss_type),'exercise':ex,'target':target,'seed':seed,'features':features,'soft_mask':soft_mask,'use_proto':use_proto,'global_mask':global_mask,'shared_adapter':shared_adapter,'map_control':map_control,'n_params':sum(p.numel() for p in m.parameters()),'n_train':len(tr),'n_val':len(va),'n_test':len(te),'split_audit':split_audit,'synthetic_data':bool(df.attrs.get('is_synthetic',False)),'val':metrics(yv,pv),'test':test,'epochs':ep+1,'train_seconds':time.time()-t0,'hyper':{'lr':lr,'lam_proto':lam_proto,'lam_supcon':lam_supcon,'lam_h1':lam_h1,'context_alpha':context_alpha,'loss':loss_type,'lop_alpha':lop_alpha if loss_type=='lop_weighted' else None,'focal_gamma':focal_gamma if loss_type=='focal' else None},'uses_holdout_identity':loss_type=='lop_weighted'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--exercise',required=True);ap.add_argument('--target',required=True);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=80);ap.add_argument('--patience',type=int,default=12);ap.add_argument('--lr',type=float,default=1.5e-3);ap.add_argument('--lam-proto',type=float,default=0.0);ap.add_argument('--lam-supcon',type=float,default=0.0);ap.add_argument('--lam-h1',type=float,default=0.0);ap.add_argument('--features',default='j');ap.add_argument('--hard-mask',action='store_true');ap.add_argument('--context-alpha',type=float,default=0.0);ap.add_argument('--no-proto-decode',action='store_true');ap.add_argument('--global-mask',action='store_true');ap.add_argument('--shared-adapter',action='store_true');ap.add_argument('--map-control',choices=['anatomy','random','permuted'],default='anatomy');ap.add_argument('--no-test',action='store_true');ap.add_argument('--checkpoint')
    ap.add_argument('--min-train-state',type=int,default=DEFAULT_MIN_TRAIN_STATE,help='Minimum training examples per criterion state (frozen protocol: 8).')
    ap.add_argument('--min-val-state',type=int,default=DEFAULT_MIN_VAL_STATE,help='Minimum validation examples per criterion state (frozen protocol: 1).')
    ap.add_argument('--loss',choices=list(LOSS_TYPES),default='bce',help="Training objective ablation. 'bce' is the reported setting. 'lop_weighted' additionally conditions on which diagnosis was held out, so it is an oracle-flavoured upper bound, not a deployable method.")
    ap.add_argument('--lop-alpha',type=float,default=0.5,help='Strength of the LOP reweighting; only used with --loss lop_weighted.')
    ap.add_argument('--focal-gamma',type=float,default=2.0,help='Focusing exponent; only used with --loss focal.')
    a=ap.parse_args()
    r=run(a.exercise,a.target,a.seed,data=a.data,epochs=a.epochs,patience=a.patience,lr=a.lr,
          lam_proto=a.lam_proto,lam_supcon=a.lam_supcon,lam_h1=a.lam_h1,features=a.features,
          soft_mask=not a.hard_mask,eval_test=not a.no_test,context_alpha=a.context_alpha,
          use_proto=not a.no_proto_decode,global_mask=a.global_mask,shared_adapter=a.shared_adapter,
          map_control=a.map_control,checkpoint=a.checkpoint,
          min_train_state=a.min_train_state,min_val_state=a.min_val_state,
          loss_type=a.loss,lop_alpha=a.lop_alpha,focal_gamma=a.focal_gamma)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
