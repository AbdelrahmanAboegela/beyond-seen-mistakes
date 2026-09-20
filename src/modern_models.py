"""Modern skeleton baselines and CAPER for the ALEX-GYM LOCO study.

The Block/Proto/Hyper/Gamba variants are compact *mechanism-matched adaptations*
for the 33-joint, two-view ALEX-GYM representation. They are not drop-in
reproductions of the authors' original NTU/FineGYM implementations. The names
used in result files therefore carry a ``_style`` suffix.

CAPER = Counterfactual Anatomical Prototype Exchange and Residualization.
CAPER is retained as an exploratory development model; the final reported method is FACT in train_fact.py.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# MediaPipe Pose body connectivity. We retain all 33 nodes with self loops;
# face/finger nodes not listed here can still communicate through learned/global paths.
EDGES=[(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
       (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
       (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
       (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),
       (24,26),(26,28),(28,30),(30,32),(28,32)]

PARENTS={
    0:0,1:0,2:1,3:2,4:0,5:4,6:5,7:3,8:6,9:0,10:0,
    11:23,12:24,13:11,14:12,15:13,16:14,17:15,18:16,19:15,20:16,21:15,22:16,
    23:23,24:24,25:23,26:24,27:25,28:26,29:27,30:28,31:29,32:30,
}

# Halpe-26 body connectivity, used for CPR-Coach.  Indices 0-16 follow COCO;
# 17-19 add head/neck/mid-hip and 20-25 the feet.  This is the published
# skeleton definition, not a choice we make.
HALPE26_EDGES=[(0,1),(0,2),(1,3),(2,4),
               (5,18),(6,18),(17,18),(18,19),(19,11),(19,12),
               (5,7),(7,9),(6,8),(8,10),
               (11,13),(13,15),(12,14),(14,16),
               (15,24),(24,20),(24,22),(16,25),(25,21),(25,23)]


def normalized_adjacency(edges, V):
    """Symmetric normalised adjacency with self loops for an arbitrary skeleton."""
    import numpy as _np
    A=_np.eye(V,dtype=_np.float32)
    for i,j in edges:
        if i<V and j<V: A[i,j]=A[j,i]=1
    deg=A.sum(1); D=_np.diag(1/_np.sqrt(_np.maximum(deg,1e-6)))
    return torch.tensor(D@A@D,dtype=torch.float32)


def physical_adjacency(V=33):
    A=np.eye(V,dtype=np.float32)
    for i,j in EDGES:
        if i<V and j<V: A[i,j]=A[j,i]=1
    deg=A.sum(1); D=np.diag(1/np.sqrt(np.maximum(deg,1e-6)))
    return torch.tensor(D@A@D,dtype=torch.float32)

def hop_adjacencies(V=33,max_hop=4):
    A=np.zeros((V,V),dtype=np.float32)
    for i,j in EDGES:
        if i<V and j<V: A[i,j]=A[j,i]=1
    # shortest paths by repeated BFS (tiny V)
    dist=np.full((V,V),999,dtype=np.int64)
    for s in range(V):
        dist[s,s]=0; frontier=[s]
        for d in range(1,max_hop+1):
            nxt=[]
            for u in frontier:
                for v in np.flatnonzero(A[u]>0):
                    if dist[s,v]>d: dist[s,v]=d; nxt.append(int(v))
            frontier=nxt
    mats=[]
    for h in range(max_hop+1):
        M=(dist==h).astype(np.float32)
        if h==0: M=np.eye(V,dtype=np.float32)
        deg=M.sum(1,keepdims=True)
        M=M/np.maximum(deg,1.0)
        mats.append(torch.tensor(M,dtype=torch.float32))
    return torch.stack(mats,0)

HOPS=hop_adjacencies()
APHYS=physical_adjacency()


def split_pose(x:torch.Tensor,views=2,joints=33,coords=3):
    """B,T,views*joints*coords -> B,views,T,joints,coords.

    Defaults are ALEX-GYM-1 (two views, MediaPipe-33, 3D).  CPR-Coach passes
    its own geometry; a bare assert would be stripped under ``python -O`` and
    the view would silently reshape wrong, so this raises.
    """
    B,T,D=x.shape
    want=views*joints*coords
    if D!=want:
        raise ValueError(f"expected {want} dims (views={views}, joints={joints}, coords={coords}), got {D}")
    return x.view(B,T,views,joints,coords).permute(0,2,1,3,4).contiguous()

def pose_features(x:torch.Tensor, mode='jbm', views=2, joints=33, coords=3):
    """Return B,views,T,V,C for joints/bones/motion combinations.

    mode: j=joint xyz, b=bone vectors, m=temporal motion.  Bone mode needs a
    parent table and is therefore MediaPipe-33 only.
    """
    if 'b' in mode and joints!=33:
        raise ValueError("bone features require the MediaPipe-33 parent table")
    p=split_pose(x,views=views,joints=joints,coords=coords)
    fs=[]
    if 'j' in mode: fs.append(p)
    if 'b' in mode:
        parent=torch.tensor([PARENTS[i] for i in range(33)],device=x.device)
        fs.append(p-p[:,:,:,parent,:])
    if 'm' in mode:
        m=torch.zeros_like(p); m[:,:,1:]=p[:,:,1:]-p[:,:,:-1]
        fs.append(m)
    return torch.cat(fs,-1)


def graph_apply_static(A:torch.Tensor,x:torch.Tensor):
    # A[V,V], x[B,T,V,D] -> B,T,V,D using batched matmul (faster than einsum on CPU)
    z=torch.matmul(x.permute(0,1,3,2),A.t())
    return z.permute(0,1,3,2).contiguous()

def graph_apply_dynamic(A:torch.Tensor,x:torch.Tensor):
    # A[B,V,V], x[B,T,V,D]
    B,T,V,D=x.shape
    xx=x.permute(0,1,3,2).reshape(B,T*D,V)
    z=torch.bmm(xx,A.transpose(1,2)).reshape(B,T,D,V).permute(0,1,3,2)
    return z.contiguous()

class TemporalJointBlock(nn.Module):
    def __init__(self,d,k=5,drop=.08):
        super().__init__()
        self.norm=nn.LayerNorm(d)
        self.dw=nn.Conv2d(d,d,(k,1),padding=(k//2,0),groups=d)
        self.pw=nn.Conv2d(d,d,1)
        self.drop=nn.Dropout(drop)
    def forward(self,x): # B,T,V,D
        z=self.norm(x).permute(0,3,1,2)
        z=F.gelu(self.pw(self.dw(z))).permute(0,2,3,1)
        return x+self.drop(z)

class HopGraphBlock(nn.Module):
    """Topology-aware grouped graph convolution inspired by graph-distance/BlockGC ideas."""
    def __init__(self,d,groups=4,max_hop=4,learn_hops=True,drop=.08):
        super().__init__(); self.d=d; self.groups=groups; self.max_hop=max_hop; self.learn_hops=learn_hops
        assert d%groups==0
        self.register_buffer('hops',HOPS[:max_hop+1].clone())
        self.theta=nn.Parameter(torch.zeros(groups,max_hop+1))
        with torch.no_grad():
            self.theta[:,0]=.2; self.theta[:,1]=1.0
            for g in range(groups):
                self.theta[g,min(g+1,max_hop)]+=.5
        if not learn_hops: self.theta.requires_grad_(False)
        self.lin=nn.ModuleList([nn.Linear(d//groups,d//groups,bias=False) for _ in range(groups)])
        self.out=nn.Linear(d,d)
        self.norm=nn.LayerNorm(d); self.temporal=TemporalJointBlock(d,5,drop); self.drop=nn.Dropout(drop)
    def forward(self,x):
        B,T,V,D=x.shape; u=self.norm(x); chunks=u.chunk(self.groups,-1); outs=[]
        for g,ch in enumerate(chunks):
            w=F.softmax(self.theta[g],-1)
            A=torch.einsum('h,hvw->vw',w,self.hops)
            z=graph_apply_static(A,ch)
            outs.append(self.lin[g](z))
        z=self.out(torch.cat(outs,-1))
        x=x+self.drop(F.gelu(z))
        return self.temporal(x)

class SkeletonEncoder(nn.Module):
    def __init__(self,d=48,features='jbm',layers=2,graph=True,learn_hops=True):
        super().__init__(); self.features=features; cin=3*len(features); self.graph=graph
        self.inp=nn.Sequential(nn.LayerNorm(cin),nn.Linear(cin,d),nn.GELU())
        self.blocks=nn.ModuleList([HopGraphBlock(d,groups=4,learn_hops=learn_hops) if graph else TemporalJointBlock(d) for _ in range(layers)])
    def forward(self,x):
        p=pose_features(x,self.features) # B,2,T,V,C
        B,W,T,V,C=p.shape; h=self.inp(p)
        h=h.reshape(B*W,T,V,-1)
        for b in self.blocks: h=b(h)
        return h.reshape(B,W,T,V,-1)

class PoolHead(nn.Module):
    def __init__(self,d,nout,drop=.1):
        super().__init__(); self.h=nn.Sequential(nn.Linear(4*d,2*d),nn.GELU(),nn.Dropout(drop),nn.Linear(2*d,nout))
    def forward(self,h): # B,2,T,V,D
        a=h.mean((2,3)); b=h.amax(2).mean(2); z=torch.cat([a,b],-1).flatten(1)
        return self.h(z)

class TopoBlockGCNStyle(nn.Module):
    def __init__(self,nout,d=32,features='jbm'):
        super().__init__(); self.enc=SkeletonEncoder(d,features,layers=2,graph=True,learn_hops=True); self.head=PoolHead(d,nout)
    def forward(self,x,return_aux=False):
        y=self.head(self.enc(x)); return (y,{}) if return_aux else y

class ProtoGCNStyle(nn.Module):
    """Prototype-reconstruction skeleton model inspired by ProtoGCN's key mechanism."""
    def __init__(self,nout,d=32,kproto=12,features='jbm'):
        super().__init__(); self.enc=SkeletonEncoder(d,features,layers=2,graph=True); self.proto=nn.Parameter(torch.randn(kproto,d)*.08)
        self.head=nn.Sequential(nn.Linear(4*d+kproto,2*d),nn.GELU(),nn.Dropout(.1),nn.Linear(2*d,nout))
    def forward(self,x,return_aux=False):
        h=self.enc(x); B,W,T,V,D=h.shape
        local=h.mean(2).reshape(B,W*V,D)
        zn=F.normalize(local,-1); pn=F.normalize(self.proto,-1)
        sim=torch.einsum('bnd,kd->bnk',zn,pn)
        a=F.softmax(sim/0.15,-1); rec=torch.einsum('bnk,kd->bnd',a,self.proto)
        rec_loss=F.mse_loss(rec,local)
        gram=pn@pn.T; eye=torch.eye(len(self.proto),device=x.device); div=((gram-eye)**2).mean()
        usage=a.mean(1)
        pmean=h.mean((2,3)); pmax=h.amax(2).mean(2); z=torch.cat([pmean,pmax],-1).flatten(1)
        logits=self.head(torch.cat([z,usage],-1))
        aux={'proto_recon':rec_loss,'proto_div':div}
        return (logits,aux) if return_aux else logits

class AdaptiveHyperBlock(nn.Module):
    def __init__(self,d,k=6,nvirtual=4,drop=.08):
        super().__init__(); self.k=k; self.nvirtual=nvirtual; self.norm=nn.LayerNorm(d); self.q=nn.Linear(d,d//2,bias=False); self.v=nn.Linear(d,d,bias=False); self.out=nn.Linear(d,d); self.temp=TemporalJointBlock(d); self.drop=nn.Dropout(drop)
        self.virtual=nn.Parameter(torch.randn(nvirtual,d)*.05)
        A=torch.zeros(33+nvirtual,33+nvirtual)
        A[:33,:33]=APHYS
        A[33:,:]=1/(33+nvirtual); A[:,33:]=1/(33+nvirtual)
        self.register_buffer('baseA',A)
    def forward(self,x): # B,T,33,D
        B,T,V,D=x.shape; virt=self.virtual[None,None].expand(B,T,-1,-1); u=torch.cat([x,virt],2); un=self.norm(u)
        # time-averaged semantic relation, then top-k adaptive topology per sample
        q=self.q(un.mean(1)); qn=F.normalize(q,-1); s=torch.einsum('bvd,bwd->bvw',qn,qn)
        kk=min(self.k,s.shape[-1]); val,idx=s.topk(kk,-1); mask=torch.zeros_like(s).scatter_(-1,idx,1.0)
        adap=F.softmax(s.masked_fill(mask==0,-1e4),-1)
        A=.55*self.baseA[None]+.45*adap
        z=graph_apply_dynamic(A,self.v(un))
        u=u+self.drop(F.gelu(self.out(z)))
        physical=self.temp(u[:,:,:33])
        vn=F.normalize(self.virtual,-1); gram=vn@vn.T; eye=torch.eye(self.nvirtual,device=x.device)
        return physical, ((gram-eye)**2).mean()

class HyperGCNStyle(nn.Module):
    def __init__(self,nout,d=28,features='jbm'):
        super().__init__(); self.features=features; cin=3*len(features); self.inp=nn.Sequential(nn.LayerNorm(cin),nn.Linear(cin,d),nn.GELU()); self.b1=AdaptiveHyperBlock(d,6,3); self.head=PoolHead(d,nout)
    def forward(self,x,return_aux=False):
        p=pose_features(x,self.features); B,W,T,V,C=p.shape; h=self.inp(p).reshape(B*W,T,V,-1)
        h,l1=self.b1(h); h=h.reshape(B,W,T,V,-1); logits=self.head(h); aux={'hyper_div':l1}
        return (logits,aux) if return_aux else logits

class SelectiveNodeSSM(nn.Module):
    def __init__(self,d):
        super().__init__(); self.norm=nn.LayerNorm(d); self.dec=nn.Linear(d,d); self.cand=nn.Linear(d,d); self.gate=nn.Linear(d,d); self.out=nn.Linear(d,d)
    def scan(self,x,rev=False): # B,V,T,D
        B,V,T,D=x.shape; s=x.new_zeros(B,V,D); ys=[]; rr=range(T-1,-1,-1) if rev else range(T)
        for t in rr:
            u=x[:,:,t]; a=torch.sigmoid(self.dec(u)); c=torch.tanh(self.cand(u)); s=a*s+(1-a)*c; ys.append(torch.sigmoid(self.gate(u))*s)
        if rev: ys=ys[::-1]
        return torch.stack(ys,2)
    def forward(self,x): # B,T,V,D
        u=self.norm(x).permute(0,2,1,3); y=.5*(self.scan(u)+self.scan(u,True)); return x+self.out(y.permute(0,2,1,3))

class GambaStyle(nn.Module):
    """Dynamic graph + selective state-space mechanism-matched adaptation."""
    def __init__(self,nout,d=32,features='jbm',types=4):
        super().__init__(); self.features=features; cin=3*len(features); self.inp=nn.Sequential(nn.LayerNorm(cin),nn.Linear(cin,d),nn.GELU()); self.ssm=SelectiveNodeSSM(d); self.type_head=nn.Linear(d,types); self.rel=nn.Parameter(torch.randn(types,types)*.05); self.proj=nn.Linear(d,d); self.temp=TemporalJointBlock(d); self.head=PoolHead(d,nout); self.register_buffer('A',APHYS)
    def one(self,h):
        h=self.ssm(h); summary=h.mean(1); assign=F.softmax(self.type_head(summary),-1) # B,V,K
        dyn=torch.einsum('bvk,kl,bwl->bvw',assign,self.rel,assign)/assign.shape[-1]
        dyn=F.softmax(dyn,-1); A=.6*self.A[None]+.4*dyn
        z=graph_apply_dynamic(A,self.proj(h)); return self.temp(h+F.gelu(z))
    def forward(self,x,return_aux=False):
        p=pose_features(x,self.features); B,W,T,V,C=p.shape; h=self.inp(p).reshape(B*W,T,V,-1); h=self.one(h).reshape(B,W,T,V,-1); logits=self.head(h); return (logits,{}) if return_aux else logits

# Criterion maps: derived from the original annotation wording and MediaPipe topology; no new labels.
# CPR-Coach criterion -> (view, Halpe-26 joints).  The view labels are NOT a
# guess: channel roles were identified from measured geometry (shoulder width
# over torso height, a 7x spread), giving ch1 as frontal and ch3 as lateral.
# The joint sets are our reading of CPR biomechanics and therefore ARE an
# authored choice, which is why the map_control='random' ablation is reported
# alongside: if the anatomy map is load-bearing, the two must differ.
# Halpe-26: 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees,
# 15/16 ankles, 17 head, 18 neck, 19 mid-hip.
ANAT_MAP_CPR=[
 ('F',[7,8,9,10]),              # Overlap Hands
 ('F',[9,10]),                  # Clenching Hands
 ('F',[5,6,7,8,9,10]),          # Single Hand
 ('L',[5,6,7,8,9,10]),          # Bending Arms
 ('F',[5,6,7,8,9,10,18]),       # Tilting Arms
 ('L',[11,12,13,14,15,16,18,19]),  # Jump Pressing
 ('L',[11,12,13,14,15,16,19]),  # Squatting
 ('L',[11,12,13,14,15,16,18,19]),  # Standing
 ('F',[5,6,9,10,18,19]),        # Wrong Position
 ('L',[5,6,7,8,9,10]),          # Insufficient Pressing
 ('L',[5,6,9,10]),              # Slow Frequency
 ('L',[5,6,7,8,9,10]),          # Excessive Pressing
 ('F',[5,6,9,10,19]),           # Random Position Pressing
]

ANAT_MAP={
'cpr':ANAT_MAP_CPR,
'squat':[
 ('F',[23,24,25,26,27,28,29,30,31,32]),
 ('L',[25,26,27,28,29,30,31,32]),
 ('L',[11,12,23,24,25,26,27,28]),
 ('L',[11,12,23,24,25,26,27,28]),
 ('L',[11,12,23,24,25,26]),
 ('L',[23,24,25,26,27,28])],
'deadlift':[
 ('F',[11,12,23,24,25,26,27,28,29,30,31,32]),
 ('L',[11,12,23,24,25,26]),
 ('L',[23,24,25,26,27,28]),
 ('L',[11,12,23,24,25,26,27,28]),
 ('L',[23,24,25,26,27,28])],
'lunges':[
 ('F',[23,24,25,26,27,28,29,30,31,32]),
 ('F',[0,1,2,3,4,5,6,7,8,11,12,23,24]),
 ('L',[23,24,25,26,27,28]),
 ('F',[11,12,13,14,15,16,23,24]),
 ('F',[23,24,25,26,27,28,29,30,31,32]),
 ('L',[11,12,23,24,25,26]),
 ('L',[23,24,25,26,27,28,29,30,31,32])]
}

class CriterionSlots(nn.Module):
    """Criterion-conditioned spatio-temporal evidence slots with soft anatomy/view priors."""
    def __init__(self,exercise,d=48,soft_masks=True,hard_views=False):
        super().__init__(); self.exercise=exercise; self.C=len(ANAT_MAP[exercise]); self.d=d; self.soft_masks=soft_masks; self.hard_views=hard_views
        self.query=nn.Parameter(torch.randn(self.C,d)*.03); self.k=nn.Linear(d,d,bias=False); self.v=nn.Linear(d,d,bias=False); self.norm=nn.LayerNorm(d)
        prior=torch.full((self.C,33),-2.0); views=[]
        view_prior=torch.zeros(self.C,2)
        for c,(v,joints) in enumerate(ANAT_MAP[exercise]):
            prior[c,joints]=2.0; vi=0 if v=='F' else 1; views.append(vi); view_prior[c,vi]=1.5
        self.register_buffer('prior_logits',prior); self.register_buffer('views',torch.tensor(views,dtype=torch.long)); self.register_buffer('view_prior',view_prior)
        self.mask_resid=nn.Parameter(torch.zeros(self.C,33)); self.view_resid=nn.Parameter(torch.zeros(self.C,2))
    def forward(self,h,disable_masks=False): # B,2,T,V,D -> B,C,D
        B,W,T,V,D=h.shape
        if self.hard_views:
            z=h[:,self.views] # B,C,T,V,D
            k=self.k(z); v=self.v(z)
            score=(k*self.query[None,:,None,None,:]).sum(-1)/math.sqrt(D)
            if not disable_masks:
                mw=torch.sigmoid(self.prior_logits+self.mask_resid) if self.soft_masks else (self.prior_logits>0).float()
                score=score+torch.log(mw.clamp_min(1e-4))[None,:,None,:]
            a=F.softmax(score.flatten(2),-1).view(B,self.C,T,V)
            slot=(a[...,None]*v).sum((2,3))
        else:
            # Compute K/V once per skeleton feature, then score all criterion queries.
            # This avoids materializing a BxCxWxTxVxD expanded tensor.
            k=self.k(h); v=self.v(h) # B,W,T,V,D
            score=torch.einsum('bwtvd,cd->bcwtv',k,self.query)/math.sqrt(D)
            if not disable_masks:
                mw=torch.sigmoid(self.prior_logits+self.mask_resid) if self.soft_masks else (self.prior_logits>0).float()
                score=score+torch.log(mw.clamp_min(1e-4))[None,:,None,None,:]
            vw=F.softmax(self.view_prior+self.view_resid,-1)
            score=score+torch.log(vw.clamp_min(1e-4))[None,:,:,None,None]
            a=F.softmax(score.flatten(2),-1).view(B,self.C,W,T,V)
            slot=torch.einsum('bcwtv,bwtvd->bcd',a,v)
        return self.norm(slot+self.query[None])

class CAPER(nn.Module):
    """Counterfactual Anatomical Prototype Exchange and Residualization.

    Prediction = criterion-local prototype evidence + gated cross-criterion residual.
    The gate is learned from features; training-only counterfactual pressure supplies a
    reliability target, so no ground-truth test diagnosis is required at inference.
    """
    def __init__(self,exercise,d=28,ed=20,features='jbm',graph=True,masks=True,prototypes=True,context=True,pressure_gate=True,soft_masks=True,hard_views=True):
        super().__init__(); self.exercise=exercise; self.C=len(ANAT_MAP[exercise]); self.d=d; self.ed=ed; self.use_masks=masks; self.use_proto=prototypes; self.use_context=context; self.pressure_gate=pressure_gate
        self.enc=SkeletonEncoder(d,features,layers=1,graph=graph,learn_hops=graph)
        self.slots=CriterionSlots(exercise,d,soft_masks=soft_masks,hard_views=hard_views)
        self.evidence=nn.Sequential(nn.Linear(d,ed),nn.GELU(),nn.LayerNorm(ed))
        self.context=nn.Sequential(nn.Linear(d,ed),nn.GELU(),nn.LayerNorm(ed))
        # A direct criterion classifier stabilizes the tiny-data regime. Prototypes
        # provide an auxiliary state geometry rather than being the sole decision rule.
        self.local_w=nn.Parameter(torch.randn(self.C,ed)*.04); self.local_b=nn.Parameter(torch.zeros(self.C))
        if prototypes:
            self.proto=nn.Parameter(torch.randn(self.C,2,ed)*.08); self.log_temp=nn.Parameter(torch.tensor(-.5)); self.proto_mix=nn.Parameter(torch.tensor(-1.5))
        if context:
            # Leave-one-criterion-out contextual prior: criterion c never sees its own
            # context slot when forming the co-occurrence residual.
            self.ctx_proj=nn.Sequential(nn.Linear(ed,ed),nn.GELU(),nn.LayerNorm(ed))
            self.ctx_w=nn.Parameter(torch.randn(self.C,ed)*.04); self.ctx_b=nn.Parameter(torch.zeros(self.C))
            self.gate=nn.Sequential(nn.Linear(2*ed+2,ed),nn.GELU(),nn.Linear(ed,1))
            self.context_scale=nn.Parameter(torch.tensor(-0.7))
        self.final_bias=nn.Parameter(torch.zeros(self.C))
    def encode_slots(self,x):
        h=self.enc(x); s=self.slots(h,disable_masks=not self.use_masks); return self.evidence(s), self.context(s)
    def local_logits(self,e):
        linear=(e*self.local_w[None]).sum(-1)+self.local_b
        if not self.use_proto: return linear
        dist=((e[:,:,None,:]-self.proto[None])**2).mean(-1); tau=F.softplus(self.log_temp)+.2
        proto_logit=(dist[:,:,0]-dist[:,:,1])/tau
        return linear+torch.sigmoid(self.proto_mix)*proto_logit
    def decode(self,e,c):
        local=self.local_logits(e)
        if not self.use_context:
            return local+self.final_bias, {'local_logits':local,'residual_logits':torch.zeros_like(local),'gate':torch.zeros_like(local)}
        C=c.shape[1]
        other=(c.sum(1,keepdim=True)-c)/max(C-1,1)
        other=self.ctx_proj(other)
        residual=(other*self.ctx_w[None]).sum(-1)+self.ctx_b
        # Context is feature-gated and explicitly conflict-aware: contradictory
        # co-occurrence evidence is downweighted before it can overturn local motion.
        agreement=torch.sigmoid(local*residual)
        gfeat=torch.sigmoid(self.gate(torch.cat([e,other,local[:,:,None],residual[:,:,None]],-1)).squeeze(-1))
        gate=gfeat*agreement
        scale=torch.sigmoid(self.context_scale)
        logits=local+scale*gate*residual+self.final_bias
        return logits, {'local_logits':local,'residual_logits':residual,'gate':gate}
    def forward(self,x,return_aux=False,return_slots=False):
        e,c=self.encode_slots(x); logits,aux=self.decode(e,c); aux.update({'evidence':e,'context':c})
        if return_slots: return logits,aux
        return (logits,aux) if return_aux else logits
    def decode_slots(self,e,c): return self.decode(e,c)


def build_modern(kind,nout,exercise=None,features='jbm',**kw):
    if kind=='blockgcn_core':
        from blockgcn_core import BlockGCNCoreAdapt
        return BlockGCNCoreAdapt(nout)
    if kind=='blockgcn_style': return TopoBlockGCNStyle(nout,features=features)
    if kind=='protogcn_style': return ProtoGCNStyle(nout,features=features)
    if kind=='hypergcn_style': return HyperGCNStyle(nout,features=features)
    if kind=='gamba_style': return GambaStyle(nout,features=features)
    if kind=='cape' or kind.startswith('caper'):
        assert exercise is not None
        cfg=dict(graph=True,masks=True,prototypes=True,context=True,pressure_gate=True,soft_masks=True,hard_views=True)
        if kind=='cape': cfg['context']=False
        elif kind=='caper_soft_view': cfg['hard_views']=False
        elif kind=='caper_hard_view': cfg['hard_views']=True
        elif kind=='caper_no_graph': cfg['graph']=False
        elif kind=='caper_no_mask': cfg['masks']=False
        elif kind=='caper_no_proto': cfg['prototypes']=False
        elif kind=='caper_no_context': cfg['context']=False
        elif kind=='caper_hard_mask': cfg['soft_masks']=False
        elif kind not in ('caper','cape','caper_no_neighbor','caper_no_pair','caper_no_exchange','caper_no_pressure','caper_hard_view','caper_soft_view'): raise ValueError(kind)
        return CAPER(exercise,features=features,**cfg)
    raise ValueError(kind)
