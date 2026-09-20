from pathlib import Path
import json,numpy as np,pandas as pd

from temporal_features import (N_RATE_FEATURES, append_constant_channels,
                               rate_features, standardize)

PREPROCESS_VERSION = "alexgym-v2-interpolate-whole-frame-missing"

CRITERIA={
'squat':['Feet Out 30 F','Whole Feet Flat On the Floor (1) L','Bend Hips and Knees Simultaniously (1) L','Hips backwards (1) L','Lower back neural (1) L','Hips are lower than knees level (1 point) L'],
'deadlift':['Balance (0.5 point) F','Straight back (0.5) L','Full back leg extended (0.5 point) L','Slow reverse movement (0.5 point) L','Support knee\xa0bend\xa015-20° (0.5 point) L'],
'lunges':['Heels shoulder width apart (0.5 point) F','Head looking forward (0.5 point)F','Bend knees slowly 90° (0.5 point) L','Swinging arms (0.5 point) F','Front foot is parallel with the back foot, not on one line with the back foot (0.5 point) F','Straight back (0.5 point) L','Back knee is just above the floor (0.5 point) L']}

def resample(seq,T=16):
 x=np.asarray(seq,dtype=np.float32)
 if len(x)==T:return x
 old=np.linspace(0,1,len(x));new=np.linspace(0,1,T);o=np.empty((T,x.shape[1],x.shape[2]),np.float32)
 for j in range(x.shape[1]):
  for c in range(x.shape[2]):o[:,j,c]=np.interp(new,old,x[:,j,c])
 return o

def fill_missing_frames(seq):
 """Linearly interpolate frames in which the entire pose is absent.

 The raw release uses all-zero frames for failed pose detections.  Interpolation
 happens before normalization and temporal resampling.  Partially missing joints
 are deliberately left untouched because the release has no confidence channel
 with which to distinguish a true zero coordinate from a missing joint.
 """
 x=np.asarray(seq,dtype=np.float32).copy()
 valid=np.abs(x).sum(axis=(1,2))>1e-8
 if not valid.any():
  return x,valid
 if not valid.all():
  t=np.arange(len(x));tv=t[valid]
  for j in range(x.shape[1]):
   for c in range(x.shape[2]):x[:,j,c]=np.interp(t,tv,x[valid,j,c])
 return x,valid

def normalize_pose(x):
 pelvis=(x[:,23,:]+x[:,24,:])/2;x=x-pelvis[:,None,:]
 shoulder=np.linalg.norm(x[:,11]-x[:,12],axis=-1);hip=np.linalg.norm(x[:,23]-x[:,24],axis=-1)
 sizes=np.r_[shoulder[shoulder>1e-5],hip[hip>1e-5]]
 scale=float(np.median(sizes)) if len(sizes) else 1.0
 return x/max(scale,1e-3)

def load_exercise(root,exercise='squat',T=16,with_rate=False):
 root=Path(root)
 # A separate cache and version tag keep the published artefact byte-identical.
 cache=root/(f'{exercise}_T16_rate.npz' if with_rate else f'{exercise}_T16.npz')
 want=PREPROCESS_VERSION+('-rate' if with_rate else '')
 if T==16 and cache.exists() and (root/f'{exercise}_df.pkl').exists():
  z=np.load(cache)
  version=str(z['preprocess_version'].item()) if 'preprocess_version' in z.files else ''
  if version==want:
   df=pd.read_pickle(root/f'{exercise}_df.pkl'); return z['X'],z['Y'],z['co'],z['g'],df
 df=pd.read_excel(root/f'{exercise}.xlsx');front=json.load(open(root/f'front_pose_{exercise}.json'));lat=json.load(open(root/f'lat_pose_{exercise}.json'))
 # A bare assert is stripped by `python -O`; zip() would then truncate to the
 # shortest input and pair each label row with the wrong pose sequence.
 if not (len(df)==len(front)==len(lat)):
  raise ValueError(f'{exercise}: workbook/front/lateral lengths differ: {len(df)}, {len(front)}, {len(lat)}. The three files must be row-aligned; see data/README.md.')
 X=[]
 quality=[];keep=[];excluded=[];rates=[]
 for row_idx,(fs,ls) in enumerate(zip(front,lat,strict=True)):
  fs,fv=fill_missing_frames(fs);ls,lv=fill_missing_frames(ls)
  if not fv.any() or not lv.any():
   excluded.append({'raw_row':int(row_idx),'front_has_valid_frame':bool(fv.any()),
                    'lateral_has_valid_frame':bool(lv.any())})
   continue
  # rate statistics come from the pre-resampling sequences, which is exactly
  # what resampling destroys for the 'slow'/'simultaneous' criteria
  rates.append(np.concatenate([rate_features(fs),rate_features(ls)]))
  f=normalize_pose(resample(fs,T));l=normalize_pose(resample(ls,T));X.append(np.concatenate([f.reshape(T,-1),l.reshape(T,-1)],-1))
  quality.append((float(fv.mean()),float(lv.mean()),int((~fv).sum()),int((~lv).sum())))
  keep.append(row_idx)
 df=df.iloc[keep].copy().reset_index(drop=True)
 X=np.stack(X)
 if with_rate: X=append_constant_channels(X,standardize(np.stack(rates)))
 Y=(df[CRITERIA[exercise]].fillna(0).to_numpy()<=0).astype(np.float32);co=np.array([''.join(map(str,r.astype(int))) for r in Y]);g=df['Num Video Frontal'].to_numpy()
 q=np.asarray(quality)
 df['_front_valid_fraction']=q[:,0];df['_lateral_valid_fraction']=q[:,1]
 df['_front_missing_frames']=q[:,2].astype(int);df['_lateral_missing_frames']=q[:,3].astype(int)
 df.attrs['preprocess_version']=want
 df.attrs['excluded_no_valid_view']=excluded
 return X,Y,co,g,df
