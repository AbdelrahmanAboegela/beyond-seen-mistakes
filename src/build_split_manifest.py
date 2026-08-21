"""Create and validate the frozen recording-pair-disjoint LOCO split manifest."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
try:
    from .alexgym_data import load_exercise, PREPROCESS_VERSION
    from .loco_split import make_loco_split
except ImportError:
    from alexgym_data import load_exercise, PREPROCESS_VERSION
    from loco_split import make_loco_split


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data')
    ap.add_argument('--protocol',default='configs/final_protocol.json')
    ap.add_argument('--out',default='configs/split_manifest_v2.json')
    ap.add_argument('--min-train-state',type=int,default=8)
    ap.add_argument('--min-val-state',type=int,default=1)
    a=ap.parse_args(); protocol=json.loads(Path(a.protocol).read_text())
    raw_manifest=Path('results/data_audit/raw_data_manifest.json')
    raw_hash=hashlib.sha256(raw_manifest.read_bytes()).hexdigest() if raw_manifest.exists() else None
    manifest={'version':'v2','data_root':a.data,'preprocess_version':PREPROCESS_VERSION,
              'raw_manifest_sha256':raw_hash,
              'support':{'min_train_state':a.min_train_state,'min_val_state':a.min_val_state},
              'dataset':{},'splits':{}}
    for ex,targets in protocol['default_targets'].items():
        X,Y,co,g,df=load_exercise(a.data,ex)
        manifest['dataset'][ex]={'n_repetitions':int(len(X)),
            'excluded_no_valid_view':df.attrs.get('excluded_no_valid_view',[])}
        manifest['splits'][ex]={}
        for target in targets:
            manifest['splits'][ex][target]={}
            for seed in protocol['seeds']:
                _,_,_,audit=make_loco_split(Y,co,g,target,seed,min_train_state=a.min_train_state,min_val_state=a.min_val_state)
                manifest['splits'][ex][target][str(seed)]=audit
    Path(a.out).write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'out':a.out,'targets':sum(map(len,protocol['default_targets'].values()))},indent=2))
if __name__=='__main__': main()
