"""Train checkpointed v2 TCN/FACT runs for the context-evidence study."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

SRC_DIR=Path(__file__).resolve().parent
ROOT_DIR=SRC_DIR.parent

BACKBONES={'tcn','gru','transformer','ssm','stgcn'}
# FACT variant name -> map_control.  'fact' is the reported configuration; the
# map controls are named so their result files cannot be pooled with it.
FACT_VARIANTS={
    'fact':              'anatomy',
    'fact_random_map':   'random',
    'fact_permuted_map': 'permuted',
}
SUPPORTED=sorted(set(FACT_VARIANTS)|BACKBONES)

def run_one(task,a):
    # a.data/a.outdir were resolved against the caller's cwd in main() before the
    # child is launched with cwd=ROOT_DIR; relative paths must not silently
    # re-anchor to the repository root.
    model,ex,target,seed=task;root=Path(a.outdir);stem=f'{model}_{ex}_{target}_s{seed}'
    result=root/'runs'/f'{stem}.json';checkpoint=root/'checkpoints'/f'{stem}.pt'
    fact_variant=model in FACT_VARIANTS
    if result.exists() and checkpoint.exists():return task,'cached',0.
    # Scripts are addressed through the repository root so the sweep works from any cwd.
    script=str(SRC_DIR/('train_fact.py' if fact_variant else 'train_backbones.py'))
    cmd=[sys.executable,script,'--data',a.data,'--exercise',ex,'--target',target,
         '--seed',str(seed),'--out',str(result),'--epochs',str(a.epochs),
         '--checkpoint',str(checkpoint),
         '--min-train-state',str(a.min_train_state),'--min-val-state',str(a.min_val_state)]
    if fact_variant:
        cmd += ['--patience',str(a.patience),'--map-control',FACT_VARIANTS[model]]
    else:
        cmd += ['--model',model,'--patience',str(a.patience)]
    env=os.environ.copy()
    env['PYTHONPATH']=str(SRC_DIR)+os.pathsep+env.get('PYTHONPATH','')
    started=time.time()
    try:
        p=subprocess.run(cmd,cwd=str(ROOT_DIR),env=env,stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE,text=True,timeout=a.timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'{task}: exceeded --timeout of {a.timeout}s; raise it for full-length runs.') from None
    if p.returncode:raise RuntimeError(f'{task}: {p.stderr[-2000:]}')
    return task,'done',time.time()-started

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--protocol',default=str(ROOT_DIR/'configs/final_protocol.json'))
    ap.add_argument('--outdir',default='results/reproduction');ap.add_argument('--models',default='tcn,gru,transformer,ssm,fact')
    ap.add_argument('--seeds',help='Comma-separated seeds; defaults to the frozen protocol seeds.')
    ap.add_argument('--exercises',help='Comma-separated exercises; defaults to every exercise with protocol targets.')
    ap.add_argument('--jobs',type=int,default=3);ap.add_argument('--epochs',type=int,default=55);ap.add_argument('--patience',type=int,default=9)
    ap.add_argument('--min-train-state',type=int,default=8);ap.add_argument('--min-val-state',type=int,default=1)
    ap.add_argument('--timeout',type=int,default=3600,help='Per-run wall-clock limit in seconds.')
    a=ap.parse_args()
    a.data=str(Path(a.data).resolve())
    a.outdir=str(Path(a.outdir).resolve())
    a.protocol=str(Path(a.protocol).resolve())
    root=Path(a.outdir)
    models=[m.strip() for m in a.models.split(',') if m.strip()]
    if unknown:=[m for m in models if m not in SUPPORTED]:
        ap.error(f'unsupported --models entries {unknown}; choose from {SUPPORTED}')
    protocol=json.loads(Path(a.protocol).read_text())
    seeds=[int(s) for s in a.seeds.split(',')] if a.seeds else protocol['seeds']
    targets_by_exercise=protocol['default_targets']
    if a.exercises:
        wanted=[e.strip() for e in a.exercises.split(',') if e.strip()]
        if unknown:=[e for e in wanted if e not in targets_by_exercise]:
            ap.error(f'unknown --exercises entries {unknown}; protocol defines {sorted(targets_by_exercise)}')
        targets_by_exercise={e:targets_by_exercise[e] for e in wanted}
    (root/'runs').mkdir(parents=True,exist_ok=True);(root/'checkpoints').mkdir(parents=True,exist_ok=True)
    tasks=[(model,ex,target,seed) for model in models
           for ex,targets in targets_by_exercise.items() for target in targets for seed in seeds]
    if not tasks:
        raise SystemExit('Nothing to run: the selected exercises have no protocol targets.')
    print(f'v2 context-evidence sweep: {len(tasks)} runs',flush=True)
    failures=[]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        fs={pool.submit(run_one,t,a):t for t in tasks}
        for f in as_completed(fs):
            try:print(f.result(),flush=True)
            except Exception as exc:failures.append(str(exc));print(f'FAILED {fs[f]}: {exc}',flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} of {len(tasks)} runs failed.')

if __name__=='__main__':main()
