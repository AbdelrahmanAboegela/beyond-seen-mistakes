"""Train checkpointed v2 TCN/FACT runs for the context-evidence study."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

def run_one(task,a):
    model,ex,target,seed=task;root=Path(a.outdir);stem=f'{model}_{ex}_{target}_s{seed}'
    result=root/'runs'/f'{stem}.json';checkpoint=root/'checkpoints'/f'{stem}.pt'
    fact_variant=model in {'fact','fact_random_map','fact_permuted_map'}
    needs_checkpoint=fact_variant or model in {'tcn','gru','transformer','ssm','stgcn'}
    if result.exists() and (checkpoint.exists() or not needs_checkpoint):return task,'cached',0.
    if fact_variant:script='src/train_fact.py'
    elif model=='independent':script='src/train_independent.py'
    elif model in {'majority','knn'}:script='src/eval_nonparametric_baselines.py'
    else:script='src/train_backbones.py'
    cmd=[sys.executable,script,'--data',a.data,'--exercise',ex,'--target',target,
         '--seed',str(seed),'--out',str(result)]
    if needs_checkpoint:cmd += ['--epochs',str(a.epochs),'--checkpoint',str(checkpoint)]
    elif model=='independent':cmd += ['--epochs',str(a.epochs),'--patience',str(a.patience)]
    if fact_variant:
        cmd += ['--patience',str(a.patience)]
        if model!='fact':cmd += ['--map-control',model.replace('fact_','').replace('_map','')]
    elif model!='independent':cmd += ['--model',model]
    env=os.environ.copy();env['PYTHONPATH']='src';started=time.time()
    p=subprocess.run(cmd,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,timeout=a.timeout)
    if p.returncode:raise RuntimeError(f'{task}: {p.stderr[-2000:]}')
    return task,'done',time.time()-started

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--protocol',default='configs/final_protocol.json')
    ap.add_argument('--outdir',default='results/reproduction');ap.add_argument('--models',default='tcn,gru,transformer,ssm,fact')
    ap.add_argument('--jobs',type=int,default=3);ap.add_argument('--epochs',type=int,default=55);ap.add_argument('--patience',type=int,default=9)
    ap.add_argument('--timeout',type=int,default=180);a=ap.parse_args();root=Path(a.outdir)
    (root/'runs').mkdir(parents=True,exist_ok=True);(root/'checkpoints').mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path(a.protocol).read_text());tasks=[]
    for model in a.models.split(','):
        for ex,targets in protocol['default_targets'].items():
            for target in targets:
                for seed in protocol['seeds']:tasks.append((model,ex,target,seed))
    print(f'v2 context-evidence sweep: {len(tasks)} runs',flush=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        fs={pool.submit(run_one,t,a):t for t in tasks}
        for f in as_completed(fs):print(f.result(),flush=True)

if __name__=='__main__':main()
