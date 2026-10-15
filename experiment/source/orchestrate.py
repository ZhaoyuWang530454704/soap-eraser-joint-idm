"""Exclusive scientific scheduler; exact process receipts, no automatic retry."""
from pathlib import Path
import os,sys,json,time,subprocess,datetime,hashlib,traceback,signal,fcntl
ROOT=Path(__file__).resolve().parents[1];SPEC=json.loads((ROOT/'contract.json').read_text())
PYTHON='/mnt/data/wangzhaoyu/omnivita/omnivita_idm_20260803/pyenv/python/bin/python3.13'
RUNTIME=Path('/mnt/data/wangzhaoyu/outputs/whiteboard_cosmos_unfrozen_eightgpu_real74_20261011_study01_trainfirst_attempt06')
sys.path.insert(0,str(ROOT/'source'))
from transport_bytes import identity,descendants,now,write,update
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def hardware():
    h=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    a=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return h,a
def live(owned):
    result=[]
    for v in owned.values():
        q=identity(v['pid'])
        if q and q['start_ticks']==v['start_ticks'] and q['state']!='Z':result.append(q)
    return result
def ready():
    h,a=hardware()
    for u in SPEC['GPU_UUIDs']:
        row=next(q for q in h.splitlines() if u in q);c=[x.strip() for x in row.split(',')]
        if u in a or int(c[3].split()[0])>16 or int(c[4].split()[0])!=0:return False
    return True
def assign_idle_GPUs():
    locks=ROOT.parent/'.joint_IDM_GPU_leases';locks.mkdir(exist_ok=True)
    while True:
        h,a=hardware();idle=[]
        for row in h.splitlines():
            c=[x.strip() for x in row.split(',')]
            if c[1] in SPEC['allowed_GPU_UUIDs'] and 'A100' in c[2] and c[1] not in a and int(c[3].split()[0])<=16 and int(c[4].split()[0])==0:idle.append(c)
        handles=[];chosen=[]
        for c in idle:
            f=(locks/(c[1]+'.lock')).open('a+')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:f.close();continue
            handles.append(f);chosen.append(c)
            if len(chosen)==2:break
        if len(chosen)==2:
            SPEC['GPU_UUIDs']=[c[1] for c in chosen]
            if ready():
                write(ROOT/'GPU_assignment.json',{'UTC_server':now(),'GPU_UUIDs':SPEC['GPU_UUIDs'],'physical_GPU_indices':[int(c[0]) for c in chosen],'hardware_snapshot':h,'compute_apps_snapshot':a,'two_actually_idle_before_assignment':True,'max_concurrent_GPUs':2,'lease_owner':identity(os.getpid())});return handles
        for f in handles:f.close()
        update(ROOT/'progress.json',{'UTC_server':now(),'stage':'queued_waiting_for_two_idle_A100','TEST_GT_read':False,'other_processes_untouched':True});time.sleep(30)
def run(unit,script,gpu=False,args=()):
    d=ROOT/'processes'/unit;d.mkdir(parents=True,exist_ok=False);owned={};start=time.monotonic()
    write(d/'launch_intent.json',{'UTC_server':now(),'unit':unit,'script_SHA256':sha(ROOT/'source'/script),'no_retry':True,'GPU_UUIDs':SPEC['GPU_UUIDs'] if gpu else []})
    while gpu and not ready():
        update(ROOT/'progress.json',{'UTC_server':now(),'stage':'waiting_assigned_GPUs_idle','unit':unit,'other_processes_untouched':True,'TEST_GT_read':False});time.sleep(15)
    site=RUNTIME/'runtime/site-packages'
    env=dict(os.environ,PYTHONPATH=str(site),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=','.join(SPEC['GPU_UUIDs']) if gpu else '',OMP_NUM_THREADS='2' if gpu else '4',OPENBLAS_NUM_THREADS='2' if gpu else '4',MKL_NUM_THREADS='2' if gpu else '4',CUBLAS_WORKSPACE_CONFIG=':4096:8',NCCL_DEBUG='WARN',MPLCONFIGDIR=str(ROOT/'matplotlib_cache'))
    env['LD_LIBRARY_PATH']=':'.join(map(str,[RUNTIME/'runtime_cuda_libs',*sorted((site/'nvidia').glob('*/lib'))]))
    command=[PYTHON,'-u',str(ROOT/'source'/script),*args] if not gpu else [PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=2','--max-restarts=0',str(ROOT/'source'/script),*args]
    update(ROOT/'progress.json',{'UTC_server':now(),'stage':unit+'_starting','TEST_GT_read':False})
    with (d/'stdout.log').open('xb') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        v=identity(p.pid);assert v;v['command']=v['command'] or ' '.join(command);write(d/'launcher_process.json',v)
        while True:
            for q in descendants(p.pid):
                k=str(q['pid'])+':'+str(q['start_ticks'])
                if k not in owned or q['command']:owned[k]=q
            update(d/'owned_processes.json',{'UTC_server':now(),'records':list(owned.values())})
            code=p.poll()
            if code is not None:break
            time.sleep(1)
    deadline=time.monotonic()+20
    while live(owned) and time.monotonic()<deadline:time.sleep(.5)
    remaining=live(owned)
    if code!=0 and remaining:
        for q in remaining:
            cur=identity(q['pid'])
            if cur and cur['start_ticks']==q['start_ticks']:os.kill(q['pid'],signal.SIGTERM)
        time.sleep(3);remaining=live(owned)
    checks=[{'recorded':q,'current':identity(q['pid'])} for q in owned.values()]
    write(d/'terminal.json',{'UTC_server':now(),'unit':unit,'exit_code':code,'elapsed_s':time.monotonic()-start,'all_owned_children_exited':not remaining,'owned_checks':checks,'remaining':remaining,'no_retry':True})
    assert code==0 and not remaining,(unit,code,remaining)
def main():
    assert (ROOT/'launch_intent.json').exists() and not (ROOT/'scheduler_terminal.json').exists()
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for rel,h in manifest['files'].items():assert sha(ROOT/rel)==h,rel
    write(ROOT/'evidence/scheduler_started.json',{'UTC_server':now(),'identity':identity(os.getpid()),'source_manifest_SHA256':sha(ROOT/'source_manifest.json'),'unique':True})
    error=None;start=time.monotonic();handles=[]
    try:
        for mode in ('send','receive'):
            p=ROOT/'transport'/mode/'terminal.json'
            if mode=='receive':assert p.is_file() and json.loads(p.read_text())['exit_code']==0
        run('CPU_preparation','prepare.py')
        handles=assign_idle_GPUs()
        run('selection_training','train.py',True,['selection'])
        run('TRAIN_epoch_selection','choose_epoch.py')
        assert json.loads((ROOT/'epoch_selection.json').read_text())['status']=='TRAIN_only_epoch_frozen_before_fresh_all50_refit_and_TEST'
        run('refit_training','train.py',True,['refit'])
        update(ROOT/'progress.json',{'UTC_server':now(),'stage':'all50_joint_training_complete_pending_independent_prediction_evaluation','TEST_GT_read':False})
    except Exception as e:error=type(e).__name__+': '+str(e);traceback.print_exc()
    for f in handles:f.close()
    write(ROOT/'scheduler_terminal.json',{'UTC_server':now(),'status':'training_complete_pending_prediction_evaluation' if error is None else 'failed_no_retry','error':error,'elapsed_s':time.monotonic()-start,'TEST_GT_read':False,'scheduler_exits_after_receipt':True})
    sys.exit(0 if error is None else 1)
if __name__=='__main__':main()
