"""Server-to-server frozen input byte transport, no array/GT interpretation."""
from pathlib import Path
import os,sys,json,time,subprocess,datetime
ROOT=Path(__file__).resolve().parents[1]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def update(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def identity(pid):
    try:
        p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'ppid':int(v[1]),'start_ticks':int(v[19]),'state':v[0],'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()}
    except (OSError,IndexError):return None
def descendants(pid):
    allp={}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            v=identity(int(p.name))
            if v:allp[v['pid']]=v
    owned={pid}
    while True:
        extra={v['pid'] for v in allp.values() if v['ppid'] in owned}-owned
        if not extra:break
        owned.update(extra)
    return [allp[k] for k in owned if k in allp]
def main():
    mode=sys.argv[1];assert mode in ('send','receive');d=ROOT/'transport'/mode;d.mkdir(parents=True,exist_ok=False)
    expected=json.loads((ROOT/'inputs/expected_inputs.json').read_text());paths=(ROOT/'inputs/transport_paths.txt').read_text().splitlines()
    assert set(paths)==set(expected) and len(paths)==120
    for rel in paths:
        p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
        assert (rel.startswith('data/episodes/sim') and p.name in ('rgb.npy','action_raw.npy')) or (rel.startswith('test_video_only/episodes/eval') and p.name in ('rgb.npy','timestamps.npy'))
    if mode=='send':
        source=Path('/g0022010zlc/wangzhaoyu/general_idm_data_20260812/runs/soap_eraser_cosmos_fullfinetune_sim50_test10_ep50_20261012_attempt01')
        for rel,v in expected.items():assert (source/rel).is_file() and (source/rel).stat().st_size==v['bytes']
        command=['tar','-czf','-','-C',str(source),'-T',str(ROOT/'inputs/transport_paths.txt')]
    else:
        destination=ROOT/'dataset';destination.mkdir(exist_ok=False)
        command=['tar','-xzf','-','--no-same-owner','-C',str(destination)]
    write(d/'launch_intent.json',{'UTC_server':now(),'mode':mode,'identity':identity(os.getpid()),'paths':paths,'no_test_GT_files':True,'original_inputs_read_only':True})
    owned={};start=time.monotonic()
    with (d/'stderr.log').open('xb') as log:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL if mode=='send' else sys.stdin.buffer,stdout=sys.stdout.buffer if mode=='send' else log,stderr=log)
        while True:
            for v in descendants(p.pid):
                k=str(v['pid'])+':'+str(v['start_ticks'])
                if k not in owned or v['command']:owned[k]=v
            update(d/'owned_processes.json',{'UTC_server':now(),'records':list(owned.values())})
            code=p.poll()
            if code is not None:break
            time.sleep(.5)
    checks=[{'recorded':v,'current':identity(v['pid'])} for v in owned.values()]
    live=[c for c in checks if c['current'] and c['current']['start_ticks']==c['recorded']['start_ticks'] and c['current']['state']!='Z']
    write(d/'terminal.json',{'UTC_server':now(),'exit_code':code,'all_owned_children_exited':not live,'checks':checks,'elapsed_s':time.monotonic()-start,'owner_exits_after_receipt':True})
    assert code==0 and not live
if __name__=='__main__':main()
