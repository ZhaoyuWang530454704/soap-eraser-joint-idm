"""Public complete RGB-to-action package. Input contains RGB only, never action labels."""
from pathlib import Path
import sys,json,hashlib,argparse
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
from common import JointAction,features_from_rgb,rot_numpy,ema,fingerprint,np,torch,SPEC
PACK=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def load_model(device):
    m=json.loads((PACK/'package_manifest.json').read_text())
    for rel,h in m['files'].items():assert sha(PACK/rel)==h,rel
    value=torch.load(PACK/'model.pt',map_location='cpu',weights_only=True);model=JointAction();model.load_state_dict(value['state'],strict=True);assert len(model.state_dict())==187
    return model.to(device).eval()
def predict(rgb,model,device):
    assert rgb.ndim==4 and rgb.shape[1:]==(256,640,3) and rgb.dtype==np.uint8 and len(rgb)>=2
    aux=features_from_rgb(rgb,PACK/'projection');n=len(rgb)-1;T=SPEC['temporal_length'];starts=np.arange(0,n,SPEC['temporal_stride']);q=[];z=[]
    actual=[];valid=[]
    for offset in range(0,len(starts),2):
        batch=starts[offset:offset+2];count=len(batch);batch=np.r_[batch,np.repeat(batch[-1],2-count)];rows=np.minimum(batch[:,None]+np.arange(T),n-1)
        x=rgb[rows];im=np.stack((x[:,:,:,:320].transpose(0,1,4,2,3),x[:,:,:,320:].transpose(0,1,4,2,3)),2)
        a=aux[rows]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output,latent=model(torch.from_numpy(im.copy()).to(device),torch.from_numpy(a.copy()).to(device))
        q.extend(output[:count].float().cpu().numpy());z.extend(latent[:count].float().cpu().numpy());actual.extend(batch.tolist());valid.extend((np.arange(2)<count).tolist())
    q=np.stack(q);z=np.stack(z);taper=np.sin(np.pi*(np.arange(T)+.5)/T)**2
    scalar=np.zeros((n,4),float);rotation=np.zeros((n,3,3),float);weights=np.zeros(n,float)
    for start,window in zip(starts,q):
        rr=start+np.arange(T);ok=rr<n;rows=rr[ok];w=taper[ok]
        scalar[rows]+=window[ok][:,[0,1,2,9]]*w[:,None];rotation[rows]+=rot_numpy(window[ok,3:9])*w[:,None,None];weights[rows]+=w
    assert np.all(weights>0);scalar/=weights[:,None];u,_,vh=np.linalg.svd(rotation/weights[:,None,None]);u[:,:,-1]*=np.linalg.det(u@vh)[:,None];r=u@vh
    raw=np.concatenate((scalar[:,:3],r[:,:,0],r[:,:,1],scalar[:,3:4]),1).astype(np.float32)
    return {'raw':raw,'ema05':ema(raw),'window_outputs':q,'window_latents':z,'auxiliary':aux,'starts':starts,'actual_batch_starts':np.asarray(actual),'actual_batch_valid':np.asarray(valid)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--rgb',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    value=predict(np.load(a.rgb,mmap_mode='r',allow_pickle=False),load_model(a.device),a.device)
    assert not a.out.exists();np.savez_compressed(a.out,**value)
if __name__=='__main__':main()
