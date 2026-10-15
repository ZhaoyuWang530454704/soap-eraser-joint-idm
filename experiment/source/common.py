"""Task-adapted Joint14/15 architecture; scientific numerical execution only on servers."""
from pathlib import Path
import os,json,hashlib,datetime,sys
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'contract.json').read_text())
if (ROOT/'GPU_assignment.json').is_file():SPEC['GPU_UUIDs']=json.loads((ROOT/'GPU_assignment.json').read_text())['GPU_UUIDs']
OUT=ROOT/'outputs'/SPEC['output_name']
sys.path.insert(0,str(ROOT/'reference_source'))
from model03 import fine_features,fit_embedding,embedding
from pls03 import fit_pls,pls_embedding,pls_input
from keypoints03 import keypoint_features,keypoint_input
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def update(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
    q.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def verify_sources():
    m=read(ROOT/'source_manifest.json')
    for rel,h in m['files'].items():assert sha(ROOT/rel)==h,rel
    assert sha(ROOT/'contract.json')==m['contract_SHA256']
    return sha(ROOT/'source_manifest.json')
def configure(rank=None):
    torch.set_num_threads(2 if rank is not None else 4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.manual_seed(SPEC['seed']);np.random.seed(SPEC['seed'])
    torch.use_deterministic_algorithms(True)
    if rank is not None:
        assert torch.cuda.device_count()==SPEC['world_size']==2 and len(SPEC['GPU_UUIDs'])==2;torch.cuda.set_device(rank);torch.cuda.manual_seed_all(SPEC['seed'])
        assert os.environ['CUDA_VISIBLE_DEVICES'].split(',')==SPEC['GPU_UUIDs']
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def rotmat6(q):
    a=F.normalize(q[...,0:3].float(),dim=-1,eps=1e-8)
    b=q[...,3:6].float();b=F.normalize(b-(a*b).sum(-1,keepdim=True)*a,dim=-1,eps=1e-8)
    return torch.stack((a,b,torch.cross(a,b,dim=-1)),dim=-1)

class JointBase(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__();self.backbone=resnet18(weights=None);self.backbone.fc=nn.Identity()
        if pretrained:
            p=ROOT/'pretrained/resnet18-f37072fd.pth';assert sha(p)==SPEC['pretrained_SHA256']
            state=torch.load(p,map_location='cpu',weights_only=True);state.pop('fc.weight');state.pop('fc.bias')
            self.backbone.load_state_dict(state,strict=True)
        self.views=nn.ModuleList([nn.Sequential(nn.Linear(4352,256),nn.LayerNorm(256),nn.GELU()) for _ in range(2)])
        self.visual_fuse=nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.GELU())
        self.aux=nn.Sequential(nn.Linear(1344,256),nn.LayerNorm(256),nn.GELU())
        self.shared=nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.05))
        self.position=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,3))
        self.orientation=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,6))
        self.closure=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
        for head in (self.position,self.orientation,self.closure):nn.init.normal_(head[-1].weight,0,.001);nn.init.zeros_(head[-1].bias)
        self.orientation[-1].bias.data.copy_(torch.tensor([1.,0,0,0,1,0]))
        self.register_buffer('xyz_mean',torch.zeros(3))
        self.register_buffer('aux_mean',torch.zeros(1344));self.register_buffer('aux_scale',torch.ones(1344))
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406]).reshape(1,3,1,1))
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
    def train(self,mode=True):
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        return self
    def forward(self,image,auxiliary):
        n=len(image);x=image.reshape(n*2,3,256,320).float()/255;x=(x-self.rgb_mean)/self.rgb_std;b=self.backbone
        x=b.maxpool(b.relu(b.bn1(b.conv1(x))));a=b.layer1(x);c=b.layer2(a);d=b.layer4(b.layer3(c))
        z=torch.cat((F.avg_pool2d(a,(16,16)).flatten(1),F.avg_pool2d(c,(8,8)).flatten(1),F.avg_pool2d(d,(8,10)).flatten(1)),1).reshape(n,2,4352)
        z=self.visual_fuse(torch.cat([self.views[i](z[:,i]) for i in range(2)],1))
        with torch.autocast('cuda',enabled=False):
            aux=(auxiliary.float()-self.aux_mean)/self.aux_scale
            latent=self.shared(torch.cat((z.float(),self.aux(aux)),1))
            xyz=self.position(latent)*.1+self.xyz_mean;r=rotmat6(self.orientation(latent))
            grip=torch.sigmoid(self.closure(latent));q=torch.cat((xyz,r[:,:,0],r[:,:,1],grip),1)
        return q,latent

def loss_rows(pred,target):
    position=torch.sqrt(((pred[:,:3]-target[:,:3]).float()**2).sum(1)+1e-12)/SPEC['position_loss_scale_m']
    rp=rotmat6(pred[:,3:9]);rt=rotmat6(target[:,3:9])
    orientation=torch.sqrt(((rp-rt)**2).sum((1,2))*.5+1e-10)/SPEC['rotation_loss_scale_rad']
    closure=F.smooth_l1_loss(pred[:,9],target[:,9],reduction='none',beta=SPEC['closure_smoothL1_beta'])/SPEC['closure_loss_scale']
    total=position+SPEC['rotation_loss_weight']*orientation+SPEC['closure_loss_weight']*closure
    return total,position,orientation,closure

def fingerprint(state):
    result=[]
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        result.append({'name':name,'shape':list(a.shape),'dtype':str(a.dtype),'bytes':a.nbytes,'SHA256':hashlib.sha256(a.tobytes()).hexdigest()})
    return {'records':result,'group_SHA256':hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()}
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def memory(rank):
    total=torch.cuda.get_device_properties(rank).total_memory/2**30;reserved=torch.cuda.max_memory_reserved(rank)/2**30
    v={'GPU_peak_allocated_GiB':torch.cuda.max_memory_allocated(rank)/2**30,'GPU_peak_reserved_GiB':reserved,'GPU_total_GiB':total,'headroom_GiB':total-reserved}
    v['actual_free_GiB']=torch.cuda.mem_get_info(rank)[0]/2**30
    assert v['headroom_GiB']>=SPEC['headroom_GiB'] and v['actual_free_GiB']>=SPEC['headroom_GiB'],v
    return v

def features_from_rgb(rgb,projection):
    n=len(rgb)-1;x=np.stack([fine_features(rgb[i]) for i in range(n)])
    points=np.stack([keypoint_features(rgb[i]) for i in range(n)]);groups=np.zeros(n,np.int32)
    pca=embedding(x,np.load(projection/'pca10.npz',allow_pickle=False))
    z=[pls_embedding(x,np.load(projection/(t+'10.npz'),allow_pickle=False)) for t in ('xyz','all10')]
    a=pls_input(np.concatenate((pca,z[0],z[1]),1),groups,'context2',352)
    return np.ascontiguousarray(np.concatenate((a,keypoint_input(points,groups)),1),dtype=np.float32)

def ema(raw):
    out=raw.astype(np.float64).copy();r=rot_numpy(raw[:,3:9]);s=r.copy()
    for i in range(1,len(raw)):
        out[i,[0,1,2,9]]=.5*raw[i,[0,1,2,9]]+.5*out[i-1,[0,1,2,9]]
        prev=Rotation.from_matrix(s[i-1]);delta=prev.inv()*Rotation.from_matrix(r[i])
        s[i]=(prev*Rotation.from_rotvec(.5*delta.as_rotvec())).as_matrix()
    out[:,3:6]=s[:,:,0];out[:,6:9]=s[:,:,1]
    return out.astype(np.float32)
def rot_numpy(q):
    a=q[:,:3].astype(float).copy();a/=np.linalg.norm(a,axis=1,keepdims=True)
    b=q[:,3:6].astype(float).copy();b-=(a*b).sum(1,keepdims=True)*a;b/=np.linalg.norm(b,axis=1,keepdims=True)
    return np.stack((a,b,np.cross(a,b)),2)
def metrics(pred,gt):
    e=(pred[:,:3].astype(float)-gt[:,:3])*1000;d=np.linalg.norm(e,axis=1)
    r=rot_numpy(gt[:,3:9]).transpose(0,2,1)@rot_numpy(pred[:,3:9])
    ang=Rotation.from_matrix(r).magnitude()*180/np.pi;c=np.abs(pred[:,9].astype(float)-gt[:,9]);agree=(pred[:,9]>=.5)==(gt[:,9]>=.5);pose=(d<=10)&(ang<=10)
    return {'frames':len(pred),'xyz_mae_mm':np.abs(e).mean(0).tolist(),'position_mean_mm':float(d.mean()),'position_median_mm':float(np.median(d)),'position_p95_mm':float(np.quantile(d,.95)),'position_max_mm':float(d.max()),'position_within5mm_percent':float((d<=5).mean()*100),'rotation_mean_deg':float(ang.mean()),'rotation_p95_deg':float(np.quantile(ang,.95)),'closure_mae':float(c.mean()),'closure_class_error_percent':float((~agree).mean()*100),'closure_class_agreement_percent':float(agree.mean()*100),'position10mm_rotation10deg_accuracy_percent':float(pose.mean()*100),'position10mm_rotation10deg_closureclass_accuracy_percent':float((pose&agree).mean()*100)}


class FineClosure(nn.Module):
    def __init__(self):
        super().__init__();layers=[];last=6
        for width in (12,24,32):
            layers.extend([nn.Conv2d(last,width,3,stride=2,padding=1),nn.GELU()]);last=width
        self.visual=nn.Sequential(*layers,nn.AvgPool2d(2),nn.Flatten(),nn.Linear(32*6*8,128),nn.GELU())
        self.closure=nn.Sequential(nn.Dropout(.05),nn.Linear(128,64),nn.GELU(),nn.Dropout(.05),nn.Linear(64,1),nn.Sigmoid())
    def forward(self,image):
        x=F.interpolate(image.reshape(len(image),6,256,320).float(),size=(96,128),mode='area')/255-.5
        return self.closure(self.visual(x)).squeeze(1)

class JointFrame(JointBase):
    def __init__(self,pretrained=False):
        super().__init__(pretrained=pretrained)
        self.fine_enabled=SPEC.get('fine_branch',False);self.dense_enabled=SPEC.get('dense_spatial',False)
        if self.fine_enabled:
            fine=FineClosure();self.fine_visual=fine.visual;self.fine_closure=fine.closure[:-1];del self.closure
            self.extra=nn.Sequential(nn.Linear(128+(256 if self.dense_enabled else 0),256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,256))
            self.closure_residual=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
            nn.init.normal_(self.extra[-1].weight,0,.001);nn.init.zeros_(self.extra[-1].bias)
            nn.init.normal_(self.closure_residual[-1].weight,0,.001);nn.init.zeros_(self.closure_residual[-1].bias)
            if self.dense_enabled:
                self.dense_reduce=nn.Conv2d(128,16,1)
                self.dense_embed=nn.Sequential(nn.Linear(2*16*8*10,256),nn.LayerNorm(256),nn.GELU())
    def forward(self,image,auxiliary):
        if not self.fine_enabled:return super().forward(image,auxiliary)
        n=len(image);x=image.reshape(n*2,3,256,320).float()/255;x=(x-self.rgb_mean)/self.rgb_std;b=self.backbone
        x=b.maxpool(b.relu(b.bn1(b.conv1(x))));a=b.layer1(x);c=b.layer2(a);d=b.layer4(b.layer3(c))
        z=torch.cat((F.avg_pool2d(a,(16,16)).flatten(1),F.avg_pool2d(c,(8,8)).flatten(1),F.avg_pool2d(d,(8,10)).flatten(1)),1).reshape(n,2,4352)
        z=self.visual_fuse(torch.cat([self.views[i](z[:,i]) for i in range(2)],1))
        dense=self.dense_embed(F.avg_pool2d(self.dense_reduce(c),(4,4)).reshape(n,-1)) if self.dense_enabled else None
        with torch.autocast('cuda',enabled=False):
            fine=F.interpolate(image.reshape(n,6,256,320).float(),size=(96,128),mode='area')/255-.5
            fine=self.fine_visual(fine)
            aux=(auxiliary.float()-self.aux_mean)/self.aux_scale
            latent=self.shared(torch.cat((z.float(),self.aux(aux)),1))
            extra=torch.cat((fine,dense.float()),1) if dense is not None else fine
            latent=latent+.1*self.extra(extra)
            xyz=self.position(latent)*.1+self.xyz_mean;r=rotmat6(self.orientation(latent))
            grip=torch.sigmoid(self.fine_closure(fine)+.1*self.closure_residual(latent))
            return torch.cat((xyz,r[:,:,0],r[:,:,1],grip),1),latent


class JointAction(JointFrame):
    def __init__(self,pretrained=False):
        super().__init__(pretrained=pretrained)
        self.temporal=nn.Sequential(nn.Conv1d(256,128,5,padding=2),nn.GELU(),nn.Conv1d(128,128,5,padding=4,dilation=2),nn.GELU(),nn.Conv1d(128,256,5,padding=8,dilation=4))
        nn.init.normal_(self.temporal[-1].weight,0,.001);nn.init.zeros_(self.temporal[-1].bias)
        self.fine_closure.register_forward_hook(self._capture_closure)
    def _capture_closure(self,module,inputs,output):self._closure_logits=output
    def forward(self,image,auxiliary):
        b,t=image.shape[:2];assert t==SPEC['temporal_length']
        _,base=super().forward(image.reshape(b*t,2,3,256,320),auxiliary.reshape(b*t,1344))
        with torch.autocast('cuda',enabled=False):
            z=base.reshape(b,t,256);latent=z+.1*self.temporal(z.transpose(1,2)).transpose(1,2)
            xyz=self.position(latent)*.1+self.xyz_mean;r=rotmat6(self.orientation(latent));grip=torch.sigmoid(self._closure_logits.reshape(b,t,1)+.1*self.closure_residual(latent))
            return torch.cat((xyz,r[:,:,:,0],r[:,:,:,1],grip),2),latent

frame_loss_rows=loss_rows
def loss_rows(pred,target):
    if pred.ndim==2:return frame_loss_rows(pred,target)
    b,t=pred.shape[:2];ls=[v.reshape(b,t) for v in frame_loss_rows(pred.reshape(-1,10),target.reshape(-1,10))]
    if SPEC['position_velocity_weight']:
        delta=(pred[:,1:,:3]-pred[:,:-1,:3])-(target[:,1:,:3]-target[:,:-1,:3]);velocity=F.pad(torch.sqrt(delta.float().square().sum(2)+1e-12)/.01,(1,0));ls[0]=ls[0]+SPEC['position_velocity_weight']*velocity;ls[1]=ls[1]+SPEC['position_velocity_weight']*velocity
    return tuple(ls)

def merge_clips(data,clip_ids,outputs,latents):
    assert len(clip_ids)==len(data) and np.array_equal(np.sort(clip_ids),np.arange(len(data)))
    n=len(data.groups);qs=np.zeros((n,4),float);rs=np.zeros((n,3,3),float);zs=np.zeros((n,256),float);weights=np.zeros(n,float)
    taper=np.sin(np.pi*(np.arange(SPEC['temporal_length'])+.5)/SPEC['temporal_length'])**2
    for ci,q,z in zip(clip_ids,outputs,latents):
        valid=data.clip_valid[ci]>0;rows=data.clip_rows[ci][valid];w=taper[valid];np.add.at(weights,rows,w);np.add.at(qs,rows,q[valid][:,[0,1,2,9]]*w[:,None]);np.add.at(rs,rows,rot_numpy(q[valid,3:9])*w[:,None,None]);np.add.at(zs,rows,z[valid]*w[:,None])
    ix=data.row_indices;assert np.all(weights[ix]>0);p=qs[ix]/weights[ix,None];z=zs[ix]/weights[ix,None]
    u,_,vh=np.linalg.svd(rs[ix]/weights[ix,None,None]);sign=np.linalg.det(u@vh);u[:,:,-1]*=sign[:,None];rotation=u@vh
    out=np.concatenate((p[:,:3],rotation[:,:,0],rotation[:,:,1],p[:,3:4]),1).astype(np.float32)
    assert np.isfinite(out).all() and np.isfinite(z).all();return out,z.astype(np.float32)

def initialize(model,phase):
    norm=np.load(ROOT/'features'/phase/'projection/normalization10.npz',allow_pickle=False)
    model.xyz_mean.copy_(torch.from_numpy(norm['xyz_mean']))
    model.aux_mean.copy_(torch.from_numpy(norm['aux_mean']));model.aux_scale.copy_(torch.from_numpy(norm['aux_scale']))
    model.orientation[-1].bias.data.copy_(torch.from_numpy(norm['rotation_mean']))
    mean=float(np.clip(norm['closure_mean'],.01,.99))
    model.fine_closure[-1].bias.data.fill_(np.log(mean/(1-mean)))
    return {'kind':'fresh_ImageNet_task_initialization','ImageNet_SHA256':SPEC['pretrained_SHA256'],'projection_scope':phase,'whiteboard_task_weights_loaded':False,'prior_soap_weights_loaded':False,'TRAIN_only_bias_and_normalization':True}

class ClipData(torch.utils.data.Dataset):
    def __init__(self,phase,partition):
        assert phase in ('selection','refit') and partition in ('train','validation','test')
        self.phase=phase;self.partition=partition;self.test=partition=='test'
        meta=np.load(ROOT/'features'/('TEST_meta.npz' if self.test else 'TRAIN_meta.npz'),allow_pickle=False)
        self.groups=meta['groups'];self.rows=meta['rows'];self.times=meta['times']
        self.y=np.zeros((len(self.groups),10),np.float32) if self.test else meta['y']
        self.ids=SPEC['test_ids'] if self.test else SPEC['validation_ids'] if partition=='validation' else SPEC['fit_ids'] if phase=='selection' else SPEC['train_ids']
        assert self.test or set(self.ids).isdisjoint(SPEC['test_ids'])
        self.row_indices=np.flatnonzero(np.isin(self.groups,self.ids));self._rgb={};self._aux=None
        self.aux_path=ROOT/'features'/('test' if self.test else phase)/'auxiliary.npy'
        clips=[];valid=[];names=[];T=SPEC['temporal_length']
        for sid in self.ids:
            rows=np.flatnonzero(self.groups==sid);assert np.array_equal(self.rows[rows],np.arange(375))
            for start in range(0,len(rows),SPEC['temporal_stride']):
                rr=start+np.arange(T);clips.append(rows[np.minimum(rr,len(rows)-1)]);valid.append(rr<len(rows));names.append(sid+':start'+str(start))
        self.clip_rows=np.asarray(clips,np.int64);self.clip_valid=np.asarray(valid,np.float32);self.sample_ids=np.asarray(names)
        self.coverage=np.bincount(self.clip_rows[self.clip_valid>0],minlength=len(self.groups));assert np.all(self.coverage[self.row_indices]>0)
        self.loss_weights=self.clip_valid/np.maximum(self.coverage[self.clip_rows],1)
        assert np.isclose(self.loss_weights.sum(),len(self.row_indices),rtol=0,atol=1e-7)
    def __getstate__(self):
        d=self.__dict__.copy();d['_rgb']={};d['_aux']=None;return d
    def __len__(self):return len(self.clip_rows)
    def __getitem__(self,index):
        index,pad_valid=index if isinstance(index,tuple) else (index,1.);rows=self.clip_rows[index];sid=str(self.groups[rows[0]])
        if sid not in self._rgb:
            self._rgb[sid]=np.load(ROOT/'dataset'/('test_video_only' if self.test else 'data')/'episodes'/sid/'rgb.npy',mmap_mode='r',allow_pickle=False)
        if self._aux is None:self._aux=np.load(self.aux_path,mmap_mode='r',allow_pickle=False)
        rgb=self._rgb[sid][self.rows[rows]]
        image=np.stack((rgb[:,:,:320].transpose(0,3,1,2),rgb[:,:,320:].transpose(0,3,1,2)),1)
        return torch.from_numpy(image.copy()),torch.from_numpy(self._aux[rows].copy()),torch.from_numpy(self.y[rows].copy()),int(index),torch.from_numpy((self.loss_weights[index]*pad_valid).astype(np.float32))

def epoch_plan(n,epoch):
    order=np.random.RandomState(SPEC['seed']+epoch).permutation(n);pad=(-n)%SPEC['global_batch']
    mask=np.r_[np.ones(n,np.float32),np.zeros(pad,np.float32)]
    if pad:order=np.r_[order,order[:pad]]
    assert len(order)%8==0 and np.array_equal(np.sort(order[mask>0]),np.arange(n))
    return order,mask




# Additional registered rotation dynamics term. Architecture and old loss remain unchanged.
_loss_without_rotation_velocity=loss_rows
def rotation_velocity_rows(pred,target):
    assert pred.ndim==target.ndim==3 and pred.shape==target.shape and pred.shape[-1]==10
    rp=rotmat6(pred[...,3:9]);rt=rotmat6(target[...,3:9])
    dp=rp[:,:-1].transpose(-1,-2)@rp[:,1:]
    dt=rt[:,:-1].transpose(-1,-2)@rt[:,1:]
    error=torch.sqrt(.5*(dp-dt).float().square().sum((-1,-2))+1e-12)-1e-6
    return F.pad(error/SPEC['rotation_velocity_loss_scale_rad'],(1,0))
def rotation_pair_weights(data):
    rows=data.clip_rows;valid=data.clip_valid.astype(bool)
    pairs=np.zeros_like(valid);pairs[:,1:]=valid[:,1:]&valid[:,:-1]&(rows[:,1:]==rows[:,:-1]+1)
    assert np.all(data.groups[rows[:,1:][pairs[:,1:]]]==data.groups[rows[:,:-1][pairs[:,1:]]])
    coverage=np.bincount(rows[pairs],minlength=len(data.groups))
    weights=pairs/np.maximum(coverage[rows],1)
    multiplier=np.divide(weights,data.loss_weights,out=np.zeros_like(weights),where=data.loss_weights>0)
    expected=len(data.row_indices)-len(data.ids)
    assert np.isclose(weights.sum(),expected,rtol=0,atol=1e-8)
    assert np.all(weights[:,0]==0) and np.all(weights[~valid]==0)
    return multiplier.astype(np.float32),weights,coverage
def loss_rows(pred,target,rotation_pair_multiplier=None):
    values=list(_loss_without_rotation_velocity(pred,target))
    if pred.ndim==2:return tuple(values)
    rv=rotation_velocity_rows(pred,target)
    if rotation_pair_multiplier is not None:
        assert rotation_pair_multiplier.shape==rv.shape
        rv=rv*rotation_pair_multiplier
    values[0]=values[0]+SPEC['rotation_loss_weight']*SPEC['rotation_velocity_weight']*rv
    return (*values,rv)
