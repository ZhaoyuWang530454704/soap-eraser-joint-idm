"""CPU-only, non-neural image embeddings and numerical regression recipes."""
import numpy as np
import cv2
from scipy.linalg import eigh

BLOCKS = [('third',5376,128),('wrist',2880,96),('fingers',768,32)]
SEED=20260812

def fine_features(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    def block(im,shape,gradshape):
        sm=cv2.resize(im,shape,interpolation=cv2.INTER_AREA).astype(np.float32)/255
        g=cv2.resize(im,gradshape,interpolation=cv2.INTER_AREA).astype(np.float32)/255
        dx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)/4
        dy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)/4
        return np.concatenate([sm.ravel(),dx.ravel(),dy.ravel()])
    third=block(gray[:230,45:275],(64,56),(32,28))
    wrist=block(gray[:240,320:640],(48,40),(24,20))
    fingers=cv2.resize(gray[170:256,325:635],(48,16),interpolation=cv2.INTER_AREA).astype(np.float32).ravel()/255
    return np.concatenate([third,wrist,fingers])

def fit_embedding(x,seed=SEED):
    rng=np.random.default_rng(seed); result={}; z=[]; start=0
    for i,(name,d,k) in enumerate(BLOCKS):
        a=x[:,start:start+d]; start+=d
        mu=a.mean(0); scale=np.maximum(a.std(0),.025)*np.sqrt(d)
        # All fit-session rows set the normalization. Only fit-session RGB sets PCA.
        # Deterministic subsampling limits memory; the same projection applies to every row.
        ids=np.linspace(0,len(a)-1,min(len(a),7000),dtype=int)
        sample=(a[ids]-mu)/scale
        q=k+16
        omega=rng.standard_normal((len(sample),q),dtype=np.float32)
        basis=np.linalg.qr(sample.T@omega,mode='reduced')[0].astype(np.float32)
        # One power iteration resolves finer components beyond the dominant lighting modes.
        basis=np.linalg.qr(sample.T@(sample@basis),mode='reduced')[0].astype(np.float32)
        reduced=sample@basis
        vals,vec=eigh((reduced.T@reduced).astype(np.float64),check_finite=False)
        proj=(basis@vec[:,-k:][:,::-1]).astype(np.float32)
        result[f'mean{i}']=mu; result[f'scale{i}']=scale; result[f'projection{i}']=proj
        result[f'PCA_eigenvalues{i}']=vals[-k:][::-1]
        z.append(((a-mu)/scale)@proj)
        print('PCA TRAIN-only block',name,'rows',len(a),'sample',len(sample),'output',k,flush=True)
    return result,np.concatenate(z,axis=1).astype(np.float32)

def embedding(x,model):
    z=[]; start=0
    for i,(_,d,_) in enumerate(BLOCKS):
        a=x[:,start:start+d]; start+=d
        z.append(((a-model[f'mean{i}'])/model[f'scale{i}'])@model[f'projection{i}'])
    return np.concatenate(z,axis=1).astype(np.float32)

def input_features(z,groups,mode,ratio):
    weight=np.ones(z.shape[1],np.float32); weight[128:224]=np.sqrt(ratio)
    a=z*weight
    if mode=='spatial':return a
    assert mode=='context2'
    before=a.copy(); after=a.copy(); before2=a.copy(); after2=a.copy()
    for sid in np.unique(groups):
        ids=np.flatnonzero(groups==sid); v=a[ids]; n=len(v); r=np.arange(n)
        before[ids]=v[np.maximum(r-1,0)]; after[ids]=v[np.minimum(r+1,n-1)]
        before2[ids]=v[np.maximum(r-2,0)]; after2[ids]=v[np.minimum(r+2,n-1)]
    # Offline, RGB-only local derivatives. No timestamps, row numbers or action inputs.
    return np.concatenate([a,.5*(after-before),.25*(after2-2*a+before2)],axis=1)

def squared_distance(a,b):
    return np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*(a@b.T),0)

def ridge_fit(x,y,relative):
    xm=x.mean(0); ym=y.mean(0); a=x-xm
    gram=(a.T@a).astype(np.float64); rhs=(a.T@(y-ym)).astype(np.float64)
    penalty=max(float(np.trace(gram)/len(gram))*relative,1e-9)
    coef=np.linalg.solve(gram+penalty*np.eye(len(gram)),rhs).astype(np.float32)
    return {'xmean':xm,'ymean':ym,'coef':coef,'penalty':np.array(penalty)}

def ridge_predict(x,m):return (x-m['xmean'])@m['coef']+m['ymean']

def neighbors(x,z,maxk=128):
    ids=[]; ds=[]
    for start in range(0,len(z),96):
        d=squared_distance(z[start:start+96],x)
        ix=np.argpartition(d,maxk-1,axis=1)[:,:maxk]
        q=np.take_along_axis(d,ix,axis=1); order=np.argsort(q,axis=1)
        ids.append(np.take_along_axis(ix,order,axis=1));ds.append(np.take_along_axis(q,order,axis=1))
    return np.concatenate(ids),np.concatenate(ds)

def positive_local(y,ids,ds,k):
    w=1/np.maximum(ds[:,:k].astype(np.float64),1e-7);w/=w.sum(1,keepdims=True)
    return np.einsum('bk,bkj->bj',w,y[ids[:,:k]].astype(np.float64)).astype(np.float32)

def local_columns(dim,mode):
    # Include both cameras and fingers rather than taking only the first view's PCs.
    if dim==24:cols=np.r_[np.arange(12),128+np.arange(8),224+np.arange(4)]
    elif dim==48:cols=np.r_[np.arange(24),128+np.arange(16),224+np.arange(8)]
    else:raise ValueError(dim)
    if mode=='context2':cols=np.r_[cols,256+cols[:16]]
    return cols

def affine_local(x,z,y,ids,ds,k,dim,relative,mode):
    cols=local_columns(dim,mode); out=[]
    for start in range(0,len(z),40):
        ix=ids[start:start+40,:k];d=ds[start:start+40,:k]
        w=1/np.maximum(d.astype(np.float64),1e-7);w/=w.sum(1,keepdims=True)
        delta=x[ix][:,:,cols].astype(np.float64)-z[start:start+len(ix),cols][:,None,:]
        target=y[ix].astype(np.float64)
        mx=np.einsum('bk,bkd->bd',w,delta);my=np.einsum('bk,bkj->bj',w,target)
        dx=delta-mx[:,None,:];dy=target-my[:,None,:]
        gram=dx.transpose(0,2,1)@(dx*w[:,:,None]);rhs=dx.transpose(0,2,1)@(dy*w[:,:,None])
        penalty=np.maximum(np.trace(gram,axis1=1,axis2=2)/len(cols)*relative,1e-8)
        coef=np.linalg.solve(gram+penalty[:,None,None]*np.eye(len(cols))[None,:,:],rhs)
        out.append(my-np.einsum('bd,bdj->bj',mx,coef))
    return np.concatenate(out).astype(np.float32)

def kernel_setup(x,z,y,base,landmarks=1024):
    ix=np.linspace(0,len(x)-1,min(landmarks,len(x)),dtype=int);centers=x[ix]
    dt=squared_distance(x,centers);dv=squared_distance(z,centers);dm=squared_distance(centers,centers)
    positive=dm[dm>1e-8]
    width=np.sqrt(np.median(positive))
    residual=y-ridge_predict(x,base)
    return centers,dt,dv,dm,float(width),residual

def kernel_fit_from_dist(centers,dt,dv,dm,width,residual,factor,alphas):
    sigma=width*factor
    k=np.exp(-dt/(2*sigma*sigma));v=np.exp(-dv/(2*sigma*sigma));m=np.exp(-dm/(2*sigma*sigma))
    vals,u=eigh(m.astype(np.float64),check_finite=False)
    keep=vals>max(float(vals[-1])*1e-7,1e-7)
    transform=(u[:,keep]/np.sqrt(vals[keep])[None,:]).astype(np.float32)
    phi=k@transform;vp=v@transform
    gram=(phi.T@phi).astype(np.float64);rhs=(phi.T@residual).astype(np.float64)
    eig,q=eigh(gram,check_finite=False);prod=q.T@rhs
    ans=[]
    for alpha in alphas:
        coef=q@(prod/(np.maximum(eig,0)+alpha)[:,None])
        mc=(transform@coef).astype(np.float32)
        ans.append((alpha,(vp@coef).astype(np.float32),{'centers':centers,'kernel_coef':mc,'sigma':np.array(sigma)}))
    return ans

def fit_recipe(x,y,recipe):
    kind=recipe['kind']
    if kind=='ridge':return ridge_fit(x,y,recipe['relative'])
    if kind in ['local','affine']:return {}
    assert kind=='kernel'
    base=ridge_fit(x,y,recipe['base_relative'])
    centers,dt,dv,dm,width,resid=kernel_setup(x,x[:1],y,base)
    model=kernel_fit_from_dist(centers,dt,dv,dm,width,resid,recipe['factor'],[recipe['alpha']])[0][2]
    model.update({'base_'+k:v for k,v in base.items()})
    return model

def predict_recipe(x,z,y,recipe,model,neighbor_cache=None):
    kind=recipe['kind']
    if kind=='ridge':return ridge_predict(z,model)
    if kind=='kernel':
        base={k[5:]:v for k,v in model.items() if k.startswith('base_')}
        out=[]
        for start in range(0,len(z),256):
            q=z[start:start+256];d=squared_distance(q,model['centers'])
            out.append(ridge_predict(q,base)+np.exp(-d/(2*float(model['sigma'])**2))@model['kernel_coef'])
        return np.concatenate(out)
    ids,ds=neighbor_cache if neighbor_cache is not None else neighbors(x,z)
    if kind=='local':return positive_local(y,ids,ds,recipe['k'])
    return affine_local(x,z,y,ids,ds,recipe['k'],recipe['dim'],recipe['relative'],recipe['mode'])
