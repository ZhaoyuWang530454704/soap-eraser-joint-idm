"""TRAIN-only supervised image projections; traditional PLS, no neural network."""
import numpy as np

def fit_pls(x,y,components=48):
    xm=x.mean(0);scale=np.maximum(x.std(0),.025)
    # Equalize the three camera/ROI feature blocks without discarding fine detail.
    scale[:5376]*=np.sqrt(5376);scale[5376:8256]*=np.sqrt(2880);scale[8256:]*=np.sqrt(768)
    a=((x-xm)/scale).astype(np.float32)
    ym=y.mean(0);ys=np.maximum(y.std(0),.001);response=(y-ym)/ys
    cross=(a.T@response).astype(np.float64)
    weights=[];loadings=[];scores=[]
    for j in range(components):
        u,_,_=np.linalg.svd(cross,full_matrices=False);w=u[:,0].astype(np.float32)
        t=a@w
        if scores:
            oldt=np.column_stack(scores);oldp=np.column_stack(loadings)
            t-=oldt@(oldp.T@w)
        tt=float(t.astype(float)@t)
        if tt<1e-9:break
        p=(a.T@t)/tt
        if scores:p-=oldp@(oldt.T@t)/tt
        q=(response.T@t)/tt
        cross-=np.outer(p.astype(float),q.astype(float))*tt
        weights.append(w);loadings.append(p.astype(np.float32));scores.append(t.astype(np.float32))
    w=np.column_stack(weights);p=np.column_stack(loadings)
    projection=w.astype(float)@np.linalg.pinv(p.astype(float).T@w)
    z=a@projection.astype(np.float32)
    scaling=np.maximum(z.std(0),1e-5)**.5
    return {'mean':xm,'scale':scale,'projection':projection.astype(np.float32),'score_scale':scaling},(z/scaling).astype(np.float32)

def pls_embedding(x,model):return (((x-model['mean'])/model['scale'])@model['projection']/model['score_scale']).astype(np.float32)

def pls_input(z,groups,mode,dim):
    a=z[:,:dim]
    if mode=='spatial':return a
    before=a.copy();after=a.copy();before2=a.copy();after2=a.copy()
    for sid in np.unique(groups):
        ix=np.flatnonzero(groups==sid);v=a[ix];n=len(v);r=np.arange(n)
        before[ix]=v[np.maximum(r-1,0)];after[ix]=v[np.minimum(r+1,n-1)]
        before2[ix]=v[np.maximum(r-2,0)];after2[ix]=v[np.minimum(r+2,n-1)]
    return np.concatenate([a,.5*(after-before),.25*(after2-2*a+before2)],axis=1)
