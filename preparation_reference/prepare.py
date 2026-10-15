"""Server CPU only: verify frozen inputs and fit task TRAIN-only visual projections."""
import shutil,time
from common import *
def main():
    configure();source=verify_sources();start=time.monotonic();f=ROOT/'features';f.mkdir(exist_ok=False)
    expected=read(ROOT/'inputs/expected_inputs.json');proof=[]
    for rel,value in expected.items():
        p=ROOT/'dataset'/rel;assert p.is_file() and p.stat().st_size==value['bytes'] and sha(p)==value['sha256'],rel
        proof.append({'path':rel,'bytes':p.stat().st_size,'SHA256':sha(p)})
    assert len(proof)==120
    pre=Path(SPEC['pretrained_path']);assert sha(pre)==SPEC['pretrained_SHA256']
    (ROOT/'pretrained').mkdir();shutil.copyfile(pre,ROOT/'pretrained'/pre.name)
    xs=[];ps=[];ys=[];groups=[];rows=[];times=[]
    for sid in SPEC['train_ids']:
        d=ROOT/'dataset/data/episodes'/sid;rgb=np.load(d/'rgb.npy',mmap_mode='r',allow_pickle=False);y=np.load(d/'action_raw.npy',allow_pickle=False)
        assert rgb.shape==(376,256,640,3) and rgb.dtype==np.uint8 and y.shape==(375,10) and y.dtype==np.float32 and np.isfinite(y).all()
        rr=rot_numpy(y[:,3:9]);assert np.allclose(rr.transpose(0,2,1)@rr,np.eye(3),atol=1e-6) and np.allclose(np.linalg.det(rr),1,atol=1e-6)
        xs.append(np.stack([fine_features(rgb[i]) for i in range(375)]));ps.append(np.stack([keypoint_features(rgb[i]) for i in range(375)]));ys.append(y)
        groups.extend([sid]*375);rows.extend(range(375));times.extend((np.arange(375)/15).tolist())
        update(ROOT/'progress.json',{'UTC_server':now(),'stage':'TRAIN_RGB_feature_preparation','prepared_episodes':len(ys),'total_episodes':50,'TEST_GT_read':False})
        print('TRAIN RGB-only features',sid,flush=True)
    x=np.concatenate(xs).astype(np.float32);points=np.concatenate(ps).astype(np.float32);y=np.concatenate(ys).astype(np.float32);groups=np.asarray(groups);rows=np.asarray(rows,np.int32);times=np.asarray(times)
    assert x.shape==(18750,9024) and points.shape==(18750,96) and y.shape==(18750,10) and set(groups).isdisjoint(SPEC['test_ids'])
    np.savez(f/'TRAIN_meta.npz',groups=groups,rows=rows,times=times,y=y)
    np.save(f/'TRAIN_fine.npy',x,allow_pickle=False);np.save(f/'TRAIN_color.npy',points,allow_pickle=False)
    for phase in ('selection','refit'):
        d=f/phase;d.mkdir();pr=d/'projection';pr.mkdir();fit=np.isin(groups,SPEC['fit_ids']) if phase=='selection' else np.ones(len(y),bool)
        basis,_=fit_embedding(x[fit]);np.savez(pr/'pca10.npz',**basis);z=embedding(x,basis);zs=[]
        for name,cols in [('xyz',slice(0,3)),('all10',slice(0,10))]:
            basis,_=fit_pls(x[fit],y[fit,cols],48);assert basis['projection'].shape[1]==48
            np.savez(pr/(name+'10.npz'),**basis);zs.append(pls_embedding(x,basis))
        a=pls_input(np.concatenate((z,zs[0],zs[1]),1),groups,'context2',352);a=np.concatenate((a,keypoint_input(points,groups)),1).astype(np.float32)
        assert a.shape==(18750,1344) and np.isfinite(a).all();np.save(d/'auxiliary.npy',a,allow_pickle=False)
        np.savez(pr/'normalization10.npz',aux_mean=a[fit].mean(0),aux_scale=np.maximum(a[fit].std(0),.01),xyz_mean=y[fit,:3].mean(0),rotation_mean=y[fit,3:9].mean(0),closure_mean=y[fit,9].mean())
        write(d/'receipt.json',{'phase':phase,'fit_rows':int(fit.sum()),'fit_ids':sorted(set(groups[fit])),'projection_files':{v.name:sha(v) for v in pr.iterdir()},'TRAIN_auxiliary_SHA256':sha(d/'auxiliary.npy'),'TEST_used':False,'no_whiteboard_projection_reuse':True})
    controls={}
    for phase,part,expected_rows,expected_clips in [('selection','train',15000,1880),('selection','validation',3750,470),('refit','train',18750,2350)]:
        ds=ClipData(phase,part);assert len(ds.row_indices)==expected_rows and len(ds)==expected_clips
        im,a,target,_,weight=ds[0];assert im.shape==(17,2,3,256,320) and a.shape==(17,1344) and target.shape==(17,10) and weight.shape==(17,)
        ids,mask=epoch_plan(len(ds),1);assert set(ds.sample_ids[ids[mask>0]])==set(ds.sample_ids)
        zero=ds[(0,0.)][-1];assert torch.count_nonzero(zero)==0
        controls[phase+'_'+part]={'rows':expected_rows,'clips':expected_clips,'inverse_coverage_total':float(ds.loss_weights.sum()),'padding_zero_weight':True}
    model=JointAction(pretrained=True);initialize(model,'selection');assert len(model.state_dict())==187
    controls['model_schema']={'state_tensors':187,'parameters':sum(p.numel() for p in model.parameters()),'actual_GPU_forward_not_run':True}
    write(ROOT/'evidence/CPU_preparation.json',{'status':'passed','source_manifest_SHA256':source,'frozen_inputs':proof,'TRAIN_rows':18750,'TRAIN_sessions':50,'selection_fitting_sessions':40,'internal_TRAIN_validation_sessions':10,'test_sessions':10,'controls':controls,'elapsed_s':time.monotonic()-start,'CPU_checks_are_not_GPU_preflight':True,'TEST_GT_read':False})
    update(ROOT/'progress.json',{'stage':'CPU_preparation_complete_pending_actual_GPU_inline_gates','UTC_server':now(),'elapsed_s':time.monotonic()-start,'TEST_GT_read':False})
if __name__=='__main__':main()
