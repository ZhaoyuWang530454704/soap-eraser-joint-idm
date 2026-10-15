"""Independent server CPU verification before any TEST ground-truth numerical load."""
from pathlib import Path
import sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from common import ROOT,SPEC,read,write,sha,configure,JointAction,fingerprint,ClipData,np,torch
from scipy.linalg import polar
from scipy.spatial.transform import Rotation,Slerp
def basis(q):
    x=q[:,:3].astype(np.float64);x=x/np.linalg.norm(x,axis=1)[:,None]
    y=q[:,3:6].astype(np.float64);y=y-np.einsum('ij,ij->i',x,y)[:,None]*x;y=y/np.linalg.norm(y,axis=1)[:,None]
    return np.stack([x,y,np.cross(x,y)],axis=2)
def fused(data,outputs):
    count=len(data.row_indices);scalar=np.zeros((count,4),np.float64);rot=np.zeros((count,3,3),np.float64);weight=np.zeros(count,np.float64)
    for i,q in enumerate(outputs):
        ok=data.clip_valid[i].astype(bool);rr=data.clip_rows[i][ok];w=np.sin(np.pi*(np.flatnonzero(ok)+.5)/17)**2
        for row,v,r,ww in zip(rr,q[ok][:,[0,1,2,9]],basis(q[ok,3:9]),w):scalar[row]+=v*ww;rot[row]+=r*ww;weight[row]+=ww
    assert np.all(weight>0);scalar/=weight[:,None];rot/=weight[:,None,None]
    matrices=np.stack([polar(v)[0] for v in rot]);assert np.min(np.linalg.det(matrices))>.99999
    raw=np.column_stack((scalar[:,:3],matrices[:,:,0],matrices[:,:,1],scalar[:,3])).astype(np.float32)
    return raw
def filtered(raw):
    q=raw.astype(float).copy();r=basis(raw[:,3:9]);smooth=r.copy()
    for i in range(1,len(q)):
        q[i,[0,1,2,9]]=(raw[i,[0,1,2,9]].astype(float)+q[i-1,[0,1,2,9]])/2
        smooth[i]=Slerp([0.,1.],Rotation.from_matrix(np.stack([smooth[i-1],r[i]])))([.5]).as_matrix()[0]
    q[:,3:6]=smooth[:,:,0];q[:,6:9]=smooth[:,:,1];return q.astype(np.float32)
def training_audit(phase):
    terminal=read(ROOT/'units'/phase/'training_terminal.json');assert terminal['status']=='complete'
    ds=ClipData(phase,'train');epochs=terminal['epochs'];records=[];inits=[];finals=[]
    for rank in range(2):
        d=ROOT/'units'/phase/f'rank{rank}';v=read(d/'training_reload_lifecycle.json');assert v['status']=='passed' and v['strict187_loaded'] and v['TRAIN2clips_fresh_reload_output_and_latent_bitwise'] and v['communication_active_at_receipt']
        assert v['memory']['headroom_GiB']>=10 and v['memory']['actual_free_GiB']>=10
        assert v['epochs']==epochs and v['optimizer_steps']==epochs*((len(ds)+(-len(ds))%8)//8)
        assert not read(d/'post_communication_shutdown.json')['distributed_initialized']
        ledger=np.load(d/'actual_sampling.npz',allow_pickle=False);assert sha(d/'actual_sampling.npz')==v['sampling_SHA256']
        for e in range(1,epochs+1):
            original=np.random.RandomState(SPEC['seed']+e).permutation(len(ds));pad=(-len(ds))%8
            order=np.r_[original,original[:pad]] if pad else original;mask=np.r_[np.ones(len(ds)),np.zeros(pad)]
            order=order.reshape(-1,8)[:,rank*4:(rank+1)*4].reshape(-1);mask=mask.reshape(-1,8)[:,rank*4:(rank+1)*4].reshape(-1)
            assert np.array_equal(ledger['clip_indices'][e-1],order) and np.array_equal(ledger['real_mask'][e-1],mask)
            assert np.array_equal(ledger['sample_ids'][e-1],ds.sample_ids[order])
        assert set(v['actual_sampled_episode_ids'])==set(ds.ids) and set(ds.ids).isdisjoint(SPEC['test_ids'])
        maps=np.load(d/'original_row_maps.npz',allow_pickle=False)
        for name in ('clip_rows','clip_valid','loss_weights','row_indices','coverage'):assert np.array_equal(maps[name],getattr(ds,name)),name
        for step in (1,2,3):
            g=read(d/f'gradient_step{step}.json');o=read(d/f'optimizer_step{step}.json');assert g['status']==o['status']=='passed' and g['all2_rank_gradient_SHA_equal'] and o['all2_rank_parameter_SHA_equal']
        control=read(d/'actual_GPU_shared_gradient_control.json');assert control['status']=='passed'
        assert set(control['three_losses_reach_shared_visual_encoder'])=={'XYZ','rotation','closure','rotation_velocity'}
        for task,values in control['three_losses_reach_shared_visual_encoder'].items():assert all(np.isfinite(v) and v>0 for v in values.values())
        rows=ds.clip_rows;ok=ds.clip_valid.astype(bool);pairs=np.zeros_like(ok);pairs[:,1:]=ok[:,:-1]&ok[:,1:]&(rows[:,1:]-rows[:,:-1]==1)
        count=np.bincount(rows[pairs],minlength=len(ds.groups));weights=pairs/np.maximum(count[rows],1)
        assert np.array_equal(maps['rotation_pair_coverage'],count) and np.array_equal(maps['rotation_pair_weights'],weights)
        assert np.isclose(weights.sum(),18700,rtol=0,atol=1e-8)
        original=np.divide(weights,ds.loss_weights,out=np.zeros_like(weights),where=ds.loss_weights>0).astype(np.float32)
        assert np.array_equal(maps['rotation_pair_multiplier'],original)
        inits.append(read(d/'initial.json')['fingerprint']);finals.append(v['final_fingerprint']);records.append({'rank':rank,'epochs':epochs,'all_indices_and_sample_IDs_recomputed':True,'original_row_coverage_verified':True,'memory':v['memory']})
    assert inits[0]==inits[1] and finals[0]==finals[1]
    cp=ROOT/'units'/phase/(('epoch%03d.pt'%epochs) if phase=='selection' else 'model.pt');assert sha(cp)==terminal['checkpoint_SHA256']
    value=torch.load(cp,map_location='cpu',weights_only=True);model=JointAction();model.load_state_dict(value['state'],strict=True);assert len(model.state_dict())==187 and fingerprint(model.state_dict())==finals[0]
    process=read(ROOT/'processes'/(phase+'_training')/'terminal.json');assert process['exit_code']==0 and process['all_owned_children_exited']
    return {'phase':phase,'records':records,'strict187_CPU_load_all_names_shapes_dtypes_valueSHAs':True,'checkpoint_SHA256':sha(cp)}
def check_archive(archive,folder):
    expected={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
    import hashlib
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(expected) and set(z.namelist())==set(expected) and not z.testzip()
        for name,h in expected.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    return len(expected)
def main():
    configure();mode=sys.argv[1] if len(sys.argv)>1 else 'initial';assert mode in ('initial','final')
    if mode=='final':
        prior=read(ROOT/'evidence/independent_prediction_audit.json');assert prior['status']=='passed_pending_actual_GPU_public_replay'
        completion=read(ROOT/'frozen_predictions/completion.json');assert sha(ROOT/'frozen_predictions/completion.json')==prior['prediction_completion_SHA256']
        entries=[]
        for rank in range(2):
            d=ROOT/'public_replay'/f'rank{rank}';r=read(d/'lifecycle.json');assert r['status']=='passed' and r['all187_state_tensors_unchanged_and_strictly_loaded'] and r['memory']['headroom_GiB']>=10 and r['memory']['actual_free_GiB']>=10
            assert not read(d/'post_communication_shutdown.json')['distributed_initialized']
            for e in r['entries']:assert sha(d/(e['episode']+'.npz'))==e['SHA256']
            entries.extend(r['entries'])
        assert sorted(e['episode'] for e in entries)==SPEC['test_ids']
        p=read(ROOT/'processes/actual_GPU_public_replay/terminal.json');assert p['exit_code']==0 and p['all_owned_children_exited']
        for name in ('actions_test10_GTfree.zip','complete_joint_IDM_model.zip'):
            key='action_ZIP_SHA256' if name.startswith('actions') else 'model_ZIP_SHA256';assert sha(ROOT/'frozen_predictions'/name)==completion[key]
        write(ROOT/'evidence/prediction_release_gate.json',{'status':'passed','CPU_independent_audit_SHA256':sha(ROOT/'evidence/independent_prediction_audit.json'),'prediction_completion_SHA256':sha(ROOT/'frozen_predictions/completion.json'),'all3750_predictions_and187_full_model_frozen':True,'actual_GPU_full_RGB_public_replay_all10':entries,'checkpoint_SHA256':completion['checkpoint_SHA256'],'action_ZIP_SHA256':completion['action_ZIP_SHA256'],'model_ZIP_SHA256':completion['model_ZIP_SHA256'],'TEST_GT_values_read':False,'GT_numerical_scoring_may_start_after_this_receipt':True})
        return
    training=[training_audit('refit')]
    assert read(ROOT/'evidence/rotation_loss_CPU_controls.json')['passed']
    assert read(ROOT/'units/refit/rank0/initial.json')['fingerprint']==SPEC['expected_initial_fingerprint']
    assert read(ROOT/'units/refit/rank1/initial.json')['fingerprint']==SPEC['expected_initial_fingerprint']
    assert sha(__import__('pathlib').Path(SPEC['baseline_root'])/'epoch_selection.json')==SPEC['baseline_epoch_selection_SHA256']
    assert read(ROOT/'units/refit/training_terminal.json')['epochs']==read(ROOT/'epoch_selection.json')['selected_epoch']
    data=ClipData('refit','test');p=ROOT/'frozen_predictions';completion=read(p/'completion.json');assert completion['TEST_GT_read'] is False
    assert sha(p/'all_clips_sorted.npz')==completion['all_sorted_clips_SHA256']
    full=np.load(p/'all_clips_sorted.npz',allow_pickle=False);assert np.array_equal(full['clip_ids'],np.arange(470)) and full['outputs'].shape==(470,17,10) and full['latents'].shape==(470,17,256)
    original_shards=[];inference=[]
    for rank in range(2):
        d=ROOT/'predictions'/f'rank{rank}';r=read(d/'inference_lifecycle.json');assert r['status']=='passed' and r['strict187_state_name_shape_dtype_value_equal'] and r['fresh_reload_output_and_latent_bitwise'] and r['communication_active_at_receipt'] and r['action_inputs_all_zero']
        assert r['memory']['headroom_GiB']>=10 and r['memory']['actual_free_GiB']>=10 and not read(d/'post_communication_shutdown.json')['distributed_initialized']
        assert sha(d/'clips.npz')==r['clips_SHA256'] and r['checkpoint_SHA256']==completion['checkpoint_SHA256']
        s=np.load(d/'clips.npz',allow_pickle=False);assert np.array_equal(s['clip_ids'],np.arange(rank,470,2))
        assert np.array_equal(s['actual_batch_clip_ids'][s['actual_batch_valid']],s['clip_ids'])
        assert int((~s['actual_batch_valid']).sum())==1 and np.all(s['actual_batch_clip_ids'][~s['actual_batch_valid']]==s['clip_ids'][-1])
        assert s['outputs'].shape==(235,17,10) and s['latents'].shape==(235,17,256) and np.isfinite(s['outputs']).all() and np.isfinite(s['latents']).all()
        original_shards.append(s);inference.append({'rank':rank,'unique_clips':235,'registered_batch_padding':1,'strict187_and_memory_lifecycle_verified':True})
    order=np.argsort(np.concatenate([s['clip_ids'] for s in original_shards]))
    for key in ('outputs','latents'):assert np.array_equal(full[key],np.concatenate([s[key] for s in original_shards])[order]),key
    independently_fused=fused(data,full['outputs']);differences=[]
    for i,sid in enumerate(SPEC['test_ids']):
        v=np.load(p/'actions'/sid/'actions.npz',allow_pickle=False);raw=independently_fused[i*375:(i+1)*375];smoothed=filtered(raw)
        assert np.array_equal(v['source_observation_row'],np.arange(375)) and np.array_equal(v['source_target_row'],np.arange(1,376))
        original=np.load(ROOT/'dataset/test_video_only/episodes'/sid/'timestamps.npy',allow_pickle=False)
        assert np.array_equal(v['action_timestamp'],original[1:]) and np.array_equal(v['observation_timestamp'],original[:-1])
        for kind,a in [('raw',raw),('ema05',smoothed)]:
            err=np.max(abs(v[kind].astype(float)-a),axis=0);assert np.max(err[:3])<=2e-7 and np.max(err[3:])<=2e-6,(sid,kind,err)
            differences.append({'episode':sid,'kind':kind,'max_absolute_error_by_column':err.tolist()})
        assert sha(p/'actions'/sid/'actions.npz')==completion['actions'][i]['actions_NPZ_SHA256']
    assert not read(p/'actions/README.json')['GT_in_package']
    action_members=check_archive(p/'actions_test10_GTfree.zip',p/'actions');model_members=check_archive(p/'complete_joint_IDM_model.zip',ROOT/'model_package')
    value=torch.load(ROOT/'model_package/model.pt',map_location='cpu',weights_only=True);public=JointAction();public.load_state_dict(value['state'],strict=True)
    expected=read(ROOT/'units/refit/rank0/training_reload_lifecycle.json')['final_fingerprint'];assert fingerprint(public.state_dict())==expected
    write(ROOT/'evidence/independent_prediction_audit.json',{'status':'passed_pending_actual_GPU_public_replay','training':training,'inference':inference,'all470_unique_clips_and_two_registered_padding_slots_verified':True,'strict187_public_full_model_CPU_load':True,'all3750_actions_independently_fused_with_SciPy_polar_and_Slerp':True,'fusion_differences':differences,'action_ZIP_exact_SHA_members':action_members,'model_ZIP_exact_SHA_members':model_members,'prediction_completion_SHA256':sha(p/'completion.json'),'TEST_GT_values_read':False})
if __name__=='__main__':main()
