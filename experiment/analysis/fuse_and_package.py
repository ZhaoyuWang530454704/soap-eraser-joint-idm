"""Server CPU: freeze video-only predictions and public complete model, before GT."""
from pathlib import Path
import sys,shutil,csv,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from common import *
def main():
    configure();verify_sources();assert read(ROOT/'processes/test_video_inference/terminal.json')['all_owned_children_exited']
    data=ClipData('refit','test');pieces=[]
    for rank in range(2):
        d=ROOT/'predictions'/f'rank{rank}';receipt=read(d/'inference_lifecycle.json');assert receipt['status']=='passed' and sha(d/'clips.npz')==receipt['clips_SHA256']
        pieces.append(np.load(d/'clips.npz',allow_pickle=False))
    ids=np.concatenate([p['clip_ids'] for p in pieces]);qs=np.concatenate([p['outputs'] for p in pieces]);zs=np.concatenate([p['latents'] for p in pieces]);order=np.argsort(ids)
    ids,qs,zs=ids[order],qs[order],zs[order];raw,z=merge_clips(data,ids,qs,zs)
    d=ROOT/'frozen_predictions';d.mkdir();action=d/'actions';action.mkdir();items=[]
    fields=['tcp_x_m','tcp_y_m','tcp_z_m','r11','r21','r31','r12','r22','r32','commanded_closure']
    np.savez_compressed(d/'all_clips_sorted.npz',clip_ids=ids,outputs=qs,latents=zs)
    for sid in SPEC['test_ids']:
        ix=np.flatnonzero(data.groups[data.row_indices]==sid);rr=raw[ix];ee=ema(rr);t=data.times[data.row_indices][ix];folder=action/sid;folder.mkdir()
        assert rr.shape==(375,10) and np.array_equal(data.rows[data.row_indices][ix],np.arange(375))
        np.savez_compressed(folder/'actions.npz',raw=rr,ema05=ee,source_observation_row=np.arange(375),source_target_row=np.arange(1,376),action_timestamp=t,observation_timestamp=np.load(ROOT/'dataset/test_video_only/episodes'/sid/'timestamps.npy',allow_pickle=False)[:-1])
        for name,a in [('raw',rr),('ema05',ee)]:
            with (folder/(name+'.csv')).open('x',newline='') as f:
                w=csv.writer(f);w.writerow(['source_observation_row','source_target_row','action_timestamp_s']+fields);w.writerows(np.column_stack((np.arange(375),np.arange(1,376),t,a)))
        items.append({'episode':sid,'frames':375,'actions_NPZ_SHA256':sha(folder/'actions.npz'),'raw_CSV_SHA256':sha(folder/'raw.csv'),'ema05_CSV_SHA256':sha(folder/'ema05.csv')})
    write(action/'README.json',{'action_spec':SPEC['action_spec'],'coordinate_frame':'robot_base','units':'XYZ meters, rotation first two columns, commanded closure0open1closed','columns':fields,'rows':3750,'GT_in_package':False,'raw_primary_EMA_secondary':True,'not_physical_execution':True,'canonical_alignment':'next endpoint already prepared; prediction row0 maps observation row0 and original planned endpoint row1, no re-shift'})
    with zipfile.ZipFile(d/'actions_test10_GTfree.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(action.rglob('*')):
            if p.is_file():archive.write(p,str(p.relative_to(action)))
    pack=ROOT/'model_package';pack.mkdir();(pack/'source').mkdir();(pack/'reference_source').mkdir();(pack/'projection').mkdir()
    cp=torch.load(ROOT/'units/refit/model.pt',map_location='cpu',weights_only=True);torch.save({'state':cp['state'],'epoch':cp['epoch'],'contract':SPEC,'source_manifest_SHA256':sha(ROOT/'source_manifest.json')},pack/'model.pt')
    assert len(cp['state'])==187
    shutil.copyfile(ROOT/'source/common.py',pack/'source/common.py');shutil.copyfile(ROOT/'contract.json',pack/'contract.json')
    for p in (ROOT/'reference_source').glob('*.py'):shutil.copyfile(p,pack/'reference_source'/p.name)
    for p in (ROOT/'features/refit/projection').iterdir():shutil.copyfile(p,pack/'projection'/p.name)
    shutil.copyfile(ROOT/'analysis/public_predict.py',pack/'public_predict.py')
    files={str(p.relative_to(pack)):sha(p) for p in pack.rglob('*') if p.is_file()};write(pack/'package_manifest.json',{'files':files,'state_tensors':187,'model_SHA256':sha(pack/'model.pt'),'full_RGB_to_action':True,'no_training_or_TEST_GT_arrays':True,'architecture':'Joint14/15 shared RGB temporal model refitted on50 soap simulations'})
    with zipfile.ZipFile(d/'complete_joint_IDM_model.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(pack.rglob('*')):
            if p.is_file():archive.write(p,str(p.relative_to(pack)))
    write(d/'completion.json',{'status':'predictions_and_full_model_SHA_frozen_pending_independent_audits_no_GT_read','checkpoint_SHA256':sha(ROOT/'units/refit/model.pt'),'public_model_SHA256':sha(pack/'model.pt'),'package_manifest_SHA256':sha(pack/'package_manifest.json'),'model_ZIP_SHA256':sha(d/'complete_joint_IDM_model.zip'),'action_ZIP_SHA256':sha(d/'actions_test10_GTfree.zip'),'all_sorted_clips_SHA256':sha(d/'all_clips_sorted.npz'),'epoch_selection_SHA256':sha(ROOT/'epoch_selection.json'),'actions':items,'full3750_actions':True,'TEST_GT_read':False})
if __name__=='__main__':main()
