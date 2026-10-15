"""Separate SciPy quaternion error computation, all rows and original GT bytes."""
from pathlib import Path
import sys,json,hashlib,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from common import ROOT,SPEC,OUT,read,write,sha,np,configure
from scipy.spatial.transform import Rotation
def rotation(q):
    first=q[:,3:6].astype(np.float64).copy();first/=np.sqrt(np.einsum('ij,ij->i',first,first))[:,None]
    second=q[:,6:9].astype(np.float64).copy();second-=np.einsum('ij,ij->i',first,second)[:,None]*first;second/=np.sqrt(np.einsum('ij,ij->i',second,second))[:,None]
    return Rotation.from_matrix(np.stack([first,second,np.cross(first,second)],axis=-1))
def independent(pred,truth):
    d=(pred[:,:3].astype(np.float64)-truth[:,:3])*1000;distance=np.linalg.norm(d,axis=-1)
    angles=(rotation(truth).inv()*rotation(pred)).magnitude()*180/np.pi
    closure=np.abs(pred[:,9].astype(np.float64)-truth[:,9]);correct=np.equal(pred[:,9]>=.5,truth[:,9]>=.5);pose=np.logical_and(distance<=10,angles<=10)
    return {'frames':len(pred),'xyz_mae_mm':np.mean(abs(d),axis=0).tolist(),'position_mean_mm':float(np.mean(distance)),'position_median_mm':float(np.percentile(distance,50)),'position_p95_mm':float(np.percentile(distance,95)),'position_max_mm':float(np.max(distance)),'position_within5mm_percent':float(np.count_nonzero(distance<=5)/len(pred)*100),'position_within10mm_percent':float(np.count_nonzero(distance<=10)/len(pred)*100),'position_within10mm_frames':int(np.count_nonzero(distance<=10)),'rotation_mean_deg':float(np.mean(angles)),'rotation_p95_deg':float(np.percentile(angles,95)),'closure_mae':float(np.mean(closure)),'closure_class_error_percent':float(np.count_nonzero(~correct)/len(pred)*100),'closure_class_agreement_percent':float(np.count_nonzero(correct)/len(pred)*100),'position10mm_rotation10deg_accuracy_percent':float(np.count_nonzero(pose)/len(pred)*100),'position10mm_rotation10deg_closureclass_accuracy_percent':float(np.count_nonzero(pose&correct)/len(pred)*100)}
def compare(actual,wanted):
    assert set(actual)==set(wanted)
    for key,v in wanted.items():
        tolerance=2e-5 if key.startswith('rotation_') else 1e-9
        assert np.allclose(actual[key],v,rtol=0,atol=tolerance),(key,actual[key],v)
def main():
    configure();report=read(OUT/'metrics.json');assert report['prediction_release_gate_SHA256']==sha(ROOT/'evidence/prediction_release_gate.json')
    manifest=read(ROOT/'analysis/gt_reference_manifest.json');truths=[];predictions={'raw':[],'ema05':[]};checks=[]
    for entry in report['episodes']:
        sid=entry['episode'];p=ROOT/'scoring_inputs'/f'{sid}.npy';assert sha(p)==manifest['files'][sid]['SHA256']==entry['GT_SHA256']
        gt=np.load(p,allow_pickle=False);assert gt.shape==(375,10);truths.append(gt);a=np.load(ROOT/'frozen_predictions/actions'/sid/'actions.npz',allow_pickle=False)
        assert np.array_equal(a['source_observation_row'],np.arange(375)) and np.array_equal(a['source_target_row'],np.arange(1,376))
        for kind in ('raw','ema05'):
            q=a[kind];predictions[kind].append(q);compare(entry['metrics'][kind],independent(q,gt))
            detail=np.loadtxt(OUT/'per_frame'/f'{sid}_{kind}.csv',delimiter=',',skiprows=1)
            assert detail.shape==(375,12) and np.array_equal(detail[:,0],np.arange(375)) and np.array_equal(detail[:,1],np.arange(1,376)) and np.array_equal(detail[:,2],a['action_timestamp'].astype(float))
            xyz=np.abs(q[:,:3].astype(float)-gt[:,:3])*1000;assert np.allclose(detail[:,3:6],xyz,atol=1e-10,rtol=0) and np.allclose(detail[:,6],np.linalg.norm(xyz,axis=1),atol=1e-10,rtol=0)
        checks.append({'episode':sid,'rows':375,'GT_original_SHA256':sha(p),'raw_EMA_errors_independently_recomputed':True})
    assert [v['episode'] for v in checks]==SPEC['test_ids'] and len(np.concatenate(truths))==3750
    for kind in predictions:compare(report['aggregate_frame_weighted'][kind],independent(np.concatenate(predictions[kind]),np.concatenate(truths)))
    frozen=read(ROOT/'frozen_predictions/completion.json')
    assert sha(OUT/'actions_test10_GTfree.zip')==frozen['action_ZIP_SHA256'] and sha(OUT/'complete_joint_IDM_model.zip')==frozen['model_ZIP_SHA256']
    with zipfile.ZipFile(OUT/'actions_test10_GTfree.zip') as z:
        expected={'README.json'}|{sid+'/'+name for sid in SPEC['test_ids'] for name in ('actions.npz','raw.csv','ema05.csv')};assert len(z.namelist())==31 and set(z.namelist())==expected and not z.testzip()
        assert json.loads(z.read('README.json'))['GT_in_package'] is False
        for name in expected:assert hashlib.sha256(z.read(name)).hexdigest()==sha(ROOT/'frozen_predictions/actions'/name)
    for p in report['all21_plots']:assert sha(OUT/p['path'])==p['SHA256']
    for rel,h in read(OUT/'evaluation_files_SHA256.json')['files'].items():assert sha(OUT/rel)==h
    write(OUT/'independent_metrics_verification.json',{'status':'passed','episodes':checks,'all3750_rows_frame_weighted_recomputed':True,'independent_rotation_method':'SciPy quaternion composition magnitude, separate from trace-arccos scoring','original_GT_preserved':True,'GT_free_action_ZIP_31_exact_members':True,'metrics_SHA256':sha(OUT/'metrics.json'),'action_ZIP_SHA256':sha(OUT/'actions_test10_GTfree.zip'),'model_ZIP_SHA256':sha(OUT/'complete_joint_IDM_model.zip'),'actual_visual_review_still_required':True})
    print(json.dumps({'status':'independent_metrics_and_GTfree_ZIP_passed','rows':3750,'plots_pending_actual_open':21}),flush=True)
if __name__=='__main__':main()
