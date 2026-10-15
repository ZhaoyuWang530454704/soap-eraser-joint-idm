"""One-shot server owner: paired refit, gated predictions, then independent simulation."""
from pathlib import Path
import json,os,sys,time,traceback,shutil
import orchestrate as control
from transport_bytes import identity,now,write,update
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def verify():
 m=read(ROOT/'source_manifest.json')
 assert control.sha(ROOT/'contract.json')==m['contract_SHA256']
 for p,h in m['files'].items():assert control.sha(ROOT/p)==h,p
def main():
 verify();write(ROOT/'owner_process.json',identity(os.getpid()));handles=[];start=time.monotonic();error=None
 try:
  control.run('rotation_loss_CPU_controls','rotation_loss_controls.py')
  handles=control.assign_idle_GPUs()
  control.run('refit_training','train.py',True,['refit'])
  control.run('test_video_inference','../analysis/infer_test.py',True)
  control.run('prediction_fusion_and_packaging','../analysis/fuse_and_package.py')
  control.run('independent_prediction_audit','../analysis/audit_predictions.py',args=['initial'])
  control.run('actual_GPU_public_replay','../analysis/replay_public.py',True)
  control.run('final_prediction_release_gate','../analysis/audit_predictions.py',args=['final'])
  for f in handles:f.close()
  handles=[]
  gate=read(ROOT/'evidence/prediction_release_gate.json');assert gate['status']=='passed'
  folder=ROOT/'scoring_inputs';folder.mkdir(exist_ok=False)
  manifest=read(ROOT/'analysis/gt_reference_manifest.json');base=Path(control.SPEC['baseline_root'])
  for sid,e in manifest['files'].items():
   original=base/'scoring_inputs'/f'{sid}.npy';assert control.sha(original)==e['SHA256']
   shutil.copyfile(original,folder/original.name);assert control.sha(folder/original.name)==e['SHA256']
  write(folder/'staging_complete.json',{'prediction_release_gate_SHA256':control.sha(ROOT/'evidence/prediction_release_gate.json'),'manifest_SHA256':control.sha(ROOT/'analysis/gt_reference_manifest.json'),'existing_server_GT_bytes_only_reused_after_new_prediction_gate':True,'files':10})
  control.run('TEST_GT_scoring_and_report','../analysis/evaluate.py')
  control.run('independent_TEST_metrics_and_ZIP_verification','../analysis/verify_metrics.py')
  update(ROOT/'progress.json',{'UTC_server':now(),'stage':'offline_metrics_verified_waiting_postprocess_manifest'})
  post=ROOT/'postprocess/source_manifest.json'
  while not post.exists():time.sleep(5)
  for p,h in read(post)['files'].items():assert control.sha(ROOT/'postprocess'/p)==h,p
  control.run('physics_and_comparison_report','../postprocess/owner.py')
  verify()
 except BaseException as e:error=type(e).__name__+': '+str(e);(ROOT/'owner_traceback.log').write_text(traceback.format_exc())
 finally:
  for f in handles:f.close()
 write(ROOT/'owner_terminal.json',{'UTC_server':now(),'status':'computed_pending_actual_visual_resource_delivery' if error is None else 'failed_preserved_no_retry','error':error,'elapsed_s':time.monotonic()-start,'owner':identity(os.getpid()),'automatic_retry':False})
 sys.exit(0 if error is None else 1)
if __name__=='__main__':main()
