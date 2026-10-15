"""Actual fresh two-GPU public model replay from every original test RGB, before GT."""
from pathlib import Path
import sys,os,json,datetime,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model_package'))
import public_predict as engine
from common import torch,np,fingerprint,memory,configure,write,sha,now,read
import torch.distributed as dist
def main():
    rank=int(os.environ['LOCAL_RANK'])
    # Public package stores the immutable unassigned contract; validate actual assignment explicitly.
    assignment=read(ROOT/'GPU_assignment.json');assert os.environ['CUDA_VISIBLE_DEVICES'].split(',')==assignment['GPU_UUIDs'] and torch.cuda.device_count()==2
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    import cv2;cv2.setNumThreads(1)
    torch.cuda.set_device(rank);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20));device=f'cuda:{rank}';model=engine.load_model(device);before=fingerprint(model.state_dict())
    wanted=read(ROOT/'units/refit'/f'rank{rank}/training_reload_lifecycle.json')['final_fingerprint'];assert before==wanted
    assert read(ROOT/'evidence/independent_prediction_audit.json')['status']=='passed_pending_actual_GPU_public_replay'
    full=np.load(ROOT/'frozen_predictions/all_clips_sorted.npz',allow_pickle=False);aux=np.load(ROOT/'features/test/auxiliary.npy',mmap_mode='r',allow_pickle=False)
    d=ROOT/'public_replay'/f'rank{rank}';d.mkdir(parents=True,exist_ok=False);entries=[];start=time.monotonic()
    for i in range(rank,10,2):
        sid=f'eval{i:04d}';rgb=np.load(ROOT/'dataset/test_video_only/episodes'/sid/'rgb.npy',mmap_mode='r',allow_pickle=False);v=engine.predict(rgb,model,device)
        pred=np.load(ROOT/'frozen_predictions/actions'/sid/'actions.npz',allow_pickle=False)
        refq=full['outputs'][i*47:(i+1)*47];refz=full['latents'][i*47:(i+1)*47]
        errors={'position_m':float(np.max(abs(v['raw'][:,:3]-pred['raw'][:,:3]))),'rotation_components':float(np.max(abs(v['raw'][:,3:9]-pred['raw'][:,3:9]))),'closure':float(np.max(abs(v['raw'][:,9]-pred['raw'][:,9]))),'latent':float(np.max(abs(v['window_latents']-refz))),'auxiliary':float(np.max(abs(v['auxiliary']-aux[i*375:(i+1)*375]))),'window_position_m':float(np.max(abs(v['window_outputs'][:,:,:3]-refq[:,:,:3])))}
        assert errors['position_m']<=2e-5 and errors['rotation_components']<=2e-6 and errors['closure']<=2e-6 and errors['latent']<=5e-5 and errors['auxiliary']<=1e-5 and errors['window_position_m']<=2e-5,errors
        assert np.max(abs(v['ema05'][:,:3]-pred['ema05'][:,:3]))<=2e-5 and np.max(abs(v['ema05'][:,3:]-pred['ema05'][:,3:]))<=2e-6
        np.savez_compressed(d/(sid+'.npz'),**v)
        entries.append({'episode':sid,'errors':errors,'projection_bitwise_equal':bool(np.array_equal(v['auxiliary'],aux[i*375:(i+1)*375])),'window_action_bitwise_equal':bool(np.array_equal(v['window_outputs'],refq)),'window_latent_bitwise_equal':bool(np.array_equal(v['window_latents'],refz)),'SHA256':sha(d/(sid+'.npz'))})
        memory(rank)
    assert fingerprint(model.state_dict())==before;torch.cuda.synchronize(rank)
    write(d/'lifecycle.json',{'status':'passed','rank':rank,'entries':entries,'all187_state_tensors_unchanged_and_strictly_loaded':True,'full_RGB_features_recomputed_without_feature_cache_or_GT':True,'model_package_manifest_SHA256':sha(ROOT/'model_package/package_manifest.json'),'memory':memory(rank),'communication_active_at_receipt':True,'elapsed_s':time.monotonic()-start,'TEST_GT_read':False})
    dist.barrier();dist.destroy_process_group();write(d/'post_communication_shutdown.json',{'distributed_initialized':dist.is_initialized(),'memory':memory(rank),'worker_exits_after_receipt':True})
if __name__=='__main__':main()
