"""Fresh full187 tensor load; two GPU ranks predict test video only."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch.distributed as dist
from common import *
def main():
    rank=int(os.environ['LOCAL_RANK']);configure(rank);verify_sources();dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20));assert dist.get_world_size()==2
    d=ROOT/'predictions'/f'rank{rank}';d.mkdir(parents=True,exist_ok=False);data=ClipData('refit','test');assert np.count_nonzero(data.y)==0
    path=ROOT/'units/refit/model.pt';train=read(ROOT/'units/refit/training_terminal.json');assert sha(path)==train['checkpoint_SHA256']
    value=torch.load(path,map_location='cpu',weights_only=True);model=JointAction();model.load_state_dict(value['state'],strict=True);assert len(value['state'])==187
    wanted=read(ROOT/'units/refit'/f'rank{rank}/training_reload_lifecycle.json')['final_fingerprint'];assert fingerprint(model.state_dict())==wanted
    model.cuda(rank).eval();ids=np.arange(rank,len(data),2);outputs=[];latents=[];actual=[];valids=[];torch.cuda.synchronize(rank);tick=time.monotonic()
    for start in range(0,len(ids),2):
        ix=ids[start:start+2];n=len(ix);ix=np.r_[ix,np.repeat(ix[-1],2-n)];samples=[data[int(i)] for i in ix]
        im=torch.stack([s[0] for s in samples]).cuda(rank);aux=torch.stack([s[1] for s in samples]).cuda(rank)
        assert all(torch.count_nonzero(s[2])==0 for s in samples)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):q,z=model(im,aux)
        assert torch.isfinite(q).all() and torch.isfinite(z).all();outputs.append(q[:n].cpu().numpy());latents.append(z[:n].cpu().numpy());actual.append(ix);valids.append(np.arange(2)<n)
        if start==0:
            reload=JointAction().cuda(rank);reload.load_state_dict(value['state'],strict=True);reload.eval()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):q2,z2=reload(im,aux)
            assert torch.equal(q,q2) and torch.equal(z,z2);del reload
        memory(rank)
    torch.cuda.synchronize(rank);elapsed=time.monotonic()-tick
    np.savez_compressed(d/'clips.npz',clip_ids=ids,outputs=np.concatenate(outputs),latents=np.concatenate(latents),actual_batch_clip_ids=np.stack(actual),actual_batch_valid=np.stack(valids))
    assert fingerprint(model.state_dict())==wanted
    write(d/'inference_lifecycle.json',{'status':'passed','rank':rank,'strict187_state_name_shape_dtype_value_equal':True,'checkpoint_SHA256':sha(path),'final_fingerprint':wanted,'unique_clips':len(ids),'batch_slots':int(np.stack(actual).size),'registered_tail_padding_slots':int((~np.stack(valids)).sum()),'fresh_reload_output_and_latent_bitwise':True,'clips_SHA256':sha(d/'clips.npz'),'elapsed_s_CUDA_synchronized':elapsed,'memory':memory(rank),'communication_active_at_receipt':True,'action_inputs_all_zero':True,'TEST_GT_read':False})
    dist.barrier();dist.destroy_process_group();write(d/'post_communication_shutdown.json',{'distributed_initialized':dist.is_initialized(),'memory':memory(rank),'worker_exits_after_receipt':True})
if __name__=='__main__':main()
