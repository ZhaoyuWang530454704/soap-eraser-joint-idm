"""Two-GPU fresh joint training; actual GPU gates inline, no separate smoke training."""
import contextlib,gc,math,time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Sampler
from common import *
class EpochSampler(Sampler):
    def __init__(self,n,rank):self.n=n;self.rank=rank;self.order=None;self.mask=None
    def set_epoch(self,epoch):
        self.global_order,self.global_mask=epoch_plan(self.n,epoch)
        self.order=self.global_order.reshape(-1,8)[:,self.rank*4:(self.rank+1)*4].reshape(-1)
        self.mask=self.global_mask.reshape(-1,8)[:,self.rank*4:(self.rank+1)*4].reshape(-1)
    def __iter__(self):return iter([(int(i),float(m)) for i,m in zip(self.order,self.mask)])
    def __len__(self):return (self.n+(-self.n)%8)//2
def all_same(value):
    out=[None]*2;dist.all_gather_object(out,value);assert len(set(out))==1,out;return out
@torch.inference_mode()
def validation(model,data,rank):
    model.eval();ids=np.arange(rank,len(data),2);qq=[];zz=[]
    for begin in range(0,len(ids),2):
        ix=ids[begin:begin+2];n=len(ix);ix=np.r_[ix,np.repeat(ix[-1],2-n)];sample=[data[int(i)] for i in ix]
        im=torch.stack([s[0] for s in sample]).cuda(rank);aux=torch.stack([s[1] for s in sample]).cuda(rank)
        with torch.autocast('cuda',dtype=torch.bfloat16):q,z=model(im,aux)
        qq.append(q[:n].cpu().numpy());zz.append(z[:n].cpu().numpy())
    pieces=[None]*2;dist.all_gather_object(pieces,(ids,np.concatenate(qq),np.concatenate(zz)))
    pred,_=merge_clips(data,np.concatenate([p[0] for p in pieces]),np.concatenate([p[1] for p in pieces]),np.concatenate([p[2] for p in pieces]))
    target=data.y[data.row_indices];value=metrics(pred,target);ls=frame_loss_rows(torch.from_numpy(pred),torch.from_numpy(target))
    value['joint_normalized_loss']=float(ls[0].mean());value['normalized_components']=[float(x.mean()) for x in ls[1:]]
    return value,pred
def main():
    phase=sys.argv[1];assert phase in ('selection','refit');rank=int(os.environ['LOCAL_RANK'])
    configure(rank);source=verify_sources();start=time.monotonic();dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20));assert dist.get_world_size()==2
    unit=ROOT/'units'/phase;unit.mkdir(parents=True,exist_ok=True);rd=unit/f'rank{rank}';rd.mkdir(exist_ok=False)
    data=ClipData(phase,'train');val=ClipData(phase,'validation') if phase=='selection' else None
    assert len(data.row_indices)==(15000 if phase=='selection' else 18750)
    pair_multiplier,pair_weights,pair_coverage=rotation_pair_weights(data)
    model=JointAction(pretrained=True);receipt=initialize(model,phase);model.cuda(rank);initial=fingerprint(model.state_dict());assert len(initial['records'])==187;all_same(initial['group_SHA256'])
    assert initial==SPEC['expected_initial_fingerprint'],'Fresh187 initial tensors differ from paired baseline'
    write(rd/'initial.json',{'rank':rank,'GPU_UUID':SPEC['GPU_UUIDs'][rank],'fingerprint':initial,'initialization':receipt,'source_manifest_SHA256':source,'TEST_GT_read':False})
    samples=[data[i] for i in range(2)];im=torch.stack([s[0] for s in samples]).cuda(rank);aux=torch.stack([s[1] for s in samples]).cuda(rank);target=torch.stack([s[2] for s in samples]).cuda(rank);model.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16):q,_=model(im,aux)
    controls={}
    for task,loss in zip(('XYZ','rotation','closure','rotation_velocity'),loss_rows(q,target)[1:]):
        gradients=torch.autograd.grad(loss.mean(),(model.backbone.conv1.weight,model.shared[0].weight,model.fine_visual[0].weight,model.temporal[0].weight),retain_graph=True)
        norms=[float(g.float().norm()) for g in gradients];assert all(math.isfinite(x) and x>0 for x in norms),(task,norms)
        controls[task]=dict(zip(('shared_visual_conv1','shared_fusion','fine_visual','temporal_visual'),norms))
    assert fingerprint(model.state_dict())==initial
    write(rd/'actual_GPU_shared_gradient_control.json',{'status':'passed','rank':rank,'three_losses_reach_shared_visual_encoder':controls,'TRAIN_only':True,'CPU_is_not_GPU_preflight':True,'memory':memory(rank)})
    del im,aux,target,q,loss,gradients;gc.collect();torch.cuda.empty_cache()
    ddp=DDP(model,device_ids=[rank],output_device=rank,broadcast_buffers=False,find_unused_parameters=False);groups={}
    for n,p in model.named_parameters():
        lr=SPEC['backbone_lr'] if n.startswith('backbone.') else SPEC['fine_branch_lr'] if n.startswith(('fine_visual.','fine_closure.')) else SPEC['temporal_lr'] if n.startswith('temporal.') else SPEC['added_layers_lr'] if n.startswith(('extra.','closure_residual.','dense_reduce.','dense_embed.')) else SPEC['new_layers_lr']
        groups.setdefault(lr,[]).append(p)
    optimizer=torch.optim.AdamW([{'params':p,'lr':lr,'initial_lr':lr} for lr,p in groups.items()],weight_decay=SPEC['weight_decay']);assert len(optimizer.state)==0
    torch.manual_seed(SPEC['seed']+rank*1000003);torch.cuda.manual_seed(SPEC['seed']+rank*1000003)
    epochs=SPEC['epochs_max'] if phase=='selection' else read(ROOT/'epoch_selection.json')['selected_epoch']
    sampler=EpochSampler(len(data),rank);sampler.set_epoch(1)
    loader=DataLoader(data,batch_size=4,sampler=sampler,num_workers=1,multiprocessing_context='spawn',persistent_workers=True,pin_memory=True,drop_last=False)
    prefixes=('backbone.','shared.','position.','orientation.','fine_visual.','fine_closure.','closure_residual.','temporal.');steps=0;history=[];orders=[];masks=[];choices=[]
    for epoch in range(1,epochs+1):
        sampler.set_epoch(epoch);orders.append(sampler.order.copy());masks.append(sampler.mask.copy());ddp.train();torch.cuda.synchronize(rank);tick=time.monotonic();totals=torch.zeros(7,device=rank,dtype=torch.float64)
        factor=SPEC['lr_floor_factor']+(1-SPEC['lr_floor_factor'])*.5*(1+math.cos(math.pi*(epoch-1)/SPEC['lr_cosine_horizon']))
        for g in optimizer.param_groups:g['lr']=g['initial_lr']*factor
        for batch_index,(im,aux,target,indices,valid) in enumerate(loader):
            assert len(im)==4;denom=valid.double().sum().cuda(rank);dist.all_reduce(denom);assert float(denom)>0;optimizer.zero_grad(set_to_none=True)
            for begin in (0,2):
                x=im[begin:begin+2].cuda(rank,non_blocking=True);a=aux[begin:begin+2].cuda(rank,non_blocking=True);y=target[begin:begin+2].cuda(rank,non_blocking=True);w=valid[begin:begin+2].cuda(rank,non_blocking=True)
                context=ddp.no_sync() if begin==0 else contextlib.nullcontext()
                with context:
                    with torch.autocast('cuda',dtype=torch.bfloat16):pred,_=ddp(x,a)
                    rm=torch.from_numpy(pair_multiplier[indices[begin:begin+2].numpy()]).cuda(rank,non_blocking=True)
                    ls=loss_rows(pred,y,rm);loss=(ls[0]*w).sum()*2/denom;assert torch.isfinite(loss);loss.backward()
                totals[0]+=w.double().sum()
                for j,v in enumerate(ls):totals[j+1]+=(v.detach().double()*w).sum()
                totals[6]+=float((w>0).sum())
            if steps<3:
                norms={n:float(p.grad.float().norm()) for n,p in model.named_parameters() if p.grad is not None}
                assert len(norms)==len(list(model.parameters())) and all(math.isfinite(v) for v in norms.values())
                for prefix in prefixes:assert any(v>0 for n,v in norms.items() if n.startswith(prefix)),prefix
                gf=fingerprint({n:p.grad for n,p in model.named_parameters()});all_same(gf['group_SHA256'])
                write(rd/f'gradient_step{steps+1}.json',{'status':'passed','finite_norms':norms,'gradient_fingerprint':gf,'all2_rank_gradient_SHA_equal':True,'three_losses_joint_in_each_backward':True,'backward_calls_per_rank':2,'real_microbatch_clips':2,'global_clips_per_update':8,'global_weight_denominator':float(denom)})
            torch.nn.utils.clip_grad_norm_(model.parameters(),SPEC['clip_grad_norm'],error_if_nonfinite=True);optimizer.step();steps+=1
            if steps<=3:
                after=fingerprint(model.state_dict());all_same(after['group_SHA256']);changed=[a['name'] for a,b in zip(after['records'],initial['records']) if a['SHA256']!=b['SHA256']]
                for prefix in prefixes:assert any(n.startswith(prefix) for n in changed),prefix
                write(rd/f'optimizer_step{steps}.json',{'status':'passed','changed_tensors':changed,'all_three_heads_and_shared_visual_updated':True,'all2_rank_parameter_SHA_equal':True,'memory':memory(rank),'TEST_GT_read':False})
            memory(rank)
            if rank==0 and (steps<=3 or batch_index%20==0):
                update(ROOT/'progress.json',{'UTC_server':now(),'stage':phase+'_training','epoch':epoch,'total_epochs':epochs,'optimizer_steps':steps,'step_within_epoch':batch_index+1,'steps_per_epoch':len(loader),'actual_GPU_inline_gates_passed':steps>=3,'TEST_GT_read':False})
        dist.all_reduce(totals);torch.cuda.synchronize(rank);duration=torch.tensor(time.monotonic()-tick,device=rank,dtype=torch.float64);dist.all_reduce(duration,op=dist.ReduceOp.MAX);elapsed=float(duration);v=totals.cpu().numpy()
        assert abs(v[0]-len(data.row_indices))<.01
        item={'epoch':epoch,'phase':phase,'optimizer_steps':steps,'TRAIN_joint_loss':float(v[1]/v[0]),'TRAIN_components':[float(v[k]/v[0]) for k in (2,3,4)],'TRAIN_rotation_velocity_loss':float(v[5]/v[0]),'weighted_unique_rows':float(v[0]),'train_epoch_s_slowest_rank':elapsed,'real_clips':len(data),'global_clips_including_zero_padding':len(sampler.global_order),'real_overlap_row_slots':int(data.clip_valid[sampler.global_order[sampler.global_mask>0]].sum()),'real_clips_per_second':len(data)/elapsed,'real_microbatch_per_rank':2,'gradient_accumulation':2,'world_size':2,'effective_batch':8,'learning_rate_factor':factor}
        if val is not None:
            values,pred=validation(model,val,rank);item['internal_TRAIN_validation']=values
            if epoch in SPEC['epoch_candidates'] and rank==0:
                path=unit/('epoch%03d.pt'%epoch);torch.save({'state':cpu_state(model),'optimizer':optimizer.state_dict(),'epoch':epoch,'phase':phase,'contract':SPEC,'source_manifest_SHA256':source,'fresh_ImageNet':True},path)
                np.save(unit/('validation_epoch%03d.npy'%epoch),pred,allow_pickle=False);choices.append({'epoch':epoch,'metrics':values,'checkpoint':path.name,'checkpoint_SHA256':sha(path)})
        dist.barrier();item['elapsed_s']=time.monotonic()-start;history.append(item)
        update(rd/'progress.json',dict(UTC_server=now(),**item,TEST_GT_read=False))
        if rank==0:
            update(ROOT/'progress.json',dict(UTC_server=now(),stage=phase+'_epoch_complete',**item,TEST_GT_read=False))
            with (unit/'training.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
            print(json.dumps(item),flush=True)
    if loader._iterator is not None:loader._iterator._shutdown_workers()
    del loader,ddp;gc.collect()
    if phase=='refit' and rank==0:torch.save({'state':cpu_state(model),'optimizer':optimizer.state_dict(),'epoch':epochs,'phase':phase,'contract':SPEC,'source_manifest_SHA256':source,'fresh_ImageNet':True},unit/'model.pt')
    torch.save({'torch_CPU_RNG':torch.get_rng_state(),'torch_CUDA_RNG':torch.cuda.get_rng_state(rank),'rank':rank,'completed_epoch':epochs},rd/'RNG_terminal.pt')
    del optimizer;gc.collect();dist.barrier();path=unit/(('epoch%03d.pt'%epochs) if phase=='selection' else 'model.pt');final=fingerprint(model.state_dict());all_same(final['group_SHA256'])
    samples=[data[i] for i in range(2)];im=torch.stack([s[0] for s in samples]).cuda(rank);aux=torch.stack([s[1] for s in samples]).cuda(rank);model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):q,z=model(im,aux)
    reload=JointAction().cuda(rank);checkpoint=torch.load(path,map_location='cpu',weights_only=True);assert checkpoint['epoch']==epochs and len(checkpoint['state'])==187
    reload.load_state_dict(checkpoint['state'],strict=True);assert fingerprint(reload.state_dict())==final;reload.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):q2,z2=reload(im,aux)
    assert torch.equal(q,q2) and torch.equal(z,z2)
    np.savez_compressed(rd/'actual_sampling.npz',clip_indices=np.stack(orders),sample_ids=data.sample_ids[np.stack(orders)],real_mask=np.stack(masks))
    np.savez_compressed(rd/'original_row_maps.npz',clip_rows=data.clip_rows,clip_valid=data.clip_valid,loss_weights=data.loss_weights,row_indices=data.row_indices,coverage=data.coverage,rotation_pair_multiplier=pair_multiplier,rotation_pair_weights=pair_weights,rotation_pair_coverage=pair_coverage)
    write(rd/'training_reload_lifecycle.json',{'status':'passed','rank':rank,'phase':phase,'epochs':epochs,'optimizer_steps':steps,'all2_rank_final_SHA_equal':True,'final_fingerprint':final,'checkpoint_SHA256':sha(path),'strict187_loaded':True,'TRAIN2clips_fresh_reload_output_and_latent_bitwise':True,'sampling_SHA256':sha(rd/'actual_sampling.npz'),'actual_sampled_episode_ids':sorted(set(str(s).split(':')[0] for s in data.sample_ids[np.stack(orders)][np.stack(masks)>0])),'memory':memory(rank),'communication_active_at_receipt':True,'elapsed_s':time.monotonic()-start,'TEST_GT_read':False})
    if rank==0:write(unit/'training_terminal.json',{'status':'complete','phase':phase,'epochs':epochs,'optimizer_steps':steps,'final_group_SHA256':final['group_SHA256'],'checkpoint_SHA256':sha(path),'registered_candidates':choices,'history':history,'elapsed_s':time.monotonic()-start,'TEST_GT_read':False})
    dist.barrier();dist.destroy_process_group();write(rd/'post_communication_shutdown.json',{'distributed_initialized':dist.is_initialized(),'memory':memory(rank),'worker_exits_after_receipt':True})
if __name__=='__main__':main()
