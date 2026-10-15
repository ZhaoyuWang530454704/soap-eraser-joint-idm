"""Server-only TEST GT scoring after independent prediction release gate."""
from pathlib import Path
import sys,csv,shutil,json,html
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from common import ROOT,SPEC,OUT,read,write,sha,np,configure
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def matrices(q):
    a=q[:,3:6].astype(float);a/=np.linalg.norm(a,axis=1)[:,None]
    b=q[:,6:9].astype(float);b-=(a*b).sum(1)[:,None]*a;b/=np.linalg.norm(b,axis=1)[:,None]
    return np.stack((a,b,np.cross(a,b)),2)
def measure(pred,truth):
    delta=(pred[:,:3].astype(float)-truth[:,:3])*1000;distance=np.sqrt(np.square(delta).sum(1));r=matrices(truth).transpose(0,2,1)@matrices(pred)
    angle=np.degrees(np.arccos(np.clip((np.trace(r,axis1=1,axis2=2)-1)*.5,-1,1)))
    closure=np.abs(pred[:,9].astype(float)-truth[:,9]);correct=(pred[:,9]>=.5)==(truth[:,9]>=.5);pose=(distance<=10)&(angle<=10)
    values={'frames':len(pred),'xyz_mae_mm':np.abs(delta).mean(0).tolist(),'position_mean_mm':float(distance.mean()),'position_median_mm':float(np.median(distance)),'position_p95_mm':float(np.quantile(distance,.95)),'position_max_mm':float(distance.max()),'position_within5mm_percent':float((distance<=5).mean()*100),'position_within10mm_percent':float((distance<=10).mean()*100),'position_within10mm_frames':int((distance<=10).sum()),'rotation_mean_deg':float(angle.mean()),'rotation_p95_deg':float(np.quantile(angle,.95)),'closure_mae':float(closure.mean()),'closure_class_error_percent':float((~correct).mean()*100),'closure_class_agreement_percent':float(correct.mean()*100),'position10mm_rotation10deg_accuracy_percent':float(pose.mean()*100),'position10mm_rotation10deg_closureclass_accuracy_percent':float((pose&correct).mean()*100)}
    details=np.column_stack([np.abs(delta),distance,angle,closure,correct,pose,pose&correct]);return values,details
def number(v):return f'{v:.4f}'
def main():
    configure();gate=read(ROOT/'evidence/prediction_release_gate.json');assert gate['status']=='passed' and gate['TEST_GT_values_read'] is False
    assert gate['prediction_completion_SHA256']==sha(ROOT/'frozen_predictions/completion.json')
    for name,key in [('actions_test10_GTfree.zip','action_ZIP_SHA256'),('complete_joint_IDM_model.zip','model_ZIP_SHA256')]:assert sha(ROOT/'frozen_predictions'/name)==gate[key]
    gt_manifest=read(ROOT/'analysis/gt_reference_manifest.json');staged=read(ROOT/'scoring_inputs/staging_complete.json');assert staged['prediction_release_gate_SHA256']==sha(ROOT/'evidence/prediction_release_gate.json') and staged['manifest_SHA256']==sha(ROOT/'analysis/gt_reference_manifest.json')
    OUT.mkdir(parents=True,exist_ok=False)
    for folder in ('plots','per_frame'): (OUT/folder).mkdir()
    entries=[];allgt=[];allpred={'raw':[],'ema05':[]};plot_names=[]
    for sid in SPEC['test_ids']:
        path=ROOT/'scoring_inputs'/f'{sid}.npy';assert sha(path)==gt_manifest['files'][sid]['SHA256']
        truth=np.load(path,allow_pickle=False);assert truth.shape==(375,10) and np.isfinite(truth).all()
        data=np.load(ROOT/'frozen_predictions/actions'/sid/'actions.npz',allow_pickle=False);times=data['action_timestamp'];assert np.max(abs(times-np.arange(1,376)/15))<2e-6
        allgt.append(truth);metrics={}
        for kind in ('raw','ema05'):
            q=data[kind];assert q.shape==truth.shape and np.isfinite(q).all();m,detail=measure(q,truth);metrics[kind]=m;allpred[kind].append(q)
            with (OUT/'per_frame'/f'{sid}_{kind}.csv').open('x',newline='') as f:
                w=csv.writer(f);w.writerow(['source_observation_row','source_target_row','target_time_s','abs_X_mm','abs_Y_mm','abs_Z_mm','distance3D_mm','SO3_deg','abs_closure','closure_class_correct','pose10mm10deg_correct','pose_and_closure_correct']);w.writerows(np.column_stack([np.arange(375),np.arange(1,376),times,detail]))
        fig,axes=plt.subplots(4,1,figsize=(14,10),sharex=True,layout='constrained')
        for k in range(3):
            axes[k].plot(times,truth[:,k]*1000,color='#242424',lw=1.5,label='GT');axes[k].plot(times,data['raw'][:,k]*1000,color='#1678c2',lw=1.3,label='Joint IDM prediction (raw)');axes[k].set_ylabel('XYZ'[k]+' (mm)');axes[k].grid(alpha=.2)
        axes[0].legend(ncol=2,loc='best');axes[3].plot(times,truth[:,9],color='#242424',lw=1.5,label='GT commanded closure');axes[3].plot(times,data['raw'][:,9],color='#1678c2',lw=1.3,label='Predicted closure');axes[3].set_ylabel('Commanded closure');axes[3].grid(alpha=.2);axes[3].set_xlabel('Original next-endpoint target time (s), no shift');axes[3].legend(ncol=2,loc='best')
        fig.suptitle(f'{sid} | 375 actions | dual-view joint IDM | 3D mean {metrics["raw"]["position_mean_mm"]:.3f} mm');name=f'{sid}_xyz_gripper.png';fig.savefig(OUT/'plots'/name,dpi=130);plt.close(fig);plot_names.append('plots/'+name)
        fig=plt.figure(figsize=(10,8),layout='constrained');ax=fig.add_subplot(111,projection='3d')
        for label,q,color in [('GT',truth,'#242424'),('Prediction (raw)',data['raw'],'#1678c2')]:
            xyz=q[:,:3]*1000;ax.plot(*xyz.T,color=color,lw=1.5,label=label);ax.scatter(*xyz[0],color=color,marker='o',s=24);ax.scatter(*xyz[-1],color=color,marker='x',s=28)
        both=np.concatenate([truth[:,:3],data['raw'][:,:3]])*1000;center=(both.min(0)+both.max(0))/2;radius=max(float(np.ptp(both,axis=0).max())*.55,1)
        ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius);ax.set_box_aspect((1,1,1));ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)');ax.set_zlabel('Z (mm)');ax.legend();ax.set_title(f'{sid} | robot-base fixed TCP | raw, no EMA');name=f'{sid}_trajectory3D.png';fig.savefig(OUT/'plots'/name,dpi=130);plt.close(fig);plot_names.append('plots/'+name)
        entries.append({'episode':sid,'frames':375,'GT_SHA256':sha(path),'metrics':metrics,'plots':['plots/'+sid+'_xyz_gripper.png','plots/'+sid+'_trajectory3D.png']})
    truth=np.concatenate(allgt);aggregate={kind:measure(np.concatenate(values),truth)[0] for kind,values in allpred.items()};assert len(truth)==3750
    selection=[json.loads(s) for s in (__import__('pathlib').Path(SPEC['baseline_root'])/'units/selection/training.jsonl').read_text().splitlines()];refit=[json.loads(s) for s in (ROOT/'units/refit/training.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(4,3,figsize=(17,11),layout='constrained');titles=['Prior baseline40 TRAIN fitting (reused)','Prior baseline10 TRAIN validation (reused)','New rotation-dynamics50 TRAIN refit'];labels=['Joint normalized loss','XYZ normalized loss','Rotation normalized loss','Closure SmoothL1 normalized loss']
    for col,history in enumerate((selection,selection,refit)):
        for row in range(4):
            if col==1:values=[q['internal_TRAIN_validation']['joint_normalized_loss'] if row==0 else q['internal_TRAIN_validation']['normalized_components'][row-1] for q in history]
            else:values=[q['TRAIN_joint_loss'] if row==0 else q['TRAIN_components'][row-1] for q in history]
            axes[row,col].plot([q['epoch'] for q in history],values,color='#e58b22' if col==1 else '#1678c2');axes[row,col].grid(alpha=.2);axes[row,col].set_ylabel(labels[row]);axes[row,col].set_xlabel('Epoch')
            if row==0:axes[row,col].set_title(titles[col])
    fig.suptitle('Rotation dynamics ablation | baseline selection curves reused; new refit adds SO3 increment loss');fig.savefig(OUT/'plots/loss.png',dpi=130);plt.close(fig);plot_names.append('plots/loss.png');assert len(plot_names)==21
    for name in ('actions_test10_GTfree.zip','complete_joint_IDM_model.zip'):shutil.copyfile(ROOT/'frozen_predictions'/name,OUT/name)
    choice=read(ROOT/'epoch_selection.json');value={'schema':'soap_joint_IDM_sim50_test10_v1','status':'computed_pending_independent_metrics_visual_resource_delivery','train_episodes':50,'internal_fit_episodes':40,'internal_validation_episodes':10,'test_episodes':10,'scored_rows':3750,'selected_epoch':choice['selected_epoch'],'architecture':SPEC['architecture'],'initialization':SPEC['initialization'],'action_spec':SPEC['action_spec'],'alignment':SPEC['canonical_alignment'],'raw_primary_EMA_secondary':True,'test_GT_read_after_prediction_release_gate':True,'prediction_release_gate_SHA256':sha(ROOT/'evidence/prediction_release_gate.json'),'GT_manifest_SHA256':sha(ROOT/'analysis/gt_reference_manifest.json'),'aggregate_frame_weighted':aggregate,'episodes':entries,'all21_plots':[{'path':p,'SHA256':sha(OUT/p)} for p in plot_names],'training':{'selection_epochs':0,'selection_reference_epochs':60,'selection_curves_reused_from_previous_TRAIN_only_decision':True,'final_refit_epochs':len(refit),'selection_train_wall_s':0,'refit_train_wall_s':sum(v['train_epoch_s_slowest_rank'] for v in refit),'selection_real_clip_slots':0,'refit_real_clip_slots':sum(v['real_clips'] for v in refit),'selection_overlap_row_slots':0,'refit_overlap_row_slots':sum(v['real_overlap_row_slots'] for v in refit),'GPUs':read(ROOT/'GPU_assignment.json')['physical_GPU_indices'],'effective_batch':8,'real_per_GPU_microbatch':2,'accumulation':2},'limits':SPEC['scope_limits'],'prediction_clipping':False,'GT_replacement_or_resmoothing':False,'physical_task_success_rate_measured':False}
    write(OUT/'metrics.json',value)
    with (OUT/'metrics.csv').open('x',newline='') as f:
        fields=list(aggregate['raw']);w=csv.writer(f);w.writerow(['scope','kind']+fields)
        for scope,m in [('ALL3750',aggregate)]+[(v['episode'],v['metrics']) for v in entries]:
            for kind,record in m.items():w.writerow([scope,kind]+[json.dumps(record[k]) if isinstance(record[k],list) else record[k] for k in fields])
    raw=aggregate['raw'];ema=aggregate['ema05'];rows=[]
    for v in entries:
        m=v['metrics']['raw'];rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [v['episode'],375,number(m['position_mean_mm']),number(m['position_p95_mm']),number(m['position_max_mm']),*[number(x) for x in m['xyz_mae_mm']],number(m['rotation_mean_deg']),number(m['closure_mae']),number(m['closure_class_agreement_percent'])+'%'])+'</tr>')
    blocks=[]
    for v in entries:blocks.append(f'<section id="{v["episode"]}"><h2>{v["episode"]} — 375 actions</h2><div class="grid">'+''.join(f'<a href="{p}"><img src="{p}" alt="{v["episode"]} {p}"></a>' for p in v['plots'])+'</div></section>')
    cards=f'<div class="cards"><div>3D mean (raw)<strong>{raw["position_mean_mm"]:.4f} mm</strong></div><div>XYZ MAE<strong>{" / ".join(number(x) for x in raw["xyz_mae_mm"])} mm</strong></div><div>Closure agreement (threshold 0.5)<strong>{raw["closure_class_agreement_percent"]:.4f}%</strong></div></div>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>板擦洗手液 · 联合IDM · 50训练10测试</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#213047;background:#f3f6fa;margin:auto;max-width:1500px;padding:26px}h1,h2{line-height:1.3}a{color:#0668ad}section,.intro{background:white;padding:22px;border-radius:14px;margin:24px 0}.grid{display:grid;grid-template-columns:1.35fr 1fr;gap:20px}.grid img,img.loss{width:100%;height:auto}.cards{display:flex;gap:16px;flex-wrap:wrap}.cards div{background:#eaf4fb;border-radius:12px;padding:16px;min-width:240px}.cards strong{display:block;font-size:24px}table{border-collapse:collapse;background:white;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid #dbe3ec;text-align:right}th:first-child,td:first-child{text-align:left}.table{overflow-x:auto}.note{color:#516071;font-size:14px}@media(max-width:950px){.grid{grid-template-columns:1fr}}</style><h1>板擦放洗手液瓶 · 新增旋转变化约束对照</h1><div class="intro">'''+cards+f'<p>固定 50 条仿真训练、10 条仿真测试；每条 375 个动作，共 3,750 行评分。双视角 RGB 输入，XYZ、旋转和夹爪联合优化。沿用上一轮 TRAIN 内40条拟合＋10条验证选出的轮数；本轮不重跑选型，全部50条从完全相同初始权重加入旋转变化约束，重新拟合 {choice["selected_epoch"]} 轮。</p><p><a href="complete_joint_IDM_model.zip">完整模型＋RGB 推理代码</a> · <a href="actions_test10_GTfree.zip">全部动作 ZIP（不含 GT）</a> · <a href="metrics.json">完整误差 JSON</a> · <a href="metrics.csv">误差 CSV</a></p><p>主图只画预测 raw 与 GT 两条线。EMA0.5 是单独的输出后处理；其 3D mean 为 {ema["position_mean_mm"]:.4f} mm、夹爪分类一致率为 {ema["closure_class_agreement_percent"]:.4f}%。</p><p class="note">动作合同：robot_base 固定 TCP，link7 沿 Z 轴平移 261.9830738 mm；XYZ 使用米保存，图中换算为毫米。夹爪 GT 是原始目标闭合指令。观测行0对应已准备好的下一端点目标行1，保持原映射。每帧均参与评分，没有裁剪、偏移搜索或筛帧。3D mean = 每帧 XYZ 欧氏距离的平均值；P95 = 95% 帧的误差不超过该值。分类一致率按闭合度 ≥ 0.5 判闭合；单种子离线仿真动作恢复指标。该10条测试曾用于历史研究，本次选轮数只用 TRAIN 内部验证。</p></div><h2>15 Hz 固定测试集逐条误差 · raw</h2><div class="table"><table><thead><tr>'+''.join('<th>'+v+'</th>' for v in ['Episode','Rows','3D mean mm','3D P95 mm','3D max mm','X MAE mm','Y MAE mm','Z MAE mm','SO3 mean deg','Closure MAE','Closure agreement'])+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><section><h2>训练曲线</h2><p class="note">左、中为上一轮训练集选型曲线，仅作冻结参照，未重新训练；右为本轮50条新训练，目标额外包含相邻帧相对旋转匹配项。</p><img class="loss" src="plots/loss.png" alt="training and internal validation loss"></section>'+''.join(blocks)+'</html>'
    (OUT/'index.html').write_text(page)
    write(OUT/'evaluation_files_SHA256.json',{'files':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()},'independent_verification_still_required':True})
    print(json.dumps({'status':'scored_pending_independent_visual_resource_audits','raw':raw,'ema05':ema,'output':str(OUT)}),flush=True)
if __name__=='__main__':main()
