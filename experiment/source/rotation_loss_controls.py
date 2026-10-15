"""Small server CPU mathematical controls; never a GPU/model preflight."""
from common import *
def encoded(angles):
 r=Rotation.from_rotvec(np.column_stack((np.zeros(len(angles)),np.zeros(len(angles)),np.deg2rad(angles)))).as_matrix()
 q=np.zeros((1,len(r),10),np.float32);q[0,:,3:6]=r[:,:,0];q[0,:,6:9]=r[:,:,1]
 return torch.tensor(q,requires_grad=True)
def main():
 configure();verify_sources();checks={}
 gt=encoded([0,5,10,15]);same=rotation_velocity_rows(gt,gt)
 checks['perfect_nonzero_rotation_path_zero_loss']=bool(torch.max(abs(same))<2e-5)
 static=encoded([0,0,0,0]);bad=rotation_velocity_rows(static,gt)
 checks['static_path_rejected_for_moving_GT']=bool(bad[:,1:].min()>1)
 reverse=encoded([0,-5,-10,-15]);checks['wrong_rotation_direction_costs_more']=bool(rotation_velocity_rows(reverse,gt).sum()>bad.sum())
 shifted=encoded([30,35,40,45]);checks['constant_left_offset_not_anchored_by_dynamics_alone']=bool(rotation_velocity_rows(shifted,gt).max()<2e-5)
 checks['per_frame_orientation_still_anchors_offset']=bool(frame_loss_rows(shifted.reshape(-1,10),gt.reshape(-1,10))[2].min()>1)
 wrap=encoded([179,181,183,185]);continuous=encoded([-1,1,3,5]);checks['Euler_wrap_does_not_create_rotation_jump']=bool(rotation_velocity_rows(wrap,continuous).max()<2e-5)
 # Compare a nontrivial torch relative-rotation cost with independent SciPy composition.
 qp=encoded([0,4,12,17]);qt=encoded([0,5,10,15]);loss=rotation_velocity_rows(qp,qt)
 a=Rotation.from_rotvec(np.column_stack((np.zeros(4),np.zeros(4),np.deg2rad([0,4,12,17]))));b=Rotation.from_rotvec(np.column_stack((np.zeros(4),np.zeros(4),np.deg2rad([0,5,10,15]))))
 error=((a[:-1].inv()*a[1:]).inv()*(b[:-1].inv()*b[1:])).magnitude()
 independent=(np.sqrt((2*np.sin(error/2))**2+1e-12)-1e-6)/SPEC['rotation_velocity_loss_scale_rad']
 checks['independent_SciPy_relative_rotation_math']=bool(np.allclose(loss.detach().numpy()[0,1:],independent,atol=2e-5,rtol=0))
 loss.sum().backward();checks['finite_nonzero_dynamics_gradient']=bool(torch.isfinite(qp.grad).all() and qp.grad.norm()>0)
 ds=ClipData('refit','train');mult,weights,coverage=rotation_pair_weights(ds)
 sums=np.zeros(len(ds.groups));np.add.at(sums,ds.clip_rows.ravel(),weights.ravel())
 checks['all18700_original_pairs_total_weight1']=bool(np.count_nonzero(coverage)==18700 and np.allclose(sums[coverage>0],1,rtol=0,atol=1e-12))
 checks['no_first_frame_tail_or_repeated_row_pairs']=bool(np.all(weights[:,0]==0) and np.all(weights[ds.clip_valid==0]==0) and np.all(weights[:,1:][ds.clip_rows[:,1:]==ds.clip_rows[:,:-1]]==0))
 checks['no_heldout_recordings_in_pairs']=bool(set(ds.ids)==set(SPEC['train_ids']) and set(ds.ids).isdisjoint(SPEC['test_ids']))
 old=SPEC['rotation_velocity_weight'];SPEC['rotation_velocity_weight']=0
 legacy=_loss_without_rotation_velocity(qp.detach(),qt.detach()) if '_loss_without_rotation_velocity' in globals() else __import__('common')._loss_without_rotation_velocity(qp.detach(),qt.detach())
 current=loss_rows(qp.detach(),qt.detach());SPEC['rotation_velocity_weight']=old
 checks['zero_extra_weight_preserves_old_total_bitwise']=bool(torch.equal(legacy[0],current[0]))
 assert all(checks.values()),checks
 write(ROOT/'evidence/rotation_loss_CPU_controls.json',{'passed':True,'checks':checks,'tests':len(checks),'GPU_or_model_preflight':False,'new_term_only_matches_GT_relative_rotations':True,'unique_valid_TRAIN_pairs':18700,'TRAIN_only':True})
 print(json.dumps({'CPU_mathematical_controls_passed':len(checks),'GPU_or_model_preflight':False}),flush=True)
if __name__=='__main__':main()
