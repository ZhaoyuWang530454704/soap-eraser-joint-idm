"""Select only from pre-registered epochs using internal TRAIN validation."""
from common import *
def main():
    verify_sources();d=read(ROOT/'units/selection/training_terminal.json');assert d['status']=='complete'
    values=d['registered_candidates'];assert [v['epoch'] for v in values]==SPEC['epoch_candidates']
    best=min(v['metrics']['joint_normalized_loss'] for v in values)
    feasible=[v for v in values if v['metrics']['joint_normalized_loss']<=best*1.01]
    choice=min(feasible,key=lambda v:v['epoch'])
    for v in values:assert sha(ROOT/'units/selection'/v['checkpoint'])==v['checkpoint_SHA256']
    write(ROOT/'epoch_selection.json',{'status':'TRAIN_only_epoch_frozen_before_fresh_all50_refit_and_TEST','selected_epoch':choice['epoch'],'selected_candidate':choice,'all_candidates':values,'rule':SPEC['selection_rule'],'fit_episodes':SPEC['fit_ids'],'validation_episodes':SPEC['validation_ids'],'TEST_used':False})
    print(json.dumps({'selected_epoch':choice['epoch'],'internal_TRAIN_validation':choice['metrics'],'TEST_used':False}),flush=True)
if __name__=='__main__':main()
