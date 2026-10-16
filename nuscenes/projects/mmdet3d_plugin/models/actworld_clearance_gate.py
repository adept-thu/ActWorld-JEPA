"""Prediction-only clearance gate; contains no ground-truth inputs."""
import numpy as np

def as_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)

def clearance_decreased(result):
    confidence=as_numpy(result.get('cls_scores',result['scores_3d']))
    keep=confidence>=0.5
    if not keep.any():return False
    boxes=as_numpy(result['boxes_3d'])[keep]
    trajectories=as_numpy(result['trajs_3d'])[keep]
    modes=as_numpy(result['trajs_score'])[keep].argmax(-1)
    future=trajectories[np.arange(len(modes)),modes]
    radius=0.5*np.linalg.norm(boxes[:,3:5],axis=-1)*1.1+0.5*np.hypot(4.08,1.73)*1.1
    baseline=as_numpy(result['actworld_baseline_planning']).reshape(12,2)
    proposed=as_numpy(result['final_planning']).reshape(12,2)
    original_clearance=np.linalg.norm(future-baseline[None],axis=-1)-radius[:,None]
    proposed_clearance=np.linalg.norm(future-proposed[None],axis=-1)-radius[:,None]
    # Refuse to reduce an already narrow predicted gap to any actor at any step.
    return bool(((original_clearance<2.0)&(proposed_clearance<original_clearance-1e-3)).any())
