import numpy as np
from .actworld_clearance_gate import as_numpy

def project_safe_steps(result, proposed):
 baseline=as_numpy(result['final_planning']).reshape(12,2)
 proposed=as_numpy(proposed).reshape(12,2)
 keep=as_numpy(result.get('cls_scores',result['scores_3d']))>=0.5
 if not keep.any():return proposed.copy(),np.ones(12)
 boxes=as_numpy(result['boxes_3d'])[keep];trajs=as_numpy(result['trajs_3d'])[keep]
 modes=as_numpy(result['trajs_score'])[keep].argmax(-1)
 future=trajs[np.arange(len(modes)),modes]
 radius=.5*np.linalg.norm(boxes[:,3:5],axis=-1)*1.1+.5*np.hypot(4.08,1.73)*1.1
 gap=np.linalg.norm(future-baseline[None],axis=-1)-radius[:,None]
 scale=np.zeros(12);pending=np.ones(12,dtype=bool)
 for value in [1.,.5,.25,.125]:
  trial=baseline+value*(proposed-baseline)
  newgap=np.linalg.norm(future-trial[None],axis=-1)-radius[:,None]
  safe=~((gap<2.)&(newgap<gap-1e-3)).any(0)
  choose=pending&safe;scale[choose]=value;pending[choose]=False
 # Spread attenuation to neighboring steps to avoid isolated spikes.
 scale=np.minimum.reduce([np.r_[scale[0],scale[:-1]],scale,np.r_[scale[1:],scale[-1]]])
 return baseline+scale[:,None]*(proposed-baseline),scale
