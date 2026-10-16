from mmdet.models import DETECTORS
from .actworld_momadv2_clearance import ActWorldMomADv2Clearance
from .actworld_candidate import CandidateOutcomeJEPA
@DETECTORS.register_module()
class ActWorldMomADv2Candidate(ActWorldMomADv2Clearance):
    def __init__(self,actworld=None,**kwargs):
        super().__init__(actworld=actworld,**kwargs)
        self.actworld=CandidateOutcomeJEPA(**(actworld or {}))
