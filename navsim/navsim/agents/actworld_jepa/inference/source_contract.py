"""Fail closed on unreviewed planner revisions; never bypass selector consent."""
import hashlib
from pathlib import Path

# Audited deployment and public release at commit 3128270. Their forward and
# _post_r94_selection ASTs are identical. The public version adds explicit
# aggregate-NAVTEST calibration consent to configuration/artifact loading.
SUPPORTED_PLANNER_SHA256 = frozenset({
    "63ab9634f720b3ea54ec2cd500bd3baa8d5432123fcb72cd67a1bf49c267bb48",
    "0cecff239883df7ca86d1a0ad8daa3a402be39f6996af9f84eb4f5442aeb66e8",
})


def validate_source(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest not in SUPPORTED_PLANNER_SHA256:
        raise RuntimeError("Unreviewed ActWorld planner source: " + digest)
    return digest
