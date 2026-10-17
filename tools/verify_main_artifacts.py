"""Verify downloaded main-model files without deserializing pickle contents."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('--benchmark', choices=['all', 'navsim-v1', 'navtest-v2', 'navhard-v2', 'nuscenes'], default='all')
parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'artifact_manifest.json')
args = parser.parse_args()
failures = []
groups = {
    'navsim-v1': {'actworld_jepa_navsim_v1_r383_bundle.tar.gz'},
    'navtest-v2': {'actworld_navsim_v2.ckpt', 'factor005.pkl', 'strict_variant_selector.pkl', 'multicomponent_gate_selector.pkl'},
    'navhard-v2': {'actworld_navsim_v2.ckpt', 'factor005.pkl', 'strict_variant_selector.pkl', 'multicomponent_gate_selector.pkl', 'temporal_fcts_pair_head.pkl', 'temporal_fcts_head.pt'},
    'nuscenes': {'actworld_nuscenes_r13.pth'},
}
for item in json.loads(args.manifest.read_text()):
    if args.benchmark != 'all' and item['filename'] not in groups[args.benchmark]:
        continue
    path = args.directory / item['filename']
    if not path.is_file() or path.stat().st_size != item['bytes']:
        failures.append(item['filename'] + ': missing or wrong size')
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != item['sha256']:
        failures.append(item['filename'] + ': hash mismatch')
    else:
        print('Verified:', item['filename'])
if failures:
    raise SystemExit('\n'.join(failures))
