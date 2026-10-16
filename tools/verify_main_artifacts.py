"""Verify downloaded main-model files without deserializing pickle contents."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'artifact_manifest.json')
args = parser.parse_args()
failures = []
for item in json.loads(args.manifest.read_text()):
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
