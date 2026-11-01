#!/usr/bin/env python3
"""Export explicit continuation and replay eligibility inventories; no mining/training.

Unique continuation rows deduplicate reused authentic twins. Pair-exposure rows
preserve every RL pair's authentic exposure, including deliberate repetitions.
These are alternative controls, not additional sequential training stages.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'v3'))
from build_sft_v3 import sft_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pools',type=Path,required=True)
    args=parser.parse_args()
    inventory={}
    for regime in ('id','cf','cm','cfcm'):
        unique={}
        exposure=[]
        with (args.pools/f'rl_{regime}.jsonl').open() as stream:
            for line in stream:
                unit=json.loads(line)
                if unit.get('fake'):
                    fake=sft_record(unit['fake'])
                    unique.setdefault(unit['fake']['image'],fake)
                    exposure.append(fake)
                real=sft_record({'type':'real','image':unit['real']['image']})
                unique.setdefault(unit['real']['image'],real)
                exposure.append(real)
        for suffix,rows in [('',list(unique.values())),('_pair_exposure',exposure)]:
            name=f'contsft_{regime}{suffix}.jsonl'
            with (args.pools/name).open('x') as stream:
                for row in rows:
                    stream.write(json.dumps(row)+'\n')
            inventory[name]={'rows':len(rows),'unique_image_paths':len({row['images'][0] for row in rows}),
                             'role':'alternative continuation control on eligible RL identities'}
        name=f'replay_eligible_{regime}.jsonl'
        with (args.pools/f'sft_{regime}.jsonl').open('rb') as src,(args.pools/name).open('xb') as dst:
            shutil.copyfileobj(src,dst)
        inventory[name]={'role':'SFT-pool replay eligibility only; no model failures mined or evaluated'}
    (args.pools/'continuation_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(inventory,indent=2))


if __name__=='__main__':
    main()
