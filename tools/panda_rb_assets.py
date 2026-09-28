#!/usr/bin/env python3
"""Explicit B01 F1 binding/export/import; never starts a training run."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panda_rb import bindings, plan, weights
from panda_rb.common import docker_required


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=plan.ROOT)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--destination',type=Path)
    p.add_argument('--paths-json',type=Path,help='Explicit artifact and native/LP path mappings')
    p=sub.add_parser('weights');p.add_argument('--binding',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--device',default='cuda')
    p=sub.add_parser('export');p.add_argument('destination',type=Path);p.add_argument('--binding',type=Path)
    p.add_argument('--weights',type=Path,help='Default common full-train cache and all six seed mappings')
    p=sub.add_parser('import');p.add_argument('package',type=Path);p.add_argument('--data-paths',required=True,type=Path)
    p.add_argument('--destination',type=Path);p.add_argument('--weights',type=Path)
    p=sub.add_parser('check');p.add_argument('--binding',type=Path)
    args=parser.parse_args();root=args.root
    docker_required()
    binding=getattr(args,'binding',None) or plan.binding_path(root)
    if args.command=='prepare':
        mapping=json.loads(args.paths_json.read_text()) if args.paths_json else {}
        output=bindings.prepare_binding(args.destination or binding,root,mapping.get('origin_paths'),mapping.get('data_paths'))
    elif args.command=='weights':
        output=weights.prepare_weights(binding,args.output or plan.weights_dir(root),args.device,root=root)
    elif args.command=='export':
        output=bindings.export_binding(binding,args.destination,args.weights or plan.weights_dir(root))
    elif args.command=='import':
        paths=json.loads(args.data_paths.read_text())
        output=bindings.import_binding(args.package,args.destination or binding,paths,root,
                                       args.weights or plan.weights_dir(root))
    else:
        _,b,_,q=bindings.validate_binding(binding)
        output=dict(common_sha256=b['common_sha256'],teacher_checkpoint_sha256=b['teacher_checkpoint_sha256'],
                    q_shape=list(q.shape),runtime_ready=False)
    print(json.dumps(output if isinstance(output,dict) else str(output),indent=2))

if __name__=='__main__':main()
