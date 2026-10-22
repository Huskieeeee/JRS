#!/usr/bin/env python3
"""Verify torch/CUDA without silently substituting CPU."""
import argparse
import importlib.util
import json
import platform

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--require-cuda',action='store_true');a=p.parse_args(argv)
    out={'python':platform.python_version(),'torch_installed':importlib.util.find_spec('torch') is not None}
    if out['torch_installed']:
        import torch
        out.update(torch=str(torch.__version__),cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available())
        if out['cuda_available']:
            out['devices']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            x=torch.eye(4,device='cuda');out['matrix_check']=bool(torch.allclose(x@x,x))
    print(json.dumps(out,indent=2))
    if a.require_cuda and not out.get('matrix_check'): raise SystemExit(2)


if __name__ == "__main__":
    main()
