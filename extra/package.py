#!/usr/bin/env python3
# This script will package the extension into a single file for inline JIT loading.

from sys import exit
from pathlib import Path
from argparse import ArgumentParser
from zlib import compress
from base64 import b64encode
from itertools import chain
import re

parser = ArgumentParser(description="Package the extension into a single file for inline JIT loading.")
parser.add_argument('-p', '--path', default=None, help="Location of source files, defaults based on this scripts location.")
parser.add_argument('-o', '--output', default = "swish_inline.py", help="File to write output to, default: ./__init__.py")
parser.add_argument('-s', '--stdout', action="store_true", help="Weite output to stdout instead of file")
args = parser.parse_args()

path = Path(__file__).absolute().parent.parent if args.path is None else Path(args.path).absolute()
if not path.exists(): exit("Input path doesn't exist.")
if not (path/'csrc').exists(): exit(f"Path doesn't appear to contain extension sources. Couldn't find {(path/'csrc').absolute()}.")
cpp_files = list((path/'csrc').glob('*.cpp'))
cu_files = list((path/'csrc').glob('*.cu'))
incs = {p.name: p.read_text() for p in chain(*[(path/d).glob(p) for d in ['csrc','external'] for p in ['*.h','*.cuh']])}

def proc_src(f):
    src = f.read_text()
    res,pos = f"\n// From {f}\n\n",0
    for m in re.finditer(r'#include "([^"]+)"\n', src):
        res += src[pos:m.start()]
        inc = m.group(1)
        if inc not in incs: exit(f"Couldn't find included file '{inc}' included in {f}'")
        res += f"\n// Include: {f}\n" + incs[inc] + f"\n// End Include: {f}\n\n"
        pos = m.end()
    res += src[pos:]
    return res

cpp_srcs = [proc_src(f) for f in cpp_files]
cu_srcs = [proc_src(f) for f in cu_files]

m = re.search(r"""version=['"]([^'"]+)['"]""", (path/'setup.py').read_text())
if not m: exit("Unable to find version in setup.py")
ver = m.group(1)

src = f"""
ALL = ['Swish','SwishFunction','swish','__version__']
from base64 import b64decode
from zlib import decompress
import torch
from torch.utils.cpp_extension import load_inline

__version__='{ver}'

def load_module():
    print("Compiling script_torch module...")
    cpp_comp = [{','.join((f"'{b64encode(compress(s.encode(),9)).decode()}'" for s in cpp_srcs))}]
    cu_comp  = [{','.join((f"'{b64encode(compress(s.encode(),9)).decode()}'" for s in cu_srcs ))}]
    cpp_srcs = [decompress(b64decode(src)).decode() for src in cpp_comp]
    cu_srcs = [decompress(b64decode(src)).decode() for src in cu_comp]

    swish_mod = load_inline("swish_torch_inline", cpp_sources=cpp_srcs, cuda_sources=cu_srcs, extra_cuda_cflags=['--expt-extended-lambda'])
    return swish_mod

if not torch.cuda.is_available():
    print("CUDA not available but is required for swish_torch")
    swish_mod = None
else:
     swish_mod = load_module()

class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return swish_mod.swish_forward(inp)
    
    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        if not ctx.needs_input_grad[0]: return (None,)
        return swish_mod.swish_backward(inp, grad_out)
        
class Swish(torch.nn.Module):
    '''Swish Activation Function - Inline PyTorch CUDA Version'''
    def forward(self, inp): return SwishFunction.apply(inp)

swish = SwishFunction.apply

if swish_mod is not None:
    print(f"Successfully loaded swish-torch inline version {{__version__}}")

"""

src = src.replace("$CPP_SRCS$", 
                  ','.join((f"'{b64encode(compress(s.encode(),9)).decode()}'" for s in cpp_srcs)))
src = src.replace("$CU_SRCS$", 
                  ','.join((f"'{b64encode(compress(s.encode(),9)).decode()}'" for s in cu_srcs)))

if args.stdout:
    print(src)
else:
    with Path(args.output).open('w') as out:
        out.write(src)
