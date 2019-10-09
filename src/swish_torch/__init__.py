ALL = ['Swish','SwishFunction','swish']

import torch # Must import torch before C extension
from ._C import swish_forward, swish_backward

class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return swish_forward(inp)
    
    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        if not ctx.needs_input_grad[0]: return (None,)
        return swish_backward(inp, grad_out)
        
class Swish(torch.nn.Module):
    """Swish Activation Function - PyTorch CUDA Version"""
    def forward(self, inp): return SwishFunction.apply(inp)

swish = SwishFunction.apply