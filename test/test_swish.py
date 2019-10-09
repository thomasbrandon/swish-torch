import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_allclose

swish_forward_pt = lambda x: x.mul(torch.sigmoid(x))

class SwishPT(torch.nn.Module):
    def forward(self, x): return swish_forward_pt(x)

def get_input_params():
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    devs = ['cuda:0'] # TODO: Allow other devices
    dev_types = [(dtype,device)
                 for dtype in [torch.float16,torch.float32,torch.float64]
                 for device in devs
                 # Basic ops not supported on CPU/Half, could test by converting but skip for now
                 if not (dtype==torch.float16 and torch.device(device).type == 'cpu')] 
    inputs = [(ndim,dtype,device)
              for (dtype,device) in dev_types
              for ndim in [1,2,3,4,8]]
    return inputs

@pytest.fixture(params=get_input_params())
def test_input(request):
    ndim,dtype,device = request.param
    sz = (2,) * (ndim-1) + (10,)
    if device == 'cpu' and dtype == torch.float16:
        t = torch.randn(*sz).half() # No randn for half on CPU
    else: t = torch.randn(*sz, device=device, dtype=dtype)
    return t + torch.randint(-1000, 1000, sz, device=device, dtype=dtype)

def test_forward(test_input):
    from swish_torch import swish_forward
    res = swish_forward(test_input)
    exp = swish_forward_pt(test_input)
    assert_allclose(res, exp)

def get_grads(inp):
    y = swish_forward_pt(inp)
    l = y.mean()
    grad_out, = torch.autograd.grad(l, y, retain_graph=True)
    exp, = torch.autograd.grad(y, inp, grad_out, retain_graph=True)
    return grad_out, exp

def test_backward(test_input):
    from swish_torch import swish_backward
    x = test_input.requires_grad_()
    grad_out,exp = get_grads(test_input)
    res = swish_backward(test_input.detach(), grad_out)
    assert_allclose(res, exp)

def test_function(test_input):
    from swish_torch import SwishFunction
    x1,x2 = (test_input.clone().requires_grad_() for i in range(2))

    y1 = swish_forward_pt(x1)
    l1 = y1.mean()
    exp, = torch.autograd.grad(l1, x1)

    y2 = SwishFunction.apply(x2)
    l2 = y2.mean()
    res, = torch.autograd.grad(l2, x2)
    assert_allclose(res, exp)

def test_module(test_input):
    from swish_torch import Swish
    x1,x2 = (test_input.clone().requires_grad_() for i in range(2))

    m1 = SwishPT()
    y1 = m1(x1)
    l1 = y1.mean()
    exp, = torch.autograd.grad(l1, x1)

    m2 = Swish()
    y2 = m2(x2)
    l2 = y2.mean()
    res, = torch.autograd.grad(l2, x2)
    assert_allclose(res, exp)

def test_gradient():
    from swish_torch import SwishFunction
    inp = torch.randn(10, 10, dtype=torch.float64, requires_grad=True, device='cuda:0')
    assert torch.autograd.gradcheck(SwishFunction.apply, inp)

def test_gradgrad():
    from swish_torch import SwishFunction
    inp = torch.randn(10, 10, dtype=torch.float64, requires_grad=True, device='cuda:0')
    assert torch.autograd.gradgradcheck(SwishFunction.apply, inp)

def test_overlapping():
    '''Test handling of overlapping output tensors'''
    from swish_torch import swish_forward
    t = torch.randn(2, 10, device='cuda:0')
    t_o = t.as_strided((3,10), (5,1)) # OVerlapping
    t_c = t_o.contiguous()             # Contiguous
    o_o = swish_forward(t_o, torch.empty_like(t_o))
    o_c = swish_forward(t_c, torch.empty_like(t_c))
    assert torch.equal(o_o, o_c)
