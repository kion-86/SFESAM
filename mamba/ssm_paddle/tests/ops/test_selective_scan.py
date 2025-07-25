# Copyright (C) 2023, Tri Dao.

import math

import paddle
import paddle.nn.functional as F
import pytest

from einops import rearrange

from mamba_ssm_paddle.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm_paddle.ops.selective_scan_interface import mamba_inner_fn, mamba_inner_ref

#######################################################################################################################################
# patch paddle.allclose
old_allclose = paddle.allclose
def allclose(a, b, **kwargs):
    return old_allclose(a.cast("float32"), b.cast("float32"), **kwargs)
paddle.allclose = allclose  

old_equal_all = paddle.equal_all
def equal_all(a, b):
    return old_equal_all(a.cast("float32"), b.cast("float32"))
paddle.equal_all = equal_all  

def requires_grad_(self, value=True):
    self.stop_gradient = not value
    return self

paddle.Tensor.requires_grad_ = requires_grad_
#######################################################################################################################################

@pytest.mark.parametrize('wtype', [paddle.float32])
# @pytest.mark.parametrize('wtype', [paddle.float32])
@pytest.mark.parametrize('itype', [paddle.float32, paddle.float16, paddle.bfloat16])
# @pytest.mark.parametrize('itype', [paddle.float32])
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 372, 512, 784, 1024, 1134, 2048, 4096])
@pytest.mark.parametrize('seqlen', [128, 256, 512, 1024, 2048, 4096])
# @pytest.mark.parametrize('seqlen', [128])
# @pytest.mark.parametrize("return_last_state", [False, True])
@pytest.mark.parametrize("return_last_state", [True])
# @pytest.mark.parametrize('has_delta_bias', [False, True])
@pytest.mark.parametrize('has_delta_bias', [True])
# @pytest.mark.parametrize('delta_softplus', [False, True])
@pytest.mark.parametrize('delta_softplus', [True])
# @pytest.mark.parametrize('has_z', [False, True])
@pytest.mark.parametrize('has_z', [True])
# @pytest.mark.parametrize('has_D', [False, True])
@pytest.mark.parametrize('has_D', [True])
@pytest.mark.parametrize("varBC_groups", [1, 2])
# @pytest.mark.parametrize("varBC_groups", [1])
# @pytest.mark.parametrize("is_variable_C", [False, True])
@pytest.mark.parametrize("is_variable_C", [True])
# @pytest.mark.parametrize("is_variable_B", [False, True])
@pytest.mark.parametrize("is_variable_B", [True])
def test_selective_scan(is_variable_B, is_variable_C, varBC_groups, has_D, has_z, has_delta_bias,
                        delta_softplus, return_last_state, seqlen, itype, wtype):
    if varBC_groups > 1 and (not is_variable_B or not is_variable_C):
        pytest.skip()  # This config is not applicable

    rtol, atol = (6e-4, 2e-3) if itype == paddle.float32 else (3e-3, 5e-3)
    if itype == paddle.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:  # If we have z, the errors on the weights seem higher
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)
    # set seed
    paddle.seed(0)
    batch_size = 2
    dim = 4
    dstate = 8
    is_complex = wtype == paddle.complex64

    if is_complex:
        A = (-0.5 * paddle.rand([dim, dstate], dtype="float32")).cast(wtype).requires_grad_()
    else:
        A = (-0.5 * paddle.rand([dim, dstate], dtype=wtype)).requires_grad_()
    if not is_variable_B:
        B_shape = (dim, dstate)
    elif varBC_groups == 1:
        B_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
    else:
        B_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    B = paddle.randn(B_shape, dtype=wtype if not is_variable_B else itype).requires_grad_()
    if not is_variable_C:
        C_shape = (dim, dstate)
    elif varBC_groups == 1:
        C_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
    else:
        C_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    C = paddle.randn(C_shape, dtype=wtype if not is_variable_C else itype).requires_grad_()
    if has_D:
        D = paddle.randn([dim,],dtype=paddle.float32).requires_grad_()
    else:
        D = None
    if has_z:
        z = paddle.randn([batch_size, dim, seqlen], dtype=itype).requires_grad_()
    else:
        z = None
    if has_delta_bias:
        delta_bias = (0.5 * paddle.rand([dim,], dtype=paddle.float32)).requires_grad_()
    else:
        delta_bias = None
    u = paddle.randn([batch_size, dim, seqlen], dtype=itype).requires_grad_()
    delta = (0.5 * paddle.rand([batch_size, dim, seqlen], dtype=itype)).requires_grad_()
    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    u_ref = u.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
    out, *rest = selective_scan_fn(
        u, delta, A, B, C, D, z=z,
        delta_bias=delta_bias, delta_softplus=delta_softplus,
        return_last_state=return_last_state
    )
    if return_last_state:
        state = rest[0]
    out_ref, *rest = selective_scan_ref(
        u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref, z=z_ref,
        delta_bias=delta_bias_ref, delta_softplus=delta_softplus,
        return_last_state=return_last_state
    )
    if return_last_state:
        state_ref = rest[0]
    # dA = paddle.exp(paddle.einsum('bdl,dn->bdln', delta, A))
    # dt_u = delta * u

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    assert paddle.allclose(out, out_ref, rtol=rtol, atol=atol)
    if return_last_state:
        print(f'State max diff: {(state - state_ref).abs().max().item()}')
        assert paddle.allclose(state, state_ref, rtol=rtol, atol=atol)

    g = paddle.randn(out.shape, dtype=out.dtype)
    out_ref.backward(g.cast(out_ref.dtype))
    out.backward(g.cast(out.dtype))

    print(f'du max diff: {(u.grad - u_ref.grad).abs().max().item()}')
    print(f'ddelta max diff: {(delta.grad - delta_ref.grad).abs().max().item()}')
    print(f'dA max diff: {(A.grad - A_ref.grad).abs().max().item()}')
    print(f'dB max diff: {(B.grad - B_ref.grad).abs().max().item()}')
    print(f'dC max diff: {(C.grad - C_ref.grad).abs().max().item()}')
    if has_D:
        print(f'dD max diff: {(D.grad - D_ref.grad).abs().max().item()}')
    if has_z:
        print(f'dz max diff: {(z.grad - z_ref.grad).abs().max().item()}')
    if has_delta_bias:
        print(f'ddelta_bias max diff: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}')

    assert paddle.allclose(u.grad, u_ref.grad.cast(dtype=itype), rtol=rtol * 2, atol=atol * 2)
    assert paddle.allclose(delta.grad, delta_ref.grad.cast(dtype=itype), rtol=rtol * 5, atol=atol * 10)
    assert paddle.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
    assert paddle.allclose(B.grad, B_ref.grad, rtol=rtolw if not is_variable_B else rtol,
                          atol=atolw if not is_variable_B else atol)
    assert paddle.allclose(C.grad, C_ref.grad, rtol=rtolw if not is_variable_C else rtol,
                          atol=atolw if not is_variable_C else atol)
    if has_D:
        assert paddle.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
    if has_z:
        assert paddle.allclose(z.grad, z_ref.grad, rtol=rtolw, atol=atolw)
    if has_delta_bias:
        assert paddle.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)


# @pytest.mark.parametrize('wtype', [paddle.float32])
@pytest.mark.parametrize('wtype', [paddle.float32])
@pytest.mark.parametrize('itype', [paddle.float32, paddle.float16, paddle.bfloat16])
# @pytest.mark.parametrize('itype', [paddle.float32])
# @pytest.mark.parametrize('seqlen', [8, 16, 32, 64, 128, 256, 372, 512, 784, 1024, 1134, 2048, 4096])
@pytest.mark.parametrize('seqlen', [128])
@pytest.mark.parametrize("is_variable_C", [False, True])
# @pytest.mark.parametrize("is_variable_C", [False])
@pytest.mark.parametrize("is_variable_B", [False, True])
# @pytest.mark.parametrize("is_variable_B", [True])
def test_mamba_inner_fn(is_variable_B, is_variable_C, seqlen, itype, wtype):
    rtol, atol = (6e-4, 2e-3) if itype == paddle.float32 else (3e-3, 5e-3)
    if itype == paddle.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # If we have z, the errors on the weights seem higher
    rtolw = max(rtolw, rtol)
    atolw = max(atolw, atol)
    # set seed
    paddle.seed(0)
    batch_size = 2
    dim = 768
    dstate = 8
    dt_rank = 48
    is_complex = wtype == paddle.complex64

    xz = paddle.randn([batch_size, 2 * dim, seqlen], dtype=itype).requires_grad_()
    conv1d_weight = paddle.randn([dim, 1, 3], dtype=paddle.float32).requires_grad_()
    conv1d_bias = paddle.randn([dim,], dtype=paddle.float32).requires_grad_()
    x_proj_weight = paddle.randn([dt_rank + (bool(is_variable_B) + bool(is_variable_C)) * dstate
                                * (1 if not is_complex else 2),
                                dim], dtype=itype).requires_grad_()
    delta_proj_weight = paddle.randn([dim, dt_rank], dtype=itype).requires_grad_()
    out_proj_weight = paddle.randn([dim // 2, dim], dtype=itype).requires_grad_()
    out_proj_bias = None

    if is_complex:
        A = (-0.5 * paddle.rand([dim, dstate], dtype="float32")).cast(wtype).requires_grad_()
    else:
        A = (-0.5 * paddle.rand([dim, dstate], dtype=wtype)).requires_grad_()
    B = (paddle.randn([dim, dstate], dtype=wtype).requires_grad_()
         if not is_variable_B else None)
    C = (paddle.randn([dim, dstate], dtype=wtype).requires_grad_()
         if not is_variable_C else None)
    D = paddle.randn([dim,], dtype=paddle.float32).requires_grad_()
    delta_bias = (0.5 * paddle.rand([dim,], dtype=paddle.float32)).requires_grad_()
    B_proj_bias = None
    C_proj_bias = None
    xz_ref = xz.detach().clone().requires_grad_()
    conv1d_weight_ref = conv1d_weight.detach().clone().requires_grad_()
    conv1d_bias_ref = conv1d_bias.detach().clone().requires_grad_()
    x_proj_weight_ref = x_proj_weight.detach().clone().requires_grad_()
    delta_proj_weight_ref = delta_proj_weight.detach().clone().requires_grad_()
    out_proj_weight_ref = out_proj_weight.detach().clone().requires_grad_()
    out_proj_bias_ref = (out_proj_bias.detach().clone().requires_grad_()
                         if out_proj_bias is not None else None)
    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_() if B is not None else None
    C_ref = C.detach().clone().requires_grad_() if C is not None else None
    D_ref = D.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
    out = mamba_inner_fn(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                         out_proj_weight, out_proj_bias,
                         A, B, C, D, delta_bias=delta_bias, delta_softplus=True)
    out_ref = mamba_inner_ref(xz_ref, conv1d_weight_ref, conv1d_bias_ref, x_proj_weight_ref,
                              delta_proj_weight_ref, out_proj_weight_ref, out_proj_bias_ref,
                              A_ref, B_ref, C_ref, D_ref,
                              delta_bias=delta_bias_ref, delta_softplus=True)
    # dA = paddle.exp(paddle.einsum('bdl,dn->bdln', delta, A))
    # dt_u = delta * u

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    assert paddle.allclose(out, out_ref, rtol=rtol, atol=atol)

    g = paddle.randn(out.shape, dtype=out.dtype)
    out_ref.backward(g.cast(out_ref.dtype))
    out.backward(g.cast(out.dtype))

    print(f'dxz max diff: {(xz.grad - xz_ref.grad).abs().max().item()}')
    print(f'dA max diff: {(A.grad - A_ref.grad).abs().max().item()}')
    if not is_variable_B:
        print(f'dB max diff: {(B.grad - B_ref.grad).abs().max().item()}')
    if not is_variable_C:
        print(f'dC max diff: {(C.grad - C_ref.grad).abs().max().item()}')
    print(f'dD max diff: {(D.grad - D_ref.grad).abs().max().item()}')
    print(f'ddelta_bias max diff: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}')
    print(f'dout_proj_weight max diff: {(out_proj_weight.grad - out_proj_weight_ref.grad).abs().max().item()}')
    print(f'ddelta_proj_weight max diff: {(delta_proj_weight.grad - delta_proj_weight_ref.grad).abs().max().item()}')
    print(f'dx_proj_weight max diff: {(x_proj_weight.grad - x_proj_weight_ref.grad).abs().max().item()}')
    print(f'dconv1d_weight max diff: {(conv1d_weight.grad - conv1d_weight_ref.grad).abs().max().item()}')
    print(f'dconv1d_bias max diff: {(conv1d_bias.grad - conv1d_bias_ref.grad).abs().max().item()}')

    # assert paddle.allclose(xz.grad, xz_ref.grad.cast(dtype=itype), rtol=rtol * 2, atol=atol * 2)
    # assert paddle.allclose(delta.grad, delta_ref.grad.cast(dtype=itype), rtol=rtol * 5, atol=atol * 10)
    # assert paddle.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
    # assert paddle.allclose(B.grad, B_ref.grad, rtol=rtolw if not is_variable_B else rtol,
    #                       atol=atolw if not is_variable_B else atol)
    # assert paddle.allclose(C.grad, C_ref.grad, rtol=rtolw if not is_variable_C else rtol,
    #                       atol=atolw if not is_variable_C else atol)
    # assert paddle.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
    # assert paddle.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)
