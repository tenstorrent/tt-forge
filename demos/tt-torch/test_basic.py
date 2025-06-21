# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn
import pytest
import math
import threading

import tt_torch
from tt_torch.tools.verify import verify_module, verify_against_golden
from tt_torch.tools.utils import CompilerConfig
from tt_torch.tools.device_manager import DeviceManager


def test_return_same_tensors():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x, y

    verify_module(Basic(), input_shapes=[(256, 256), (256, 256)])


def test_return_rt_tensor_and_torch_tensors():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y, z):
            output1 = y
            output2 = torch.abs(x)
            return output1, output2, z

    verify_module(
        Basic(),
        input_shapes=[(5, 10), (5, 10), (5, 10)],
    )


def test_abs():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.abs(x)

    verify_module(Basic(), input_shapes=[(256, 256)])


def test_add():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.add(x, y)

    verify_module(Basic(), input_shapes=[(256, 256)] * 2)


# Runs the AddOp on all detected chips in parallel
def test_add_multidevice():
    class AddOp(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.add(x, y)

    num_devices = DeviceManager.get_num_available_devices()
    parent, device_list = DeviceManager.acquire_available_devices()
    assert (
        len(device_list) == num_devices
    ), "Number of devices is not equal to expected."
    threads = []
    compiler_configs = [CompilerConfig() for _ in range(num_devices)]
    for i in range(num_devices):
        cc = compiler_configs[i]
        device = device_list[i]
        mod = AddOp()
        input_shapes = [(256, 256)] * 2
        thread = threading.Thread(
            target=verify_module,
            kwargs={
                "mod": mod,
                "input_shapes": input_shapes,
                "compiler_config": cc,
                "devices": [device],
            },
        )
        threads.append(thread)

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    DeviceManager.release_parent_device(parent, cleanup_sub_devices=True)
    assert (
        len(DeviceManager.get_parent_devices()) == 0
    ), "Some devices are not released."


@pytest.mark.parametrize(
    ("input"),
    [
        (torch.tensor([3.0], dtype=torch.bfloat16)),
        (torch.tensor([6], dtype=torch.int32)),
        (torch.tensor([[1], [2]], dtype=torch.float32)),
    ],
)
def test_broadcast(input):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.expand(2, 8)

    verify_module(Basic(), inputs=[input])


def test_clamp_int_bound():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.clamp(x, min=-1, max=1)

    verify_module(Basic(), input_shapes=[(32, 32)], input_range=(-5, 5))


def test_clamp_float_bound():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.clamp(x, min=-1.0, max=1.0)

    verify_module(Basic(), input_shapes=[(32, 32)], input_range=(-5, 5))


def test_clamp_tensor():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, min, max):
            return torch.clamp(x, min=min, max=max)

    verify_module(
        Basic(), input_shapes=[(32, 32), (32, 32), (32, 32)], input_range=(-5, 5)
    )


def test_concat_dim0():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat((x, y), dim=0)

    verify_module(Basic(), input_shapes=[(32, 32), (64, 32)])


def test_concat_dim1():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat((x, y), dim=1)

    verify_module(Basic(), input_shapes=[(32, 32), (32, 64)])


def test_concat_dim2():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat((x, y), dim=2)

    verify_module(Basic(), input_shapes=[(32, 32, 32), (32, 32, 64)])


def test_concat_dim3():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.cat((x, y), dim=3)

    verify_module(Basic(), input_shapes=[(32, 32, 32, 32), (32, 32, 32, 64)])


@pytest.mark.skip(
    "Torch keeps the 'value' as dialect resource which are not processed."
)
def test_constant_ones():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.tensor([1.0, 1.0, 1.0, 1.0])

    verify_module(Basic(), input_shapes=[(1, 1)])


def test_convert():
    class Basic_toFloat(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.to(torch.float32)

    class Basic_toInt(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.to(torch.int32)

    verify_module(
        Basic_toFloat(), input_shapes=[(4, 4)], input_data_types=[torch.int32]
    )
    verify_module(
        Basic_toFloat(), input_shapes=[(4, 4)], input_data_types=[torch.float32]
    )
    verify_module(Basic_toInt(), input_shapes=[(4, 4)], input_data_types=[torch.int32])
    verify_module(
        Basic_toInt(),
        input_shapes=[(4, 4)],
        input_data_types=[torch.float32],
        input_range=(0, 60),
    )


@pytest.mark.parametrize(
    ("input_range", "input_shapes", "input_type"),
    [
        ((-0.5, 0.5), [(2, 2), (2, 2)], [torch.float32, torch.float32]),
        ((1, 10), [(32, 32), (32, 32)], [torch.bfloat16, torch.bfloat16]),
        ((1, 10), [(32, 32), (32, 32)], [torch.float32, torch.float32]),
    ],
)
def test_div(input_range, input_shapes, input_type):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x / y

    verify_module(
        Basic(),
        input_shapes=input_shapes,
        input_data_types=input_type,
        input_range=input_range,
        required_atol=0.1,
    )


def test_div_zero():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x / y

    input1 = torch.tensor([1, -2, 3, -4, 5, 6, -7, 18], dtype=torch.float32)
    input2 = torch.tensor([1, 0, 3, 0, -9, 10, -7, 12], dtype=torch.float32)
    verify_module(Basic(), inputs=[input1, input2])


def test_exp():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.exp(x)

    verify_module(Basic(), input_shapes=[(2, 2)], required_atol=3e-2)


def test_floor():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.floor(x)

    verify_module(Basic(), input_shapes=[(32, 32)], input_data_types=[torch.float32])


def test_linear():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(32, 64, bias=False)
            self.linear_b = nn.Linear(64, 64, bias=False)

        def forward(self, x):
            x = self.linear_a(x)
            x = self.linear_b(x)
            return x

    verify_module(Basic(), input_shapes=[(32, 32)])


def test_linear_with_bias():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = nn.Linear(32, 32)

        def forward(self, x):
            x = self.linear_a(x)
            return x

    verify_module(Basic(), input_shapes=[(32, 32)])


@pytest.mark.parametrize(
    ("input_type"),
    [
        ([torch.float32]),
        ([torch.bfloat16]),
        ([torch.int32]),
    ],
)
def test_constant_add(input_type):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + 1.0

    verify_module(Basic(), input_shapes=[(1, 768)], input_data_types=input_type)


@pytest.mark.parametrize(
    ("input_type"),
    [
        ([torch.float32]),
        ([torch.bfloat16]),
        ([torch.int32]),
    ],
)
def test_constant_multiply(input_type):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x * 3.0

    verify_module(Basic(), input_shapes=[(1, 768)], input_data_types=input_type)


def test_maximum():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.maximum(x, y)

    verify_module(Basic(), input_shapes=[(32, 32), (32, 32)], input_range=(-6, 6))


def test_multiply():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x * y

    verify_module(Basic(), input_shapes=[(32, 32), (32, 32)])


def test_negate():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return -x

    verify_module(Basic(), input_shapes=[(32, 32)], input_range=(-6, 6))


@pytest.mark.skip("keepdim=False is not supported")
def test_reduce_max():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.max(x)

    verify_module(Basic(), input_shapes=[(32, 32)], input_range=(-6, 6))


@pytest.mark.skip("keepdim=False is not supported")
def test_reduce_sum():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sum(x)

    verify_module(Basic(), input_shapes=[(32, 32)], input_range=(-6, 6))


def test_relu():
    pytest.xfail()

    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.relu(x)

    verify_module(Basic(), input_shapes=[(32, 32)])


def test_rsqrt():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.rsqrt(x)

    verify_module(
        Basic(), input_shapes=[(32, 32)], required_atol=3e-2, input_range=(0.1, 1)
    )


def test_sqrt():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sqrt(x)

    verify_module(
        Basic(), input_shapes=[(32, 32)], required_atol=3e-2, input_range=(0.1, 1)
    )


dim0_cases = []
for begin in torch.arange(10).tolist():
    for end in torch.arange(90, 100).tolist():
        dim0_cases.append((begin, end, 0))

dim1_cases = []
for begin in torch.arange(10).tolist():
    for end in torch.arange(90, 100).tolist():
        dim1_cases.append((begin, end, 1))

dim2_cases = []
for begin in torch.arange(0, 64, 32).tolist():
    for end in torch.arange(64, 128, 32).tolist():
        dim2_cases.append((begin, end, 2))

dim3_cases = []
for begin in torch.arange(0, 64, 32).tolist():
    for end in torch.arange(64, 128, 32).tolist():
        dim3_cases.append((begin, end, 3))


@pytest.mark.parametrize(
    "begin, end, dim", [*dim2_cases, *dim3_cases, *dim0_cases, *dim1_cases]
)
def test_slice(begin, end, dim):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a):
            if dim == 0:
                return a[begin:end, :, :, :]
            elif dim == 1:
                return a[:, begin:end, :, :]
            elif dim == 2:
                return a[:, :, begin:end, :]
            else:
                return a[:, :, :, begin:end]

    shape = [10, 10, 10, 10]
    shape[dim] = 128
    verify_module(Basic(), input_shapes=[shape])


def test_subtract():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x - y

    verify_module(Basic(), input_shapes=[(32, 32), (32, 32)], input_range=(-6, 6))


def test_transpose_2d():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.transpose(x, 0, 1)

    verify_module(Basic(), input_shapes=[(4, 8)], input_range=(-6, 6))


@pytest.mark.skip("TTNN does not support transpose for higher ranks/dimensions.")
def test_transpose_3d():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.transpose(x, 0, 1)

    verify_module(Basic(), input_shapes=[(4, 8, 4)], input_range=(-6, 6))


def test_multiple_ops():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            y = x + x
            z = y + y
            z = torch.argmax(z)
            return z

    cc = CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    verify_module(
        Basic(), input_shapes=[(256, 256)], compiler_config=cc, do_assert=False
    )


def test_unused_output():
    class Basic_var_only(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            var, mean = torch.var_mean(x)
            return var

    class Basic_mean_only(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            var, mean = torch.var_mean(x)
            return mean

    for module in [Basic_var_only, Basic_mean_only]:
        cc = CompilerConfig()
        cc.compile_depth = tt_torch.tools.utils.CompileDepth.COMPILE_OP_BY_OP
        verify_module(module(), input_shapes=[(256, 256)], compiler_config=cc)


def test_multiple_users():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x2 = x + x  # add op
            y1 = x2 + x  # user 1 of add op
            y2 = x2 + x  # user 2 of add op
            z = y1 + y2
            return z

    cc = CompilerConfig()
    cc.compile_depth = tt_torch.tools.utils.CompileDepth.EXECUTE_OP_BY_OP
    verify_module(
        Basic(), input_shapes=[(256, 256)], compiler_config=cc, do_assert=False
    )


@pytest.mark.parametrize(
    ("input_range", "input_shapes", "input_type"),
    [
        ((1, 10), [(32, 32), (32, 32)], [torch.float32, torch.float32]),
        ((1, 10), [(32, 32), (32, 32)], [torch.bfloat16, torch.bfloat16]),
        ((1, 10), [(3, 3), (3, 3)], [torch.float32, torch.float32]),
        ((1, 100), [(32, 32), (32, 32)], [torch.float32, torch.float32]),
        # This set of parameter can fail when we generate a right hand operand
        # which contains a 0. TTNN returns LHS operand instead of NaN in such
        # case. Issue: https://github.com/tenstorrent/tt-metal/issues/16394
        ((-100, 100), [(32, 32), (32, 32)], [torch.float32, torch.float32]),
    ],
)
def test_remainder_op(input_range, input_shapes, input_type):
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.remainder(x, y)

    verify_module(
        Basic(),
        input_shapes=input_shapes,
        input_data_types=input_type,
        input_range=input_range,
        required_atol=1,
    )


@pytest.mark.xfail(
    reason="TTNN returns LHS operand instead of NaN if divisor is 0, see https://github.com/tenstorrent/tt-metal/issues/16394",
)
def test_remainder_op_zero():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.remainder(x, y)

    input1 = torch.tensor([1, -2, 3, -4, 5, 6, -7, 18], dtype=torch.float32)
    input2 = torch.tensor([1, 0, 3, 0, -9, 10, -7, 12], dtype=torch.float32)
    verify_module(Basic(), inputs=[input1, input2])


def test_log_op():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.log(x)

    verify_module(
        Basic(), input_shapes=[(32, 32)], input_range=(0.1, 60), required_atol=0.02
    )


def test_ceil_op():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.ceil(x)

    verify_module(
        Basic(), input_shapes=[(32, 32)], input_range=(-6, 6), required_atol=0.02
    )


def test_sine_op():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sin(x)

    verify_module(
        Basic(), input_shapes=[(32, 32)], input_range=(-2 * math.pi, 2 * math.pi)
    )


def test_cosine_op():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.cos(x)

    verify_module(
        Basic(), input_shapes=[(32, 32)], input_range=(-2 * math.pi, 2 * math.pi)
    )


@pytest.mark.parametrize("assert_pcc", [True, False])
@pytest.mark.parametrize("assert_atol", [True, False])
def test_verify_against_golden_low_atol_high_pcc(assert_pcc, assert_atol):
    # High PCC and Low ATOL. Perfect numeric match case.
    # Under no conditions should an assertion error be raised

    # True ATOL: 0
    # True PCC: 1

    golden_tensors = (torch.tensor([1.0, 2.0, 3.0]),)
    calculated_tensors = (torch.tensor([1.0, 2.0, 3.0]),)
    pccs, atols, passed_pcc, passed_atol = verify_against_golden(
        golden_tensors,
        calculated_tensors,
        assert_pcc=assert_pcc,
        assert_atol=assert_atol,
        required_atol=0.1,
    )
    assert passed_pcc is True
    assert passed_atol is True


@pytest.mark.parametrize("assert_pcc", [True, False])
@pytest.mark.parametrize("assert_atol", [True, False])
def test_verify_against_golden_high_atol_high_pcc(assert_pcc, assert_atol):
    # High ATOL and High PCC  (correlated, but numerically different tensors).
    # Expected to raise ATOL asssertion if asserting atol, but not raise if ATOL not asserted

    # True ATOL: 0.5
    # True PCC: 1

    golden_tensors = (torch.tensor([1.0, 2.0, 3.0]),)
    calculated_tensors = (torch.tensor([1.5, 2.5, 3.5]),)

    if assert_atol:
        # Assert that an AssertionError is raised if we are asserting ATOL
        # when comparing tensors with a true high ATOL
        with pytest.raises(AssertionError) as e:
            pccs, atols, passed_pcc, passed_atol = verify_against_golden(
                golden_tensors,
                calculated_tensors,
                assert_pcc=assert_pcc,
                assert_atol=assert_atol,
                required_atol=0.1,
            )
    else:
        pccs, atols, passed_pcc, passed_atol = verify_against_golden(
            golden_tensors,
            calculated_tensors,
            assert_pcc=assert_pcc,
            assert_atol=assert_atol,
            required_atol=0.1,
        )

        # If ATOL is bad and PCC is good, and we don't assert on ATOL,
        # we expect no AssertionError to be raised by verify_against_golden

        # An implicit check here is that the above call to verify_against_golden
        # does not raise an AssertionError. If it does, the pytest will fail.

        assert passed_pcc is True
        assert passed_atol is False


@pytest.mark.parametrize("assert_pcc", [True, False])
@pytest.mark.parametrize("assert_atol", [True, False])
def test_verify_against_golden_low_atol_low_pcc(assert_pcc, assert_atol):
    # Low ATOL and Low PCC  (uncorrelated, but numerically close tensors)
    # Expected to raise PCC assert if asserting PCC, but not raise if PCC not asserted

    # True ATOL: 0.2
    # True PCC: 0.97 (Considered for this test as "LOW" PCC)

    golden_tensors = (torch.tensor([1.0, 2.0, 3.0]),)
    calculated_tensors = (torch.tensor([1.2, 1.8, 3.2]),)

    if assert_pcc:
        # Assert that an AssertionError is raised if we are asserting PCC
        # when comparing tensors with a true low PCC
        with pytest.raises(AssertionError) as e:
            pccs, atols, passed_pcc, passed_atol = verify_against_golden(
                golden_tensors,
                calculated_tensors,
                assert_pcc=assert_pcc,
                assert_atol=assert_atol,
                required_pcc=0.99,
                required_atol=0.25,
            )
    else:
        pccs, atols, passed_pcc, passed_atol = verify_against_golden(
            golden_tensors,
            calculated_tensors,
            assert_pcc=assert_pcc,
            assert_atol=assert_atol,
            required_pcc=0.99,
            required_atol=0.25,
        )
        assert passed_pcc is False
        assert passed_atol is True


@pytest.mark.parametrize("assert_pcc", [True, False])
@pytest.mark.parametrize("assert_atol", [True, False])
def test_verify_against_golden_high_atol_low_pcc(assert_pcc, assert_atol):
    # High ATOL and Low PCC  (uncorrelated and numerically different = completely different tensors)
    # Expected to fail if either PCC or ATOL is asserted

    # True ATOL: 2997
    # True PCC: 0.397

    golden_tensors = (torch.tensor([1.0, 2.0, 3.0]),)
    calculated_tensors = (torch.tensor([1000.0, -2000.0, 3000.0]),)

    if assert_pcc or assert_atol:
        # Assert that an AssertionError is raised if we are asserting either PCC or ATOL
        # In this case, both ATOL and PCC are truly bad, so asserting on either
        # should cause an AssertionError to be raised
        with pytest.raises(AssertionError) as e:
            pccs, atols, passed_pcc, passed_atol = verify_against_golden(
                golden_tensors,
                calculated_tensors,
                assert_pcc=assert_pcc,
                assert_atol=assert_atol,
                required_pcc=0.99,
                required_atol=0.1,
            )
    else:
        pccs, atols, passed_pcc, passed_atol = verify_against_golden(
            golden_tensors,
            calculated_tensors,
            assert_pcc=assert_pcc,
            assert_atol=assert_atol,
            required_pcc=0.99,
            required_atol=0.1,
        )

        assert passed_pcc is False
        assert passed_atol is False
