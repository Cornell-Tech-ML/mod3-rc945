# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function for CUDA device execution."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile a function for CUDA execution."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """See `tensor_ops.py`"""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Practice sum kernel for preparing reduction."""
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0
    cuda.syncthreads()
    j = 1
    while j < BLOCK_DIM:
        stride = j * 2
        if pos % stride == 0 and (pos + j) < BLOCK_DIM:
            cache[pos] += cache[pos + j]
        cuda.syncthreads()
        j = stride
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice sum kernel for preparing reduction."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)
            acc = reduce_value
            reduce_size = a_shape[reduce_dim]
            padded_size = 1
            while padded_size < reduce_size:
                padded_size *= 2
            for s in range(padded_size):
                if s < reduce_size:
                    out_index[reduce_dim] = s
                    j = index_to_position(out_index, a_strides)
                    acc = fn(acc, a_storage[j])
                else:
                    acc = fn(acc, reduce_value)
            cache[pos] = acc
        else:
            cache[pos] = reduce_value
        cuda.syncthreads()
        j = 1
        while j < BLOCK_DIM:
            stride = j * 2
            if pos % stride == 0 and (pos + j) < BLOCK_DIM:
                cache[pos] += cache[pos + j]
            cuda.syncthreads()
            j = stride
        if pos == 0:
            to_index(out_pos, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            out[o] = cache[0]

    return jit(_reduce)


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * BLOCK_DIM + ty
    col = cuda.blockIdx.x * BLOCK_DIM + tx

    if row < size and col < size:
        acc = 0.0
        for tile in range(0, size, BLOCK_DIM):
            if row < size and (tile + tx) < size:
                a_shared[ty, tx] = a[row * size + (tile + tx)]
            else:
                a_shared[ty, tx] = 0.0
            if col < size and (tile + ty) < size:
                b_shared[ty, tx] = b[(tile + ty) * size + col]
            else:
                b_shared[ty, tx] = 0.0
            cuda.syncthreads()
            for k in range(BLOCK_DIM):
                acc += a_shared[ty, k] * b_shared[k, tx]
            cuda.syncthreads()
        if row < size and col < size:
            out[row * size + col] = acc


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice square MM kernel to prepare for matmul."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function."""
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Initialize the accumulator for the dot product
    acc = 0.0

    # Iterate over tiles of the input matrices
    for tile in range(0, a_shape[-1], BLOCK_DIM):
        # Load a tile of matrix `a` into shared memory
        if i < a_shape[-2] and (tile + pj) < a_shape[-1]:
            a_index = (
                batch * a_batch_stride + i * a_strides[1] + (tile + pj) * a_strides[2]
            )
            a_shared[pi, pj] = a_storage[a_index]
        else:
            a_shared[pi, pj] = 0.0

        # Load a tile of matrix `b` into shared memory
        if j < b_shape[-1] and (tile + pi) < b_shape[-2]:
            b_index = (
                batch * b_batch_stride + (tile + pi) * b_strides[1] + j * b_strides[2]
            )
            b_shared[pi, pj] = b_storage[b_index]
        else:
            b_shared[pi, pj] = 0.0

        # Synchronize threads to ensure all data is loaded
        cuda.syncthreads()

        # Perform the dot product for the current tile
        if i < out_shape[-2] and j < out_shape[-1]:
            for k in range(BLOCK_DIM):
                acc += a_shared[pi, k] * b_shared[k, pj]
            cuda.syncthreads()

    # Write the result to the output matrix
    if i < out_shape[-2] and j < out_shape[-1]:
        out_index = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[out_index] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)


@cuda.jit
def _tensor_matrix_multiply_backward(
    grad_output_storage: Storage,
    grad_output_shape: Shape,
    grad_output_strides: Strides,
    grad_a_storage: Storage,
    grad_a_shape: Shape,
    grad_a_strides: Strides,
    grad_b_storage: Storage,
    grad_b_shape: Shape,
    grad_b_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA kernel for backward pass of matrix multiplication."""
    batch, row, col = cuda.grid(3)

    if (
        batch >= grad_output_shape[0]
        or row >= grad_output_shape[1]
        or col >= grad_output_shape[2]
    ):
        return

    # Calculate gradients for a
    acc_a = 0.0
    for k in range(b_shape[1]):
        b_val = b_storage[batch * b_strides[0] + k * b_strides[1] + col * b_strides[2]]
        grad_out = grad_output_storage[
            batch * grad_output_strides[0]
            + row * grad_output_strides[1]
            + k * grad_output_strides[2]
        ]
        acc_a += grad_out * b_val
    grad_a_storage[
        batch * grad_a_strides[0] + row * grad_a_strides[1] + col * grad_a_strides[2]
    ] = acc_a

    # Calculate gradients for b
    acc_b = 0.0
    for k in range(a_shape[1]):
        a_val = a_storage[batch * a_strides[0] + row * a_strides[1] + k * a_strides[2]]
        grad_out = grad_output_storage[
            batch * grad_output_strides[0]
            + k * grad_output_strides[1]
            + col * grad_output_strides[2]
        ]
        acc_b += grad_out * a_val
    grad_b_storage[
        batch * grad_b_strides[0] + row * grad_b_strides[1] + col * grad_b_strides[2]
    ] = acc_b


tensor_matrix_multiply_backward = cuda.jit(_tensor_matrix_multiply_backward)


@cuda.jit
def _broadcast_backward(
    out_storage: Storage,
    out_shape: Shape,
    out_strides: Strides,
    grad_storage: Storage,
    grad_shape: Shape,
    grad_strides: Strides,
    reduce_dims: Shape,
) -> None:
    """CUDA kernel for backward pass of broadcast operations."""
    idx = cuda.grid(1)

    if idx >= len(out_storage):
        return

    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    to_index(idx, out_shape, out_index)

    grad_index = cuda.local.array(MAX_DIMS, numba.int32)
    broadcast_index(out_index, out_shape, grad_shape, grad_index)

    grad_pos = index_to_position(grad_index, grad_strides)
    out_storage[idx] = grad_storage[grad_pos]


broadcast_backward = cuda.jit(_broadcast_backward)


def get_grid_dim(size: int, block_dim: int) -> int:
    """Calculate grid dimension ensuring full coverage of the input size."""
    return (size + block_dim - 1) // block_dim


def launch_kernel(kernel: FakeCUDAKernel, size: int, *args: Any) -> None:
    """Launch CUDA kernel with optimal thread configuration."""
    threads_per_block = 256  # This can be tuned based on your GPU
    blocks_per_grid = get_grid_dim(size, threads_per_block)
    kernel[blocks_per_grid, threads_per_block](*args)
