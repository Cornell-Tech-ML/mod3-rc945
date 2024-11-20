# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html

You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (164)
---------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                          |
        out: Storage,                                                                  |
        out_shape: Shape,                                                              |
        out_strides: Strides,                                                          |
        in_storage: Storage,                                                           |
        in_shape: Shape,                                                               |
        in_strides: Strides,                                                           |
    ) -> None:                                                                         |
        # TODO: Implement for Task 3.1.                                                |
        if np.array_equal(out_strides, in_strides) and len(in_storage) >= len(out):    |
            for i in prange(len(out)):-------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                             |
        else:                                                                          |
            for i in prange(len(out)):-------------------------------------------------| #3
                out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)------------------| #0
                in_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)-------------------| #1
                to_index(i, out_shape, out_index)                                      |
                broadcast_index(out_index, out_shape, in_shape, in_index)              |
                o = index_to_position(out_index, out_strides)                          |
                j = index_to_position(in_index, in_strides)                            |
                out[o] = fn(in_storage[j])                                             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)



Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (178) is hoisted out
of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (179) is hoisted out
of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: in_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (212)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (212)
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              |
        out: Storage,                                                      |
        out_shape: Shape,                                                  |
        out_strides: Strides,                                              |
        a_storage: Storage,                                                |
        a_shape: Shape,                                                    |
        a_strides: Strides,                                                |
        b_storage: Storage,                                                |
        b_shape: Shape,                                                    |
        b_strides: Strides,                                                |
    ) -> None:                                                             |
        # TODO: Implement for Task 3.1.                                    |
        if (                                                               |
            np.array_equal(out_strides, a_strides)                         |
            and len(a_storage) >= len(out)                                 |
            and np.array_equal(a_strides, b_strides)                       |
            and len(b_storage) >= len(out)                                 |
        ):                                                                 |
            for i in prange(len(out)):-------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                    |
        else:                                                              |
            for i in prange(len(out)):-------------------------------------| #8
                out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)------| #4
                a_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------| #5
                b_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------| #6
                to_index(i, out_shape, out_index)                          |
                o = index_to_position(out_index, out_strides)              |
                broadcast_index(out_index, out_shape, a_shape, a_index)    |
                j = index_to_position(a_index, a_strides)                  |
                broadcast_index(out_index, out_shape, b_shape, b_index)    |
                k = index_to_position(b_index, b_strides)                  |
                out[o] = fn(a_storage[j], b_storage[k])                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)



Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (234) is hoisted out
of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (235) is hoisted out
of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (236) is hoisted out
of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: b_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (269)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (269)
---------------------------------------------------------------------|loop #ID
    def _reduce(                                                     |
        out: Storage,                                                |
        out_shape: Shape,                                            |
        out_strides: Strides,                                        |
        a_storage: Storage,                                          |
        a_shape: Shape,                                              |
        a_strides: Strides,                                          |
        reduce_dim: int,                                             |
    ) -> None:                                                       |
        # TODO: Implement for Task 3.1.                              |
        for i in prange(len(out)):-----------------------------------| #10
            out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)----| #9
            reduce_size = a_shape[reduce_dim]                        |
            to_index(i, out_shape, out_index)                        |
            o = index_to_position(out_index, out_strides)            |
            acc = out[o]                                             |
            for s in range(reduce_size):                             |
                out_index[reduce_dim] = s                            |
                j = index_to_position(out_index, a_strides)          |
                acc = fn(acc, a_storage[j])                          |
            out[o] = acc                                             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)



Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (280) is hoisted out
of the parallel loop labelled #10 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (294)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/ryanchen/workspace/mod3-rc945/minitorch/fast_ops.py (294)
--------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                          |
    out: Storage,                                                                     |
    out_shape: Shape,                                                                 |
    out_strides: Strides,                                                             |
    a_storage: Storage,                                                               |
    a_shape: Shape,                                                                   |
    a_strides: Strides,                                                               |
    b_storage: Storage,                                                               |
    b_shape: Shape,                                                                   |
    b_strides: Strides,                                                               |
) -> None:                                                                            |
    """NUMBA tensor matrix multiply function.                                         |
                                                                                      |
    Should work for any tensor shapes that broadcast as long as                       |
                                                                                      |
    ```                                                                               |
    assert a_shape[-1] == b_shape[-2]                                                 |
    ```                                                                               |
                                                                                      |
    Optimizations:                                                                    |
                                                                                      |
    * Outer loop in parallel                                                          |
    * No index buffers or function calls                                              |
    * Inner loop should have no global writes, 1 multiply.                            |
                                                                                      |
                                                                                      |
    Args:                                                                             |
    ----                                                                              |
        out (Storage): storage for `out` tensor                                       |
        out_shape (Shape): shape for `out` tensor                                     |
        out_strides (Strides): strides for `out` tensor                               |
        a_storage (Storage): storage for `a` tensor                                   |
        a_shape (Shape): shape for `a` tensor                                         |
        a_strides (Strides): strides for `a` tensor                                   |
        b_storage (Storage): storage for `b` tensor                                   |
        b_shape (Shape): shape for `b` tensor                                         |
        b_strides (Strides): strides for `b` tensor                                   |
                                                                                      |
    Returns:                                                                          |
    -------                                                                           |
        None : Fills in `out`                                                         |
                                                                                      |
    """                                                                               |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                            |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                            |
                                                                                      |
    # TODO: Implement for Task 3.2.                                                   |
    m = a_shape[-2]  # Rows of A                                                      |
    n = b_shape[-1]  # Columns of B                                                   |
    k = a_shape[-1]  # Shared dimension                                               |
    for i in prange(len(out)):--------------------------------------------------------| #11
        batch = (i // (m * n)) if len(out_shape) > 2 else 0                           |
        r = (i // n) % m                                                              |
        c = i % n                                                                     |
                                                                                      |
        acc = 0.0                                                                     |
        for j in range(k):                                                            |
            a_pos = batch * a_batch_stride + r * a_strides[-2] + j * a_strides[-1]    |
            b_pos = batch * b_batch_stride + j * b_strides[-2] + c * b_strides[-1]    |
            acc += a_storage[a_pos] * b_storage[b_pos]                                |
                                                                                      |
        out[i] = acc                                                                  |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
Simple cpu
```
Epoch  0  loss  3.1626240169760202 correct 44 time 18.386573791503906
Epoch  10  loss  1.9135774949948257 correct 48 time 0.9120252132415771
Epoch  20  loss  1.024653362878196 correct 49 time 0.8055679798126221
Epoch  30  loss  0.4755160067226361 correct 50 time 0.8667922019958496
Epoch  40  loss  0.39266169760557196 correct 49 time 0.8039829730987549
Epoch  50  loss  0.5298750221841649 correct 50 time 0.8045721054077148
Epoch  60  loss  0.35059566127911945 correct 50 time 0.8088963031768799
Epoch  70  loss  1.672400227465523 correct 50 time 0.9865195751190186
Epoch  80  loss  0.5325783354036671 correct 50 time 0.8107705116271973
Epoch  90  loss  0.040020120629786486 correct 50 time 0.8081347942352295
Epoch  100  loss  0.05199265739649817 correct 50 time 1.404463768005371
Epoch  110  loss  0.04858543013534456 correct 50 time 0.8079681396484375
Epoch  120  loss  0.28154772007692475 correct 50 time 0.8001449108123779
Epoch  130  loss  0.4547800550730256 correct 50 time 0.7956905364990234
Epoch  140  loss  0.020238487191109756 correct 50 time 0.7910075187683105
Epoch  150  loss  0.2772892749883238 correct 50 time 0.8068044185638428
Epoch  160  loss  0.01706888292799087 correct 50 time 0.7999508380889893
Epoch  170  loss  0.01788801019971458 correct 50 time 1.5112018585205078
Epoch  180  loss  0.060786772845282416 correct 50 time 0.8062188625335693
Epoch  190  loss  0.2869385900783748 correct 50 time 0.8097500801086426
Epoch  200  loss  0.014310497492315033 correct 50 time 0.804821252822876
Epoch  210  loss  0.322109562999112 correct 50 time 0.821152925491333
Epoch  220  loss  0.10270424093418416 correct 50 time 0.8044784069061279
Epoch  230  loss  0.17649649841455203 correct 50 time 0.827538013458252
Epoch  240  loss  0.29538668471016394 correct 50 time 1.5099186897277832
Epoch  250  loss  0.08358819165924121 correct 50 time 0.8070518970489502
Epoch  260  loss  0.027541426067433598 correct 50 time 0.808873176574707
Epoch  270  loss  0.21630295751976134 correct 50 time 0.9651284217834473
Epoch  280  loss  0.2147590213578748 correct 50 time 0.8014655113220215
Epoch  290  loss  0.09645849943123876 correct 50 time 0.8007340431213379
Epoch  300  loss  0.004134098025004724 correct 50 time 1.7272543907165527
Epoch  310  loss  0.11038833969125973 correct 50 time 0.8028888702392578
Epoch  320  loss  0.12623258699371673 correct 50 time 0.8070952892303467
Epoch  330  loss  0.14924023470800643 correct 50 time 0.805065393447876
Epoch  340  loss  0.21040580737812462 correct 50 time 0.8103446960449219
Epoch  350  loss  0.07439065838284631 correct 50 time 1.4424943923950195
Epoch  360  loss  0.19433763266224982 correct 50 time 0.803516149520874
Epoch  370  loss  0.003438866195370395 correct 50 time 0.8057324886322021
Epoch  380  loss  0.10247144635800048 correct 50 time 0.7911677360534668
Epoch  390  loss  0.11873469581459123 correct 50 time 0.7995164394378662
Epoch  400  loss  0.023428236766334334 correct 50 time 0.8066303730010986
Epoch  410  loss  0.13291005168731176 correct 50 time 0.8126194477081299
Epoch  420  loss  0.21301934689356194 correct 50 time 1.1210358142852783
Epoch  430  loss  0.12162720846363391 correct 50 time 0.8036820888519287
Epoch  440  loss  0.16012733060756526 correct 50 time 0.8013119697570801
Epoch  450  loss  0.17161175388865282 correct 50 time 1.2515368461608887
Epoch  460  loss  0.004995150591797067 correct 50 time 0.8098039627075195
Epoch  470  loss  0.003008699326275551 correct 50 time 0.7879974842071533
Epoch  480  loss  0.023944267065781673 correct 50 time 0.8048434257507324
Epoch  490  loss  0.03533912910435058 correct 50 time 0.799358606338501
```
simple gpu
```
Epoch  0  loss  4.378489495411926 correct 40 time 9.99264907836914
Epoch  10  loss  1.9279086689019909 correct 47 time 0.3753237724304199
Epoch  20  loss  2.0371178291730185 correct 48 time 0.3740830421447754
Epoch  30  loss  0.237235908608112 correct 50 time 0.3751029968261719
Epoch  40  loss  0.603761711901372 correct 49 time 0.38115501403808594
Epoch  50  loss  1.6669349588965454 correct 49 time 0.36682605743408203
Epoch  60  loss  0.990253643939725 correct 49 time 0.3777139186859131
Epoch  70  loss  0.1999822788344603 correct 50 time 0.3705940246582031
Epoch  80  loss  0.7782431043621397 correct 50 time 0.36463379859924316
Epoch  90  loss  0.11309375314148085 correct 50 time 0.36631298065185547
Epoch  100  loss  0.06036096571652118 correct 49 time 0.41874194145202637
Epoch  110  loss  0.08064745335180129 correct 50 time 0.3763008117675781
Epoch  120  loss  0.18300886812395098 correct 49 time 0.37854790687561035
Epoch  130  loss  1.047600593096785 correct 49 time 0.3697049617767334
Epoch  140  loss  0.6233548162316381 correct 49 time 0.3822000026702881
Epoch  150  loss  0.20813135228755247 correct 50 time 0.3715178966522217
Epoch  160  loss  0.5847746671574272 correct 49 time 0.42801475524902344
Epoch  170  loss  0.1513084090048782 correct 49 time 0.37445497512817383
Epoch  180  loss  0.6776938818191628 correct 49 time 0.372053861618042
Epoch  190  loss  0.6639917209924203 correct 50 time 0.3735208511352539
Epoch  200  loss  1.0446133461070821 correct 50 time 0.4044499397277832
Epoch  210  loss  0.290397985212438 correct 49 time 0.37520599365234375
Epoch  220  loss  0.19003411411752735 correct 50 time 0.3740828037261963
Epoch  230  loss  0.8541012767435149 correct 50 time 0.4183478355407715
Epoch  240  loss  0.45105077546435945 correct 50 time 0.3739011287689209
Epoch  250  loss  0.471092006232462 correct 49 time 0.38414478302001953
Epoch  260  loss  0.8511309946951314 correct 50 time 0.36591315269470215
Epoch  270  loss  0.06324139609066415 correct 49 time 0.3709690570831299
Epoch  280  loss  0.18469938782772347 correct 50 time 0.36800217628479004
Epoch  290  loss  0.5584956332714264 correct 50 time 0.37105798721313477
Epoch  300  loss  0.18320540360139792 correct 50 time 0.36904311180114746
Epoch  310  loss  0.4550787370498923 correct 49 time 0.4336690902709961
Epoch  320  loss  0.08007790763238602 correct 50 time 0.3833789825439453
Epoch  330  loss  0.002014795751600665 correct 50 time 0.3762779235839844
Epoch  340  loss  0.039900796181834064 correct 50 time 0.38905882835388184
Epoch  350  loss  0.9210736061205074 correct 50 time 0.3682596683502197
Epoch  360  loss  0.03020497678761159 correct 50 time 0.3630790710449219
Epoch  370  loss  0.04322177395040902 correct 50 time 0.3734090328216553
Epoch  380  loss  0.0432200268366313 correct 49 time 0.37781500816345215
Epoch  390  loss  0.553342115419156 correct 50 time 0.36559271812438965
Epoch  400  loss  0.7399504396245166 correct 50 time 0.3643488883972168
Epoch  410  loss  0.8114214538747404 correct 50 time 0.3708338737487793
Epoch  420  loss  0.07726958266357356 correct 50 time 0.36328601837158203
Epoch  430  loss  0.007987021830365922 correct 49 time 0.36746978759765625
Epoch  440  loss  0.006594130218126309 correct 50 time 0.3619530200958252
Epoch  450  loss  0.005178922729074319 correct 50 time 0.4436619281768799
Epoch  460  loss  0.0842163766092263 correct 50 time 0.40578198432922363
Epoch  470  loss  0.029557698826562003 correct 50 time 0.3669290542602539
Epoch  480  loss  0.7131573576834861 correct 50 time 0.3667318820953369
Epoch  490  loss  0.7756344811752431 correct 50 time 0.40368103981018066
```
split cpu
```
Epoch  0  loss  6.3238076091986075 correct 23 time 18.415794610977173
Epoch  10  loss  4.269255348129892 correct 46 time 0.9089338779449463
Epoch  20  loss  3.6702469475348845 correct 45 time 0.8121097087860107
Epoch  30  loss  3.758155418508016 correct 42 time 0.8618545532226562
Epoch  40  loss  2.9265411568414055 correct 47 time 0.7945926189422607
Epoch  50  loss  2.534279552312569 correct 49 time 0.8069314956665039
Epoch  60  loss  1.440983439556574 correct 49 time 0.8081021308898926
Epoch  70  loss  1.9433767624556084 correct 48 time 1.0245392322540283
Epoch  80  loss  1.5693890743800194 correct 50 time 0.810513973236084
Epoch  90  loss  2.2716073919290634 correct 48 time 0.804344892501831
Epoch  100  loss  1.7507936971751834 correct 50 time 1.4294962882995605
Epoch  110  loss  1.3699668864233896 correct 49 time 0.7879157066345215
Epoch  120  loss  0.3187453764634903 correct 49 time 0.816856861114502
Epoch  130  loss  0.25448144796912653 correct 50 time 0.808372974395752
Epoch  140  loss  1.5446764847976133 correct 50 time 0.8068227767944336
Epoch  150  loss  0.830751592043018 correct 50 time 0.8211147785186768
Epoch  160  loss  0.8734371882053978 correct 50 time 0.8073709011077881
Epoch  170  loss  0.5802417426625033 correct 50 time 1.4356281757354736
Epoch  180  loss  0.13503263852898473 correct 50 time 0.7929656505584717
Epoch  190  loss  0.9514237969517765 correct 50 time 0.8126349449157715
Epoch  200  loss  0.4249175838643865 correct 50 time 0.8049037456512451
Epoch  210  loss  1.384219238630622 correct 50 time 0.8107993602752686
Epoch  220  loss  0.8492731805278904 correct 50 time 0.8201999664306641
Epoch  230  loss  0.49633087423383787 correct 50 time 0.8113267421722412
Epoch  240  loss  0.15468533874816118 correct 50 time 1.3360133171081543
Epoch  250  loss  0.8627267439500891 correct 50 time 0.7926239967346191
Epoch  260  loss  0.36029217416497983 correct 50 time 0.8324053287506104
Epoch  270  loss  0.684469925423946 correct 50 time 0.8120763301849365
Epoch  280  loss  0.7228818155040235 correct 50 time 0.8059239387512207
Epoch  290  loss  0.6035359380661378 correct 50 time 0.8051962852478027
Epoch  300  loss  0.215686090727972 correct 50 time 0.786566972732544
Epoch  310  loss  0.5528502103179591 correct 50 time 1.6758861541748047
Epoch  320  loss  0.8567713448669784 correct 50 time 0.8048267364501953
Epoch  330  loss  0.13468907026514637 correct 50 time 0.8081479072570801
Epoch  340  loss  0.1216131878269146 correct 50 time 0.8035507202148438
Epoch  350  loss  0.5172845800478035 correct 50 time 0.8029003143310547
Epoch  360  loss  0.09613623419131268 correct 50 time 0.8015637397766113
Epoch  370  loss  0.09335317483188378 correct 50 time 0.8027079105377197
Epoch  380  loss  0.33096603310042183 correct 50 time 1.2711904048919678
Epoch  390  loss  0.24689345928809378 correct 50 time 0.7948644161224365
Epoch  400  loss  0.3024354288337298 correct 50 time 0.7941219806671143
Epoch  410  loss  0.07258214690368815 correct 50 time 1.288356065750122
Epoch  420  loss  0.33667778619791816 correct 50 time 0.8087484836578369
Epoch  430  loss  0.2227311026737961 correct 50 time 0.8032732009887695
Epoch  440  loss  0.11542533935201231 correct 50 time 0.8019256591796875
Epoch  450  loss  0.45846659183530825 correct 50 time 0.8299582004547119
Epoch  460  loss  0.1008002808901957 correct 50 time 0.8058984279632568
Epoch  470  loss  0.11433890099982175 correct 50 time 0.7944555282592773
Epoch  480  loss  0.4418163715008422 correct 50 time 0.8115129470825195
Epoch  490  loss  0.23711731905431407 correct 50 time 1.0963194370269775
```
split gpu
```
Epoch  0  loss  7.945405488105599 correct 23 time 10.004431962966919
Epoch  10  loss  6.238369488709076 correct 33 time 0.3932018280029297
Epoch  20  loss  5.617331378630725 correct 41 time 0.37520694732666016
Epoch  30  loss  3.486818750196696 correct 40 time 0.3871128559112549
Epoch  40  loss  4.742429398632613 correct 36 time 0.37733888626098633
Epoch  50  loss  4.133783377863446 correct 44 time 0.36904382705688477
Epoch  60  loss  4.896970219151994 correct 47 time 0.37683606147766113
Epoch  70  loss  3.961255536146835 correct 47 time 0.379702091217041
Epoch  80  loss  2.1010705591180034 correct 48 time 0.3880331516265869
Epoch  90  loss  2.834195083190949 correct 49 time 0.43248581886291504
Epoch  100  loss  1.7792897508993495 correct 45 time 0.3792991638183594
Epoch  110  loss  2.554350719313012 correct 48 time 0.3833179473876953
Epoch  120  loss  3.3539286895137113 correct 48 time 0.38200902938842773
Epoch  130  loss  0.911698501539757 correct 48 time 0.37702178955078125
Epoch  140  loss  1.0501862969486868 correct 49 time 0.38130784034729004
Epoch  150  loss  2.533361391840603 correct 49 time 0.3994109630584717
Epoch  160  loss  2.447428301005152 correct 47 time 0.3769369125366211
Epoch  170  loss  1.0078337956497714 correct 50 time 0.37496185302734375
Epoch  180  loss  1.0891190637683592 correct 50 time 0.375385046005249
Epoch  190  loss  1.5756194478965826 correct 50 time 0.3757510185241699
Epoch  200  loss  1.5090326464513186 correct 50 time 0.37291693687438965
Epoch  210  loss  0.46507015530756507 correct 50 time 0.36673617362976074
Epoch  220  loss  0.6078873600802163 correct 50 time 0.3678429126739502
Epoch  230  loss  0.8025790296576139 correct 50 time 0.3681190013885498
Epoch  240  loss  0.2677684293710117 correct 50 time 0.36591482162475586
Epoch  250  loss  0.821038018028071 correct 50 time 0.36634230613708496
Epoch  260  loss  0.5700834783331729 correct 50 time 0.38416600227355957
Epoch  270  loss  0.3824397001371795 correct 50 time 0.36653685569763184
Epoch  280  loss  0.7293691595758135 correct 50 time 0.36618614196777344
Epoch  290  loss  0.21866636659599234 correct 50 time 0.36631083488464355
Epoch  300  loss  0.7036365965537603 correct 50 time 0.36779189109802246
Epoch  310  loss  0.37651509220524293 correct 50 time 0.36545467376708984
Epoch  320  loss  0.41765006550342726 correct 50 time 0.36845970153808594
Epoch  330  loss  0.35555655201652375 correct 50 time 0.36527490615844727
Epoch  340  loss  0.13717139218709848 correct 50 time 0.3681671619415283
Epoch  350  loss  0.3152615660927212 correct 50 time 0.36745381355285645
Epoch  360  loss  0.33864993444115055 correct 50 time 0.37141990661621094
Epoch  370  loss  0.48559670255786047 correct 50 time 0.3669891357421875
Epoch  380  loss  0.7389235477360508 correct 50 time 0.3661372661590576
Epoch  390  loss  0.1588504735600371 correct 50 time 0.37189388275146484
Epoch  400  loss  0.38080232786825063 correct 50 time 0.3665947914123535
Epoch  410  loss  0.1955011377103912 correct 50 time 0.3673830032348633
Epoch  420  loss  0.1702826081955639 correct 50 time 0.3656308650970459
Epoch  430  loss  0.16802138904548247 correct 50 time 0.3659968376159668
Epoch  440  loss  0.2399342819613357 correct 50 time 0.36551594734191895
Epoch  450  loss  0.11726640046392484 correct 50 time 0.3678300380706787
Epoch  460  loss  0.3101219083686918 correct 50 time 0.36598896980285645
Epoch  470  loss  0.18474876899704898 correct 50 time 0.3661789894104004
Epoch  480  loss  0.25355735759414516 correct 50 time 0.3657400608062744
Epoch  490  loss  0.2693718872095852 correct 50 time 0.36897778511047363
```
xor cpu
```
Epoch  0  loss  8.576277658596071 correct 25 time 18.517525911331177
Epoch  10  loss  3.946852705734696 correct 41 time 0.9286439418792725
Epoch  20  loss  3.6940450914994107 correct 46 time 0.7967078685760498
Epoch  30  loss  4.344224552290742 correct 41 time 0.8099961280822754
Epoch  40  loss  1.7712114444322078 correct 46 time 0.806682825088501
Epoch  50  loss  3.297684973118515 correct 44 time 0.8058395385742188
Epoch  60  loss  2.4018204885729912 correct 45 time 0.8037869930267334
Epoch  70  loss  2.6628491995884205 correct 48 time 1.4597084522247314
Epoch  80  loss  1.5656646242151535 correct 46 time 0.8076674938201904
Epoch  90  loss  1.3065880829903707 correct 46 time 0.793675422668457
Epoch  100  loss  1.5754865679718404 correct 48 time 0.8007335662841797
Epoch  110  loss  1.0494052396416313 correct 48 time 0.7936201095581055
Epoch  120  loss  3.7107449195578135 correct 49 time 0.8075647354125977
Epoch  130  loss  1.421405039209054 correct 47 time 0.8130440711975098
Epoch  140  loss  1.3044474475418166 correct 48 time 1.2856483459472656
Epoch  150  loss  0.709908601561297 correct 49 time 0.8112397193908691
Epoch  160  loss  1.968585815844672 correct 48 time 0.8046789169311523
Epoch  170  loss  0.7602024581866647 correct 49 time 1.1241505146026611
Epoch  180  loss  1.410643232217226 correct 50 time 0.7986133098602295
Epoch  190  loss  1.6282744098499773 correct 49 time 0.8114516735076904
Epoch  200  loss  0.5389116525201689 correct 50 time 0.8073780536651611
Epoch  210  loss  0.6899064529742183 correct 49 time 0.9213676452636719
Epoch  220  loss  1.1052214832877625 correct 50 time 0.8081598281860352
Epoch  230  loss  0.348996538610735 correct 50 time 0.8144593238830566
Epoch  240  loss  1.189739486577284 correct 50 time 1.45562744140625
Epoch  250  loss  0.7306107778189564 correct 50 time 0.8117122650146484
Epoch  260  loss  0.6078141929294881 correct 50 time 0.8146741390228271
Epoch  270  loss  1.2333196830094784 correct 49 time 0.8177368640899658
Epoch  280  loss  1.168621498469806 correct 49 time 0.8100852966308594
Epoch  290  loss  0.32465694085644115 correct 50 time 0.8182463645935059
Epoch  300  loss  0.236542910301833 correct 50 time 0.7994608879089355
Epoch  310  loss  0.6188081943098289 correct 50 time 1.5248544216156006
Epoch  320  loss  0.18212613614197035 correct 50 time 0.8057270050048828
Epoch  330  loss  0.09441537507068977 correct 50 time 0.8134546279907227
Epoch  340  loss  0.382274216227547 correct 50 time 0.8341350555419922
Epoch  350  loss  0.3634614603984609 correct 50 time 0.7946064472198486
Epoch  360  loss  0.5070310712889597 correct 50 time 0.8096129894256592
Epoch  370  loss  0.17461578142357176 correct 50 time 0.8095555305480957
Epoch  380  loss  0.670759040614084 correct 50 time 1.4903223514556885
Epoch  390  loss  0.30930569923644446 correct 50 time 0.8185238838195801
Epoch  400  loss  0.4522159160694273 correct 50 time 0.8052792549133301
Epoch  410  loss  0.15797531954134855 correct 50 time 0.8895354270935059
Epoch  420  loss  0.18882554913893798 correct 50 time 0.7966222763061523
Epoch  430  loss  0.2676906353136448 correct 50 time 0.8113195896148682
Epoch  440  loss  0.5894424503115442 correct 50 time 0.8356752395629883
Epoch  450  loss  0.10665354223046244 correct 50 time 1.136230230331421
Epoch  460  loss  0.1669353585789121 correct 50 time 0.8112738132476807
Epoch  470  loss  0.671923444705814 correct 50 time 0.8103060722351074
Epoch  480  loss  0.4701532774953231 correct 50 time 1.2754535675048828
Epoch  490  loss  0.16218260524492165 correct 50 time 0.7986617088317871
```
xor gpu
```
Epoch  0  loss  8.692501408344423 correct 21 time 9.70314908027649
Epoch  10  loss  4.424851701867676 correct 43 time 0.41849207878112793
Epoch  20  loss  4.119206497976287 correct 45 time 0.3755960464477539
Epoch  30  loss  3.2892152064767637 correct 46 time 0.372730016708374
Epoch  40  loss  2.726151472004572 correct 43 time 0.3672678470611572
Epoch  50  loss  2.4805382931843205 correct 48 time 0.3783590793609619
Epoch  60  loss  1.8775826712806423 correct 48 time 0.3686079978942871
Epoch  70  loss  4.304306531848049 correct 48 time 0.37343287467956543
Epoch  80  loss  2.126133768328788 correct 48 time 0.3657071590423584
Epoch  90  loss  3.725934709984071 correct 46 time 0.3664219379425049
Epoch  100  loss  3.347400780959992 correct 43 time 0.3835141658782959
Epoch  110  loss  2.667429538025516 correct 48 time 0.38031888008117676
Epoch  120  loss  1.1315534432016865 correct 48 time 0.375568151473999
Epoch  130  loss  0.9893010510721072 correct 48 time 0.37641215324401855
Epoch  140  loss  1.8528653272642417 correct 47 time 0.38531994819641113
Epoch  150  loss  2.640700866958813 correct 48 time 0.37639617919921875
Epoch  160  loss  2.5341852809681913 correct 47 time 0.376446008682251
Epoch  170  loss  1.0543997812046524 correct 48 time 0.39060401916503906
Epoch  180  loss  0.7538171443778187 correct 48 time 0.37674808502197266
Epoch  190  loss  1.7058783343954984 correct 48 time 0.37313079833984375
Epoch  200  loss  1.3375272639208236 correct 45 time 0.4003121852874756
Epoch  210  loss  2.2413618473439354 correct 48 time 0.3774261474609375
Epoch  220  loss  0.7830343569473872 correct 48 time 0.37537598609924316
Epoch  230  loss  1.6210755601565945 correct 47 time 0.38988590240478516
Epoch  240  loss  0.8328203546139056 correct 48 time 0.3665778636932373
Epoch  250  loss  3.4669272780316094 correct 48 time 0.36841821670532227
Epoch  260  loss  2.5690110664569783 correct 47 time 0.3612947463989258
Epoch  270  loss  2.3310071524337688 correct 48 time 0.3618738651275635
Epoch  280  loss  2.7245426597498748 correct 48 time 0.36246705055236816
Epoch  290  loss  1.4099414092106488 correct 47 time 0.364743709564209
Epoch  300  loss  0.6516411546396373 correct 48 time 0.3854348659515381
Epoch  310  loss  0.49715880264270934 correct 48 time 0.3617100715637207
Epoch  320  loss  1.1067373966910963 correct 48 time 0.3620328903198242
Epoch  330  loss  0.5408092371927395 correct 48 time 0.36076903343200684
Epoch  340  loss  0.9908742489676274 correct 47 time 0.3614461421966553
Epoch  350  loss  0.0697664906629108 correct 48 time 0.36597394943237305
Epoch  360  loss  2.902790425959142 correct 48 time 0.3603520393371582
Epoch  370  loss  0.19282311013584028 correct 48 time 0.36541104316711426
Epoch  380  loss  0.715626534510228 correct 48 time 0.3622429370880127
Epoch  390  loss  0.3455675103666206 correct 47 time 0.362152099609375
Epoch  400  loss  1.2226452757283557 correct 48 time 0.3676028251647949
Epoch  410  loss  2.0942989425262812 correct 48 time 0.36517810821533203
Epoch  420  loss  0.9072353369242769 correct 44 time 0.4043309688568115
Epoch  430  loss  0.6690840136515669 correct 48 time 0.37328672409057617
Epoch  440  loss  2.733986064488458 correct 46 time 0.3745608329772949
Epoch  450  loss  1.7692480705282925 correct 49 time 0.37238001823425293
Epoch  460  loss  2.32979380037449 correct 47 time 0.3747849464416504
Epoch  470  loss  0.8730353977864656 correct 48 time 0.3746340274810791
Epoch  480  loss  1.352557700779619 correct 50 time 0.38489198684692383
Epoch  490  loss  0.395242248396923 correct 48 time 0.37325024604797363
```
big simple cpu (200, 0.01)
```
Epoch  0  loss  6.733858454881348 correct 40 time 20.430898904800415
Epoch  10  loss  1.8017873962562614 correct 46 time 2.9788527488708496
Epoch  20  loss  2.986896167708056 correct 48 time 4.1179187297821045
Epoch  30  loss  1.647127339165352 correct 49 time 2.8353018760681152
Epoch  40  loss  1.0370764262229768 correct 49 time 3.44385027885437
Epoch  50  loss  1.2331157973497333 correct 49 time 2.8096706867218018
Epoch  60  loss  1.0521504761168938 correct 49 time 2.816053867340088
Epoch  70  loss  0.9608405803596355 correct 49 time 2.8269426822662354
Epoch  80  loss  0.2587101671407276 correct 49 time 2.832472324371338
Epoch  90  loss  0.26517477008231777 correct 49 time 3.3243303298950195
Epoch  100  loss  0.7710039020951125 correct 49 time 2.8255414962768555
Epoch  110  loss  1.6240815351700857 correct 49 time 4.086138963699341
Epoch  120  loss  0.5744101424587826 correct 49 time 2.833861827850342
Epoch  130  loss  0.03132403331919633 correct 49 time 4.044171333312988
Epoch  140  loss  0.36543112736013006 correct 49 time 2.846937417984009
Epoch  150  loss  1.0827089555545943 correct 49 time 3.2435171604156494
Epoch  160  loss  0.008860428069638515 correct 49 time 2.841409921646118
Epoch  170  loss  0.41028926201722216 correct 49 time 2.8086771965026855
Epoch  180  loss  0.4231906659761487 correct 49 time 2.8585877418518066
Epoch  190  loss  1.5521577943028628 correct 49 time 2.8690924644470215
Epoch  200  loss  0.2425255485527762 correct 49 time 3.3859777450561523
Epoch  210  loss  1.2155069642721115 correct 49 time 2.8118433952331543
Epoch  220  loss  0.5169471222301169 correct 49 time 4.136273622512817
Epoch  230  loss  0.1796890473449706 correct 49 time 2.8385603427886963
Epoch  240  loss  0.26099773185911535 correct 49 time 4.0699169635772705
Epoch  250  loss  0.5946687438243058 correct 49 time 2.8356831073760986
Epoch  260  loss  0.4324675949631754 correct 49 time 3.2688050270080566
Epoch  270  loss  0.6484358779143333 correct 49 time 2.8323862552642822
Epoch  280  loss  0.5293079121075129 correct 49 time 2.826270818710327
Epoch  290  loss  0.34974167092942066 correct 49 time 2.8374722003936768
Epoch  300  loss  0.38297634131470343 correct 49 time 2.838642120361328
Epoch  310  loss  1.067318268227078 correct 49 time 3.513394832611084
Epoch  320  loss  0.37900214459081194 correct 49 time 2.821638584136963
Epoch  330  loss  1.462346419723473 correct 49 time 4.105454921722412
Epoch  340  loss  0.22673900841582315 correct 49 time 2.833369493484497
Epoch  350  loss  0.16654432413201106 correct 49 time 3.895742177963257
Epoch  360  loss  0.3750799540167905 correct 49 time 2.8169898986816406
Epoch  370  loss  0.18937472780471 correct 49 time 3.909172296524048
Epoch  380  loss  0.239317753134706 correct 49 time 2.8166539669036865
Epoch  390  loss  0.10552878562070446 correct 49 time 2.8350324630737305
Epoch  400  loss  0.6932276881133893 correct 49 time 2.835188865661621
Epoch  410  loss  0.526368060377009 correct 49 time 2.848191499710083
Epoch  420  loss  1.3113515167663197 correct 49 time 3.349860191345215
Epoch  430  loss  0.13100338599972383 correct 49 time 2.8424885272979736
Epoch  440  loss  0.5521163816364393 correct 49 time 4.064100027084351
Epoch  450  loss  0.06180393725661055 correct 49 time 2.843130350112915
Epoch  460  loss  1.0471509265862766 correct 49 time 4.12450647354126
Epoch  470  loss  0.6122518638175742 correct 49 time 2.8487043380737305
Epoch  480  loss  0.15108015625042573 correct 49 time 3.3810653686523438
Epoch  490  loss  0.11111856583795418 correct 49 time 2.8357577323913574
```
big simple gpu (200, 0.01)
```
Epoch  0  loss  3.1626240169760202 correct 44 time 18.386573791503906
Epoch  10  loss  1.9135774949948257 correct 48 time 0.9120252132415771
Epoch  20  loss  1.024653362878196 correct 49 time 0.8055679798126221
```
