https://github.com/tinygrad/tinygrad
tinygrad: For something between PyTorch and karpathy/micrograd. Maintained by tiny corp.
tinygrad is an end-to-end deep learning stack:

    Tensor library with autograd
    IR and compiler that fuse and lower kernels
    JIT + graph execution
    nn / optim / datasets for real training

It’s inspired by PyTorch (ergonomics), JAX (functional transforms and IR-based AD), and TVM (scheduling and codegen), but stays intentionally tiny and hackable.