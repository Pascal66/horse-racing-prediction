https://github.com/tinygrad/tinygrad
tinygrad: For something between PyTorch and karpathy/micrograd. Maintained by tiny corp.
tinygrad is an end-to-end deep learning stack:

    Tensor library with autograd
    IR and compiler that fuse and lower kernels
    JIT + graph execution
    nn / optim / datasets for real training

It’s inspired by PyTorch (ergonomics), JAX (functional transforms and IR-based AD), and TVM (scheduling and codegen), but stays intentionally tiny and hackable.

https://github.com/mthorrell/gbnet

What is GBNet?
Gradient boosting (GBM) libraries like XGBoost and LightGBM are excellent for tabular data but can be cumbersome to extend with custom losses or model architectures because you must supply gradients and Hessians by hand.

GBNet wraps GBM libraries in PyTorch Modules so you can:

Define losses and architectures in plain PyTorch
Let PyTorch autograd compute gradients / Hessians
Use XGBoost / LightGBM / boosted linear layers as building blocks inside larger models
At the core of GBNet are three PyTorch Modules:

gbnet.xgbmodule.XGBModule – XGBoost as a PyTorch Module
gbnet.lgbmodule.LGBModule – LightGBM as a PyTorch Module
gbnet.gblinear.GBLinear – a linear PyTorch Module trained with boosting instead of via gradient descent methods
On top of these, GBNet ships higher-level models in gbnet.models, including forecasting, ordinal regression and survival models.

https://github.com/COGS108/Group106_WI26