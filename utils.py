import random
import numpy as np
import pandas as pd
import torch
from lib.loss_metrics import RMSE, ND, QuantileLoss, MAPE, sMAPE, NRMSE
import importlib
import time
import operator
from numbers import Number
from collections import OrderedDict

# import functorch
# from functorch import grad, vjp
# from functools import partial

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# Calculate and show metrics
def calc_metrics(yhat, y, quantiles):
    df = pd.DataFrame(columns=['RMSE','NRMSE','ND','MAPE','sMAPE','QuantileLoss','Quantile'])
    df.loc[:, 'Quantile'] = quantiles
    for q, quantile in enumerate(quantiles):
        df.loc[q, 'RMSE'] = RMSE(y, yhat[q])
        df.loc[q, 'NRMSE'] = NRMSE(y, yhat[q])
        df.loc[q, 'ND'] = ND(y, yhat[q])
        df.loc[q, 'MAPE'] = MAPE(y, yhat[q])
        df.loc[q, 'sMAPE'] = sMAPE(y, yhat[q])
        df.loc[q, 'QuantileLoss'] = QuantileLoss(y, yhat[q], quantile)
    q = 4
    print(f"         RMSE/NRMSE/ND/MAPE/sMAPE loss: {df['RMSE'][q]:.2f}/{df['NRMSE'][q]:.2f}/{df['ND'][q]:.3f}/{df['MAPE'][q]:.3f}/{df['sMAPE'][q]:.3f}")
    print(f"         p10/p50/p90/mp50 loss: {df['QuantileLoss'][0]:.3f}/{df['QuantileLoss'][4]:.3f}/{df['QuantileLoss'][8]:.3f}/{df['QuantileLoss'].mean():.3f}")
    return df

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Instantiate model based on string algorithm input
def instantiate_model(algorithm):
    model_class = importlib.import_module('algorithms.'+algorithm)
    model = getattr(model_class, algorithm)
    
    return model

# Read experiment csv
def read_table(filename):
    for x in range(0, 10):
        try:
            table = pd.read_csv(filename, sep=';')
            error = None
        except Exception as error:
            pass
        
        if error:
            time.sleep(5)
        else:
            break
    return table


class ParamDict(OrderedDict):
    """A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def fish_step(meta_weights, inner_weights, meta_lr):
    meta_weights, weights = ParamDict(meta_weights), ParamDict(inner_weights)
    meta_weights += meta_lr * sum([weights - meta_weights], 0 * meta_weights)
    return meta_weights

