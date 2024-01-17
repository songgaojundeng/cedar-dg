import numpy as np
 
def MAPE(y, yhat):
    yhat = yhat[y != 0]
    y = y[y != 0]
    loss = 1 / y.shape[0] * np.sum( np.abs( (yhat - y) / y ) )
    return loss

def RMSE(y, yhat):
    loss = np.sqrt(np.mean(np.square(y - yhat)))
    return loss

def NRMSE(y, yhat):
    loss_rmse = RMSE(y, yhat)
    yabsmean = np.mean(np.abs(y))
    flag = yabsmean == 0
    return np.divide(loss_rmse * (1 - flag), yabsmean + flag)

def ND(y, yhat):
    abs_error = np.sum(np.abs(y - yhat))
    yabssum = np.sum(np.abs(y))
    flag = yabssum == 0  
    return np.divide(abs_error * (1 - flag), yabssum + flag)
    
def MAE(y, yhat):
    loss = 1 / y.shape[0] * np.sum(np.abs(yhat - y))
    return loss

def sMAPE(y, yhat):
    # https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/evaluation/_base.py
    denominator = np.abs(y) + np.abs(yhat)
    flag = denominator == 0
    smape = 2 * np.mean((np.abs(y - yhat) * (1 - flag)) / (denominator + flag))
    return smape

def MASE(y, yhat):
    nom = np.sum(np.abs(yhat - y))
    denom = y.shape[0] / (y.shape[0] - 1) * np.sum(np.abs( y[1:] - y[:-1] ))
    loss = nom / denom
    return loss

def RMSPE(y, yhat):
    loss = np.sqrt(1 / y.shape[0] * np.sum( ((yhat - y) / y)**2 ) )
    return loss

def QuantileLoss(y, yhat, quantile):       
    # Source: https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/evaluation/_base.py
    return np.divide((2.0 * np.sum(np.abs((yhat - y) * ((y <= yhat) - quantile)))) , np.sum(np.abs(y)))

def loss_metrics(Y, Yhat):
    loss_MAE, loss_sMAPE = np.zeros((Y.shape[0], Y.shape[2], Y.shape[3])), np.zeros((Y.shape[0], Y.shape[2], Y.shape[3]))
    for series in range(Y.shape[0]):
        for t in range(Y.shape[2]):
            for output in range(Y.shape[3]):
                yhat = Yhat[series, :, t, output]
                y = Y[series, :, t, output]
                loss_MAE[series, t, output] = MAE(y, yhat)
                loss_sMAPE[series, t, output] = sMAPE(y, yhat)
                
    return loss_MAE, loss_sMAPE

 