# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://github.com/XinJCheng/CSPN/blob/b3e487bdcdcd8a63333656e69b3268698e543181/cspn_pytorch/utils.py#L19
# - https://web.eecs.umich.edu/~fouhey/2016/evalSN/evalSN.html
#

from math import radians
import torch


class MetricFunction():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.total_size = 0
        self.error_sum = {}
        self.error_avg = {}

    def evaluate(self, predictions, targets):
        normal_p = predictions
        normal_gt = targets
        
        error_val = evaluate_error_normal(normal_p, normal_gt)
        
        self.total_size += self.batch_size
        self.error_avg = avg_error(self.error_sum, error_val, self.total_size, self.batch_size)
        return self.error_avg
    
    def show(self):
        error = self.error_avg
        format_str = ('======NORMALS=======\nMSE=%.4f\tRMSE=%.4f\tMAE=%.4f\tMME=%.4f\nTANGLE11.25=%.4f\tTANGLE22.5=%.4f\tTANGLE30.0=%.4f')
        return format_str % (error['N_MSE'], error['N_RMSE'], error['N_MAE'],  error['N_MME'], \
                         error['N_TANGLE11.25'], error['N_TANGLE22.5'], error['N_TANGLE30.0'])


def evaluate_error_normal(pred_normal, gt_normal):
    error = {}
    
    dot_product = torch.mul(pred_normal, gt_normal).sum(dim=1)
    angular_error = torch.acos(torch.minimum(torch.tensor(1, device=pred_normal.device), 
                                             torch.maximum(torch.tensor(-1, device=pred_normal.device), dot_product)))

    error['N_MSE'] = torch.mean(torch.mul(angular_error, angular_error))
    error['N_RMSE'] = torch.sqrt(error['N_MSE'])
    error['N_MAE'] = torch.mean(angular_error)
    error['N_MME'] = torch.median(angular_error)
    
    error['N_TANGLE11.25'] = torch.mean((angular_error <= radians(11.25)).float())
    error['N_TANGLE22.5'] = torch.mean((angular_error <= radians(22.5)).float())
    error['N_TANGLE30.0'] = torch.mean((angular_error <= radians(30.0)).float())
    
    return error


# avg the error
def avg_error(error_sum, error_val, total_size, batch_size):
    error_avg = {}
    for item, value in error_val.items():
        error_sum[item] = error_sum.get(item, 0) + value * batch_size
        error_avg[item] = error_sum[item] / float(total_size)
    return error_avg


def print_single_error(epoch, loss, error):
    format_str = ('%s\nEpoch: %d, loss=%s\n%s\n')
    print (format_str % ('eval_avg_error', epoch, loss, error))