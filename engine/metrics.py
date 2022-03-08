def mse(output, gt_mask):
    '''mean squared error'''
    return ((output - gt_mask)**2).sum() / output.numel()