import torch
from torch.nn import functional as F

from einops import rearrange, parse_shape

def l2_normalize(v, dim=-1, epsilon=1e-12):
    norms = torch.sqrt(torch.sum(torch.square(v), dim=dim, keepdims=True) + epsilon)
    return v / norms

def squared_difference(x, y):
    return torch.square(x-y)

def batched_select(params, indices, dim=None, batch_dims=0):
    params_shape, indices_shape = list(params.shape), list(indices.shape)
    assert params_shape[:batch_dims] == indices_shape[:batch_dims]
   
    def _permute(dim, dim1, dim2):
        permute = []
        for i in range(dim):
            if i == dim1:
                permute.append(dim2)
            elif i == dim2:
                permute.append(dim1)
            else:
                permute.append(i)
        return permute

    if dim is not None and dim != batch_dims:
        params_permute = _permute(len(params_shape), dim1=batch_dims, dim2=dim)
        indices_permute = _permute(len(indices_shape), dim1=batch_dims, dim2=dim)
        prams = torch.permute(params, params_permute)
        indices = torch.permute(indices, params_permute)
        params_shape, indices_shape = list(params.shape), list(indices.shape)

    params, indices = torch.reshape(params, params_shape[:batch_dims+1] + [-1]), torch.reshape(indices, list(indices_shape[:batch_dims]) + [-1, 1])

    # indices = torch.tile(indices, params.shape[-1:])
    indices = indices.repeat([1] * (params.ndim - 1) + [params.shape[-1]])

    batch_params = torch.gather(params, batch_dims, indices.to(dtype=torch.int64))

    output_shape = params_shape[:batch_dims] + indices_shape[batch_dims:] + params_shape[batch_dims+1:]

    if dim is not None and dim != batch_dims:
        prams = torch.permute(params, params_permute)
        indices = torch.permute(indices, params_permute)
        
    return torch.reshape(batch_params, output_shape)

def guard_batch(x, mask, input_lens, sep=10):
    is_1d = (x.ndim == 3)
    
    new_mask = torch.stack([
        torch.cat([F.pad(mask[i,:l], [0,sep], value=0), mask[i,l:]], dim=0) for i, l in enumerate(input_lens)],dim=0)
    
    if is_1d:
        new_x = torch.stack([
            torch.cat([F.pad(x[i,:l], [0,0,0,sep], value=0.), x[i,l:]], dim=0) for i, l in enumerate(input_lens)],dim=0)
    else:
        new_x = torch.stack([
            torch.cat([
                F.pad(torch.cat([F.pad(x[i,:l,:l], [0,0,0,sep], value=0.), x[i,:l,l:]], dim=1), [0,0,0,0,0,sep], value=0.),
                torch.cat([F.pad(x[i,l:,:l], [0,0,0,sep], value=0.), x[i,l:,l:]], dim=1)], dim=0) for i, l in enumerate(input_lens)], dim=0)


    return new_x, new_mask

def unguard_batch(x, mask, input_lens, sep=10):
    is_1d = (x.ndim == 3)
    
    if is_1d:
        new_x = torch.stack([
            torch.cat([x[i,:l], x[i,l+sep:]], dim=0) for i, l in enumerate(input_lens)], dim=0)
    else:
        new_x = torch.stack([
            torch.cat([
                torch.cat([x[i,:l,:l], x[i,:l,l+sep:]], dim=1),
                torch.cat([x[i,l+sep:,:l], x[i,l+sep:,l+sep:]], dim=1)], dim=0) for i, l in enumerate(input_lens)], dim=0)

    
    return new_x

def lddt(pred_points, true_points, points_mask, cutoff=15.):
    """Computes the lddt score for a batch of coordinates.
        https://academic.oup.com/bioinformatics/article/29/21/2722/195896
        Inputs: 
        * pred_coords: (b, l, d) array of predicted 3D points.
        * true_points: (b, l, d) array of true 3D points.
        * points_mask : (b, l) binary-valued array. 1 for points that exist in
            the true points
        * cutoff: maximum inclusion radius in reference struct.
        Outputs:
        * (b, l) lddt scores ranging between 0 and 1
    """
    assert len(pred_points.shape) == 3 and pred_points.shape[-1] == 3
    assert len(true_points.shape) == 3 and true_points.shape[-1] == 3

    eps = 1e-10

    # Compute true and predicted distance matrices. 
    pred_cdist = torch.sqrt(torch.sum(
        torch.square(
            rearrange(pred_points, 'b l c -> b l () c') -
            rearrange(pred_points, 'b l c -> b () l c')),
        dim=-1,
        keepdims=False))
    true_cdist = torch.sqrt(torch.sum(
        torch.square(
            rearrange(true_points, 'b l c -> b l () c') -
            rearrange(true_points, 'b l c -> b () l c')),
        dim=-1,
        keepdims=False))
   
    cdist_to_score = ((true_cdist < cutoff) *
            (rearrange(points_mask, 'b i -> b i ()') *rearrange(points_mask, 'b j -> b () j')) *
            (1.0 - torch.eye(true_cdist.shape[1], device=points_mask.device)))  # Exclude self-interaction

    # Shift unscored distances to be far away
    dist_l1 = torch.abs(true_cdist - pred_cdist)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * sum([dist_l1 < t for t in (0.5, 1.0, 2.0, 4.0)])

    # Normalize over the appropriate axes.
    return (torch.sum(cdist_to_score * score, dim=-1) + eps)/(torch.sum(cdist_to_score, dim=-1) + eps)
