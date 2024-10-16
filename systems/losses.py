import torch


def rgb_loss(gt_rgb, pred_rgb, rays_valid=None):
    if rays_valid is not None:
        l1_loss = ((gt_rgb - pred_rgb).abs() * rays_valid).mean()
    else:
        l1_loss = (gt_rgb - pred_rgb).abs().mean()
    return l1_loss

# def eikonal_loss(sdf_grad, diff_pts=1.0):
#     gradient_error = (torch.linalg.norm(sdf_grad, ord=2, dim=-1) - 1.) ** 2 * diff_pts
#     gradient_error = gradient_error[torch.nonzero(gradient_error)]
#     # gradient_error = torch.masked_select(gradient_error, ~torch.isnan(gradient_error))
    
#     # gradient_error = gradient_error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
#     return torch.mean(gradient_error)

def eikonal_loss(sdf_grad, relax_inside_sphere, diff_pts=1.0):
    gradient_error = (torch.linalg.norm(sdf_grad, ord=2, dim=-1) - 1.) ** 2 * diff_pts
    gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
    return gradient_error

def curvature_loss(laplace):
    laplace = torch.abs(laplace)
    laplace = torch.masked_select(laplace, ~torch.isnan(laplace))
    # laplace = laplace.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    return torch.mean(laplace)

def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()

def psnr(input, target, mask=None):
    rgb_error = (input - target) ** 2
    if mask is not None:
        rgb_error = rgb_error[mask]
    score = -10.0 * torch.mean(rgb_error).log10()
    return score