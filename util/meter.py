import torch
from monai.metrics import (
    compute_dice,
    compute_hausdorff_distance,
    compute_average_surface_distance,
)
from monai.networks import one_hot


class DiceMeter:
    def __init__(self, args):
        self.num_classes = args.num_classes
        if self.num_classes > 1:
            self.include_background = False
        else:
            self.include_background = True
        self.reset()

    def reset(self):
        self.count, self.sum = torch.zeros(self.num_classes), torch.zeros(
            self.num_classes
        )
        self.all_scores = [] #added, store all individual dice scores

    def one_hot(self, pred, gt):
        if self.num_classes > 1:
            pred_one_hot = one_hot(pred, num_classes=self.num_classes + 1)
            gt_one_hot = one_hot(
                gt, num_classes=self.num_classes + 1
            )  # (1, C, H, W, D)
        else:
            pred_one_hot = pred
            gt_one_hot = gt
        return pred_one_hot, gt_one_hot

    def update_ori(self, pred, gt):
        """
        :param pred_binary: (1, 1, H, W, D)
        :param gt: (1, 1, H, W, D)
        """
        pred_one_hot, gt_one_hot = self.one_hot(pred, gt)
        mean_dice = compute_dice(
            pred_one_hot, gt_one_hot, self.include_background, ignore_empty=True
        ).sum(dim=0)
        nan = torch.isnan(mean_dice)
        n_n = (~nan).float()
        mean_dice[nan] = 0
        self.sum += mean_dice.cpu()
        self.count += n_n.cpu()

    def update(self, pred, gt):
        """
        :param pred_binary: (1, 1, H, W, D)
        :param gt: (1, 1, H, W, D)
        """
        pred_one_hot, gt_one_hot = self.one_hot(pred, gt)
        dice_scores = compute_dice(
            pred_one_hot, gt_one_hot, self.include_background, ignore_empty=True
        )
        
        # Store individual scores for later variance calculation
        self.all_scores.append(dice_scores.cpu())
        
        # Continue with existing functionality
        mean_dice = dice_scores.sum(dim=0)
        nan = torch.isnan(mean_dice)
        n_n = (~nan).float()
        mean_dice[nan] = 0
        self.sum += mean_dice.cpu()
        self.count += n_n.cpu()

    def get_average(self):
        self.count[self.count == 0] = -1
        mean = self.sum / self.count
        metric = torch.mean(mean[self.count > 0])

        # added, Calculate variability statistics
        variability_stats = self.compute_variability()

        self.reset()
        return mean.numpy(), metric.numpy(), variability_stats

    def compute_variability(self):
        """Calculate variability statistics of dice scores"""
        if not self.all_scores:
            return {"std": None, "percentiles": None}
            
        # Stack all scores and compute statistics
        all_scores_tensor = torch.cat(self.all_scores, dim=0)
        
        # Calculate standard deviation per class
        std_per_class = torch.std(all_scores_tensor, dim=0)
        
        # Calculate percentiles (25%, 50%, 75%)
        percentiles = {}
        for p in [25, 50, 75]:
            percentiles[f"p{p}"] = torch.quantile(all_scores_tensor, p/100.0, dim=0)
            
        return {
            "std": std_per_class.numpy(),
            "percentiles": {k: v.numpy() for k, v in percentiles.items()}
        }        

class HausdorffMeter(DiceMeter):
    def __init__(self, args):
        super(HausdorffMeter, self).__init__(args)
        self.spacing = args.spacing
        self.reset()

    def update(self, pred, gt):
        """
        :param pred_binary: (1, 1, H, W, D)
        :param gt: (1, 1, H, W, D)
        """
        pred_one_hot, gt_one_hot = self.one_hot(pred, gt)
        mean_dice = compute_hausdorff_distance(
            pred_one_hot,
            gt_one_hot,
            self.include_background,
            percentile=95,
            spacing=self.spacing,
        ).sum(dim=0)
        nan = torch.isnan(mean_dice)
        n_n = (~nan).float()
        mean_dice[nan] = 0
        self.sum += mean_dice.cpu()
        self.count += n_n.cpu()


class SurfaceDistanceMeter(DiceMeter):
    def __init__(self, args):
        super(SurfaceDistanceMeter, self).__init__(args)
        self.spacing = args.spacing
        self.reset()

    def update(self, pred, gt):
        """
        :param pred_binary: (1, 1, H, W, D)
        :param gt: (1, 1, H, W, D)
        """
        pred_one_hot, gt_one_hot = self.one_hot(pred, gt)
        mean_dice = compute_average_surface_distance(
            pred_one_hot, gt_one_hot, self.include_background, spacing=self.spacing
        ).sum(dim=0)
        nan = torch.isnan(mean_dice)
        n_n = (~nan).float()
        mean_dice[nan] = 0
        self.sum += mean_dice.cpu()
        self.count += n_n.cpu()
