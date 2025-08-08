import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cripser


class TopologicalLoss(nn.Module):
    """
    Simple MONAI-style topological loss.

    What it does (per batch, per class):
      1) Runs persistent homology on the probability map (sublevel on 1 - p).
      2) Compares predicted features to a target:
         - If `match_gt=True`: uses PH of y_true for that class.
         - Else: enforces a fixed target Betti counts you specify (e.g., beta0=1).
      3) Builds two sparse maps:
         - weight_map: where to apply topology nudges (mostly zeros)
         - ref_map:    values to push toward (0, 1, or paired voxel prob)
      4) Loss = MSE on those sparse points only.

    Args:
        classes: list/tuple of class indices to enforce (e.g., organs) in y_pred[:, C, ...]
        match_gt: if True, match PH of y_true; else use `target_betti`.
        target_betti: dict {class_idx: {dim: count}}, e.g. {1: {0:1}} means “1 component” for class 1.
                      Ignored if match_gt=True. If not provided, defaults to {dim 0: 1}.
        dims_to_enforce: which homology dimensions to use. For 2D, typical is (0,1); for 3D, (0,1,2).
        reduction: 'mean' | 'sum' | 'none' for the batch-wise final loss.
        eps: numerical epsilon to avoid division by zero.
    """

    def __init__(
        self,
        classes,
        match_gt: bool = True,
        target_betti: dict | None = None,
        dims_to_enforce=(0,),
        reduction: str = "mean",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.classes = list(classes)
        self.match_gt = match_gt
        self.target_betti = target_betti or {}  # {cls: {dim: count}}
        self.dims_to_enforce = tuple(dims_to_enforce)
        assert reduction in {"mean", "sum", "none"}
        self.reduction = reduction
        self.eps = eps

    # ------- Helpers: PH parsing & planning which features to keep/remove -------
    @staticmethod
    def _preprocess_ph(info: np.ndarray):
        """
        cripser.computePH returns rows:
        [dim, birth, death, bx, by, bz, dx, dy, dz]  (z fields are 0 for 2D)
        This groups them per dim and clips death<=1 (for numerical stability).
        """
        # clamp death > 1 to 1
        info = info.copy()
        info[info[:, 2] > 1, 2] = 1.0

        pd, bcp, dcp = {}, {}, {}
        dims = np.unique(info[:, 0]).astype(int)
        for d in dims:
            mask = info[:, 0] == d
            rows = info[mask]
            pd[str(d)] = rows[:, 1:3]  # (N,2)
            bcp[str(d)] = rows[:, 3:6]  # (N,3)
            dcp[str(d)] = rows[:, 6:9]  # (N,3)
        return pd, bcp, dcp

    @staticmethod
    def _keep_and_remove_indices(pd_pred: dict, pd_tgt: dict):
        """
        Decide which predicted features to keep (match to target count) and which to remove,
        per homology dim. Keep the most persistent ones.
        """
        idx_fix, idx_remove = {}, {}
        for d in pd_pred.keys():
            pers = np.abs(pd_pred[d][:, 1] - pd_pred[d][:, 0])
            order = np.argsort(pers)[::-1]  # descending persistence
            n_pred = len(pers)
            n_tgt = len(pd_tgt.get(d, [])) if d in pd_tgt else 0

            if n_pred > n_tgt:
                idx_fix[d] = order[:n_tgt]
                idx_remove[d] = order[n_tgt:]
            else:
                idx_fix[d] = order
                idx_remove[d] = np.array([], dtype=int)
        return idx_fix, idx_remove

    @staticmethod
    def _target_pd_from_betti(betti_counts: dict):
        """
        Build a fake target PD with [0,1] intervals, count times.
        Ex: { '0': 1 } -> array([[0,1]])
        """
        pd_tgt = {}
        for d_str, cnt in betti_counts.items():
            if cnt <= 0:
                continue
            pd_tgt[str(int(d_str))] = np.tile(np.array([[0.0, 1.0]]), (int(cnt), 1))
        return pd_tgt

    # ------- Core: make sparse maps for a single (B=1, C=1) probability map -------
    def _sparse_maps_for_one(
        self,
        prob_map: np.ndarray,  # shape: (Z,H,W) or (H,W)
        gt_map: np.ndarray | None,  # same shape, binary; optional
        spatial_dims: int,
    ):
        """
        Returns (weight_map, ref_map) same shape as prob_map.
        """
        # 1) predicted PH
        info_pred = cripser.computePH(1.0 - prob_map, maxdim=spatial_dims)
        pd_pred, bcp_pred, dcp_pred = self._preprocess_ph(info_pred)

        # 2) target PD
        if self.match_gt and gt_map is not None:
            info_gt = cripser.computePH(1.0 - gt_map, maxdim=spatial_dims)
            pd_tgt, _, _ = self._preprocess_ph(info_gt)
        else:
            # default to “one component” if nothing specified
            betti_cfg = {}
            for d in self.dims_to_enforce:
                # use user-specified betti if present; else default beta0=1, others 0
                cnt = 0
                if d == 0:
                    cnt = 1
                # allow override per class via target_betti at caller level
                betti_cfg[str(d)] = cnt
            pd_tgt = self._target_pd_from_betti(betti_cfg)

        # reduce to enforced dims only
        pd_pred = {
            str(d): pd_pred[str(d)]
            for d in pd_pred.keys()
            if int(d) in self.dims_to_enforce
        }
        pd_tgt = {
            str(d): pd_tgt[str(d)]
            for d in pd_tgt.keys()
            if int(d) in self.dims_to_enforce
        }

        # 3) decide keep/remove
        idx_fix, idx_remove = self._keep_and_remove_indices(pd_pred, pd_tgt)

        # 4) build sparse maps (push births toward 0, deaths toward 1; remove spurious by cross-tying)
        w = np.zeros_like(prob_map, dtype=np.float32)
        r = np.zeros_like(prob_map, dtype=np.float32)
        shape = prob_map.shape

        def in_bounds(pt):
            for i, m in enumerate(pt):
                if m < 0 or m >= shape[i]:
                    return False
            return True

        # NOTE: cripser returns (bx,by,bz); if 2D, bz will be 0—safe to index (Z,H,W) or (H,W)
        for d_str in idx_fix.keys():
            # keep: strengthen features -> push birth to 0, death to 1
            for k in idx_fix[d_str]:
                bpt = tuple(int(x) for x in bcp_pred[d_str][k])
                dpt = tuple(int(x) for x in dcp_pred[d_str][k])
                # trim to dimensionality
                bpt = bpt[-prob_map.ndim :]  # (H,W) or (Z,H,W)
                dpt = dpt[-prob_map.ndim :]

                if in_bounds(bpt):
                    w[bpt] = 1.0
                    r[bpt] = 0.0
                if in_bounds(dpt):
                    w[dpt] = 1.0
                    r[dpt] = 1.0

            # remove: tie birth↔death probabilities to annihilate short-lived features
            for k in idx_remove[d_str]:
                bpt = tuple(int(x) for x in bcp_pred[d_str][k])
                dpt = tuple(int(x) for x in dcp_pred[d_str][k])
                bpt = bpt[-prob_map.ndim :]
                dpt = dpt[-prob_map.ndim :]

                b_ok = in_bounds(bpt)
                d_ok = in_bounds(dpt)
                if b_ok and d_ok:
                    # set each to the other's current value
                    w[bpt] = 1.0
                    r[bpt] = prob_map[dpt]
                    w[dpt] = 1.0
                    r[dpt] = prob_map[bpt]
                elif b_ok and not d_ok:
                    w[bpt] = 1.0
                    r[bpt] = 1.0  # push birth up to kill spurious short feature
                elif d_ok and not b_ok:
                    w[dpt] = 1.0
                    r[dpt] = 0.0

        return w, r

    # ------- Public API -------

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor | None = None):
        """
        y_pred: [B, C, H, W] or [B, C, D, H, W] — probabilities (after sigmoid/softmax).
        y_true: same shape, one-hot (0/1). Required if match_gt=True.
        """
        if y_pred.dim() not in (4, 5):
            raise ValueError("y_pred must be [B,C,H,W] or [B,C,D,H,W]")
        if self.match_gt and y_true is None:
            raise ValueError("match_gt=True requires y_true")

        B, C = y_pred.shape[:2]
        spatial_dims = (
            1 if y_pred.dim() == 4 else 2
        )  # cripser's maxdim: 2D -> 1 (β0,β1); 3D -> 2 (β0,β1,β2)
        # NOTE: cripser uses maxdim = number of homology dims (2D -> up to 1; 3D -> up to 2)
        maxdim = 1 if y_pred.dim() == 4 else 2
        if any(d > maxdim for d in self.dims_to_enforce):
            raise ValueError(
                f"dims_to_enforce {self.dims_to_enforce} incompatible with input dimensionality."
            )

        total = y_pred.new_tensor(0.0)
        per_item = []

        for b in range(B):
            loss_b = y_pred.new_tensor(0.0)
            count_terms = 0
            for cls in self.classes:
                prob = y_pred[b, cls]  # (H,W) or (D,H,W)
                gt = None
                if self.match_gt and y_true is not None:
                    gt = y_true[b, cls]

                # to numpy for PH
                prob_np = prob.detach().float().cpu().numpy()
                gt_np = gt.detach().float().cpu().numpy() if gt is not None else None

                # build sparse maps
                w_np, r_np = self._sparse_maps_for_one(
                    prob_np, gt_np, spatial_dims=maxdim
                )

                # cast back to torch
                w = torch.from_numpy(w_np).to(prob).float()
                r = torch.from_numpy(r_np).to(prob).float()

                # skip if nothing to update
                denom = w.sum()
                if denom <= self.eps:
                    continue

                # sparse MSE on critical voxels
                # normalize by number of updated points to keep scale stable across batches
                mse = F.mse_loss(prob * w, r, reduction="sum") / (denom + self.eps)
                loss_b = loss_b + mse
                count_terms += 1

            if count_terms > 0:
                loss_b = loss_b / count_terms
            per_item.append(loss_b)
            total = total + loss_b

        if self.reduction == "mean":
            return total / max(len(per_item), 1)
        elif self.reduction == "sum":
            return total
        else:
            return torch.stack(per_item)
