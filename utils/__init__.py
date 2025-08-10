import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_erosion
import logging
logging.basicConfig(level=logging.INFO)

def convex_hull_mask_3d(
    mask,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    method="centers",
    qhull_options=None,
):
    """
    Return a boolean mask equal to the convex hull of a 3D binary mask.

    Parameters
    ----------
    mask : (Z, Y, X) bool/0-1 ndarray
    spacing : (sx, sy, sz) voxel size in world units
    origin : (ox, oy, oz) world coords of grid origin at index (0,0,0) *corner*
    method : "corners" (exact hull of union of cubes) or "centers" (approx hull)
    qhull_options : str or None, e.g. "QJ" to joggle if precision issues

    Returns
    -------
    hull_mask : (Z, Y, X) bool ndarray
    """
    mask = mask.astype(bool)

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    # sx, sy, sz = map(float, spacing)
    # sx, sy, sz = spacing[0,0], spacing[1,1], spacing[2,2]  # spacing is a tensor of shape (3,3)
    sx, sy, sz = (1,1,1)  # spacing is a tensor of shape (3,3)
    ox, oy, oz = map(float, origin)
    
    # --- choose points to hull ---
    if method == "corners":
        # keep only surface voxels before expanding to corners to reduce points
        shell = mask & ~binary_erosion(
            mask, structure=np.ones((3, 3, 3), bool), border_value=0
        )
        zyx = np.argwhere(shell)
        if zyx.size == 0:
            zyx = np.argwhere(mask)  # fall back if everything eroded away

        # 8 cube-corner offsets in (z,y,x) index space
        offs = (
            np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij"))
            .reshape(3, -1)
            .T
        )  # (8,3)
        # unique corners -> convert to (x,y,z) index space
        pts_idx_xyz = np.unique(
            (zyx[:, None, :] + offs[None, :, :]).reshape(-1, 3), axis=0
        )[:, ::-1]
    elif method == "centers":
        # voxel centers = indices + 0.5
        zyx = np.argwhere(mask)
        pts_idx_xyz = zyx[:, ::-1] + 0.5
    else:
        raise ValueError("method must be 'corners' or 'centers'")

    # indices -> world coords
    pts_world = np.empty_like(pts_idx_xyz, dtype=float)
    pts_world[:, 0] = pts_idx_xyz[:, 0] * sx + ox
    pts_world[:, 1] = pts_idx_xyz[:, 1] * sy + oy
    pts_world[:, 2] = pts_idx_xyz[:, 2] * sz + oz

    # convex hull in world space
    hull = ConvexHull(pts_world, qhull_options=qhull_options)
    A = hull.equations[:, :3]  # outward normals
    b = hull.equations[:, 3]  # offsets; inside satisfies A @ p + b <= 0

    # --- voxelize the hull back onto the original grid (centers test) ---
    Z, Y, X = mask.shape

    # bbox in index space to limit computation
    mins = np.floor(pts_idx_xyz.min(axis=0)).astype(int)
    maxs = np.ceil(pts_idx_xyz.max(axis=0)).astype(int)

    x0, x1 = np.clip([mins[0], maxs[0]], 0, X)
    y0, y1 = np.clip([mins[1], maxs[1]], 0, Y)
    z0, z1 = np.clip([mins[2], maxs[2]], 0, Z)

    # voxel centers in that bbox (index space -> world)
    zs = np.arange(z0, z1)
    ys = np.arange(y0, y1)
    xs = np.arange(x0, x1)
    Zg, Yg, Xg = np.meshgrid(zs + 0.5, ys + 0.5, xs + 0.5, indexing="ij")  # (z,y,x)

    Xw = Xg * sx + ox
    Yw = Yg * sy + oy
    Zw = Zg * sz + oz
    P = np.stack([Xw, Yw, Zw], axis=-1).reshape(-1, 3)  # (N,3), xyz

    inside = np.all(P @ A.T + b <= 1e-9, axis=1)  # tolerance
    hull_mask = np.zeros_like(mask, dtype=bool)
    inside = inside.reshape(Zg.shape)
    hull_mask[z0:z1, y0:y1, x0:x1] = inside
    return hull_mask
