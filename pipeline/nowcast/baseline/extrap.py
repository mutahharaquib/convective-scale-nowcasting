"""Baselines we score against (D1): persistence + optical-flow extrapolation.
Uses pysteps (Lucas-Kanade + semi-Lagrangian) when installed, else OpenCV Farneback
with the same semi-Lagrangian advection."""
import numpy as np, cv2


def persistence(x, n_out):
    return np.repeat(x[-1:], n_out, axis=0)


def motion_field(x):
    """Mean Farneback flow over the last 3 frame pairs. Returns (H,W,2) px/frame."""
    f = []
    for a, b in zip(x[-4:-1], x[-3:]):
        a8, b8 = [np.clip(v, 0, 255).astype(np.uint8) for v in (a, b)]
        f.append(cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 3, 15, 3, 5, 1.2, 0))
    return np.mean(f, 0)


def optical_flow(x, n_out):
    try:
        from pysteps import motion, nowcasts
        V = motion.get_method("LK")(x[-4:])
        return np.nan_to_num(nowcasts.get_method("extrapolation")(x[-1], V, n_out))
    except ImportError:
        pass
    flow = motion_field(x)
    H, W = x.shape[1:]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mx, my = gx - flow[..., 0], gy - flow[..., 1]      # backward trajectory, 1 step
    out, cur = [], x[-1].astype(np.float32)
    for _ in range(n_out):
        cur = cv2.remap(cur, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out.append(cur)
    return np.stack(out)
