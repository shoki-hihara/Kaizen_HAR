import torch
import numpy as np

class RandomScaling:
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, x):
        # x: [T, C] or [C, T] の想定（WISDM の形に合わせて調整）
        factor = torch.empty(1).uniform_(*self.scale_range)
        return x * factor

class Random3DRotation:
    def __init__(self, max_angle=np.pi):
        self.max_angle = max_angle

    def __call__(self, x):
        # x: [T, 3] を想定（加速度3軸など）
        if x.shape[-1] != 3:
            return x
        angle = torch.empty(3).uniform_(-self.max_angle, self.max_angle)
        cx, cy, cz = torch.cos(angle)
        sx, sy, sz = torch.sin(angle)

        Rx = torch.tensor([[1, 0, 0],
                           [0, cx, -sx],
                           [0, sx, cx]])
        Ry = torch.tensor([[cy, 0, sy],
                           [0, 1, 0],
                           [-sy, 0, cy]])
        Rz = torch.tensor([[cz, -sz, 0],
                           [sz,  cz, 0],
                           [0,   0,  1]])

        R = Rz @ Ry @ Rx  # [3, 3]
        return x @ R.T

class TimeWarp:
    def __init__(self, sigma=0.2, knot=4):
        self.sigma = sigma
        self.knot = knot

    def __call__(self, x):
        # x: [T, C] を想定
        import numpy as np

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        device = x.device
        x_np = x.detach().cpu().numpy()
        T = x_np.shape[0]
        if T <= 1:
            return x

        # ランダムな warp 曲線を生成（Um et al. 系の簡易版）
        orig_steps = np.arange(T)
        tt = np.linspace(0, T - 1, self.knot)
        # 1.0 周辺のランダム倍率 を積分して単調増加な warp を作る
        random_factors = np.random.normal(loc=1.0, scale=self.sigma, size=self.knot)
        cum = np.cumsum(random_factors)
        cum = (cum - cum[0]) / (cum[-1] - cum[0]) * (T - 1)
        # 補間して各 t に対応する warped t' を作る
        warped_steps = np.interp(orig_steps, tt, cum)

        # 線形補間で新しい系列をサンプリング
        x_warped = np.zeros_like(x_np)
        for c in range(x_np.shape[1]):
            x_c = x_np[:, c]
            x_warped[:, c] = np.interp(orig_steps, warped_steps, x_c)

        return torch.as_tensor(x_warped, device=device, dtype=x.dtype)
