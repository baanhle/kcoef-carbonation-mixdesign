"""Table 9 check: Jiang 2026 [70] Eq.(8) fly-ash Abrams' law, W = 175 kg/m3."""
from math import exp, log
W, MU = 175.0, 0.20
SIZE = 0.95  # GB/T 50081-2019 cl. 5.0.5(3): 100 mm cube -> 150 mm cube, below C60 (Jiang tested to this standard)
SERIES = {"C-II-FA-I": (4.922, 2.321, -0.037), "C-I-FA-II": (5.100, 2.587, 0.237)}
MIXES = [("CI-optimal", 0.385, 0.03), ("Knee", 0.389, 0.325), ("Min-CO2 end", 0.467, 0.25),
         ("Conv", 0.50, 0.0), ("Conv", 0.50, 0.20), ("Conv", 0.50, 0.30)]

def fc(wb, f, k1, k2, k3):
    B = W / wb; C, FA = B * (1 - f), B * f
    return exp(k1 - k2 * W / (C + MU * FA) - k3 * f), C

if __name__ == "__main__":
    assert abs(fc(0.5, 0, 1, 1, 0)[0] - exp(0.5)) < 1e-12  # f=0 -> pure Abrams
    for name, wb, f in MIXES:
        vals = [fc(wb, f, *k) for k in SERIES.values()]
        assert max(SIZE * v[0] for v in vals) < 60  # 0.95 is only prescribed below C60
        print(f"{name:12s} {wb:.3f} {f:.3f} C={vals[0][1]:5.0f}  100 mm " + "  ".join(f"{v[0]:5.1f}" for v in vals)
              + "  150 mm " + "  ".join(f"{SIZE * v[0]:5.1f}" for v in vals))
