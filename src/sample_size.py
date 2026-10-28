"""阶段1：最小样本量计算（两比例 z 检验）。

公式：
    n = [ Z_{α/2}·√(2·p̄·(1-p̄)) + Z_β·√(p0·(1-p0) + p1·(1-p1)) ]² / (p1 - p0)²

各项含义：
  - p0 = 对照组基线 CTR（12%）
  - p1 = p0 + MDE，最小可检测效应的目标 CTR（13%）
  - p̄ = (p0 + p1)/2 = 12.5%，两组合并比例
  - Z_{α/2}：α=0.05 双侧分位数 1.96，控制第一类错误（把没效果的版本误判为有效）
  - Z_β：power=0.8 对应分位数 0.8416，控制第二类错误（有效果却检不出）
  - 分子第一项是原假设下的抽样波动，第二项是备择假设下的抽样波动
"""
import math

from scipy.stats import norm

BASELINE_CTR = 0.12   # 基线 CTR
MDE = 0.01            # 最小可检测效应：1 个百分点（相对约 8.3%）
ALPHA = 0.05          # 显著性水平
POWER = 0.80          # 统计功效
DAILY_USERS = 6000    # 每日可进入实验的 DAU（用于估算实验周期）


def required_n_per_group(p0: float, mde: float, alpha: float, power: float) -> float:
    p1 = p0 + mde
    pbar = (p0 + p1) / 2
    z_alpha = norm.ppf(1 - alpha / 2)   # 双侧
    z_beta = norm.ppf(power)
    numerator = z_alpha * math.sqrt(2 * pbar * (1 - pbar)) + \
        z_beta * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    return numerator ** 2 / (mde ** 2)


def main():
    n = required_n_per_group(BASELINE_CTR, MDE, ALPHA, POWER)
    print("== 样本量估算 ==")
    print(f"基线 CTR p0        = {BASELINE_CTR:.1%}")
    print(f"最小可检测效应 MDE = {MDE:.1%}（绝对 +1pp）")
    print(f"显著性水平 α       = {ALPHA}")
    print(f"统计功效 power     = {POWER}")
    print(f"每组所需样本量     = {n:.0f} 人")
    print(f"总样本量           = {2 * n:.0f} 人")
    print(f"预计实验周期       = {math.ceil(2 * n / DAILY_USERS)} 天（按日进 DAU {DAILY_USERS} 估算）")
    print("\n注：样本量指「去重用户数」，不是曝光次数。单位是用户，才能对应用户级 CTR。")


if __name__ == "__main__":
    main()
