"""一键跑通：样本量估算 → 生成数据 → 假设检验 → 可视化。

用法：python -m src.main
"""
from src import analysis, gen_data, sample_size, visualize


def main():
    print("=" * 70)
    print("Step 1 · 样本量估算")
    print("=" * 70)
    sample_size.main()

    print("\n" + "=" * 70)
    print("Step 2 · 生成模拟数据")
    print("=" * 70)
    gen_data.main()

    print("\n" + "=" * 70)
    print("Step 3 · 假设检验")
    print("=" * 70)
    analysis.main()

    print("\n" + "=" * 70)
    print("Step 4 · 可视化")
    print("=" * 70)
    visualize.main()


if __name__ == "__main__":
    main()
