dfs = kw.getdf_xlsx()
df = dfs["test"]
form_df = dfs["form"]

import matplotlib.pyplot as plt

# データを'model'と'test_no'でグループ化し、それぞれのグループに対してグラフを作成します
models = df["model"].unique()

for model in models:
    df_model = df[df["model"] == model]

    # プロットの作成
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = "tab:red"
    ax1.set_xlabel("capasity (l/min)")
    ax1.set_ylabel("rotation_speed (rpm)", color=color)
    (line1,) = ax1.plot(
        df_model["capasity"],
        df_model["rotation_speed"],
        color=color,
        label="rotation_speed",
    )
    ax1.plot(df_model["capasity"], df_model["rotation_speed"], color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xlim(
        0,
    )  # rotation_speedの範囲を設定
    ax1.set_ylim(0, 5000)  # rotation_speedの範囲を設定
    ax1.grid(True)

    # 2つ目のy軸の作成
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("total_head (m)", color=color)
    (line2,) = ax2.plot(
        df_model["capasity"], df_model["total_head"], color=color, label="total_head"
    )
    ax2.plot(df_model["capasity"], df_model["total_head"], color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 100)  # total_headの範囲を設定

    # 3つ目のy軸の作成
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 45))  # y軸を右に60ポイント移動
    color = "tab:green"
    ax3.set_ylabel("output (kW)", color=color)
    (line3,) = ax3.plot(
        df_model["capasity"], df_model["output"], color=color, label="output"
    )
    ax3.plot(df_model["capasity"], df_model["output"], color=color)
    ax3.tick_params(axis="y", labelcolor=color)
    ax3.set_ylim(0, 10)  # total_headの範囲を設定

    # 凡例の作成
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc="best")

    plt.title(f"Model: {model}")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

# データを'model'と'test_no'でグループ化し、それぞれのグループに対してグラフを作成します
models = df["model"].unique()

# サブプロットの行列数を設定します
num_rows = 3
num_cols = 3

# Figure and array of axes are created
fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
axs = axs.ravel()  # 将2D网格转换为1D数组，以便轻松循环遍历

for i, model in enumerate(models):
    df_model = df[df["model"] == model]

    # Current axes are selected
    ax1 = axs[i]

    color = "tab:red"
    ax1.set_xlabel("capasity (l/min)")
    ax1.set_ylabel("rotation_speed (rpm)", color=color)
    (line1,) = ax1.plot(
        df_model["capasity"],
        df_model["rotation_speed"],
        color=color,
        label="rotation_speed",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xlim(
        0,
    )  # rotation_speedの範囲を設定
    ax1.set_ylim(0, 5000)  # rotation_speedの範囲を設定
    ax1.grid(True)

    # 2つ目のy軸の作成
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("total_head (m)", color=color)
    (line2,) = ax2.plot(
        df_model["capasity"], df_model["total_head"], color=color, label="total_head"
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 100)  # total_headの範囲を設定

    # 3つ目のy軸の作成
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # y軸を右に60ポイント移動
    color = "tab:green"
    ax3.set_ylabel("output (kW)", color=color)
    (line3,) = ax3.plot(
        df_model["capasity"], df_model["output"], color=color, label="output"
    )
    ax3.tick_params(axis="y", labelcolor=color)
    ax3.set_ylim(0, 10)  # outputの範囲を設定

    # 凡例の作成
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    ax1.set_title(f"Model: {model}")

# 使用しないサブプロットを削除
for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axs[j])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


import matplotlib.pyplot as plt

# データを'model'と'test_no'でグループ化し、それぞれのグループに対してグラフを作成します
models = df["model"].unique()

# プロットの作成
fig, ax1 = plt.subplots(figsize=(8, 6))

for model in models:
    df_model = df[df["model"] == model]

    color = "tab:red"
    ax1.set_xlabel("capasity (l/min)")
    ax1.set_ylabel("rotation_speed (rpm)", color=color)
    (line1,) = ax1.plot(
        df_model["capasity"],
        df_model["rotation_speed"],
        color=color,
        label=f"{model}_rotation_speed",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xlim(
        0,
    )  # rotation_speedの範囲を設定
    ax1.set_ylim(0, 5000)  # rotation_speedの範囲を設定
    ax1.grid(True)

    # 2つ目のy軸の作成
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("total_head (m)", color=color)
    (line2,) = ax2.plot(
        df_model["capasity"],
        df_model["total_head"],
        color=color,
        label=f"{model}_total_head",
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 100)  # total_headの範囲を設定

    # 3つ目のy軸の作成
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 45))  # y軸を右に60ポイント移動
    color = "tab:green"
    ax3.set_ylabel("output (kW)", color=color)
    (line3,) = ax3.plot(
        df_model["capasity"], df_model["output"], color=color, label=f"{model}_output"
    )
    ax3.tick_params(axis="y", labelcolor=color)
    ax3.set_ylim(0, 10)  # outputの範囲を設定

# 凡例の作成
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
plt.legend(lines, labels, loc="best")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


import matplotlib.pyplot as plt

# 描画したいモデルのリストを定義します
selected_models = ["model1", "model2", "model3"]  # ここに選択したいモデルの名前を入力してください

# プロットの作成
fig, ax1 = plt.subplots(figsize=(8, 6))

colors = ["tab:red", "tab:blue", "tab:green"]  # 必要なだけ色を追加します

lines = []
labels = []

for model, color in zip(selected_models, colors):
    df_model = df[df["model"] == model]

    ax1.set_xlabel("capasity (l/min)")
    ax1.set_ylabel("rotation_speed (rpm)", color=color)
    (line1,) = ax1.plot(df_model["capasity"], df_model["rotation_speed"], color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xlim(0, 0.7)  # rotation_speedの範囲を設定
    ax1.set_ylim(0, 5000)  # rotation_speedの範囲を設定
    ax1.grid(True)

    # 2つ目のy軸の作成
    ax2 = ax1.twinx()
    ax2.set_ylabel("total_head (m)", color=color)
    ax2.plot(df_model["capasity"], df_model["total_head"], color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 100)  # total_headの範囲を設定

    # 3つ目のy軸の作成
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # y軸を右に60ポイント移動
    ax3.set_ylabel("output (kW)", color=color)
    ax3.plot(df_model["capasity"], df_model["output"], color=color)
    ax3.tick_params(axis="y", labelcolor=color)
    ax3.set_ylim(0, 10)  # outputの範囲を設定

    lines.append(line1)
    labels.append(model)

# 凡例の作成
plt.legend(lines, labels, loc="best")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
