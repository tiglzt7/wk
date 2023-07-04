import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データを作成
data = {
    "A": ["Group1"] * 5 + ["Group2"] * 5 + ["Group3"] * 5,
    "B": np.random.randint(10, 50, size=15),
    "C": ["Label" + str(i) for i in range(1, 6)] * 3,
}

# データフレームを作成
df = pd.DataFrame(data)

print(df)


# 100%棒グラフ
def draw_stacked_bar(ax, df, group_col, value_col, label_col, group_label):
    """
    Stacked bar plot を描画する関数

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        描画対象の Axes
    df: pandas.DataFrame
        データフレーム
    group_col: str
        グループ名が含まれている列名
    value_col: str
        値が含まれている列名
    label_col: str
        ラベルが含まれている列名
    group_label: str
        描画対象のグループのラベル名
    """
    # グループごとの値の合計を計算
    total_sum = df.groupby(group_col)[value_col].sum()

    # 特定のグループのデータを取得
    group = df[df[group_col] == group_label]
    # 値の降順に並べ替え
    group = group.sort_values(value_col, ascending=False)
    # 値をパーセントに変換
    group[value_col] = group[value_col] / total_sum[group_label] * 100

    # 棒グラフの底部を描画するための初期位置
    bottom = 0

    for value, label in zip(group[value_col], group[label_col]):
        # 棒グラフを描画。指定のカラムの値をラベルとして使用。
        ax.bar(group_label, value, bottom=bottom, label=label)

        # 次の棒グラフの底部を更新
        bottom += value

    # タイトルを設定
    ax.set_title(f"{group_col}: {group_label}")
    ax.set_ylabel("Percent")


# subplots
# フォントを設定
plt.rcParams["font.family"] = "Meiryo"

group_labels = df["A"].unique()

n = len(group_labels)
cols = 3
rows = n // cols
rows += n % cols

fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 6))

for ax, label in zip(axs.ravel(), group_labels):
    draw_stacked_bar(ax, df, "A", "B", "C", label)
    # y軸の範囲を0から100に設定
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel("Percent")
    # 凡例の設定
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

# 余分なsubplotを削除
for i in range(len(group_labels), rows * cols):
    fig.delaxes(axs.flatten()[i])

fig.tight_layout()  # レイアウトを自動調整

plt.show()


# 連続出力
# フォントを設定
plt.rcParams["font.family"] = "Meiryo"

group_labels = df["A"].unique()

for label in group_labels:
    fig, ax = plt.subplots(figsize=(2, 6))
    draw_stacked_bar(ax, df, "A", "B", "C", label)
    # y軸の範囲を0から100に設定
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel("Percent")
    # 凡例の設定
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )
    fig.tight_layout()  # レイアウトを自動調整

    plt.show()


# パーセンテージ棒グラフ
def draw_bar(ax, df, group_col, value_col, label_col, group_label):
    """
    ax: Axes object to draw the graph on.
    df: DataFrame containing the data.
    group_col: Column name in df which contains group names.
    value_col: Column name in df which contains value data.
    label_col: Column name in df which contains label names.
    group_label: The specific label of the group to be drawn.
    """
    # グループごとの値の合計を計算
    total_sum = df.groupby(group_col)[value_col].sum()

    # 特定のグループのデータを取得
    group = df[df[group_col] == group_label]
    # 値の降順に並べ替え
    group = group.sort_values(value_col, ascending=False)
    # 値をパーセントに変換
    group[value_col] = group[value_col] / total_sum[group_label] * 100

    # 棒グラフを描画。指定のカラムの値をラベルとして使用。
    ax.bar(group[label_col], group[value_col], label=group_label)

    # タイトルを設定
    ax.set_title(f"{group_col}: {group_label}")
    ax.set_ylabel("%")


# 一括出力
# フォントを設定
plt.rcParams["font.family"] = "Meiryo"

group_labels = df["A"].unique()

n = len(group_labels)
cols = 2
rows = n // cols
rows += n % cols

fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))

for ax, label in zip(axs.ravel(), group_labels):
    draw_bar(ax, df, "A", "B", "C", label)
    # 凡例の設定
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

# 余分なsubplotを削除
for i in range(len(group_labels), rows * cols):
    fig.delaxes(axs.flatten()[i])

fig.tight_layout()  # レイアウトを自動調整

plt.show()


# 連続出力
# フォントを設定
plt.rcParams["font.family"] = "Meiryo"

group_labels = df["A"].unique()

for label in group_labels:
    fig, ax = plt.subplots(figsize=(6, 4))
    draw_bar(ax, df, "A", "B", "C", label)
    # 凡例の設定
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )
    fig.tight_layout()  # レイアウトを自動調整

    plt.show()


def draw_stacked_bar(ax, df, group_col, value_col, label_col, group_label):
    """
    Stacked bar plot を描画する関数

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        描画対象の Axes
    df: pandas.DataFrame
        データフレーム
    group_col: str
        グループ名が含まれている列名
    value_col: str
        値が含まれている列名
    label_col: str
        ラベルが含まれている列名
    group_label: str
        描画対象のグループのラベル名
    """
    # グループごとの値の合計を計算
    total_sum = df.groupby(group_col)[value_col].sum()

    # 特定のグループのデータを取得
    group = df[df[group_col] == group_label]
    # 値の降順に並べ替え
    group = group.sort_values(value_col, ascending=False)
    # 値をパーセントに変換
    group[value_col] = group[value_col] / total_sum[group_label] * 100

    # 棒グラフの底部を描画するための初期位置
    bottom = 0

    for value, label in zip(group[value_col], group[label_col]):
        # 棒グラフを描画。指定のカラムの値をラベルとして使用。
        ax.barh(group_label, value, left=bottom, label=label)

        # 次の棒グラフの底部を更新
        bottom += value

    # タイトルを設定
    ax.set_title(f"{group_col}: {group_label}")
    ax.set_xlabel("Percent")


# subplots
# フォントを設定
plt.rcParams["font.family"] = "Meiryo"

group_labels = df["A"].unique()

n = len(group_labels)
cols = 3
rows = n // cols
rows += n % cols

fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 2))

for ax, label in zip(axs.ravel(), group_labels):
    draw_stacked_bar(ax, df, "A", "B", "C", label)
    # x軸の範囲を0から100に設定
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel("Percent")
    # 凡例の設定
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

# 余分なsubplotを削除
for i in range(len(group_labels), rows * cols):
    fig.delaxes(axs.flatten()[i])

fig.tight_layout()  # レイアウトを自動調整

plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データの作成
data = {
    "A": ["Group1"] * 5 + ["Group2"] * 5 + ["Group3"] * 5,
    "B": np.random.randint(10, 50, size=15),
    "C": ["Label" + str(i) for i in range(1, 6)] * 3,
}

# データフレームの作成
df = pd.DataFrame(data)

# フォントの設定
plt.rcParams["font.family"] = "Meiryo"


def draw_graph(ax, df, group_col, value_col, label_col, group_label, graph_type="bar"):
    """
    特定のグラフを描画する関数

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        描画対象の Axes
    df: pandas.DataFrame
        データフレーム
    group_col: str
        グループ名が含まれている列名
    value_col: str
        値が含まれている列名
    label_col: str
        ラベルが含まれている列名
    group_label: str
        描画対象のグループのラベル名
    graph_type: str
        描画するグラフの種類 ('bar' または 'stacked_bar')
    """
    # グループごとの値の合計を計算
    total_sum = df.groupby(group_col)[value_col].sum()

    # 特定のグラフを描画
    if graph_type == "bar":
        ax.bar(
            df[label_col],
            df[value_col] / total_sum[group_label] * 100,
            label=group_label,
        )
        ax.set_ylabel("%")
    elif graph_type == "stacked_bar":
        bottom = 0
        for _, row in df.iterrows():
            ax.bar(
                group_label,
                row[value_col] / total_sum[group_label] * 100,
                bottom=bottom,
                label=row[label_col],
            )
            bottom += row[value_col]

    # タイトルを設定
    ax.set_title(f"{group_col}: {group_label}")


group_labels = df["A"].unique()
n = len(group_labels)
cols = 3
rows = n // cols
rows += n % cols

# 積み上げ棒グラフの描画
fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 2))
for ax, label in zip(axs.ravel(), group_labels):
    draw_graph(ax, df[df["A"] == label], "A", "B", "C", label, "stacked_bar")
    # 凡例の設定
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )
    # 余分なsubplotを削除
for i in range(len(group_labels), rows * cols):
    fig.delaxes(axs.flatten()[i])
fig.tight_layout()  # レイアウトを自動調整
plt.show()

# 棒グラフの描画
fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 2))
for ax, label in zip(axs.ravel(), group_labels):
    draw_graph(ax, df[df["A"] == label], "A", "B", "C", label, "bar")
    # 凡例の設定
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )
    # 余分なsubplotを削除
for i in range(len(group_labels), rows * cols):
    fig.delaxes(axs.flatten()[i])
fig.tight_layout()  # レイアウトを自動調整
plt.show()
