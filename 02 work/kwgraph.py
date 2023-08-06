import numpy as np
from numpy import polyfit, poly1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splrep, splev
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import seaborn as sns
import itertools
import kw


# 基本的な軸の設定
x_axis = "capacity"
x_axis_unit = "m3/min"
xlim_min = 0
xlim_max = None

y_axis1 = "total_head"
y_axis_unit1 = "m"

y_axis2 = "pump_eff"
y_axis_unit2 = "%"

y_axis3 = "shaft_power"
y_axis_unit3 = "kW"


def draw_plot(
    df,
    selected_models,
    x_axis=x_axis,
    X_axis_unit=x_axis_unit,
    y_axes=[y_axis1, y_axis2, y_axis3],
    y_axis_units=[y_axis_unit1, y_axis_unit2, y_axis_unit3],
    y_axis_lims=[(0, None), (0, None), (0, None)],
    types_of_plot=["scatter", "polyfit"],
    polyfit_degree=5,
    num_rows=2,
):
    # モデル毎の色を指定
    model_colors = ["k", "r", "b"]

    # 軸毎の色を指定
    axis_colors = ["k", "k", "k"]

    # モデルの数に基づいてサブプロットの列数を計算
    num_cols = len(selected_models) // num_rows
    if len(selected_models) % num_rows != 0:  # モデルの数が行数で割り切れない場合
        num_cols += 1  # 列数を1つ増やす

    # Figureとarray of axesを作成
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
    axs = axs.ravel()

    for i, models in enumerate(selected_models):
        color_index = 0  # ここでcolor_indexをリセット
        x_max = 0  # x_maxを初期化
        for model in models:
            color = model_colors[color_index]  # model_colors listから色を選択
            color_index += 1

            df_model = df[df["model"] == model]

            # 現在のモデルに対するxの最大値を取得
            x_max_model = df_model[x_axis].max()

            # これが現在のx_maxより大きければ、x_maxを更新
            if x_max_model > x_max:
                x_max = x_max_model

            # 現在のaxesを選択
            ax = axs[i]
            ax.set_xlabel(f"{x_axis} ({X_axis_unit})")
            ax.grid(True)

            # 複数のy軸をプロット
            for j, y_axis in enumerate(y_axes):
                y_axis_unit = y_axis_units[j]
                y_axis_lim = y_axis_lims[j]

                if j > 0:  # 2つ目以降のy軸の場合、新しいAxesを作成
                    ax = ax.twinx()

                    # Y軸の位置を調整
                    ax.spines["left"].set_position(("axes", -0.1 * j))  # 軸の位置を調整
                    ax.yaxis.set_label_position("left")  # ラベルの位置を調整
                    ax.yaxis.set_ticks_position("left")  # ティックの位置を調整

                ax.set_ylabel(f"{y_axis} ({y_axis_unit})", color=axis_colors[j])
                ax.set_ylim(y_axis_lim)

                # データ点のプロット
                if "scatter" in types_of_plot:
                    ax.scatter(df_model[x_axis], df_model[y_axis], color=color)

                # Polyfit
                if "polyfit" in types_of_plot:
                    poly_coeffs = polyfit(
                        df_model[x_axis], df_model[y_axis], deg=polyfit_degree
                    )
                    poly = poly1d(poly_coeffs)
                    xs = np.linspace(
                        df_model[x_axis].min(), df_model[x_axis].max(), 500
                    )
                    ax.plot(xs, poly(xs), color=color, label=model)

                # スプライン補間
                if "spline" in types_of_plot:
                    tck = splrep(df_model[x_axis], df_model[y_axis], k=3)
                    xs = np.linspace(
                        df_model[x_axis].min(), df_model[x_axis].max(), 500
                    )
                    ys = splev(xs, tck)
                    ax.plot(xs, ys, color=color, label=model)

                ax.tick_params(axis="y", labelcolor=axis_colors[j])
                ax.legend(loc="best")

            ax.set_title(f'Models: {", ".join(models)}')

        # すべてのモデルについてループが終わった後に、このサブプロットのxlimを設定
        ax.set_xlim(0, x_max + x_max * 0.05)

    # 使用しないサブプロットを削除
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axs[j])

    fig.tight_layout()
    plt.show()


# fill betweenのグラフ
def draw_plot2(
    df,
    selected_models,
    x_axis=x_axis,
    X_axis_unit=x_axis_unit,
    y_axes=[y_axis1, y_axis2, y_axis3],
    y_axis_units=[y_axis_unit1, y_axis_unit2, y_axis_unit3],
    y_axis_lims=[(0, None), (0, None), (0, None)],
    types_of_plot=["scatter", "polyfit"],
    polyfit_degree=5,
    num_rows=2,
):
    # モデル毎の色を指定
    model_colors = ["k", "r", "b"]

    # 軸毎の色を指定
    axis_colors = ["k", "k", "k"]

    # モデルの数に基づいてサブプロットの列数を計算
    num_cols = len(selected_models) // num_rows
    if len(selected_models) % num_rows != 0:  # モデルの数が行数で割り切れない場合
        num_cols += 1  # 列数を1つ増やす

    # Figureとarray of axesを作成
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
    axs = axs.ravel()

    for i, models in enumerate(selected_models):
        color_index = 0  # ここでcolor_indexをリセット
        x_max = 0  # x_maxを初期化
        for model in models:
            color = model_colors[color_index]  # model_colors listから色を選択
            color_index += 1

            df_model = df[df["model"] == model]

            # 現在のモデルに対するxの最大値を取得
            x_max_model = df_model[x_axis].max()

            # これが現在のx_maxより大きければ、x_maxを更新
            if x_max_model > x_max:
                x_max = x_max_model

            # 現在のaxesを選択
            ax = axs[i]
            ax.set_xlabel(f"{x_axis} ({X_axis_unit})")
            ax.grid(True)

            # 複数のy軸をプロット
            for j, y_axis in enumerate(y_axes):
                y_axis_unit = y_axis_units[j]
                y_axis_lim = y_axis_lims[j]

                if j > 0:  # 2つ目以降のy軸の場合、新しいAxesを作成
                    ax = ax.twinx()

                    # Y軸の位置を調整
                    ax.spines["left"].set_position(("axes", -0.1 * j))  # 軸の位置を調整
                    ax.yaxis.set_label_position("left")  # ラベルの位置を調整
                    ax.yaxis.set_ticks_position("left")  # ティックの位置を調整

                ax.set_ylabel(f"{y_axis} ({y_axis_unit})", color=axis_colors[j])
                ax.set_ylim(y_axis_lim)

                # データ点のプロット
                if "scatter" in types_of_plot:
                    ax.scatter(df_model[x_axis], df_model[y_axis], color=color)

                # Polyfit
                if "polyfit" in types_of_plot:
                    poly_coeffs = np.polyfit(
                        df_model[x_axis], df_model[y_axis], deg=polyfit_degree
                    )
                    poly = np.poly1d(poly_coeffs)
                    xs = np.linspace(
                        df_model[x_axis].min(), df_model[x_axis].max(), 500
                    )
                    ax.plot(xs, poly(xs), color=color, label=model)
                    ax.fill_between(
                        xs, poly(xs), color=color, alpha=0.1
                    )  # Add fill between plot and x-axis

                # スプライン補間
                if "spline" in types_of_plot:
                    tck = splrep(df_model[x_axis], df_model[y_axis], k=3)
                    xs = np.linspace(
                        df_model[x_axis].min(), df_model[x_axis].max(), 500
                    )
                    ys = splev(xs, tck)
                    ax.plot(xs, ys, color=color, label=model)
                    ax.fill_between(
                        xs, ys, color=color, alpha=0.1
                    )  # Add fill between plot and x-axis

                ax.tick_params(axis="y", labelcolor=axis_colors[j])
                ax.legend(loc="best")

            ax.set_title(f'Models: {", ".join(models)}')

        # すべてのモデルについてループが終わった後に、このサブプロットのxlimを設定
        ax.set_xlim(0, x_max + x_max * 0.05)

    # 使用しないサブプロットを削除
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axs[j])

    fig.tight_layout()
    plt.show()


def draw_plot4(
    df,
    selected_models,
    x_axis=x_axis,
    X_axis_unit=x_axis_unit,
    y_axes=[y_axis1, y_axis2, y_axis3],
    y_axis_units=[y_axis_unit1, y_axis_unit2, y_axis_unit3],
    y_axis_lims=[(0, None), (0, None), (0, None)],
    types_of_plot=["scatter", "polyfit"],
    polyfit_degree=5,
    num_rows=2,
    colormap="viridis",  # Added argument for colormap
    use_colormap=True,  # Added argument to switch between colormap and color list
):
    flattened_models = [model for sublist in selected_models for model in sublist]

    # 軸毎の色を指定
    axis_colors = ["k", "k", "k"]

    # モデルの数に基づいてサブプロットの列数を計算
    num_cols = len(selected_models) // num_rows
    if len(selected_models) % num_rows != 0:  # モデルの数が行数で割り切れない場合
        num_cols += 1  # 列数を1つ増やす

    # Figureとarray of axesを作成
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
    axs = axs.ravel()

    for i, models in enumerate(selected_models):
        x_max = 0  # x_maxを初期化

        if use_colormap:  # If colormap is to be used
            cmap = cm.get_cmap(colormap, len(models))
            model_colors = iter([cmap(i) for i in range(cmap.N)])
        else:
            model_colors = itertools.cycle(["k", "r", "b"])  # cycleで無限リストを生成

        for model in models:
            color = next(model_colors)
            df_model = df[df["model"] == model]

            # 現在のモデルに対するxの最大値を取得
            x_max_model = df_model[x_axis].max()

            # これが現在のx_maxより大きければ、x_maxを更新
            if x_max_model > x_max:
                x_max = x_max_model

            # 現在のaxesを選択
            ax = axs[i]
            ax.set_xlabel(f"{x_axis} ({X_axis_unit})")
            ax.grid(True)

            # 複数のy軸をプロット
            for j, y_axis in enumerate(y_axes):
                y_axis_unit = y_axis_units[j]
                y_axis_lim = y_axis_lims[j]

                if j > 0:  # 2つ目以降のy軸の場合、新しいAxesを作成
                    ax = ax.twinx()

                    # Y軸の位置を調整
                    ax.spines["left"].set_position(("axes", -0.1 * j))  # 軸の位置を調整
                    ax.yaxis.set_label_position("left")  # ラベルの位置を調整
                    ax.yaxis.set_ticks_position("left")  # ティックの位置を調整

                ax.set_ylabel(f"{y_axis} ({y_axis_unit})", color=axis_colors[j])
                ax.set_ylim(y_axis_lim)

                # データ点のプロット
                if "scatter" in types_of_plot:
                    ax.scatter(df_model[x_axis], df_model[y_axis], color=color)

                # Polyfit
                if "polyfit" in types_of_plot:
                    poly_coeffs = polyfit(
                        df_model[x_axis], df_model[y_axis], deg=polyfit_degree
                    )
                    poly = np.poly1d(poly_coeffs)
                    xs = np.linspace(
                        df_model[x_axis].min(), df_model[x_axis].max(), 500
                    )
                    ax.plot(xs, poly(xs), color=color, label=model)

                # スプライン補間
                if "spline" in types_of_plot:
                    tck = splrep(df_model[x_axis], df_model[y_axis], k=3)
                    xs = np.linspace(
                        df_model[x_axis].min(), df_model[x_axis].max(), 500
                    )
                    ys = splev(xs, tck)
                    ax.plot(xs, ys, color=color, label=model)

                ax.tick_params(axis="y", labelcolor=axis_colors[j])
                handles, labels = ax.get_legend_handles_labels()
                print(labels)
                ax.legend(loc="best")

        # すべてのモデルについてループが終わった後に、このサブプロットのxlimを設定
        ax.set_xlim(0, x_max + x_max * 0.05)

        # モデルの数が多い場合、タイトルを簡略化
        if len(models) > 5:  # 5はタイトルが長くなりすぎると判断するモデル数の閾値
            ax.set_title(f"Number of Models: {len(models)}")
        else:
            ax.set_title(f'Models: {", ".join(models)}')

    # 使用しないサブプロットを削除
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axs[j])

    fig.tight_layout()
    plt.show()


def draw_plot4b(
    df,
    selected_models,
    x_axis=x_axis,
    X_axis_unit=x_axis_unit,
    y_axes=[y_axis1, y_axis2, y_axis3],
    y_axis_units=[y_axis_unit1, y_axis_unit2, y_axis_unit3],
    y_axis_lims=[(0, None), (0, None), (0, None)],
    types_of_plot=["scatter", "polyfit"],
    polyfit_degree=5,
    num_rows=2,
    colormap="viridis",
    use_colormap=True,
):
    flattened_models = [model for sublist in selected_models for model in sublist]

    # 軸毎の色を指定
    axis_colors = ["k", "k", "k"]

    # モデルの数に基づいてサブプロットの列数を計算
    num_cols = len(selected_models) // num_rows
    if len(selected_models) % num_rows != 0:  # モデルの数が行数で割り切れない場合
        num_cols += 1  # 列数を1つ増やす

    # Figureとarray of axesを作成
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
    axs = axs.ravel()

    for i, models in enumerate(selected_models):
        model_color_map = {}  # このサブプロットで使用されるモデルと色のマッピング

        x_max = 0  # x_maxを初期化
        plotted_models = set()

        if use_colormap:  # If colormap is to be used
            cmap = cm.get_cmap(colormap, len(models))
            model_colors = iter([cmap(i) for i in range(cmap.N)])
        else:
            model_colors = itertools.cycle(["k", "r", "b"])  # cycleで無限リストを生成

        for model in models:
            color = next(model_colors)
            model_color_map[model] = color
            df_model = df[df["model"] == model]

            # 現在のモデルに対するxの最大値を取得
            x_max_model = df_model[x_axis].max()

            # これが現在のx_maxより大きければ、x_maxを更新
            if x_max_model > x_max:
                x_max = x_max_model

            # 現在のaxesを選択
            ax = axs[i]
            ax.set_xlabel(f"{x_axis} ({X_axis_unit})")
            ax.grid(True)

            # 複数のy軸をプロット
            for j, y_axis in enumerate(y_axes):
                y_axis_unit = y_axis_units[j]
                y_axis_lim = y_axis_lims[j]

                if j > 0:  # 2つ目以降のy軸の場合、新しいAxesを作成
                    ax = ax.twinx()

                    # Y軸の位置を調整
                    ax.spines["left"].set_position(("axes", -0.1 * j))  # 軸の位置を調整
                    ax.yaxis.set_label_position("left")  # ラベルの位置を調整
                    ax.yaxis.set_ticks_position("left")  # ティックの位置を調整

                ax.set_ylabel(f"{y_axis} ({y_axis_unit})", color=axis_colors[j])
                ax.set_ylim(y_axis_lim)

                if model not in plotted_models:
                    label = model
                    plotted_models.add(model)
                else:
                    label = None  # 既にこのモデルはプロットされているので、レジェンドには追加しない

                # データ点のプロット
                if "scatter" in types_of_plot:
                    ax.scatter(df_model[x_axis], df_model[y_axis], color=color)

                # Polyfit
                if "polyfit" in types_of_plot:
                    poly_coeffs = polyfit(
                        df_model[x_axis], df_model[y_axis], deg=polyfit_degree
                    )
                    poly = np.poly1d(poly_coeffs)
                    xs = np.linspace(
                        df_model[x_axis].min(), df_model[x_axis].max(), 500
                    )
                    ax.plot(xs, poly(xs), color=color)

                # スプライン補間
                if "spline" in types_of_plot:
                    tck = splrep(df_model[x_axis], df_model[y_axis], k=3)
                    xs = np.linspace(
                        df_model[x_axis].min(), df_model[x_axis].max(), 500
                    )
                    ys = splev(xs, tck)
                    ax.plot(xs, ys, color=color)

                # ax.tick_params(axis="y", labelcolor=axis_colors[j])
                # handles, labels = ax.get_legend_handles_labels()
                # print(labels)
                # ax.legend(loc="best")

        handles = [
            Line2D([0], [0], color=color, lw=2)
            for model, color in model_color_map.items()
        ]
        labels = list(model_color_map.keys())
        ax.legend(handles, labels, loc="best")

        # すべてのモデルについてループが終わった後に、このサブプロットのxlimを設定
        ax.set_xlim(0, x_max + x_max * 0.05)

        # モデルの数が多い場合、タイトルを簡略化
        if len(models) > 5:  # 5はタイトルが長くなりすぎると判断するモデル数の閾値
            ax.set_title(f"Number of Models: {len(models)}")
        else:
            ax.set_title(f'Models: {", ".join(models)}')

    # 使用しないサブプロットを削除
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axs[j])

    fig.tight_layout()
    plt.show()


def plot_scatter(ax, df_model, x_axis, y_axis, color):
    ax.scatter(df_model[x_axis], df_model[y_axis], color=color)


def plot_polyfit(ax, df_model, x_axis, y_axis, color, polyfit_degree=5):
    poly_coeffs = polyfit(df_model[x_axis], df_model[y_axis], deg=polyfit_degree)
    poly = np.poly1d(poly_coeffs)
    xs = np.linspace(df_model[x_axis].min(), df_model[x_axis].max(), 500)
    ax.plot(xs, poly(xs), color=color)


def plot_spline(ax, df_model, x_axis, y_axis, color):
    tck = splrep(df_model[x_axis], df_model[y_axis], k=3)
    xs = np.linspace(df_model[x_axis].min(), df_model[x_axis].max(), 500)
    ys = splev(xs, tck)
    ax.plot(xs, ys, color=color)


def draw_plot4c(
    df,
    selected_models,
    x_axis=x_axis,
    X_axis_unit=x_axis_unit,
    y_axes=[y_axis1, y_axis2, y_axis3],
    y_axis_units=[y_axis_unit1, y_axis_unit2, y_axis_unit3],
    y_axis_lims=[(0, None), (0, None), (0, None)],
    types_of_plot=["scatter", "polyfit"],
    polyfit_degree=5,
    num_rows=2,
    colormap="viridis",
    use_colormap=True,
):
    """
    複数のモデルと複数のy軸でのデータを表示するプロットを生成する関数。

    Parameters:
    - df: データフレーム (pandas DataFrame)
    - selected_models: 各サブプロットで表示するモデルのリストのリスト
    - x_axis: x軸のデータのカラム名
    - X_axis_unit: x軸の単位の文字列
    - y_axes: y軸のデータのカラム名のリスト
    - y_axis_units: y軸の単位のリスト
    - y_axis_lims: y軸の限界値のリスト (tuple形式)
    - types_of_plot: 使用するプロットのタイプ ("scatter", "polyfit" など)
    - polyfit_degree: polyfitの次数 (整数)
    - num_rows: サブプロットの行数
    - colormap: 使用するカラーマップの名前
    - use_colormap: カラーマップを使用するかどうかのブール値

    Returns:
    - None (直接プロットが表示される)
    """

    flattened_models = [model for sublist in selected_models for model in sublist]

    # 軸毎の色を指定
    axis_colors = ["k", "k", "k"]

    # モデルの数に基づいてサブプロットの列数を計算
    num_cols = len(selected_models) // num_rows
    if len(selected_models) % num_rows != 0:  # モデルの数が行数で割り切れない場合
        num_cols += 1  # 列数を1つ増やす

    # Figureとarray of axesを作成
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
    axs = axs.ravel()

    for i, models in enumerate(selected_models):
        model_color_map = {}  # このサブプロットで使用されるモデルと色のマッピング

        x_max = 0  # x_maxを初期化
        plotted_models = set()

        # 線の色を設定
        if use_colormap:  # If colormap is to be used
            cmap = cm.get_cmap(colormap, len(models))
            model_colors = iter([cmap(i) for i in range(cmap.N)])
        else:
            model_colors = itertools.cycle(["k", "r", "b"])  # cycleで無限リストを生成

        for model in models:
            color = next(model_colors)
            model_color_map[model] = color
            df_model = df[df["model"] == model]

            # 現在のモデルに対するxの最大値を取得
            x_max_model = df_model[x_axis].max()

            # これが現在のx_maxより大きければ、x_maxを更新
            if x_max_model > x_max:
                x_max = x_max_model

            # 現在のaxesを選択
            ax = axs[i]
            ax.set_xlabel(f"{x_axis} ({X_axis_unit})")
            ax.grid(True)

            # 複数のy軸をプロット
            for j, y_axis in enumerate(y_axes):
                y_axis_unit = y_axis_units[j]
                y_axis_lim = y_axis_lims[j]

                if j > 0:  # 2つ目以降のy軸の場合、新しいAxesを作成
                    ax = ax.twinx()

                    # Y軸の位置を調整
                    ax.spines["left"].set_position(("axes", -0.1 * j))  # 軸の位置を調整
                    ax.yaxis.set_label_position("left")  # ラベルの位置を調整
                    ax.yaxis.set_ticks_position("left")  # ティックの位置を調整

                ax.set_ylabel(f"{y_axis} ({y_axis_unit})", color=axis_colors[j])
                ax.set_ylim(y_axis_lim)

                if model not in plotted_models:
                    label = model
                    plotted_models.add(model)
                else:
                    label = None  # 既にこのモデルはプロットされているので、レジェンドには追加しない

                # データ点のプロット
                if "scatter" in types_of_plot:
                    ax.scatter(df_model[x_axis], df_model[y_axis], color=color)

                # Polyfit
                if "polyfit" in types_of_plot:
                    poly_coeffs = polyfit(
                        df_model[x_axis], df_model[y_axis], deg=polyfit_degree
                    )
                    poly = np.poly1d(poly_coeffs)
                    xs = np.linspace(
                        df_model[x_axis].min(), df_model[x_axis].max(), 500
                    )
                    ax.plot(xs, poly(xs), color=color)

                # スプライン補間
                if "spline" in types_of_plot:
                    tck = splrep(df_model[x_axis], df_model[y_axis], k=3)
                    xs = np.linspace(
                        df_model[x_axis].min(), df_model[x_axis].max(), 500
                    )
                    ys = splev(xs, tck)
                    ax.plot(xs, ys, color=color)

                # ax.tick_params(axis="y", labelcolor=axis_colors[j])
                # handles, labels = ax.get_legend_handles_labels()
                # print(labels)
                # ax.legend(loc="best")

        handles = [
            Line2D([0], [0], color=color, lw=2)
            for model, color in model_color_map.items()
        ]
        labels = list(model_color_map.keys())
        ax.legend(handles, labels, loc="best")

        # すべてのモデルについてループが終わった後に、このサブプロットのxlimを設定
        ax.set_xlim(0, x_max + x_max * 0.05)

        # モデルの数が多い場合、タイトルを簡略化
        if len(models) > 5:  # 5はタイトルが長くなりすぎると判断するモデル数の閾値
            ax.set_title(f"Number of Models: {len(models)}")
        else:
            ax.set_title(f'Models: {", ".join(models)}')

    # 使用しないサブプロットを削除
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axs[j])

    fig.tight_layout()
    plt.show()


def initialize_subplots(selected_models, num_rows):
    """サブプロットを初期化する関数"""
    num_cols = len(selected_models) // num_rows
    if len(selected_models) % num_rows != 0:
        num_cols += 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
    return fig, axs.ravel()


def plot_for_model(
    ax,
    df,
    model,
    x_axis,
    X_axis_unit,
    y_axes,
    y_axis_units,
    y_axis_lims,
    types_of_plot,
    polyfit_degree,
    colormap,
    use_colormap,
):
    """指定されたモデルのためのプロットを生成する関数"""
    # 以前のコードのプロットに関する部分をここに移動...


def set_subplot_title(ax, models):
    if len(models) > 5:
        ax.set_title(f"Number of Models: {len(models)}")
    else:
        ax.set_title(f'Models: {", ".join(models)}')


def create_subplot(num_rows, num_cols):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
    return fig, axs.ravel()


def configure_axis(
    ax, x_axis, X_axis_unit, y_axis, y_axis_unit, y_axis_lim, axis_color
):
    ax.set_xlabel(f"{x_axis} ({X_axis_unit})")
    ax.grid(True)
    ax.set_ylabel(f"{y_axis} ({y_axis_unit})", color=axis_color)
    ax.set_ylim(y_axis_lim)


import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from numpy.polynomial.polynomial import polyfit


def plot_scatter(ax, df_model, x_axis, y_axis, color):
    ax.scatter(df_model[x_axis], df_model[y_axis], color=color)


def plot_polyfit(ax, df_model, x_axis, y_axis, color, polyfit_degree=5):
    poly_coeffs = polyfit(df_model[x_axis], df_model[y_axis], deg=polyfit_degree)
    poly = np.poly1d(poly_coeffs)
    xs = np.linspace(df_model[x_axis].min(), df_model[x_axis].max(), 500)
    ax.plot(xs, poly(xs), color=color)


def plot_spline(ax, df_model, x_axis, y_axis, color):
    tck = splrep(df_model[x_axis], df_model[y_axis], k=3)
    xs = np.linspace(df_model[x_axis].min(), df_model[x_axis].max(), 500)
    ys = splev(xs, tck)
    ax.plot(xs, ys, color=color)
