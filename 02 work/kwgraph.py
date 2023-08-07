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
    colormap="jet",
    use_colormap=True,
    linestyles=["-", "--", "-.", ":"],
    markers=["o", "s", "^", "v"],
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
        model_linestyle_map = {}
        model_marker_map = {}

        x_max = 0  # x_maxを初期化

        # 線の色を設定
        if use_colormap:  # If colormap is to be used
            cmap = cm.get_cmap(colormap, len(models))
            model_colors = iter([cmap(i) for i in range(cmap.N)])
        else:
            model_colors = itertools.cycle(["k", "r", "b"])  # cycleで無限リストを生成

        # 線種を設定
        model_linestyles = itertools.cycle(linestyles)
        # マーカーを設定
        model_markers = itertools.cycle(markers)

        for model in models:
            color = next(model_colors)
            linestyle = next(model_linestyles)
            marker = next(model_markers)

            model_color_map[model] = color
            model_linestyle_map[model] = linestyle
            model_marker_map[model] = marker

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
                    ax.scatter(
                        df_model[x_axis], df_model[y_axis], color=color, marker=marker
                    )

                # Polyfit
                if "polyfit" in types_of_plot:
                    poly_coeffs = polyfit(
                        df_model[x_axis], df_model[y_axis], deg=polyfit_degree
                    )
                    poly = np.poly1d(poly_coeffs)
                    xs = np.linspace(
                        df_model[x_axis].min(), df_model[x_axis].max(), 500
                    )
                    ax.plot(xs, poly(xs), color=color, linestyle=linestyle)

                # スプライン補間
                if "spline" in types_of_plot:
                    tck = splrep(df_model[x_axis], df_model[y_axis], k=3)
                    xs = np.linspace(
                        df_model[x_axis].min(), df_model[x_axis].max(), 500
                    )
                    ys = splev(xs, tck)
                    ax.plot(xs, ys, color=color, linestyle=linestyle)

        handles = [
            Line2D(
                [0],
                [0],
                color=model_color_map[model],
                linestyle=model_linestyle_map[model],
                marker=model_marker_map[model],
                lw=2,
            )  # ラインスタイルとマーカーを反映
            for model in models
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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
from matplotlib import cm
from scipy.interpolate import splrep, splev


def draw_plot4d(
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
    colormap="jet",
    use_colormap=True,
    linestyles=["-", "--", "-.", ":"],
    markers=["o", "s", "^", "v"],
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

    # df が期待するカラムを持っているかのチェック
    required_columns = [x_axis] + y_axes + ["model"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in dataframe.")

    # selected_models の形式チェック
    if not all(isinstance(models, list) for models in selected_models):
        raise TypeError("'selected_models' should be a list of lists.")
    if not all(
        isinstance(model, str) for sublist in selected_models for model in sublist
    ):
        raise TypeError("All models in 'selected_models' should be strings.")

    # types_of_plot の形式チェック
    valid_plot_types = ["scatter", "polyfit", "spline"]
    for plot_type in types_of_plot:
        if plot_type not in valid_plot_types:
            raise ValueError(
                f"Unknown plot type '{plot_type}'. Supported types are {', '.join(valid_plot_types)}."
            )

    def calculate_num_cols(selected_models, num_rows):
        """Calculate the number of columns for subplots."""
        cols = len(selected_models) // num_rows
        return cols + 1 if len(selected_models) % num_rows else cols

    def get_model_style_maps(models):
        """Return style maps for given models."""
        model_color_map, model_linestyle_map, model_marker_map = {}, {}, {}

        model_colors = (
            [cm.get_cmap(colormap, len(models))(i) for i in range(len(models))]
            if use_colormap
            else itertools.cycle(["k", "r", "b"])
        )
        model_linestyles = itertools.cycle(linestyles)
        model_markers = itertools.cycle(markers)

        for model in models:
            model_color_map[model] = next(model_colors)
            model_linestyle_map[model] = next(model_linestyles)
            model_marker_map[model] = next(model_markers)

        return model_color_map, model_linestyle_map, model_marker_map

    num_cols = calculate_num_cols(selected_models, num_rows)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
    axs = axs.ravel()

    for i, models in enumerate(selected_models):
        model_color_map, model_linestyle_map, model_marker_map = get_model_style_maps(
            models
        )
        ax = axs[i]
        x_max = df[df["model"].isin(models)][x_axis].max()

        for j, y_axis in enumerate(y_axes):
            y_unit = y_axis_units[j]
            ax = ax if j == 0 else ax.twinx()

            if j > 0:
                ax.spines["left"].set_position(("axes", -0.1 * j))
                ax.yaxis.set_label_position("left")
                ax.yaxis.set_ticks_position("left")

            ax.set_ylabel(f"{y_axis} ({y_unit})", color="k")
            ax.set_ylim(y_axis_lims[j])

            for model in models:
                df_model = df[df["model"] == model]

                if "scatter" in types_of_plot:
                    ax.scatter(
                        df_model[x_axis],
                        df_model[y_axis],
                        color=model_color_map[model],
                        marker=model_marker_map[model],
                    )

                if "polyfit" in types_of_plot:
                    poly_coeffs = np.polyfit(
                        df_model[x_axis], df_model[y_axis], polyfit_degree
                    )
                    poly = np.poly1d(poly_coeffs)
                    xs = np.linspace(0, x_max, 500)
                    ax.plot(
                        xs,
                        poly(xs),
                        color=model_color_map[model],
                        linestyle=model_linestyle_map[model],
                    )

                if "spline" in types_of_plot:
                    tck = splrep(df_model[x_axis], df_model[y_axis], k=3)
                    xs = np.linspace(0, x_max, 500)
                    ys = splev(xs, tck)
                    ax.plot(
                        xs,
                        ys,
                        color=model_color_map[model],
                        linestyle=model_linestyle_map[model],
                    )

            handles = [
                Line2D(
                    [0],
                    [0],
                    color=model_color_map[model],
                    linestyle=model_linestyle_map[model],
                    marker=model_marker_map[model],
                    lw=2,
                )
                for model in models
            ]
            labels = models
            ax.legend(handles, labels, loc="best")

            ax.set_xlim(0, x_max + 0.05 * x_max)

        ax.set_title(
            f'Models: {", ".join(models)}'
            if len(models) <= 5
            else f"Number of Models: {len(models)}"
        )

    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axs[j])

    fig.tight_layout()
    plt.show()
