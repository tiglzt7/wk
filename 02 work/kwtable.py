import re
import numpy as np
from scipy.interpolate import UnivariateSpline
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import kw

Q = "capacity"
H = "total_head"
A = "current"
W = "output"
P = "shaft_power"


def transform_linedata(before_df):
    """
    指定されたDataFrameを変換します。変換は以下の手順で行います：
    1. 共通のカラムとテストごとのカラムを分けます。
    2. 新しい行を作成し、各テストの結果を新しいカラムにマッピングします。
    3. 新しいDataFrameを作成し、変換後のデータを保存します。

    Parameters
    ----------
    before_df : DataFrame
        変換前のDataFrame

    Returns
    -------
    after_df : DataFrame
        変換後のDataFrame
    """

    # テスト結果に関するカラムのリスト
    test_cols_list = ["Q", "H", "A", "W", "P"]

    # テスト結果のカラム名を変換する辞書
    test_cols_name_dict = {
        "Q": Q,
        "H": H,
        "A": A,
        "W": W,
        "P": P,
    }

    # 'Q', 'H', 'A', 'W', 'P'の各カラムを除く共通のカラムを取得
    common_cols_names = [
        col
        for col in before_df.columns
        if not re.match(r"^(%s)\d" % "|".join(test_cols_list), col)
    ]

    # 'Q'で始まるカラムの数をテストの数として数える
    num_tests = sum(1 for col in before_df.columns if col.startswith(test_cols_list[0]))

    # 新しい行を作成するための補助関数
    def create_rows(row):
        for j in range(1, num_tests + 1):
            # 各テスト結果のカラムを抽出し、カラム名を変換
            test_cols = {
                test_cols_name_dict[col]: row[col + str(j)] for col in test_cols_list
            }

            # 共通カラムとテスト結果のカラムを結合して新しい行を作成
            new_row = {**row[common_cols_names], **{"test_no": j}, **test_cols}
            yield pd.Series(new_row)

    # リスト内包表記を用いて各行に対してcreate_rows関数を適用し、新しい行を作成
    new_data = [
        row for idx in before_df.index for row in create_rows(before_df.loc[idx])
    ]

    # new_dataリストから新しいDataFrameを作成
    after_df = pd.DataFrame(new_data)

    # カラムの順序を設定
    column_order = common_cols_names + ["test_no"] + list(test_cols_name_dict.values())
    after_df = after_df[column_order]

    return after_df


def transform_DAdata(before_df):
    """
    データフレームを変換する関数

    以下のステップを実行します：
    1. 最大値の版数で絞り込みます。
    2. テスト結果の列名を変更します。
    3. 品番ごとにデータフレームを作成し、リストに追加します。
    4. 最終的なデータフレームを作成します。

    Args:
        before_df (DataFrame): 変換前のデータフレーム

    Returns:
        DataFrame: 変換後のデータフレーム
    """

    # テスト結果の列名を変更する辞書
    test_cols_name_dict = {
        "2": "total_head",
        "3": "revolution",
        "4": "current",
        "5": "shaft_power",
        "6": "input",
        "7": "input2",
        "8": "voltage",
        "9": "theorical_power",
        "10": "motor_eff",
        "11": "pump_eff",
    }

    # 最大値の版数で絞り込む
    df_main = before_df[
        before_df["版数"] == before_df.groupby("品番")["版数"].transform("max")
    ]

    # 品番の一意のリストを作成
    models = df_main["品番"].unique()

    df_list = []

    # 各品番に対してデータフレームを作成し、リストに追加
    for model in models:
        df = df_main[df_main["品番"] == model]
        df_capacity = df[df["Y軸計測項目区分"] == " "].reset_index(drop=True)
        df_capacity = df_capacity.rename(columns={"X軸項目値": "capacity"})

        for key, value in test_cols_name_dict.items():
            df_capacity[value] = df[df["Y軸計測項目区分"] == key]["Y軸計測値"].reset_index(
                drop=True
            )

        df_list.append(df_capacity)

    # 最終的なデータフレームを作成
    df_final = pd.concat(df_list, ignore_index=True)

    return df_final


def transform_df(before_df):
    """
    データフレームを変換する関数

    以下のステップを実行します：
    1. 最大値の版数で絞り込みます。
    2. テスト結果の列名を変更します。
    3. 品番ごとにデータフレームを作成し、リストに追加します。
    4. 最終的なデータフレームを作成します。

    Args:
        before_df (DataFrame): 変換前のデータフレーム

    Returns:
        DataFrame: 変換後のデータフレーム
    """

    # 最大値の版数で絞り込む
    df_main = before_df[
        before_df["版数"] == before_df.groupby("品番")["版数"].transform("max")
    ]

    # テスト結果の列名を変更する辞書
    test_cols_name_dict = {
        "2": "total_head",
        "3": "revolution",
        "4": "current",
        "5": "shaft_power",
        "6": "input",
        "7": "input2",
        "8": "voltage",
        "9": "theorical_power",
        "10": "motor_eff",
        "11": "pump_eff",
    }

    # 品番の一意のリストを作成
    models = df_main["品番"].unique()

    df_list = []

    # 各品番に対してデータフレームを作成し、リストに追加
    for model in models:
        df = df_main[df_main["品番"] == model]
        df_capacity = df[df["Y軸計測項目区分"] == " "].reset_index(drop=True)
        df_capacity = df_capacity.rename(columns={"X軸項目値": "capacity"})

        for key, value in test_cols_name_dict.items():
            df_capacity[value] = df[df["Y軸計測項目区分"] == key]["Y軸計測値"].reset_index(
                drop=True
            )

            df_list.append(df_capacity)

    # 最終的なデータフレームを作成
    df_final = pd.concat(df_list)

    return df_final


def VC_dupricate(df):
    """
    重複チェック+削除
    """
    df.loc[(df.duplicated(subset=["先品"], keep=False)) & (df["L1"] == 1)]
    df = df.drop(
        df[(df.duplicates(subset=["先品"], keep="False")) & (df["L1"] != 1)].index
    )

    return df


def compute_value_counts(df):
    """
    各列のユニークな値とその頻度を計算する関数。

    Parameters
    ----------
    df : pandas.DataFrame
        処理するデータフレーム。

    Returns
    -------
    value_counts_df : pandas.DataFrame
        各列のユニークな値とその頻度を持つデータフレーム。
        'column_name'列には対象の列名、'value'列にはユニークな値、'count'列にはその頻度が格納されています。

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 2, 3, 3, 3], 'B': ['a', 'a', 'b', 'b', 'b', 'c']})
    >>> compute_value_counts(df)
      value  count column_name
    0     3      3          A
    1     2      2          A
    2     1      1          A
    3     b      3          B
    4     a      2          B
    5     c      1          B
    """
    value_counts_list = []

    for col in df.columns:
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = ["value", "count"]
        value_counts["column_name"] = col
        value_counts_list.append(value_counts)

    value_counts_df = pd.concat(value_counts_list, ignore_index=True)

    return value_counts_df
