import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog


def _tk_instance():
    """tkinterのインスタンスを作成し、rootを返します。"""

    root = tk.Tk()
    root.withdraw()

    # topmost指定
    root.attributes("-topmost", True)
    root.withdraw()
    root.lift()
    root.focus_force()

    return root


def get_file_path(file_type=None):
    """
    ファイル選択ダイアログを表示し、ユーザーが選択したファイルのパスを取得します。

    Args:
        file_type (str, optional): 選択可能なファイルの種類。例えば、"txt"を指定すると.txtファイルのみが選択可能になります。
        file_typeがNoneまたは指定されていない場合、全ての種類のファイルが選択可能になります。

    Returns:
        str or None: ユーザーが選択したファイルのパス。ファイルが選択されなかった場合はNoneを返します。
    """

    _tk_instance()

    # ファイル選択ダイアログを表示し、選択したファイルのパスを取得
    if file_type:
        filetypes = ((file_type, f"*.{file_type}"),)
    else:
        filetypes = (("All files", "*.*"),)

    file_path = filedialog.askopenfilename(
        initialdir="..", title="Select file", filetypes=filetypes
    )

    # ファイルが選択されていない場合はNoneを返す
    if not file_path:
        print("ファイルが選択されていません")
        return None

    return file_path


def get_folder_path():
    """
    フォルダ選択ダイアログを表示し、ユーザーが選択したフォルダのパスを取得します。

    Returns:
        str or None: ユーザーが選択したフォルダのパス。フォルダが選択されなかった場合はNoneを返します。
    """

    _tk_instance()

    # フォルダ選択ダイアログを表示し、選択したフォルダのパスを取得
    folder_path = filedialog.askdirectory(initialdir="..", title="Select folder")

    # フォルダが選択されていない場合はNoneを返す
    if not folder_path:
        print("フォルダが選択されていません")
        return None

    return folder_path


# def save_file_path(defaultextension):
#     _tk_instance()

#     # ファイル選択ダイアログを表示し、ファイルを保存するパスを取得
#     file_path = filedialog.asksaveasfilename(
#         initialdir="./", title="Save as", defaultextension=defaultextension
#     )

#     return file_path


def getdf_xlsx(file_path=None):
    """
    指定されたxlsxファイルをpandas DataFrameに読み込みます。file_pathが指定されていない場合は、ファイル選択ダイアログが表示されます。

    Args:
        file_path (str, optional): xlsxファイルのパス。指定されていない場合、ファイル選択ダイアログが表示されます。

    Returns:
        dict: キーがシート名、値が各シートの内容を表すDataFrameの辞書。
    """
    # file_pathが指定されていない場合はファイル選択ダイアログを開く
    if file_path is None:
        file_path = get_file_path("xlsx")

    # 選択されたファイルをpandasで読み込む。品番をゼロ埋めするためstrで読み込む。
    xls = pd.ExcelFile(file_path)
    dfs = {
        sheet_name: xls.parse(sheet_name, dtype={"元品番": str, "先品番": str})
        for sheet_name in xls.sheet_names
    }

    return dfs


def getdf_xlsx_test(file_path=None):
    """
    指定されたxlsxファイルをpandas DataFrameに読み込みます。file_pathが指定されていない場合は、ファイル選択ダイアログが表示されます。

    Args:
        file_path (str, optional): xlsxファイルのパス。指定されていない場合、ファイル選択ダイアログが表示されます。

    Returns:
        dict: キーがシート名、値が各シートの内容を表すDataFrameの辞書。
    """
    # file_pathが指定されていない場合はファイル選択ダイアログを開く
    if file_path is None:
        file_path = get_file_path("xlsx")

    # 選択されたファイルをpandasで読み込む。品番をゼロ埋めするためstrで読み込む。
    xls = pd.ExcelFile(file_path)
    dfs = {
        sheet_name: xls.parse(sheet_name, dtype={"元品番": str, "先品番": str})
        for sheet_name in xls.sheet_names
    }

    # 辞書のキー（シート名）の一覧をプリント
    print(f"シート名の一覧: {list(dfs.keys())}")

    return dfs


def writedf_xlsx(df_dict):
    """
    引数で指定された辞書に格納されたデータフレームを、指定されたExcelファイルに書き込みます。

    Args:
        df_dict (dict): シート名をキーとし、対応するデータフレームを値とする辞書。
        file_path (str): データを保存するExcelファイルのパス。

    Note:
        辞書の各エントリーに対応するExcelのシートが作成され、そのシートに対応するデータフレームのデータが書き込まれます。
        すでに同名のシートが存在する場合は、そのシートは上書きされます。
        また、データフレームのインデックスはExcelに書き込まれません。
    """
    _tk_instance()

    file_name = filedialog.asksaveasfilename(
        initialdir="..", title="Save as", defaultextension="xlsx"
    )

    with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def VC_dupricate(df):
    """
    重複チェック+削除
    """
    df.loc[(df.duplicated(subset=["先品"], keep=False)) & (df["L1"] == 1)]
    df = df.drop(
        df[(df.duplicates(subset=["先品"], keep="False")) & (df["L1"] != 1)].index
    )

    return df


def getdf(file_path, dtype={"元品": str, "先品": str}):
    xls = pd.ExcelFile(file_path)
    df = {
        sheet_name: xls.parse(sheet_name, dtype=dtype) for sheet_name in xls.sheet_names
    }
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


# sheet_df_dict = {}  # シート名とデータフレームを関連付ける辞書


def add_df_to_dict(sheet_name, df):
    """
    指定されたシート名とデータフレームを辞書に追加します。

    Args:
        sheet_name (str): シート名。
        df (DataFrame): シート名に関連付けるデータフレーム。
    """
    sheet_df_dict[sheet_name] = df
