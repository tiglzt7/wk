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
    品番重複チェック+削除
    """
    df.loc[(df.duplicated(subset=["先品番"], keep=False)) & (df["レベル1"] == 1)]
    df = df.drop(
        df[(df.duplicates(subset=["先品番"], keep="False")) & (df["レベル1"] != 1)].index
    )

    return df


def getdf(file_path, dtype={"元品番": str, "先品番": str}):
    xls = pd.ExcelFile(file_path)
    df = {
        sheet_name: xls.parse(sheet_name, dtype=dtype) for sheet_name in xls.sheet_names
    }
    return df
