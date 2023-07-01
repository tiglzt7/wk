import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

kw_dtype = {"品番": str, "図番": str, "元品番": str, "先品番": str}
kw_encoding = "cp932"


def get_path(file_type=None, is_directory=False):
    """
    ユーザーにファイルまたはフォルダを選択させ、選択したパスを返します。

    Args:
        file_type (str, optional): 選択可能なファイルの種類。指定するとその種類のファイルのみが選択可能になります。
        is_directory (bool, optional): Trueの場合、フォルダ選択ダイアログが表示されます。

    Returns:
        str: ユーザーが選択したファイルまたはフォルダのパス。

    Raises:
        FileNotFoundError: ユーザーがファイルまたはフォルダを選択しなかった場合。
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    if is_directory:
        path = filedialog.askdirectory(initialdir="..", title="Select folder")
        if not path:
            raise FileNotFoundError("Folder was not selected")
    else:
        if file_type:
            filetypes = ((file_type, f"*.{file_type}"),)
        else:
            filetypes = (("All files", "*.*"),)

        path = filedialog.askopenfilename(
            initialdir="..", title="Select file", filetypes=filetypes
        )

        if not path:
            raise FileNotFoundError("File was not selected")

    return path


def get_dataframe(file_path=None, file_type=None):
    """
    CSVファイルまたはExcelファイルを読み込み、pandas DataFrameまたはシートごとのDataFrameを持つ辞書を返します。

    Args:
        file_path (str, optional): CSVまたはExcelファイルのパス。
        file_type (str, optional): ファイルの種類（"csv"または"xlsx"）。

    Returns:
        DataFrame or dict: CSVの内容を保持するpandas DataFrame、またはExcelの各シートの内容を保持するDataFrameの辞書。

    Raises:
        ValueError: file_typeが"csv"または"xlsx"でない場合。
    """
    if file_path is None:
        file_path = get_path(file_type)

    if file_type == "csv":
        df = pd.read_csv(file_path, dtype=kw_dtype, encoding=kw_encoding)
    elif file_type == "xlsx":
        xls = pd.ExcelFile(file_path)
        df = {
            sheet_name: xls.parse(sheet_name, dtype=kw_dtype)
            for sheet_name in xls.sheet_names
        }
    else:
        raise ValueError("Invalid file type")

    return df


def write_dataframe(df_dict, file_path=None):
    """
    指定した辞書に基づいてExcelファイルを作成します。

    Args:
        df_dict (dict): キーがシート名、値がそのシートの内容を保持するpandas DataFrameの辞書。
        file_path (str, optional): Excelファイルの保存先パス。
    """
    if file_path is None:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.asksaveasfilename(
            initialdir="..", title="Save as", defaultextension="xlsx"
        )

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def csv_to_excel(directory_path=None, output_file_name="output.xlsx"):
    """
    指定したフォルダ内のすべてのCSVファイルをExcelファイルに変換します。

    Args:
        directory_path (str, optional): CSVファイルが保存されているフォルダのパス。
        output_file_name (str, optional): 作成するExcelファイルの名前。
    """
    if directory_path is None:
        directory_path = get_path(is_directory=True)

    df_dict = {
        os.path.splitext(filename)[0]: get_dataframe(
            os.path.join(directory_path, filename), "csv"
        )
        for filename in os.listdir(directory_path)
        if filename.endswith(".csv")
    }

    write_dataframe(df_dict, output_file_name)


def remove_duplicates(df):
    """
    DataFrameから重複行を削除します。

    Args:
        df (DataFrame): 重複行を削除する対象のDataFrame。

    Returns:
        DataFrame: 重複行を削除したDataFrame。
    """
    return df.drop_duplicates(subset="先品")
