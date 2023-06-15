import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog


class DataHandler:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.attributes("-topmost", True)
        self.root.withdraw()
        self.root.lift()
        self.root.focus_force()

    def get_file_path(self, initialdir, title, filetypes):
        file_path = filedialog.askopenfilename(
            initialdir=initialdir, title=title, filetypes=filetypes
        )
        if not file_path:
            print("ファイルが選択されていません")
            return None
        return file_path

    def getdf_csv(self):
        file_path = self.get_file_path("./", "Select file", (("csv file", "*.csv"),))
        if file_path is None:
            return None
        df = pd.read_csv(file_path, dtype={"元品": str, "先品": str}, encoding="cp932")
        return df

    def getdf_xlsx(self):
        file_path = self.get_file_path(
            "./", "Select file", (("Excel Files", "*.xlsx"),)
        )
        if file_path is None:
            return None
        xls = pd.ExcelFile(file_path)
        dfs = {
            sheet_name: xls.parse(sheet_name, dtype={"元品": str, "先品": str})
            for sheet_name in xls.sheet_names
        }
        return dfs

    def writedf_xlsx(self, df_dict):
        file_name = filedialog.asksaveasfilename(
            initialdir="./", title="Save as", defaultextension=".xlsx"
        )
        with pd.ExcelWriter(file_name) as writer:
            for sheet_name, df in df_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        self.root.destroy()

    @staticmethod
    def VC_duplicate(df):
        df.loc[(df.duplicated(subset=["先品"], keep=False)) & (df["1"] == 1)]
        df = df.drop(
            df[(df.duplicated(subset=["先品"], keep="False")) & (df["1"] != 1)].index
        )
        return df
