# csv_selector.py
import os
import glob
import pandas as pd
from tkinter import Tk, StringVar, Label, OptionMenu, Button


def load_csv(root, var):
    if os.path.isfile(var.get()):
        df = pd.read_csv(var.get())
        print(df)
        root.destroy()  # ウィンドウを閉じます


def select_and_load_csv():
    # ワーキングディレクトリ以下のすべての.csvファイルを検索します
    csv_files = glob.glob(os.path.join(os.getcwd(), "**/*.csv"), recursive=True)

    # tkinterのルートウィンドウを作成
    root = Tk()

    # ウィンドウを前面に表示
    root.attributes("-topmost", 1)

    # CSVファイル選択のためのドロップダウンメニューを作成
    var = StringVar(root)
    var.set(csv_files[0])  # デフォルト値を設定
    OptionMenu(root, var, *csv_files).pack()

    # 選択したCSVファイルを読み込むボタンを作成
    Button(root, text="Load CSV", command=lambda: load_csv(root, var)).pack()

    # ウィンドウのメインループを開始
    root.mainloop()
