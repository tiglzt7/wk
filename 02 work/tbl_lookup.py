# データフレームの作成
import pandas as pd

# db1の作成
db1 = pd.DataFrame(
    {
        "A": ["a1", "a2", "a3"],
        "B": ["b1", "b2", "b3"],
        "C": ["c1", "c2", "c3"],
    }
)

# db2の作成
db2 = pd.DataFrame(
    {
        "A": ["a4", "a5", "a6"],
        "B": ["b4", "b5", "b6"],
        "C": ["c4", "c5", "c6"],
    }
)

# db3の作成
db3 = pd.DataFrame(
    {
        "B": ["a1", "a2", "a3", "a4", "a5", "a6"],
        "C": ["a7", "a8", "a9", "a10", "a11", "a12"],
    }
)

# db1とdb3をカラムA（db1）とカラムB（db3）を基に結合
db1 = db1.merge(db3, left_on="A", right_on="B", how="left")

# db1のカラムAをdb3のカラムCの値で更新
db1["A"] = db1["C_y"]

# db2とdb3をカラムA（db2）とカラムC（db3）を基に結合
db2 = db2.merge(db3, left_on="A", right_on="C", how="left")

# db2のカラムAをdb3のカラムBの値で更新
db2["A"] = db2["B_y"]

# それぞれのデータフレームをカラムAでグループ化
grouped1 = db1.groupby("A")
grouped2 = db2.groupby("A")

# カラムDの値を比較し、片方にしかない値を取り出す
unique_to_db1 = grouped1["D"].apply(lambda x: x[~x.isin(grouped2["D"])])
unique_to_db2 = grouped2["D"].apply(lambda x: x[~x.isin(grouped1["D"])])

print(unique_to_db1)
print(unique_to_db2)

# ここから違う関数
# db3を辞書形式に変換
lookup_dict = db3.set_index("B")["C"].to_dict()

# db1, db2のカラムAの値をルックアップテーブルで置き換え
db1["A"] = db1["A"].map(lookup_dict)
db2["A"] = db2["A"].map(lookup_dict)

# それぞれのデータフレームをカラムAでグループ化
grouped1 = db1.groupby("A")
grouped2 = db2.groupby("A")

# カラムDの値を比較し、片方にしかない値を取り出す
unique_to_db1 = grouped1["D"].apply(lambda x: x[~x.isin(grouped2["D"])])
unique_to_db2 = grouped2["D"].apply(lambda x: x[~x.isin(grouped1["D"])])

print(unique_to_db1)
print(unique_to_db2)
