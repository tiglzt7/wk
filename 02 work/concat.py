def get_unique_rows(df1, df2, group_columns, unique_columns):
    # df1とdf2を連結し、重複する行を除去（df1にもdf2にも存在する行を除去）
    concat_df = pd.concat([df1, df2])
    concat_df = concat_df.drop_duplicates(
        subset=group_columns + unique_columns, keep=False
    )

    # 連結したデータフレームから、df1とdf2の元のインデックスに基づいてそれぞれの固有の行を抽出
    unique_to_df1 = concat_df[concat_df.index.isin(df1.index)]
    unique_to_df2 = concat_df[concat_df.index.isin(df2.index)]

    return unique_to_df1, unique_to_df2


# 以下のように呼び出します
unique_to_df1, unique_to_df2 = get_unique_rows(df1, df2, ["A", "B"], ["C"])

# テストデータフレームの作成
df1 = pd.DataFrame(
    {
        "A": ["group1", "group1", "group2", "group2"],
        "B": ["value1", "value2", "value3", "value4"],
        "C": ["c1", "c2", "c3", "c4"],
        "D": ["d1", "d2", "d3", "d4"],
    }
)

df2 = pd.DataFrame(
    {
        "A": ["group1", "group1", "group2", "group2"],
        "B": ["value1", "value5", "value6", "value7"],
        "C": ["c5", "c6", "c7", "c8"],
        "D": ["d5", "d6", "d7", "d8"],
    }
)

# グループ化と固有の行の取得
group_columns = ["A", "B"]
unique_columns = ["C", "D"]
unique_to_df1, unique_to_df2 = get_unique_rows(df1, df2, group_columns, unique_columns)

print("Unique to df1:")
print(unique_to_df1)

print("\nUnique to df2:")
print(unique_to_df2)
