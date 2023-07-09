df_aaa = dfs["Sheet1"]
df_bbb = dfs["Sheet2"]

# 'rho'と'g'の値を定義します
rho = 1000  # 水の密度 (kg/m^3)
g = 9.81  # 重力加速度 (m/s^2)

# 'keiski'列でデータフレームをマージします
df_merged = pd.merge(df_aaa, df_bbb, on="keiski")

# 流量QをL/minからm^3/sに変換します
df_merged["q_m3s"] = df_merged["q"] / (1000 * 60)  # 1000 L = 1 m^3, 60 s = 1 min

# ワット単位で出力Wを計算します
df_merged["w_output"] = (
    rho * g * df_merged["q_m3s"] * df_merged["h"] * 1000
)  # 1 kW = 1000 W

# 効率ηを計算します
df_merged["eta"] = df_merged["w_output"] / df_merged["w"]

# 'keiski'カテゴリをkWとDの値の降順で並べ替えます
keiski_sorted = df_merged.sort_values(by=["kW", "D"], ascending=False)[
    "keiski"
].unique()

# プロットのための図と軸を作成します
fig, ax = plt.subplots(figsize=(10, 5))

# kWとDの値が降順の各'keiski'カテゴリについて
for keiski in keiski_sorted:
    # 現在の'keiski'カテゴリのデータをフィルタリングします
    df_keiski = df_merged[df_merged["keiski"] == keiski]

    # h対qnの折れ線グラフを作成し、線の下の領域をα = 1で塗りつぶします
    ax.fill_between(df_keiski["q"], df_keiski["h"], label=keiski, alpha=1)

# プロットのタイトルとラベルを設定します
ax.set_title("各keiskiのh対qn、ηで塗りつぶし")
ax.set_xlabel("qn")
ax.set_ylabel("h")

# 凡例を追加します
ax.legend(title="keiski")

# プロットを表示します
plt.show()
