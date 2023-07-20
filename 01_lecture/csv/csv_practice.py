import csv
with open("example.csv") as f:
    reader = csv.reader(f)
    # csvファイルの各行がgeneratorで返ってくる
    # 一つ目がheader
    for line in reader:
        print(line)

# リストだとわかりにくいので，dictionaryで処理をする
with open("example.csv") as f:
    reader = csv.DictReader(f)
    for line in reader:
        print(line)

# writeも同様で，csv.writerを使ってリストで書き込みが可能
with open("sample.csv", mode='w') as f:
    writer = csv.writer(f)
    writer.writerow(['value1', 'value2'])
    writer.writerow(['value3', 'value4'])


# csv.DictWriterを使ってdictionaryで書き込みが可能
with open("sample.csv", mode='w') as f:
    fieldnames = ['col1', 'col2']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    # .writeheader()をcallすることでheaderを追加できる
    writer.writeheader()
    writer.writerow({'col1': 'value1', 'col2': 'value2'})
    writer.writerow({'col1': 'value3', 'col2': 'value4'})
