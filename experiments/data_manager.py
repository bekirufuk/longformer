import pandas as pd

df = pd.read_csv('data/patentsview/example/train.csv', usecols=['text', 'label'])
df.to_csv('data/patentsview/example/train.csv', index=False)
