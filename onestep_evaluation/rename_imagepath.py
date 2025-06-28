import pandas as pd 
import sys
import os
dataname = sys.argv[2] # this is the downloaded parquet file name
working_dir = sys.argv[1]
df = pd.read_parquet(f'{working_dir}/data/{dataname}.parquet')

base = working_dir + f'/data/{dataname}_'
df['image'] = df['image'].apply(lambda ls: [base+ee for ee in ls])
df.to_parquet(f'{working_dir}/data/{dataname}.parquet')

print(f"[check] {df['image'].iloc[0]}")
for imglist in df['image']:
    for img in imglist:
        assert os.path.exists(img)
print(f"[checked] all files exist.")

