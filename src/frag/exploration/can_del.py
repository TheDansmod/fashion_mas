
import pyarrow.parquet as pq

def setup_metadata_lookup():
    path = r"/mnt/windows/Users/lordh/Documents/Svalbard/Data/fashion-gen/fashion_gen_metadata.parquet"
    table = pq.read_table(path)
    df = table.to_pandas()
    print(df.head())
    # df.set_index("index_2", inplace=True)
    metadata_lookup = df.to_dict(orient='index')
    print(list(metadata_lookup[0].keys()))

if __name__ == '__main__':
    setup_metadata_lookup()
