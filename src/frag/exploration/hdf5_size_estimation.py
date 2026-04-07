import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

def get_sizes():
    # Datasets that represent images vs. metadata
    IMAGE_KEYS  = {'input_image'}
    # Everything else is considered metadata
    
    results = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        def collect_sizes(name, obj):
            if isinstance(obj, h5py.Dataset):
                results[name] = {
                    'shape':        obj.shape,
                    'dtype':        str(obj.dtype),
                    'nbytes':       obj.nbytes,                  # in-memory size
                    'storage_size': obj.id.get_storage_size(),   # on-disk size
                }
        f.visititems(collect_sizes)
    
    # Separate images from metadata
    image_nbytes   = sum(v['nbytes']       for k, v in results.items() if k in IMAGE_KEYS)
    image_disk     = sum(v['storage_size'] for k, v in results.items() if k in IMAGE_KEYS)
    meta_nbytes    = sum(v['nbytes']       for k, v in results.items() if k not in IMAGE_KEYS)
    meta_disk      = sum(v['storage_size'] for k, v in results.items() if k not in IMAGE_KEYS)
    
    def fmt(b):
        return f"{b / 1e9:.3f} GB"
    
    print("=== Per-dataset breakdown ===")
    for k, v in results.items():
        label = "IMAGE" if k in IMAGE_KEYS else "META "
        print(f"[{label}] {k:<30} | shape: {str(v['shape']):<25} | "
              f"RAM: {fmt(v['nbytes'])} | Disk: {fmt(v['storage_size'])}")
    
    print("\n=== Totals ===")
    print(f"Images   — RAM: {fmt(image_nbytes)},  Disk: {fmt(image_disk)}")
    print(f"Metadata — RAM: {fmt(meta_nbytes)},  Disk: {fmt(meta_disk)}")



# get all the items
def get_all_items():
    with h5py.File(hdf5_path, 'r') as f:
        def print_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
        f.visititems(print_info)

def inspect_chunking(filepath):
    def human_size(nbytes):
        for unit in ["B", "KB", "MB", "GB"]:
            if nbytes < 1024:
                return f"{nbytes:.1f} {unit}"
            nbytes /= 1024
        return f"{nbytes:.1f} TB"

    def visit_dataset(name, obj):
        if not isinstance(obj, h5py.Dataset):
            print('found non-dataset')
            return

        chunks     = obj.chunks          # None if contiguous
        shape      = obj.shape
        dtype      = obj.dtype
        layout     = "CHUNKED" if chunks else "CONTIGUOUS"
        compression = obj.compression    # e.g. 'gzip', None
        shuffle    = obj.shuffle

        chunk_bytes = None
        if chunks:
            import math
            elements = math.prod(chunks)
            chunk_bytes = elements * dtype.itemsize

        print(f"\nDataset : /{name}")
        print(f"  Shape       : {shape}")
        print(f"  Dtype       : {dtype}")
        print(f"  Layout      : {layout}")
        print(f"  Chunk shape : {chunks}")
        if chunk_bytes:
            print(f"  Chunk size  : {human_size(chunk_bytes)} (uncompressed)")
        print(f"  Compression : {compression}  |  Shuffle: {shuffle}")

    with h5py.File(filepath, "r") as f:
        print(f"File: {filepath}")
        print(f"Top-level keys: {list(f.keys())}")
        f.visititems(visit_dataset)


def re_chunk_hdf5_file(old_file_path, new_file_path):
    SRC = old_file_path
    DST = new_file_path
    DATASET = "input_image"
    BATCH = 500  # process this many images at a time to control RAM usage
    
    with h5py.File(SRC, "r") as src, h5py.File(DST, "w") as dst:
        src_ds = src[DATASET]
        n, h, w, c = src_ds.shape   # (260490, 256, 256, 3)
    
        # Create destination dataset with one image per chunk
        dst_ds = dst.create_dataset(
            DATASET,
            shape=src_ds.shape,
            dtype=src_ds.dtype,
            chunks=(1, h, w, c),     # ← one complete image per chunk
            compression="lzf",       # keep the same compression
            shuffle=True,
        )
    
        # Copy metadata/attributes if any
        for key, val in src_ds.attrs.items():
            dst_ds.attrs[key] = val
    
        # Copy image data in batches to avoid loading 14 GB into RAM
        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            dst_ds[start:end] = src_ds[start:end]
            print(f"  Copied {end}/{n} images...", end="\r")
    
        # Copy all other datasets/groups unchanged
        for key in src.keys():
            if key != DATASET:
                src.copy(key, dst)
    
    print("\nDone. Verify with:")
    print(f"  h5py.File('{DST}')['/{DATASET}'].chunks")

def create_hdf5_file(
    output_path: str,
    image_paths: list[str],
    descriptions: list[str]
) -> None:
    """
    Creates an HDF5 file with two datasets:
      - 'images'       : image pixel data as a uint8 array
      - 'descriptions' : variable-length UTF-8 strings

    Parameters
    ----------
    output_path  : path for the output .h5 file
    image_paths  : list of file paths to images (must all be the same shape)
    descriptions : list of text strings, one per image
    """
    assert len(image_paths) == len(descriptions), \
        "image_paths and descriptions must have the same length."

    # --- Load all images into a numpy array ---
    images = np.stack(
        [np.array(Image.open(p).convert("RGB")) for p in image_paths],
        axis=0
    )  # shape: (N, H, W, 3), dtype: uint8

    # --- Write to HDF5 ---
    with h5py.File(output_path, "w") as hf:

        # Dataset 1: images
        # gzip compression (level 4) + shuffle filter for better ratio
        hf.create_dataset(
            "images",
            data=images,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        # Dataset 2: descriptions — variable-length UTF-8 strings
        dt = h5py.string_dtype(encoding="utf-8")
        hf.create_dataset(
            "descriptions",
            data=np.array(descriptions, dtype=object),
            dtype=dt,
        )

    print(f"Saved {len(image_paths)} entries → '{output_path}'")
    print(f"  images shape : {images.shape}")
    print(f"  descriptions : {len(descriptions)} strings")

def explore_hdf5_file():
    # this is to figure out how to access the contents of the sample hdf5 file while interacting with it on s3
    index = 10
    save_path = "/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/src/frag/exploration/saved_image_{idx}.png"
    with h5py.File(hdf5_path, "r") as f:
        img = Image.fromarray(f['input_image'][index])
        img.save(save_path.format(idx=index))
        print(f'saved image {index}')
        print(f['input_description'][index][0].decode('latin-1'))

def get_all_description_text():
    # i need to know how many tokens all the text in the descriptions amounts to - so will be putting all the descriptions together in a file
    save_path = "/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/src/frag/exploration/all_descriptions.txt"
    with h5py.File(hdf5_path, "r") as fr, open(save_path, "w") as fw:
        for idx in tqdm(range(260_490)):
            descr = fr['input_description'][idx][0].decode('latin-1')
            fw.write(f"{descr}\n")

if __name__ == '__main__':
    hdf5_path = "/mnt/windows/Users/lordh/Documents/Svalbard/Data/fashion-gen/fashiongen_256_256_train.h5"
    new_hdf5_path = r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/src/frag/exploration/sample_dataset.h5"
    get_all_description_text()
    # image_paths = [
    #     r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Images/Screenshots/screenshot_001.png",
    #     r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Images/Screenshots/screenshot_002.png",
    #     r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Images/Screenshots/screenshot_003.png",
    #     r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Images/Screenshots/screenshot_004.png",
    #     r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Images/Screenshots/screenshot_005.png"
    # ]
    # descriptions = [
    #     "01: cryptomator",
    #     "02: transaction number",
    #     "03: icici bank",
    #     "04: 4:41 PM",
    #     "05: 4:50 PM",
    # ]
    # create_hdf5_file(new_hdf5_path, image_paths, descriptions)
