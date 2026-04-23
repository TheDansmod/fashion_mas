"""Config for exploration, not required in production."""

from pydantic import BaseModel, ConfigDict, FilePath, DirectoryPath


class ExplorationConfig(BaseModel):
    """Exploration Config"""

    model_config = ConfigDict(frozen=True, validate_default=True)

    # this is the name of the bucket on s3 - used for testing how upload / download parsing works - it contains the sample hdf5 file
    s3_bucket_name: str = "my-personal-bucket-callingmicron"

    # this is the path to the hdf5 sample file I created to test s3 upload / download, and random access
    sample_hdf5_file_path: FilePath = "/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/src/frag/exploration/dataset_wrangling/sample_dataset.h5"

    # this is the path to the sample hdf5 file inside the s3 bucket
    sample_hdf5_s3_path: str = "datasets/sample_dataset.h5"

    # this is the name of the images dataset in the sample hdf5 file
    hdf5_images_dataset_name: str = "images"

    # this is the name of the descriptions dataset in the sample hdf5 file
    hdf5_descriptions_dataset_name: str = "descriptions"

    # this is the path where the random-access image fetched from the hdf5 file should be saved
    random_access_image_path: str = "/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/src/frag/exploration/random_access_image_{index}.png"

    # the location where to write the parquet metadata file extracted from the h5py file
    fashion_gen_metadata_path: str = r"/mnt/windows/Users/lordh/Documents/Svalbard/Data/fashion-gen/fashion_gen_metadata.parquet"

    # this is the compression to use for the parquet metadata file (options are snappy and zstd). zstd gives better compression and is often considered the better default
    parquet_compression: str = "zstd"

    # this is the folder into which individual images from the hdf5 file are to be individually extracted. this folder is meant to be deleted once the upload to AWS is completed
    image_extraction_folder_path: DirectoryPath = r"/mnt/windows/Users/lordh/Documents/Svalbard/Data/fashion-gen/extracted_images"

    # this is the batch size used to extract the images - i tried 1024 and 1024 * 5 and it did not make much of a difference in the timing, only the cpu use was higher
    image_extraction_batch_size: int = 1024

    # this is the number of workers to use to parallely extract and write images to a local folder
    image_extraction_max_workers: int = 8

    # this is the max size of the pool to use while uploading images to s3
    s3_upload_max_pool_size: int = 20
