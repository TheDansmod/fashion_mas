"""Config for exploration, not required in production."""

from pydantic import BaseModel, ConfigDict, FilePath


class ExplorationConfig(BaseModel):
    """Exploration Config"""

    model_config = ConfigDict(frozen=True, validate_default=True)

    # this is the name of the bucket on s3 - used for testing how upload / download parsing works - it contains the sample hdf5 file
    s3_bucket_name: str = "my-personal-bucket-callingmicron"

    # this is the path to the hdf5 sample file I created to test s3 upload / download, and random access
    sample_hdf5_file_path: FilePath = "/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/src/frag/exploration/sample_dataset.h5"

    # this is the path to the sample hdf5 file inside the s3 bucket
    sample_hdf5_s3_path: str = "datasets/sample_dataset.h5"

    # this is the name of the images dataset in the sample hdf5 file
    hdf5_images_dataset_name: str = "images"

    # this is the name of the descriptions dataset in the sample hdf5 file
    hdf5_descriptions_dataset_name: str = "descriptions"

    # this is the path where the random-access image fetched from the hdf5 file should be saved
    random_access_image_path: str = "/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/src/frag/exploration/random_access_image_{index}.png"
