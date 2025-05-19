from util.data import DataItem


def bio_tag_dictionary(sample_dataset: list[DataItem]) -> tuple[dict[str, int], dict[int, str]]:
    """Constructs and returns the dictionary of bio tags and their corresponding integer encoding values for training from the provided sample dataset.

    Args:
        sample_dataset (list[list[str], list[str]]): The dataset created by loading the corresponding files.

    Returns:
        tuple[dict[str, int], dict[int, str]]: The dictionary with bio tags as keys and their respective values and the inverted dictionary for decoding.
    """

    # Get the unique tags sorted from the provided sample dataset so as to not hard code.
    unique_tags = sorted(
        {tag for item in sample_dataset for tag in item.get()[1]})

    # Build the dictionary for encoding
    bio_tag_dict = {tag: i for i, tag in enumerate(unique_tags)}
    bio_tag_dict["<PAD>"] = len(bio_tag_dict)

    # Inverted dictionary for faster decoding.
    bio_tag_inverted_dict = {v: k for k, v in bio_tag_dict.items()}

    return (bio_tag_dict, bio_tag_inverted_dict)
