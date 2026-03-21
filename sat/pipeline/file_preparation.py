"""Input file preparation: sanitize filenames for the inference pipeline."""

import os
import argparse


def sanitize_filenames(input_folder):
    """Replace dashes and spaces with underscores in filenames.

    The inference pipeline and Hydra configs don't handle special characters
    well in filenames. This normalizes them before processing.
    """
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        if not os.path.isfile(filepath):
            continue

        new_filename = filename.replace('-', '_').replace(' ', '_')
        if new_filename != filename:
            new_filepath = os.path.join(input_folder, new_filename)
            os.rename(filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sanitize filenames (replace dashes/spaces with underscores).')
    parser.add_argument('input_folder', type=str)
    args = parser.parse_args()
    sanitize_filenames(args.input_folder)
