"""Read/write PLY point cloud files via plyfile.

Optimized for large point clouds (100M+ points):
- Uses numpy structured arrays directly, avoiding row-by-row Python iteration
- Reads PLY structured data without unnecessary copies
"""

import numpy as np
import pandas as pd
from plyfile import PlyElement, PlyData


def ply_to_pandas(ply_file_path, csv_file_path=None):
    """Read a PLY file and return a pandas DataFrame."""
    ply_content = PlyData.read(ply_file_path)

    available_elements = [elem.name for elem in ply_content.elements]
    if 'vertex' in available_elements:
        point_element_name = 'vertex'
    elif 'point' in available_elements:
        point_element_name = 'point'
    else:
        raise ValueError(f"No vertex/point element in PLY file: {ply_file_path}")

    point_data = ply_content[point_element_name].data
    property_names = point_data.dtype.names

    # Build DataFrame directly from structured array columns (no vstack/transpose)
    df = pd.DataFrame({name: np.asarray(point_data[name]) for name in property_names})

    if csv_file_path is not None:
        df.to_csv(csv_file_path, index=False)

    return df


def ply_to_numpy(ply_file_path):
    """Read a PLY file and return (structured_array, property_names).

    Faster than ply_to_pandas for cases where DataFrame overhead is unnecessary.
    """
    ply_content = PlyData.read(ply_file_path)

    available_elements = [elem.name for elem in ply_content.elements]
    if 'vertex' in available_elements:
        point_element_name = 'vertex'
    elif 'point' in available_elements:
        point_element_name = 'point'
    else:
        raise ValueError(f"No vertex/point element in PLY file: {ply_file_path}")

    data = ply_content[point_element_name].data
    return data, data.dtype.names


def pandas_to_ply(df, output_file_path):
    """Convert a pandas DataFrame to a PLY file.

    Optimized: builds numpy structured array column-by-column instead of
    iterating rows with map(tuple, ...), which is ~100x faster for large files.
    """
    df = df.loc[:, ~df.columns.duplicated()]
    columns = [col.replace(' ', '_') for col in df.columns]

    # Build structured array column-by-column (fast path)
    dtype = np.dtype([(col, np.float32) for col in columns])
    structured = np.empty(len(df), dtype=dtype)
    for i, col in enumerate(columns):
        structured[col] = df.iloc[:, i].values.astype(np.float32)

    vertex = PlyElement.describe(structured, 'vertex')
    PlyData([vertex], text=False).write(output_file_path)


def numpy_to_ply(data, column_names, output_file_path):
    """Write a numpy structured array or dict of arrays to PLY.

    Args:
        data: dict mapping column names to 1D numpy arrays, or a structured array.
        column_names: list of column names (used if data is a dict).
        output_file_path: output .ply path.
    """
    if isinstance(data, dict):
        dtype = np.dtype([(name, np.float32) for name in column_names])
        structured = np.empty(len(next(iter(data.values()))), dtype=dtype)
        for name in column_names:
            structured[name] = data[name].astype(np.float32)
    else:
        structured = data

    vertex = PlyElement.describe(structured, 'vertex')
    PlyData([vertex], text=False).write(output_file_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert PLY to CSV.')
    parser.add_argument('ply_path', type=str)
    parser.add_argument('csv_path', type=str)
    args = parser.parse_args()
    ply_to_pandas(args.ply_path, args.csv_path)
