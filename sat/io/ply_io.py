"""Read/write PLY point cloud files via plyfile."""

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
    data_arrays = [point_data[prop_name] for prop_name in property_names]
    all_points = np.vstack(data_arrays).T

    if csv_file_path is not None:
        np.savetxt(csv_file_path, all_points, delimiter=',',
                   header=','.join(property_names), comments='', fmt='%s')

    return pd.DataFrame(all_points, columns=property_names)


def pandas_to_ply(df, output_file_path):
    """Convert a pandas DataFrame to a PLY file.

    Args:
        df: DataFrame with point cloud columns.
        output_file_path: Output .ply file path.
    """
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = [col.replace(' ', '_') for col in df.columns]

    dtypes = [(col, 'f4') for col in df.columns]
    data = np.array(list(map(tuple, df.to_records(index=False))), dtype=dtypes)

    vertex = PlyElement.describe(data, 'vertex')
    PlyData([vertex], text=False).write(output_file_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert PLY to CSV.')
    parser.add_argument('ply_path', type=str)
    parser.add_argument('csv_path', type=str)
    args = parser.parse_args()
    ply_to_pandas(args.ply_path, args.csv_path)
