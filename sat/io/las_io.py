"""Read/write LAS/LAZ point cloud files via laspy.

Optimized for large point clouds (100M+ points):
- Reads dimensions directly into dict of numpy arrays (no vstack/transpose)
- Provides both pandas and numpy-native interfaces
- LAZ compression uses lazrs (Rust) for multi-threaded parallel encoding
"""

import time

import numpy as np
import pandas as pd
import laspy


def las_to_numpy(las_file_path):
    """Read a LAS/LAZ file and return (dict_of_arrays, column_names).

    Fastest path — returns a dict of 1D numpy arrays keyed by dimension name.
    Avoids DataFrame construction overhead for large files.
    """
    las = laspy.read(las_file_path)

    columns = {}
    for dim in las.point_format.dimension_names:
        if hasattr(las, dim.lower()):
            columns[dim] = np.asarray(getattr(las, dim.lower()))

    for dim in las.point_format.extra_dimension_names:
        if dim not in columns:
            columns[dim] = np.asarray(getattr(las, dim))

    return columns, list(columns.keys()), las


def las_to_pandas(las_file_path, csv_file_path=None):
    """Read a LAS/LAZ file and return a pandas DataFrame with all dimensions."""
    columns, col_names, _ = las_to_numpy(las_file_path)

    points_df = pd.DataFrame(columns)

    if csv_file_path is not None:
        points_df.to_csv(csv_file_path, index=False, header=True, sep=',')

    return points_df


def pandas_to_las(df, output_file_path, do_compress=False, verbose=False):
    """Convert a pandas DataFrame to a LAS/LAZ file.

    Args:
        df: DataFrame with point cloud data (x/X, y/Y, z/Z columns required).
        output_file_path: Output .las or .laz file path.
        do_compress: If True, write as .laz (uses lazrs parallel compression).
        verbose: Print status messages.
    """
    t0 = time.time()

    # Rename in place — avoid full df.copy() for 214M+ point DataFrames
    col_renames = {}
    if 'x' in df.columns:
        col_renames['x'] = 'X'
    if 'y' in df.columns:
        col_renames['y'] = 'Y'
    if 'z' in df.columns:
        col_renames['z'] = 'Z'
    if 'scan_angle_rank' in df.columns:
        col_renames['scan_angle_rank'] = 'scan_angle'
    if col_renames:
        df = df.rename(columns=col_renames)

    standard_columns_with_data_types = {
        'X': 'int32', 'Y': 'int32', 'Z': 'int32',
        'intensity': 'uint16',
        'return_number': 'uint8', 'number_of_returns': 'uint8',
        'synthetic': 'uint8', 'key_point': 'uint8', 'withheld': 'uint8',
        'overlap': 'uint8', 'scanner_channel': 'uint8',
        'scan_direction_flag': 'uint8', 'edge_of_flight_line': 'uint8',
        'classification': 'uint8', 'user_data': 'uint8',
        'scan_angle': 'uint16', 'point_source_id': 'uint16',
        'gps_time': 'float64',
        'red': 'uint16', 'green': 'uint16', 'blue': 'uint16',
    }

    extended_columns_with_data_types = {
        'Amplitude': 'float64', 'Pulse_width': 'float64',
        'Reflectance': 'float64', 'Deviation': 'int32',
        'PredSemantic': 'uint8', 'PredInstance': 'uint16',
    }

    # Use numpy arrays directly for min/max (faster than pandas Series methods)
    x_vals = df['X'].values
    y_vals = df['Y'].values
    z_vals = df['Z'].values

    scale = [0.001, 0.001, 0.001]
    offset = [float(x_vals.min()), float(y_vals.min()), float(z_vals.min())]

    las_header = laspy.LasHeader(point_format=6, version="1.4")
    las_header.scale = scale
    las_header.offset = offset
    las_header.min = offset
    las_header.max = [float(x_vals.max()), float(y_vals.max()), float(z_vals.max())]

    standard_columns = list(las_header.point_format.dimension_names)
    columns_which_match = [c for c in standard_columns if c in df.columns and c not in ('X', 'Y', 'Z')]

    extra_columns = [c for c in df.columns if c not in standard_columns]
    for column in extra_columns:
        if column not in extended_columns_with_data_types:
            extended_columns_with_data_types[column] = df[column].dtype
        las_header.add_extra_dim(laspy.ExtraBytesParams(
            name=column, type=extended_columns_with_data_types[column]
        ))

    las_file = laspy.LasData(las_header)

    # Use numpy for coordinate scaling (avoid pandas Series overhead)
    las_file.X = ((x_vals - offset[0]) / scale[0]).astype(np.int32)
    las_file.Y = ((y_vals - offset[1]) / scale[1]).astype(np.int32)
    las_file.Z = ((z_vals - offset[2]) / scale[2]).astype(np.int32)

    for column in columns_which_match:
        target_dtype = standard_columns_with_data_types[column]
        col_data = df[column].values
        if target_dtype == 'uint16':
            col_data = np.nan_to_num(col_data, nan=0.0)
            col_data = np.clip(col_data, 0, 65535).round()
        las_file[column] = col_data.astype(target_dtype)

    for column in extra_columns:
        target_dtype = extended_columns_with_data_types[column]
        col_data = df[column].values
        if target_dtype == 'uint16':
            col_data = np.nan_to_num(col_data, nan=0.0)
            col_data = np.clip(col_data, 0, 65535).round()
        las_file[column] = col_data.astype(target_dtype)

    t_prep = time.time() - t0

    if do_compress:
        output_file_path = output_file_path.replace('.las', '.laz')
        las_file.write(output_file_path, do_compress=True)
    else:
        las_file.write(output_file_path, do_compress=False)

    if verbose:
        t_total = time.time() - t0
        print(f'File saved as: {output_file_path} (prep: {t_prep:.1f}s, total: {t_total:.1f}s)')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert LAS/LAZ to pandas DataFrame.')
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, default=None)
    args = parser.parse_args()
    las_to_pandas(args.input_file, args.output_file)
