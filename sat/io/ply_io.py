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

    Uses the fast direct-to-disk path (numpy tofile) which bypasses plyfile.
    """
    df = df.loc[:, ~df.columns.duplicated()]
    columns = [col.replace(' ', '_') for col in df.columns]

    dtype = np.dtype([(col, np.float32) for col in columns])
    structured = np.empty(len(df), dtype=dtype)
    for i, col in enumerate(columns):
        structured[col] = df.iloc[:, i].values.astype(np.float32)

    numpy_to_ply_fast(structured, columns, output_file_path)


def numpy_to_ply(data, column_names, output_file_path):
    """Write a numpy structured array or dict of arrays to PLY.

    Uses the fast direct-to-disk path (numpy tofile) by default, which
    bypasses plyfile and saturates disk bandwidth on NVMe (~2-3 GB/s).

    Args:
        data: dict mapping column names to 1D numpy arrays, or a structured array.
        column_names: list of column names (used if data is a dict).
        output_file_path: output .ply path.
    """
    numpy_to_ply_fast(data, column_names, output_file_path)


_PLY_TYPE_MAP = {
    np.dtype('float32'): 'float',
    np.dtype('float64'): 'double',
    np.dtype('int8'): 'char',
    np.dtype('uint8'): 'uchar',
    np.dtype('int16'): 'short',
    np.dtype('uint16'): 'ushort',
    np.dtype('int32'): 'int',
    np.dtype('uint32'): 'uint',
}


def numpy_to_ply_fast(data, column_names, output_file_path):
    """Write arrays to binary PLY bypassing plyfile for maximum throughput.

    Writes the PLY header manually and uses numpy's tofile() for the data,
    which uses buffered C-level I/O and can saturate NVMe bandwidth (~2-3 GB/s).

    When data is a dict of arrays, writes columns sequentially to avoid
    building a large intermediate structured array, reducing memory copies.

    Args:
        data: dict mapping column names to 1D numpy arrays, or a structured array.
        column_names: list of column names (used if data is a dict).
        output_file_path: output .ply path.
    """
    if isinstance(data, dict):
        n_points = len(next(iter(data.values())))
        # Determine dtype per column (default float32)
        col_dtypes = []
        for name in column_names:
            arr = data[name]
            dt = np.dtype('float32') if arr.dtype not in _PLY_TYPE_MAP else arr.dtype
            col_dtypes.append((name, dt))

        # Write header
        header_lines = ['ply', 'format binary_little_endian 1.0',
                        f'element vertex {n_points}']
        for name, dt in col_dtypes:
            header_lines.append(f'property {_PLY_TYPE_MAP[dt]} {name}')
        header_lines.append('end_header')
        header = '\n'.join(header_lines) + '\n'

        # Build structured array and write in chunks to limit memory pressure
        chunk_size = min(n_points, 10_000_000)
        dtype = np.dtype(col_dtypes)

        with open(output_file_path, 'wb') as f:
            f.write(header.encode('ascii'))
            for start in range(0, n_points, chunk_size):
                end = min(start + chunk_size, n_points)
                chunk = np.empty(end - start, dtype=dtype)
                for name, dt in col_dtypes:
                    chunk[name] = data[name][start:end].astype(dt)
                chunk.tofile(f)
    else:
        structured = data
        n_points = len(structured)

        header_lines = ['ply', 'format binary_little_endian 1.0',
                        f'element vertex {n_points}']
        for field_name, (field_dtype, _) in structured.dtype.fields.items():
            ply_type = _PLY_TYPE_MAP.get(field_dtype)
            if ply_type is None:
                raise ValueError(f"Unsupported dtype {field_dtype} for field {field_name}")
            header_lines.append(f'property {ply_type} {field_name}')
        header_lines.append('end_header')
        header = '\n'.join(header_lines) + '\n'

        with open(output_file_path, 'wb') as f:
            f.write(header.encode('ascii'))
            structured.tofile(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert PLY to CSV.')
    parser.add_argument('ply_path', type=str)
    parser.add_argument('csv_path', type=str)
    args = parser.parse_args()
    ply_to_pandas(args.ply_path, args.csv_path)
