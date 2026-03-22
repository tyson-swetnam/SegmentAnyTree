"""Read/write PLY point cloud files via plyfile + fast binary paths.

Optimized for large point clouds (100M+ points):
- Uses numpy structured arrays directly, avoiding row-by-row Python iteration
- Fast binary reader/writer bypass plyfile for known binary_little_endian files
- Reads PLY structured data without unnecessary copies
"""

import numpy as np
import pandas as pd
from plyfile import PlyElement, PlyData

# Map PLY type strings to numpy dtypes
_NUMPY_DTYPE_MAP = {
    'float': np.float32, 'float32': np.float32,
    'double': np.float64, 'float64': np.float64,
    'char': np.int8, 'int8': np.int8,
    'uchar': np.uint8, 'uint8': np.uint8,
    'short': np.int16, 'int16': np.int16,
    'ushort': np.uint16, 'uint16': np.uint16,
    'int': np.int32, 'int32': np.int32,
    'uint': np.uint32, 'uint32': np.uint32,
}


def ply_to_pandas(ply_file_path, csv_file_path=None):
    """Read a PLY file and return a pandas DataFrame."""
    data, property_names = ply_to_numpy(ply_file_path)

    # Build DataFrame directly from structured array columns (no vstack/transpose)
    df = pd.DataFrame({name: np.asarray(data[name]) for name in property_names})

    if csv_file_path is not None:
        df.to_csv(csv_file_path, index=False)

    return df


def _try_read_binary_ply(ply_file_path):
    """Try to read a binary_little_endian PLY directly via numpy.fromfile.

    Returns (structured_array, property_names) or None if the file is not
    a simple binary_little_endian PLY (e.g. ASCII, big-endian, or list props).
    """
    with open(ply_file_path, 'rb') as f:
        # Parse header
        header_bytes = b''
        while True:
            line = f.readline()
            header_bytes += line
            if line.strip() == b'end_header':
                break
            if len(header_bytes) > 10000:
                return None  # Header too large, bail

        header = header_bytes.decode('ascii')
        lines = header.strip().split('\n')

        # Check format
        if 'format binary_little_endian 1.0' not in header:
            return None

        n_vertices = 0
        fields = []
        in_vertex = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] == 'element' and parts[1] == 'vertex':
                n_vertices = int(parts[2])
                in_vertex = True
            elif parts[0] == 'element':
                in_vertex = False
            elif in_vertex and parts[0] == 'property':
                if parts[1] == 'list':
                    return None  # List properties not supported in fast path
                dtype = _NUMPY_DTYPE_MAP.get(parts[1])
                if dtype is None:
                    return None
                fields.append((parts[2], dtype))

        if not fields or n_vertices == 0:
            # Empty file — return valid empty structured array
            if fields:
                dtype = np.dtype(fields)
                return np.empty(0, dtype=dtype), tuple(name for name, _ in fields)
            return None

        dtype = np.dtype(fields)
        data = np.fromfile(f, dtype=dtype, count=n_vertices)
        return data, tuple(name for name, _ in fields)


def ply_to_numpy(ply_file_path):
    """Read a PLY file and return (structured_array, property_names).

    Uses a fast binary reader for binary_little_endian files (our output format),
    falling back to plyfile for other formats.
    """
    result = _try_read_binary_ply(ply_file_path)
    if result is not None:
        return result

    # Fallback to plyfile for ASCII or other formats
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
