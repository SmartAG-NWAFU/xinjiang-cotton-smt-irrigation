import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def setup_paths():
    """Set up paths for input and output directories."""
    src_path = os.path.dirname(os.path.abspath(__file__))
    paths = {
        'input_dir': [
            os.path.join(src_path, "../../data/grid_10km/weather/2022-01-01_2032-12-31"),
            os.path.join(src_path, "../../data/grid_10km/weather/2033-01-01_2043-12-31"),
            os.path.join(src_path, "../../data/grid_10km/weather/2044-01-01_2054-12-31"),
            os.path.join(src_path, "../../data/grid_10km/weather/2055-01-01_2065-12-31"),
            os.path.join(src_path, "../../data/grid_10km/weather/2066-01-01_2076-12-31"),
            os.path.join(src_path, "../../data/grid_10km/weather/2077-01-01_2081-12-31")
        ],
        'output_dir': os.path.join(src_path, "../../data/grid_10km/weather/2022-01-01_2081-12-31_summary")
    }
    return paths

def merge_file_sequence(file_index, input_dirs, output_dir):
    """Merge the same-index files from multiple directories into one output file."""
    output_path = os.path.join(output_dir, f"{file_index}.csv")
    if os.path.exists(output_path):
        return f"skip {file_index}.csv, already exists"

    dfs = []
    for input_dir in input_dirs:
        filepath = os.path.join(input_dir, f"{file_index}.csv")
        try:
            df = pd.read_csv(filepath)
            dfs.append(df)
        except FileNotFoundError:
            return f"missing: {filepath}"
        except Exception as e:
            return f"error reading {filepath}: {e}"

    if dfs:
        try:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_csv(output_path, index=False)
            return None
        except Exception as e:
            return f"error writing {file_index}.csv: {e}"
    else:
        return f"no data for {file_index}.csv"

def merge_file_parallel(input_dirs, output_dir, total_files=1337, num_workers=61):
    """Run merge_file_sequence in parallel using multiple processes."""
    os.makedirs(output_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(merge_file_sequence, file_index, input_dirs, output_dir): file_index
            for file_index in range(total_files)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Merging CSVs"):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                idx = futures[future]
                print(f"Exception in file {idx}.csv: {e}")

def test():
    """Test merging a single file to verify behavior."""
    paths = setup_paths()
    result = merge_file_sequence(0, paths['input_dir'], paths['output_dir'])
    print(result)

def main():
    paths = setup_paths()
    merge_file_parallel(paths['input_dir'], paths['output_dir'], total_files=1337, num_workers=8)

if __name__ == "__main__":
    main()
    # test()  # Uncomment to run test mode