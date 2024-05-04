input_file_path = 'n:/YOLO/yolov9seg/traces/trace_summary.txt'
output_file_path = 'n:/YOLO/yolov9seg/traces/clean_trace_summary.txt'
ignored_prefixes = [
    '.conda', 
    'AppData',
    'importlib',
    'decorator-gen',
    'filename: <string>',
    'filename: <frozen zipimport>',
    'filename: config-3.py',
    'filename: config.py',
    'functions called:'
]


def should_ignore_line(line):
    """Return True if the line contains any of the ignored prefixes."""
    for prefix in ignored_prefixes:
        if prefix in line:
            return True
    return False


try:
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    filtered_sorted_lines = sorted(
        line for line in lines if not should_ignore_line(line))

    with open(output_file_path, 'w') as file:
        file.writelines(filtered_sorted_lines)

    print(f"Cleaned and sorted trace summary written to {output_file_path}")
except Exception as e:
    print(f"Error processing trace files: {e}")
