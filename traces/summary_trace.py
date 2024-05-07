import trace
import sys
import os

output_dir = "n:/YOLO/yolov9seg/traces"
script_path = 'n:/YOLO/yolov9seg/predict.py'

try:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
except Exception as e:
    print(f"Error creating directory: {e}")
    exit(1)

tracer = trace.Trace(
    ignoredirs=[sys.prefix, sys.exec_prefix],
    trace=1,
    count=0,
    countfuncs=1,
)

try:
    with open(script_path, "rb") as source_file:
        code = compile(source_file.read(), script_path, "exec")
        tracer.runctx(code, globals(), locals())
except Exception as e:
    print(f"Error executing script: {e}")
    exit(1)

r = tracer.results()
summary_file = os.path.join(output_dir, 'trace_summary.txt')

try:
    with open(summary_file, 'w') as f:
        original_stdout = sys.stdout 
        sys.stdout = f 
        r.write_results(show_missing=True, summary=True)
        sys.stdout = original_stdout
    print(f"Trace summary written to {summary_file}")
except Exception as e:
    print(f"Error writing to file: {e}")
