"""Process-boundary regression: ready and done can arrive in the same read."""
from pathlib import Path
import subprocess
import sys
import time
from services.fast_video_pipeline.mlx_fast_video_pipeline import MLXFastVideoPipeline


def test_buffered_completion_is_consumed_without_waiting_for_process_exit(tmp_path: Path) -> None:
    pipe = MLXFastVideoPipeline(model_dir=tmp_path, gemma_dir=tmp_path)
    code = 'import sys,time; sys.stdout.write(\'{"event":"ready"}\\n{"event":"done","id":"check","output":"unused"}\\n\'); sys.stdout.flush(); time.sleep(8)'
    proc = subprocess.Popen([sys.executable, '-u', '-c', code], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    pipe._proc = proc
    pipe._start_output_reader(proc)
    try:
        assert pipe._read_until({'ready'}, timeout=3) == {'event': 'ready'}
        start = time.monotonic()
        result = pipe._read_job_until_done(job_id='check', output_path='unused')
        assert result is not None and result['event'] == 'done'
        assert time.monotonic() - start < 3
    finally:
        pipe._kill()
        proc.wait(timeout=5)
        if proc.stdin:
            proc.stdin.close()
        if proc.stdout:
            proc.stdout.close()
