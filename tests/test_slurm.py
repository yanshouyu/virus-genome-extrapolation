import os
from pathlib import Path
import shlex
import subprocess


def test_training_failure_propagates_from_job_script(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path/'venv/bin'
    fake_bin.mkdir(parents=True)
    (fake_bin/'activate').write_text(
        'export PATH=' + shlex.quote(str(fake_bin) + ':' + os.environ['PATH']) + '\n')
    (fake_bin/'module').write_text('#!/bin/bash\nexit 0\n')
    (fake_bin/'python').write_text(
        '#!/bin/bash\nif [[ "$*" == *"baculo_tss_probe.train"* ]]; then exit 42; fi\nexit 0\n')
    (fake_bin/'python').chmod(0o755)
    (fake_bin/'module').chmod(0o755)
    env = dict(os.environ, PROBE_ENV=str(tmp_path/'venv'), SLURM_SUBMIT_DIR=str(root),
               SLURM_JOB_ID='synthetic-test', PATH=str(fake_bin) + ':' + os.environ['PATH'])
    result = subprocess.run(['bash', str(root/'scripts/run_probe.slurm')], cwd=root,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 42
    assert 'Training and final evaluation completed.' not in result.stdout
