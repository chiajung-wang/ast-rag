import subprocess
import os


def test_make_check_target_is_defined():
    result = subprocess.run(
        ["make", "-n", "check"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert "pytest" in result.stdout.decode()
