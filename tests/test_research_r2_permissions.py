"""Linux regressions for system-disk -> data-disk runtime staging."""
import errno
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from vdc.research_r2_linux import link_or_copy, readable_directories

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX permission semantics')


@pytest.mark.parametrize('mode', [0o755, 0o700])
def test_cross_device_copy_keeps_executable_without_changing_source(tmp_path, mode):
    source = tmp_path/'source'
    source.write_text('#!/bin/sh\nprintf executable-ok')
    source.chmod(mode)
    target = tmp_path/'copy'
    with patch('vdc.research_r2_linux.os.link', side_effect=OSError(errno.EXDEV, 'cross-device')):
        link_or_copy(source, target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o555
    assert stat.S_IMODE(source.stat().st_mode) == mode
    assert subprocess.check_output([str(target)], text=True) == 'executable-ok'


def test_private_data_copy_does_not_gain_execution_or_change_source(tmp_path):
    source = tmp_path/'private'
    source.write_bytes(b'data')
    source.chmod(0o600)
    link_or_copy(source, tmp_path/'copy')
    assert stat.S_IMODE((tmp_path/'copy').stat().st_mode) == 0o444
    assert stat.S_IMODE(source.stat().st_mode) == 0o600


def test_directories_traversable_without_touching_hardlinks_or_symlink_targets(tmp_path):
    source = tmp_path/'outside'
    source.mkdir(mode=0o700)
    file = source/'file'
    file.write_text('unchanged')
    file.chmod(0o600)
    root = tmp_path/'jail'
    root.mkdir(mode=0o700)
    (root/'nested').mkdir(mode=0o700)
    os.link(file, root/'hardlink')
    (root/'symlink').symlink_to(source, target_is_directory=True)
    readable_directories(root)
    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert stat.S_IMODE((root/'nested').stat().st_mode) == 0o755
    assert stat.S_IMODE(file.stat().st_mode) == 0o600
    assert stat.S_IMODE(source.stat().st_mode) == 0o700
