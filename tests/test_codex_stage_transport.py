import importlib.util
from pathlib import Path
import subprocess
from unittest.mock import patch
import pytest

spec=importlib.util.spec_from_file_location('advance_stage',Path(__file__).resolve().parents[1]/'scripts/advance_codex_stage.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_safe_staging_retries_timeout_then_success():
    ok=subprocess.CompletedProcess(['ssh'],0,stdout='ok',stderr='')
    with patch.object(mod.subprocess,'run',side_effect=[subprocess.TimeoutExpired(['ssh'],90),ok]) as call, patch.object(mod.time,'sleep'):
        assert mod.run_stage_process(['ssh'],'mkdir',True).stdout=='ok'
        assert call.call_count==2
        assert call.call_args.kwargs['stdin']==subprocess.DEVNULL

def test_no_retry_for_uncertain_mutation():
    with patch.object(mod.subprocess,'run',side_effect=subprocess.TimeoutExpired(['ssh'],90)) as call:
        with pytest.raises(mod.StageTransportError,match='uncertain'):
            mod.run_stage_process(['ssh'],'apply_phase')
        assert call.call_count==1

def test_safe_network_retries_are_bounded():
    failure=subprocess.CompletedProcess(['ssh'],255,stdout='',stderr='Connection reset')
    with patch.object(mod.subprocess,'run',return_value=failure) as call,patch.object(mod.time,'sleep'):
        with pytest.raises(mod.StageTransportError,match='transport failed'):
            mod.run_stage_process(['ssh'],'snapshot',True,input_text='{}')
        assert call.call_count==3
        assert call.call_args.kwargs['input']=='{}'

def test_remote_validation_error_is_not_retried():
    failure=subprocess.CompletedProcess(['ssh'],1,stdout='',stderr='State mismatch')
    with patch.object(mod.subprocess,'run',return_value=failure) as call:
        with pytest.raises(mod.StageTransportError,match='State mismatch'):
            mod.run_stage_process(['ssh'],'read',True)
        assert call.call_count==1
