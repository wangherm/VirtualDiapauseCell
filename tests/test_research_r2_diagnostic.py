from unittest.mock import patch
import pytest
from vdc.research_r2_linux import execute_linux


@pytest.mark.parametrize('extra',[
    {'inputs':{'data':'private'}}, {'model':'model'}, {'repo':'repo'},
    {'network':True}, {'config':{'diagnostic_trace':True}},
])
def test_tracing_rejects_research_or_data_mounts_before_launch(extra,tmp_path):
    args={'code':tmp_path,'inputs':{},'out':tmp_path/'out','command':['python'],
          'config':{'diagnostic_trace':True,'probe_mode':True},**extra}
    with patch('vdc.research_r2_linux.runtime_info') as info:
        with pytest.raises(ValueError):execute_linux(**args)
        info.assert_not_called()


def test_missing_tracer_is_explicit_not_a_silent_untraced_run(tmp_path):
    with patch('vdc.research_r2_linux.shutil.which',return_value=None):
        with pytest.raises(RuntimeError,match='requires strace'):
            execute_linux(tmp_path,{},tmp_path/'out',['python'],
                          {'diagnostic_trace':True,'probe_mode':True})
