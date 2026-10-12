from unittest.mock import patch
from pathlib import Path
import pytest
from vdc.io import write_json
from vdc.research_r2_devices import select_device,configure_devices,gpu_receipt,gpu_process_memory


def test_ai_selects_each_device_without_mutating_capabilities():
    config={'device_policy':'ai','gpu_available':True,'available_devices':['cpu','cuda']}
    assert select_device(config,'cuda')['gpu'] is True
    assert select_device(config,'cpu')['gpu'] is False
    assert 'gpu' not in config


def test_cuda_never_falls_back_to_cpu():
    with pytest.raises(ValueError,match='no CPU fallback'):select_device({'device_policy':'cpu'},'cuda')
    with pytest.raises(ValueError):select_device({},'auto')


def test_cpu_policy_does_not_load_cuda():
    with patch('vdc.research_r2_devices.gpu_receipt') as p:
        config=configure_devices({},'cpu')
    p.assert_not_called()
    assert config['available_devices']==['cpu'] and not config['gpu']


def test_gpu_probe_bound_to_runtime_and_actual_hardware(tmp_path):
    hardware={'devices':[{'name':'software mock','uuid':'test-gpu'}]}
    evidence={'forward_backward':{'status':'passed'},'reload':{'status':'passed'}}
    config={'runtime_root':str(tmp_path)}
    with patch('vdc.research_r2_linux.runtime_info',return_value={'runtime_id':'runtime-1'}), \
         patch('vdc.research_r2_devices.gpu_inventory',return_value=hardware):
        with pytest.raises(ValueError):gpu_receipt(config)
        write_json(tmp_path/'gpu_probe_passed.json',{'runtime_id':'different','status':'passed','hardware':hardware})
        with pytest.raises(ValueError):gpu_receipt(config)
        write_json(tmp_path/'gpu_probe_passed.json',{'runtime_id':'runtime-1','status':'passed','hardware':hardware,**evidence})
        assert configure_devices(config,'ai')['gpu_available']
        write_json(tmp_path/'gpu_probe_passed.json',{'runtime_id':'runtime-1','status':'passed','hardware':{'devices':[{'uuid':'wrong'}]},**evidence})
        with pytest.raises(ValueError,match='identity changed'):gpu_receipt(config)


def test_gpu_measurement_counts_only_candidate_processes():
    from subprocess import CompletedProcess
    with patch('vdc.research_r2_devices.subprocess.run',return_value=CompletedProcess([],0,'101, 200\n202, 900\n','')):
        assert gpu_process_memory([101])==200*1024**2
        assert gpu_process_memory([303])==0
