import json
import sys
import subprocess

import pytest

from vdc.research_r2_linux import memory_limits


def test_cpu_and_cuda_keep_the_same_resident_budget():
    resident, virtual=memory_limits({'memory':'4g','gpu':False})
    assert resident==4*1024**3 and virtual>resident
    assert memory_limits({'memory':'4g','gpu':True})==(resident,0)
    with pytest.raises(ValueError):memory_limits({'memory':'0'})


@pytest.mark.skipif(sys.platform!='linux',reason='Linux virtual address limits')
def test_large_reservation_with_small_rss_survives_corrected_cpu_policy():
    code='''import json,mmap,resource,sys
limit=int(sys.argv[1]);resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
try:
    m=mmap.mmap(-1,8*1024**3,prot=0)
except OSError:
    print(json.dumps({'reserved':False}))
else:
    print(json.dumps({'reserved':True,'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
    m.close()
'''
    resident, virtual=memory_limits({'memory':'4g','gpu':False})
    def run(limit):
        return json.loads(subprocess.check_output([sys.executable,'-c',code,str(limit)],text=True))
    assert run(resident)['reserved'] is False
    fixed=run(virtual)
    assert fixed['reserved'] is True and fixed['rss_bytes']<resident
