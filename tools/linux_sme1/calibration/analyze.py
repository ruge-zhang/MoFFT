"""CAL-03 observation/environment gates; unchanged CAL-02 statistics core."""
import json
from pathlib import Path
from cal02_analyze import analyze as frozen_analyze


def analyze(root):
    root=Path(root)
    acceptance,profile=frozen_analyze(root)
    start=json.loads((root/'target_start.json').read_text())
    reference=json.loads((root/'cpu_reference.json').read_text())
    assert start['original_affinity']==start['affinity']==[0]
    assert start['cpu_binding']['midr']==reference['cpu_binding']['midr']
    assert start['cache']==reference['cache'] and start['uname'][2:]==reference['uname'][2:]
    for key in ['scaling_governor','scaling_min_freq','scaling_max_freq']:
        assert start['frequency'][key]==reference['frequency'][key]
    rows=[json.loads(x) for x in (root/'samples.jsonl').read_text().splitlines()][1:-1]
    addresses=None;observations=[]
    for x in rows:
        if x['kind']!='layout':
            assert x['observation'] is None
            continue
        v=x['observation'];assert isinstance(v,dict)
        for key in ['thread_start_ns','thread_end_ns','thread_cpu_ns','source_address','destination_address','rows']:
            assert type(v[key]) is int and v[key]>0
        assert v['thread_end_ns']-v['thread_start_ns']==v['thread_cpu_ns']
        for key in ['nvcsw','nivcsw','minflt','majflt']:
            assert type(v[key]) is int and v[key]>=0
        for when in ['before','after']:
            f,e=v['freq_'+when+'_khz'],v['freq_'+when+'_errno']
            assert type(e) is int and e>=0
            assert (e>0 and f is None) or (e==0 and type(f) is int and f>0)
        pair=(v['source_address'],v['destination_address'])
        assert pair[0]!=pair[1] and all(p%128==0 for p in pair)
        if addresses is None:addresses=pair
        assert addresses==pair
        assert v['rows']==262144//(x['columns']*x['batch'])
        observations.append(v)
    assert len(observations)==960
    acceptance['layout_observations']=dict(count=960,frequency_missing=sum(v['freq_before_khz'] is None or v['freq_after_khz'] is None for v in observations),addresses=addresses,filtering_applied=False)
    profile['name']='linux-sme1-cal03-'+start['host']
    profile['source']='MOFFT-LINUX-CAL-03 raw_sha256='+acceptance['raw_sha256']
    profile['measurement_metadata'].update(calibration_id='MOFFT-LINUX-CAL-03',preserved_cal02_probe_machine_code=False,probe_source_rebuilt=True,layout_observation_outside_raw=True,observation_may_perturb_surrounding_state=True)
    return acceptance,profile
