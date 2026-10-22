"""Login-only strict full sample acceptance and explicitly scoped profile."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics as st
import sys

def percentile(a,q):
    a=sorted(a);p=(len(a)-1)*q;i=int(p)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(p-i)
def summary(a):
    med=st.median(a)
    return dict(n=len(a),mean=st.mean(a),median=med,p95=percentile(a,.95),stddev=st.stdev(a),iqr=percentile(a,.75)-percentile(a,.25))
def read(p):return json.loads(p.read_text())
def size(x):return int(x[:-1])*{'K':1024,'M':1024**2,'G':1024**3}[x[-1]] if x[-1].isalpha() else int(x)
def analyze(root):
    root=Path(root);cfg=read(root/'contract.json');manifest=read(root/'kernel_manifest.json')
    start=read(root/'target_start.json');end=read(root/'target_end.json')
    assert end['returncode']==0 and start['host']==cfg['target'] and len(start['affinity'])==1 and start['original_affinity']==start['affinity'] and start['environment']['OMP_NUM_THREADS']==start['environment']['OPENBLAS_NUM_THREADS']=='1'
    raw=(root/'samples.jsonl').read_bytes();rows=[json.loads(x) for x in raw.splitlines()]
    ident=rows.pop(0);assert ident['kind']=='identity' and ident['host']==start['host'] and ident['job']==start['job'] and ident['cpu']==start['affinity'][0]
    assert ident['sve_bytes']==ident['sme_bytes']==64 and rows.pop()=={'kind':'complete'}
    assert ident['hwcap'] & (1<<22)
    mask=(1<<23)|(1<<25)|(1<<29)
    assert ident['hwcap2'] & mask == mask
    nk=len(manifest);total=nk+24+32
    assert len(rows)==3*total*10
    groups={}; cursor=0;last_end=0
    for block in range(3):
        for pos in range(total):
            case=(pos+block*37)%total
            for sample in range(-1,9):
                x=rows[cursor];cursor+=1
                assert (x['case'],x['block'],x['position'],x['sample'])==(case,block,pos,sample)
                assert isinstance(x['elapsed_ns'],int) and x['elapsed_ns']>0
                assert x['guard_ok'] is True and x['cpu']==ident['cpu']
                assert isinstance(x['start_ns'],int) and x['end_ns']-x['start_ns']==x['elapsed_ns'] and x['start_ns']>last_end
                last_end=x['end_ns']
                if case<nk:
                    m=manifest[case]
                    assert x['kind']=='instruction'
                    assert (x['operation'],x['precision'],x['ilp'],x['instructions_per_loop'])==(m['operation'],m['precision'],m['ilp'],m['instructions_per_loop'])
                    assert x['loops']==(2000 if sample==-1 else 200000)
                elif case<nk+24:
                    k=case-nk;num=[16384,65536,262144,1048576,4194304,16777216,67108864,268435456][k//3];mode=k%3
                    assert x['kind']=='memory' and x['operation']==['read','write','copy'][mode]
                    loops=1 if sample==-1 else max(1,(64*1024**2+num-1)//num)
                    assert (x['loops'],x['working_set_bytes'],x['traffic_bytes'])==(loops,num*(2 if mode==2 else 1),num*loops*(2 if mode==2 else 1))
                else:
                    k=case-nk-24;loops=1 if sample==-1 else 8
                    assert (x['kind'],x['columns'],x['batch'],x['operation'],x['loops'],x['traffic_bytes'],x['working_set_bytes'])==('layout',8<<((k//8)%4),1<<((k//2)%4),['linear_transpose','blocked_transpose'][k%2],loops,8388608*loops,8388608)
                if sample>=0:groups.setdefault(case,[]).append(x)
    stats=[];bad=[]
    for case,xs in groups.items():
        a=[x['elapsed_ns'] for x in xs];v=summary(a);blocks=[st.median([x['elapsed_ns'] for x in xs if x['block']==b]) for b in range(3)]
        v.update(case=case,operation=xs[0]['operation'],precision=xs[0]['precision'],ilp=xs[0]['ilp'],block_medians=blocks,iqr_ratio=v['iqr']/v['median'],block_ratio=max(blocks)/min(blocks))
        v['stable']=v['iqr_ratio']<=.15 and v['block_ratio']<=1.15
        if not v['stable']:bad.append(case)
        stats.append(v)
    # Roles are frozen before capture; old small-address stores are diagnostics only.
    blocking=[c for c in bad if c>=nk or manifest[c]['calibration_role']=='model']
    inst={}; mapping={'add':'sve_add','fmla':'sve_fmla','fmopa':'fmopa','load1_seq':'sve_load','load2_seq':'sve_load2','store1_seq':'sve_store','store2_seq':'sve_store2','za_h':'za_extract_h','za_v':'za_extract_v','transition':'smstart_stop'}
    for op in mapping:
        for precision in (['mode'] if op=='transition' else ['fp32','fp64']):
            matches=[v for v in stats if v['case']<nk and v['operation']==op and v['precision']==precision]
            rates=[manifest[v['case']]['instructions_per_loop']*200000/v['median'] for v in matches]
            peak=max(rates);i1=next(v for v in matches if v['ilp']==1)
            key=mapping[op]+('' if precision=='mode' else '_'+precision)
            inst[key]=dict(throughput=peak,latency=i1['median']/(200000*manifest[i1['case']]['instructions_per_loop']),saturation_ilp=min(v['ilp'] for v,r in zip(matches,rates) if r>=.95*peak),resource={'add':'vector','fmla':'vector','fmopa':'matrix','load1_seq':'load','load2_seq':'load','store1_seq':'store','store2_seq':'store','za_h':'za-transfer','za_v':'za-transfer','transition':'streaming-state'}[op],inferred=True)
    for precision in ['fp32','fp64']:
        h=inst.pop('za_extract_h_'+precision);v=inst.pop('za_extract_v_'+precision)
        inst['za_extract_'+precision]=dict(h if h['throughput']<=v['throughput'] else v,latency=max(h['latency'],v['latency']))
    cache=start['cache'];l1=next(x for x in cache if x['level']=='1' and x['type'] in ['Data','Unified']);l2=next(x for x in cache if x['level']=='2')
    mem=[(groups[v['case']][0],v) for v in stats if nk<=v['case']<nk+24]
    def bw(op,capacity):
        choices=[(x,v) for x,v in mem if x['operation']==op and x['working_set_bytes']<=capacity]
        assert choices,('missing_memory_case',op,capacity)
        x,v=max(choices,key=lambda a:a[0]['working_set_bytes'])
        return x['traffic_bytes']/v['median']
    levels={}
    for name,capacity,lat in [('l1d',size(l1['size']),4.),('l2',size(l2['size']),16.),('dram',1<<62,100.)]:
        levels[name]=dict(capacity_bytes=capacity,read_bandwidth_bytes_per_cost_unit=bw('read',capacity),write_bandwidth_bytes_per_cost_unit=bw('write',capacity),mixed_bandwidth_bytes_per_cost_unit=bw('copy',capacity),latency_cost=lat,inferred=True)
    unclamped=json.loads(json.dumps(levels))
    for key in ['read_bandwidth_bytes_per_cost_unit','write_bandwidth_bytes_per_cost_unit','mixed_bandwidth_bytes_per_cost_unit']:
        levels['l2'][key]=min(levels['l1d'][key],levels['l2'][key]);levels['dram'][key]=min(levels['l2'][key],levels['dram'][key])
    copy=[(x,v) for x,v in mem if x['operation']=='copy'];layout=[]
    for v in stats:
        if v['case']<nk+24:continue
        x=groups[v['case']][0];c,cv=min(copy,key=lambda z:abs(math.log2(z[0]['working_set_bytes']/x['working_set_bytes'])))
        ratio=(x['traffic_bytes']/v['median'])/(c['traffic_bytes']/cv['median'])
        layout.append(dict(operation=x['operation'],working_set_bytes=x['working_set_bytes'],columns=x['columns'],batch=x['batch'],relative_bandwidth=min(2.,max(.05,ratio)),inferred=True))
    acceptance=dict(status='CALIBRATION_ACCEPTED' if not blocking else 'CALIBRATION_REJECTED_UNSTABLE',raw_sha256=hashlib.sha256(raw).hexdigest(),samples=27*total,warmups=3*total,unstable_cases=bad,blocking_unstable_cases=blocking,statistics=stats,unclamped_memory=unclamped)
    profile=dict(schema_version=3,name='linux-sme1-cal02-'+start['host'],architecture='aarch64-linux+sme',streaming_vector_bits=512,measured=True,cost_unit='nanoseconds',source='MOFFT-LINUX-CAL-02 raw_sha256='+acceptance['raw_sha256'],instructions=inst,memory_hierarchy=dict(cache_line_bytes=int(l1['coherency_line_size']),levels=levels),layout_model=dict(bandwidth_basis='relative-to-sequential-copy',cases=layout),measurement_metadata=dict(machine=start['host'],job=start['job'],formal_measurement=False,accepted=not blocking,scope='separate LD1/LD2/ST1/ST2 sequential-ring cost profile; no kernel or plan wisdom',separate_memory_ops=True,calibration_id='MOFFT-LINUX-CAL-02',memory_operation_note='8192-byte sequential ring; register stream count is not dependency ILP; load dump overhead retained; full predicate proxy, tails not calibrated',latency_definition='issue-rate or wrapper-call proxy; not hardware latency',memory_note='working-set proxies; dram key is model compatibility only, not residency proof; latency penalties 4/16/100 remain inferred',nonexclusive=True,raw_sha256=acceptance['raw_sha256'],identity=start,aliases='upstream mul->fmla; sub/neg->add unchanged',resource_claims='inferred model grouping, not measured execution ports',warmups='one 2000-loop instruction warmup per case/block; one pass memory/layout',samples_per_case=27))
    return acceptance,profile

if __name__=='__main__':
    assert socket.gethostname().startswith('login')
    root=Path(sys.argv[1]);out=root/'analysis_01';out.mkdir(exist_ok=False)
    a,p=analyze(root)
    (out/'acceptance.json').write_text(json.dumps(a,indent=2))
    (out/('profile.json' if a['status']=='CALIBRATION_ACCEPTED' else 'rejected_profile_diagnostic.json')).write_text(json.dumps(p,indent=2))
    print(json.dumps({k:v for k,v in a.items() if k not in ['statistics','unclamped_memory']}))
