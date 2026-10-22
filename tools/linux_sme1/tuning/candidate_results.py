"""Strict candidate log parser; it never turns unaccepted logs into wisdom."""
import math
import re
import statistics


def parse(text, expected_cases):
    labels={}
    for c in expected_cases:
        label=(f"r{c['radix']}/{c['precision']}/{c['direction']}/{c['stage']}/"
               f"{c['pattern']}/vg{c['matrix_vg_width']}/rot{int(c['rotate_temp_tiles'])}/"
               f"pipe{c['batch_pipeline_depth']}/{c['context']}")
        if label in labels:raise ValueError('DUPLICATE_EXPECTED_LABEL')
        labels[label]=c
    raw={label:[] for label in labels};med={label:[] for label in labels}
    confirmations=[];repeats=set();actual_order=[]
    for line in text.splitlines():
        if line.startswith('validated '):
            m=re.fullmatch(r'validated (\d+) generated candidates; failures=(\d+)',line)
            if not m:raise ValueError('MALFORMED_VALIDATION_SUMMARY')
            confirmations.append((int(m[1]),int(m[2])))
        elif line.startswith(('raw_rounds,','benchmark,')):
            fields=line.split(',');kind,label,repeat=fields[:3]
            if label not in labels:raise ValueError('UNKNOWN_CANDIDATE_LABEL')
            repeat=int(repeat)
            if repeat<=0:raise ValueError('INVALID_REPEAT')
            repeats.add(repeat);values=[float(x) for x in fields[3:]]
            if not values or not all(math.isfinite(x) and x>0 for x in values):
                raise ValueError('INVALID_TIMING_VALUE')
            if kind=='raw_rounds':
                if len(values)!=9:raise ValueError('EXPECTED_NINE_ROUNDS')
                raw[label].append(values);actual_order.append(label)
            else:
                if len(values)!=1:raise ValueError('EXPECTED_ONE_MEDIAN')
                med[label].append(values[0])
    if confirmations!=[(len(labels),0)]:raise ValueError('MISSING_OR_FAILED_NUMERICAL_SUMMARY')
    if len(repeats)!=1:raise ValueError('MIXED_OR_MISSING_LOCALITY_REPEAT')
    expected_order=[]
    for reverse in (False,True,False):
        for pre in ('fp32','fp64'):
            group=[label for label,c in labels.items() if c['precision']==pre]
            expected_order.extend(reversed(group) if reverse else group)
    if actual_order!=expected_order:raise ValueError('WRONG_FORWARD_REVERSE_FORWARD_ORDER')
    result=[]
    for label,c in labels.items():
        if len(raw[label])!=3 or len(med[label])!=3:raise ValueError('INCOMPLETE_THREE_PASS_CAPTURE')
        for rounds,m in zip(raw[label],med[label]):
            if abs(statistics.median(rounds)-m)>.00050001:
                raise ValueError('RAW_MEDIAN_MISMATCH')
        result.append(dict(**c,raw_round_nanoseconds=raw[label],pass_nanoseconds=med[label],
            median_nanoseconds=statistics.median(med[label]),locality_repeat=next(iter(repeats))))
    return dict(status='CANDIDATE_LOG_STRUCTURALLY_COMPLETE_NOT_ACCEPTED_WISDOM',
        entries=result,native_qualification_granted=False,formal_measurement=False)
