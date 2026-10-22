"""Read-only Donau scheduler adapter used by login acceptance; never submits."""
import json
import subprocess


def query(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return dict(command=args, exit_code=result.returncode,
                stdout=result.stdout, stderr=result.stderr)


def decoded(row):
    if row['exit_code'] != 0:
        raise RuntimeError('SCHEDULER_QUERY_FAILED:'+str(row))
    value = json.loads(row['stdout'])
    if value.get('code') != 'success':
        raise RuntimeError('SCHEDULER_RESPONSE_REJECTED:'+str(value))
    return value['data']
