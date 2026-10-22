"""SME1-only post-generation correction of locality dispatch statements."""
import hashlib
import json
from pathlib import Path
import re
import sys

PATTERN = re.compile(r'(mofft_kernel_r[0-9]+_fp(?:32|64)_(?:fwd|bwd)_other_'
                     r'(?:direct_)?broadcast_r(?:medium|large|huge)'
                     r'\(input, stride, input_repeat, tw, tw_stride, tw_repeat, output, batch\))(?= else\b)')


def repair(path):
    path = Path(path)
    original = path.read_bytes()
    text = original.decode('utf-8')
    corrected, count = PATTERN.subn(r'\1;', text)
    if corrected.replace(';', '') != text.replace(';', ''):
        raise ValueError('NON_TERMINATOR_CHANGE')
    if count:
        path.write_bytes(corrected.encode('utf-8'))
    proof = dict(sme1_only=True, statement_terminators_added=count,
                 before_sha256=hashlib.sha256(original).hexdigest(),
                 after_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 arithmetic_or_call_arguments_changed=False)
    path.with_suffix('.syntax.json').write_text(json.dumps(proof, indent=2)+'\n', encoding='utf-8')
    return proof


if __name__ == '__main__':
    print(json.dumps(repair(sys.argv[1])))
