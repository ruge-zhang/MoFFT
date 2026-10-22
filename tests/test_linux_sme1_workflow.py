"""Platform-independent rejection/provenance checks for the opt-in workflow."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT/'tools/linux_sme1'
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT/'compiler'))
import workflow
from common import check_manifest, file_manifest
from registry_terminators import repair
from submit import decoded
from mofft_compiler.targets.arm_sme import arm_sme_target


class LinuxWorkflowTests(unittest.TestCase):
    def test_versioned_c_compiler_does_not_form_invalid_cxx_name(self):
        with tempfile.TemporaryDirectory() as directory:
            cc, cxx = Path(directory)/'clang-19', Path(directory)/'clang++'
            cc.write_text('C driver')
            cxx.write_text('C++ driver')
            actual_cc, actual_cxx = workflow.compiler_paths(cc)
            self.assertEqual(actual_cc, str(cc.absolute()))
            self.assertEqual(actual_cxx, str(cxx.absolute()))
            self.assertNotIn('clang-19++', actual_cxx)

    def test_distinct_sme1_memory_cost_keys_preserve_sme2(self):
        sme1 = dict(arm_sme_target(sme1=True).instruction_model_keys)
        sme2 = dict(arm_sme_target().instruction_model_keys)
        self.assertEqual((sme1['load2'], sme1['store2']), ('sve_load2', 'sve_store2'))
        self.assertEqual((sme2['load2'], sme2['store2']), ('sve_load', 'sve_store'))

    def test_registry_repair_is_terminators_only_and_idempotent(self):
        source = ('if (x) mofft_kernel_r2_fp32_fwd_other_broadcast_rmedium'
                  '(input, stride, input_repeat, tw, tw_stride, tw_repeat, output, batch) else fallback();\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'registry.c'
            path.write_text(source)
            self.assertEqual(repair(path)['statement_terminators_added'], 1)
            self.assertEqual(path.read_text().replace(';',''), source.replace(';',''))
            once = path.read_bytes()
            self.assertEqual(repair(path)['statement_terminators_added'], 0)
            self.assertEqual(path.read_bytes(), once)

    def test_manifest_rejects_modified_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source'
            path.write_text('original')
            manifest = file_manifest([path])
            path.write_text('changed')
            with self.assertRaises(AssertionError):
                check_manifest(manifest)

    def test_phase_scripts_quote_paths_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='mofft workflow ') as directory:
            root = Path(directory)
            workflow.write_capture_scripts(root)
            scripts = list(root.glob('run_*.sh'))
            self.assertEqual(len(scripts), 4)
            for script in scripts:
                text = script.read_text()
                self.assertIn('set -euo pipefail', text)
                self.assertIn(' capture ', text)
                self.assertNotIn('timeout', text)
                self.assertNotIn('dsub', text)
            with self.assertRaises(FileExistsError):
                workflow.write_capture_scripts(root)

    def test_wrong_fftw_source_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'untrusted.tar.gz'
            archive.write_bytes(b'not the pinned FFTW source')
            with self.assertRaisesRegex(ValueError, 'HASH_MISMATCH'):
                workflow.build_fftw(Path(directory), archive, '/unused/clang', None)
            self.assertFalse((Path(directory)/'fftw_source').exists())

    def test_failed_scheduler_query_is_not_acceptance(self):
        with self.assertRaises(RuntimeError):
            decoded(dict(exit_code=1, stdout='', stderr='denied'))
        with self.assertRaises(RuntimeError):
            decoded(dict(exit_code=0, stdout=json.dumps({'code':'failure'}), stderr=''))


if __name__ == '__main__':
    unittest.main()
