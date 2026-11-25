"""Execute shell setup with fake server/Conda/Python; no package install or network."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]


class SetupCompatibilityTests(unittest.TestCase):
    def fixture(self, folder, exists=True, fail_pip=False):
        root = Path(folder)
        project = root/'project'
        plugin = project/'qat_plugin'
        plugin.mkdir(parents=True)
        (project/'2026-11-24_QAT_install_manifest.json').write_text('{}')
        shutil.copy2(ROOT/'qat_plugin/server_setup_2026-11-24.sh', plugin/'setup.sh')
        prefix = root/'envs'/'gesture_aimet_qat'
        (prefix/'bin').mkdir(parents=True)
        fake_bin = root/'bin'
        fake_bin.mkdir()
        (fake_bin/'python').symlink_to(sys.executable)
        (fake_bin/'uname').write_text('#!/bin/sh\necho Linux\n')
        (fake_bin/'uname').chmod(0o755)
        state = root/'envs.json'
        state.write_text(json.dumps({'envs': [str(prefix)] if exists else []}))
        log = root/'calls.jsonl'
        conda = fake_bin/'conda'
        conda.write_text('#!' + sys.executable + '\n' + textwrap.dedent('''\
            import json,os,sys
            from pathlib import Path
            args=sys.argv[1:]
            with open(os.environ['SETUP_TEST_LOG'],'a') as f: f.write(json.dumps(['conda']+args)+'\\n')
            state=Path(os.environ['SETUP_TEST_STATE'])
            if args==['env','list','--json']: print(state.read_text())
            elif args[:1]==['create']: state.write_text(json.dumps({'envs':[os.environ['SETUP_TEST_PREFIX']]}))
            else: sys.exit('Old Conda only supports env list/create in this test')
        '''))
        conda.chmod(0o755)
        target_python = prefix/'bin/python'
        target_python.write_text('#!' + sys.executable + '\n' + textwrap.dedent('''\
            import json,os,sys
            args=sys.argv[1:]
            with open(os.environ['SETUP_TEST_LOG'],'a') as f: f.write(json.dumps(['target-python']+args)+'\\n')
            if os.environ.get('SETUP_TEST_FAIL_PIP')=='1' and args[:3]==['-m','pip','install']: sys.exit(17)
        '''))
        target_python.chmod(0o755)
        env = dict(os.environ, PATH=str(fake_bin)+os.pathsep+os.environ.get('PATH',''),
                   CONDA_EXE=str(conda), SETUP_TEST_LOG=str(log), SETUP_TEST_STATE=str(state),
                   SETUP_TEST_PREFIX=str(prefix), SETUP_TEST_FAIL_PIP='1' if fail_pip else '0')
        return plugin/'setup.sh', env, log

    def test_existing_env_and_check_do_not_use_conda_run(self):
        with tempfile.TemporaryDirectory() as folder:
            script, env, log = self.fixture(folder)
            for mode in ('--install','--check'):
                result = subprocess.run(['bash',str(script),mode], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            calls = [json.loads(x) for x in log.read_text().splitlines()]
            self.assertTrue(any(c[:4]==['target-python','-m','pip','install'] for c in calls))
            self.assertTrue(any(c[:4]==['target-python','-m','unittest','discover'] for c in calls))
            self.assertTrue(all(c[:2]!=['conda','run'] for c in calls))
            self.assertTrue(all(c[:2]!=['conda','create'] for c in calls))

    def test_missing_env_is_created_once_then_uses_target_python(self):
        with tempfile.TemporaryDirectory() as folder:
            script, env, log = self.fixture(folder, exists=False)
            result = subprocess.run(['bash',str(script),'--install'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            calls = [json.loads(x) for x in log.read_text().splitlines()]
            self.assertEqual(sum(c[:2]==['conda','create'] for c in calls),1)

    def test_failed_pip_stops_before_remaining_install_and_check(self):
        with tempfile.TemporaryDirectory() as folder:
            script, env, log = self.fixture(folder, fail_pip=True)
            result = subprocess.run(['bash',str(script),'--install'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,17)
            calls = [json.loads(x) for x in log.read_text().splitlines()]
            self.assertEqual(sum(c[:4]==['target-python','-m','pip','install'] for c in calls),1)
            self.assertFalse(any(c[:4]==['target-python','-m','pip','check'] for c in calls))


if __name__ == '__main__':
    unittest.main()
