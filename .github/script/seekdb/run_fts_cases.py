import argparse
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / ".github" / "script" / "seekdb"))
import mysqltest_for_seekdb as runner

args = argparse.Namespace(
    seekdb=repo / "build_release" / "src" / "observer" / "seekdb",
    obclient=repo / "deps" / "3rd" / "u01" / "obclient" / "bin" / "obclient",
    mysqltest=repo / "deps" / "3rd" / "u01" / "obclient" / "bin" / "mysqltest",
    base_dir=repo / ".seekdb_fts_test" / "instance",
    work_dir=repo / ".seekdb_fts_test" / "work",
    host="127.0.0.1",
    port=2882,
    slice_index=0,
    slice_count=1,
)
deploy_dir = repo / "tools" / "deploy"
sdb_script = repo / ".github" / "script" / "seekdb" / "sdb.py"

cases = [case for case in runner.discover_cases(repo)
         if case.name.startswith("fts_index.")]
if not cases:
    raise SystemExit("no fts_index cases discovered")
print("fts_index cases: {}".format([case.name for case in cases]), flush=True)

args.work_dir.mkdir(parents=True, exist_ok=True)
runner.prepare_instance(args, repo, sdb_script, deploy_dir)
tmp_dir = args.work_dir / "tmp"
log_dir = args.work_dir / "mysqltest_log"
tmp_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

failed = []
try:
    for index, case in enumerate(cases):
        code = None
        output = ""
        for retry_index in range(runner.MAX_CASE_RETRIES + 1):
            if retry_index > 0:
                print("[ RETRY   ] {} ({}/{})".format(
                    case.name, retry_index, runner.MAX_CASE_RETRIES), flush=True)
                runner.prepare_instance(args, repo, sdb_script, deploy_dir)
            code, output = runner.run_case(args, deploy_dir, case, tmp_dir, log_dir)
            if code == 0:
                break
        if code != 0:
            failed.append(case.name)
            runner.save_case_failure(args, case.name, output)
            if index + 1 < len(cases):
                runner.prepare_instance(args, repo, sdb_script, deploy_dir)
finally:
    runner.destroy_instance(sdb_script, args.base_dir, repo, check=False)

print("fts_index failed cases: {}".format(failed), flush=True)
sys.exit(1 if failed else 0)
