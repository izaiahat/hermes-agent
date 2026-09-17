"""Actual tick coverage after task-scoped CWD replaced global env mutation."""
from collections import Counter

def test_all_workdir_classes_reach_existing_guarded_pool(tmp_path, monkeypatch):
 import cron.scheduler as s
 import cron.executions as executions
 import tools.mcp_tool as mcp
 jobs=[{'id':'agent-workdir','name':'agent','workdir':str(tmp_path),'no_agent':False},
       {'id':'script-workdir','name':'script','workdir':str(tmp_path),'no_agent':True},
       {'id':'agent-without-workdir','name':'agent-no-wd','no_agent':False}]
 called=[]
 s._running_job_ids.clear()
 monkeypatch.setattr(s,'_hermes_home',tmp_path)
 monkeypatch.setattr(s,'_should_yield_tick_to_fresh_gateway',lambda:None)
 monkeypatch.setattr(s,'_maybe_run_worktree_maintenance',lambda:None)
 monkeypatch.setattr(executions,'recover_interrupted_executions',lambda:0)
 monkeypatch.setattr(mcp,'_kill_orphaned_mcp_children',lambda:None)
 monkeypatch.setattr(s,'get_due_jobs',lambda:jobs)
 monkeypatch.setattr(s,'advance_next_runs',lambda ids:len(ids))
 monkeypatch.setattr(s,'load_config',lambda:{})
 monkeypatch.setattr(s,'create_execution',lambda id,**kw:{'id':'execution-'+id})
 monkeypatch.setattr(s,'claim_job_for_fire',lambda id,**kw:dict(next(j for j in jobs if j['id']==id)))
 monkeypatch.setattr(s,'run_one_job',lambda job,**kw:called.append(job['id']) or True)
 try:
  assert s.tick(verbose=False,sync=True)==3
  assert Counter(called)==Counter(j['id'] for j in jobs)
  assert not s.get_running_job_ids()
 finally:s._shutdown_parallel_pool()
