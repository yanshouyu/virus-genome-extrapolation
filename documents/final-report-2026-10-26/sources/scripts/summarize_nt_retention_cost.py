"""Summarize preserved SLURM accounting without hiding early engineering failures."""
import csv,json
from datetime import datetime
from pathlib import Path
P=Path('documents/retention-2026-10-25')

def main():
 rows=list(csv.DictReader((P/'jobs.psv').open(),delimiter='|'));mainrows={r['JobID']:r for r in rows if '.' not in r['JobID']}
 phases={'train feature extraction':['25004631']+[f'25004643_{i}' for i in range(6)],'benchmark test feature extraction':[f'25004644_{i}' for i in range(7)],'fresh process replay':['25004863'],'engineering GPU smoke':['25004444','25004455']}
 gpu={job:phase for phase,ids in phases.items() for job in ids};records=[];events=[]
 for job,phase in gpu.items():
  r=mainrows[job];assert r['State']=='COMPLETED' and r['ExitCode']=='0:0',r
  seconds=int(r['ElapsedRaw']);records.append(dict(job=job,phase=phase,allocated_gpu_seconds=seconds))
  events.extend([(datetime.fromisoformat(r['Start']),1),(datetime.fromisoformat(r['End']),-1)])
 active=0;maximum=0
 for _,change in sorted(events):active+=change;maximum=max(maximum,active)
 assert maximum<=2
 summary=dict(gpu_jobs=records,total_allocated_gpu_hours=sum(r['allocated_gpu_seconds'] for r in records)/3600,
              max_simultaneous_study_gpus=maximum,phase_allocated_gpu_minutes={phase:sum(r['allocated_gpu_seconds'] for r in records if r['phase']==phase)/60 for phase in phases},
              noncompleted_main_jobs=[dict(job=k,state=v['State'],exit_code=v['ExitCode']) for k,v in mainrows.items() if v['State']!='COMPLETED'],
              accounting_timezone='UTC',interpretation='Allocation wall time with one H100 per GPU job; no dollar-cost estimate. Early engineering failures remain recorded.')
 (P/'compute-cost.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['| Phase | H100 allocation minutes |','|---|---:|']+[f'| {k} | {v:.2f} |' for k,v in summary['phase_allocated_gpu_minutes'].items()]
 lines+=['',f"Total: {summary['total_allocated_gpu_hours']:.3f} allocated GPU hours; maximum simultaneous study GPUs: {maximum}.",'','All GPU jobs use one full H100. Time includes startup/checks/I/O as well as inference. This is allocation time, not measured accelerator utilization or a monetary price. Initial CPU data-audit failures are preserved in jobs.psv.']
 (P/'compute-cost.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
