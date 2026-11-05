"""冻结确认观察器入口；仅root显式调用，导入不派发。"""
from pathlib import Path
import sys

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if not args:raise SystemExit('usage: observe.py D:/.../confirmation_suite --suite | --one job | --child job')
    control=Path(args[0]).absolute();sys.path.insert(0,str(control/'code'))
    from futsalmot import bbox_confirmation_lifecycle as life
    if len(args)==2 and args[1]=='--suite':return life.observe_suite(control)
    if len(args)==3 and args[1]=='--one':return life.observe_job(control,args[2])
    if len(args)==3 and args[1]=='--child':return life.run_job(control,args[2])
    raise SystemExit('明确指定--suite/--one/--child；不默认启动')

if __name__=='__main__':raise SystemExit(main())
