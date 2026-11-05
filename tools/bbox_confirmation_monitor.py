"""原训练观察台的同截止只读评价接口；不控制或重启任何训练/评价。"""
from pathlib import Path
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import argparse,json,os,sys,time
from urllib.parse import urlsplit
import psutil
from training_monitor import confirmation_snapshot
R=Path(__file__).resolve().parents[1]
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_POST(self):self.send_error(405,'read only')
 def do_GET(self):
  host=self.headers.get('Host','').lower();origin=self.headers.get('Origin','').lower()
  if host not in ('127.0.0.1:8766','localhost:8766') or origin and origin not in ('http://127.0.0.1:8765','http://localhost:8765'):self.send_error(403);return
  path=urlsplit(self.path)
  if path.query or path.path not in ('/api/health','/api/confirmation'):self.send_error(404);return
  value=self.server.health if path.path=='/api/health' else confirmation_snapshot(R,Path('G:/FutsalMOT_Training'))
  raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf8');self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)));self.send_header('X-Content-Type-Options','nosniff')
  if origin:self.send_header('Access-Control-Allow-Origin',origin);self.send_header('Vary','Origin')
  self.end_headers();self.wfile.write(raw)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--deadline-utc',required=True);args=parser.parse_args();deadline=datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00'));assert deadline.tzinfo and time.time()<deadline.timestamp()
 # 不延原已注册观察台的截止，不删除或覆盖登记。
 old=json.loads((R/'.training_monitor.json').read_bytes());assert deadline==datetime.fromisoformat(old['deadline_utc'])
 server=ThreadingHTTPServer(('127.0.0.1',8766),Handler);server.timeout=.5;server.daemon_threads=True
 server.health={'service':'futsalmot-confirmation-monitor','version':1,'pid':os.getpid(),'create_time':psutil.Process().create_time(),'command':psutil.Process().cmdline(),'read_only':True,'deadline_utc':deadline.isoformat()}
 try:
  while time.time()<deadline.timestamp():server.handle_request()
 finally:server.server_close()
if __name__=='__main__':main()
