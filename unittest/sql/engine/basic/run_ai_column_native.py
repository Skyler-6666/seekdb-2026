"""Run native AI-column SQL, transaction and crash-recovery regressions.

Requires pymysql and a built seekdb. Uses a dedicated local test database directory.
"""
from pathlib import Path
import argparse, os, runpy, subprocess, sys, tempfile, time
import pymysql
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[4]
options = argparse.ArgumentParser()
options.add_argument('--binary', type=Path, default=ROOT/'build_release/src/observer'/('seekdb.exe' if os.name == 'nt' else 'seekdb'))
options.add_argument('--base-dir', type=Path)
options.add_argument('--port', type=int, default=2885)
options.add_argument('--file-data-dir', type=Path, help='also run file SQL on this same server; directory must not exist')
options.add_argument('--git', default='git')
args = options.parse_args()
base = args.base_dir or Path(tempfile.mkdtemp(prefix='seekdb-ai-column-native-'))
base.mkdir(parents=True, exist_ok=True)
if args.file_data_dir:args.file_data_dir.resolve().parent.mkdir(parents=True,exist_ok=True)
os.environ['SEEKDB_TEST_PORT'] = str(args.port)
os.environ['SEEKDB_AI_COLUMN_TEST_STATE'] = str(base/'mock')
regression = Path(__file__).with_name('run_ai_column_regression.py')
helpers = runpy.run_path(str(regression))

def connect():
 return pymysql.connect(host='127.0.0.1',port=args.port,user='root',autocommit=True,read_timeout=40,charset='utf8mb4')

def start():
 with (base/'console.log').open('ab') as log:
  process = subprocess.Popen([str(args.binary.resolve()),'--nodaemon','--base-dir',str(base.resolve()),'--port',str(args.port),
   '--parameter','memory_budget=2G','--parameter','cpu_count=2','--parameter','datafile_size=128M',
   '--parameter','datafile_maxsize=1G','--parameter','log_disk_size=2G','--log-level','INFO'] +
   (['--variable','secure_file_priv='+str(args.file_data_dir.resolve().parent)] if args.file_data_dir else []),
   stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
 for attempt in range(900):
  if process.poll() is not None:raise RuntimeError('seekdb exited; inspect '+str(base/'console.log'))
  try:
   connection=connect();connection.close();return process
  except pymysql.MySQLError:time.sleep(.1)
 process.kill();process.wait(timeout=10)
 raise RuntimeError('seekdb startup timed out')

class Echo(BaseHTTPRequestHandler):
 calls = 0
 def do_POST(self):
  request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  value=next(message['content'] for message in request['messages'] if message['role']=='user')
  type(self).calls += 1
  data=json.dumps({'choices':[{'message':{'content':value}}]}).encode()
  self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def log_message(self,*arguments):pass

process = start()
connection = None
runner = None
def query(sql,params=None):
 with connection.cursor() as q:q.execute(sql,params);return q.fetchall()
def expect_error(sql,codes):
 try:query(sql)
 except pymysql.MySQLError as error:assert error.args[0] in codes,(sql,error)
 else:raise AssertionError('expected failure: '+sql)
def wait_empty(table,seconds=30):
 end=time.monotonic()+seconds
 sql="SELECT COUNT(*) FROM oceanbase.__all_ai_column_task j JOIN oceanbase.__all_table t ON t.table_id=j.table_id JOIN oceanbase.__all_database d ON d.database_id=t.database_id WHERE d.database_name='ai_column_edges' AND t.table_name=%s"
 while time.monotonic()<end:
  if query(sql,(table,))==((0,),):return
  time.sleep(.1)
 raise AssertionError('tasks did not drain for '+table)
try:
 if args.file_data_dir:
  connection=connect()
  assert query('SELECT @@session.enable_file_sql,@@session.ai_column_strict_read')==((0,0),)
  print('PASS same native server registers enable_file_sql and ai_column_strict_read, both default OFF',flush=True)
  connection.close();connection=None
  subprocess.run([sys.executable,str(Path(__file__).with_name('run_file_sql_regression.py')),
   '--port',str(args.port),'--prepare-data','--data-dir',str(args.file_data_dir.resolve()),'--git',args.git],check=True)
 subprocess.run([sys.executable,str(regression)],check=True)
 connection=connect();runner=helpers['Runner'](connection);runner.command('python ai_column_mock.py start 19124')
 query('DROP DATABASE IF EXISTS ai_column_edges');query('CREATE DATABASE ai_column_edges');query('USE ai_column_edges')
 for sql in ["CALL DBMS_AI_SERVICE.DROP_AI_MODEL_ENDPOINT('ai_column_edge_ep')","CALL DBMS_AI_SERVICE.DROP_AI_MODEL('ai_column_edge_model')"]:
  try:query(sql)
  except pymysql.MySQLError as e:assert e.args[0] in (11112,11118)
 query("CALL DBMS_AI_SERVICE.CREATE_AI_MODEL('ai_column_edge_model','{\"type\":\"completion\",\"model_name\":\"mock-complete\"}')")
 query("CALL DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('ai_column_edge_ep','{\"ai_model_name\":\"ai_column_edge_model\",\"provider\":\"openai\",\"url\":\"http://127.0.0.1:19124/v1/chat/completions\",\"access_key\":\"edge-key\"}')")
 query("CREATE TABLE composite(id VARCHAR(80),part INT,content TEXT,result VARCHAR(400) AI COLUMN(AI_COMPLETE('ai_column_edge_model',content)) WITH(max_retries=0),PRIMARY KEY(id,part))")
 key="a'\\汉"
 query('INSERT INTO composite(id,part,content) VALUES(%s,7,%s)',(key,'initial'))
 query("INSERT INTO composite(id,part,content) VALUES('null',8,NULL)")
 wait_empty('composite')
 assert query('SELECT result FROM composite WHERE id=%s AND part=7',(key,))==(('mock:initial',),)
 query('SET SESSION ai_column_strict_read=ON')
 assert query("SELECT result FROM composite WHERE id='null'")==((None,),)
 query('SET SESSION ai_column_strict_read=OFF')
 expect_error("UPDATE composite SET result=NULL",(1235,))
 expect_error("INSERT INTO composite(id,part,content,result) VALUES('bad',1,'bad',NULL)",(1235,))
 query("CREATE PROCEDURE forbidden_ai_write() BEGIN UPDATE composite SET result='bad'; END")
 expect_error('CALL forbidden_ai_write()',(1235,))
 runner.command('python ai_column_mock.py reset 19124')
 query('UPDATE composite SET content=content WHERE id=%s',(key,));time.sleep(.3)
 assert helpers['mock'].read_events(19124)==[]
 query('BEGIN');query('UPDATE composite SET id=%s,content=%s WHERE id=%s',('new-key','rolled',key))
 query('SET SESSION ai_column_strict_read=ON');expect_error('SELECT result FROM composite',(11120,))
 assert query('SELECT COUNT(*) FROM composite')==((2,),)
 query('ROLLBACK');query('SET SESSION ai_column_strict_read=OFF');time.sleep(.3)
 assert helpers['mock'].read_events(19124)==[]
 assert query('SELECT result FROM composite WHERE id=%s',(key,))==(('mock:initial',),)
 query('BEGIN')
 query("INSERT INTO composite(id,part,content) VALUES('CaseKey',99,'uncommitted')")
 query("UPDATE composite SET id='casekey' WHERE id='CaseKey' AND part=99")
 query("DELETE FROM composite WHERE id='casekey' AND part=99")
 tasks=query("SELECT COUNT(*) FROM oceanbase.__all_ai_column_task j JOIN oceanbase.__all_table t ON t.table_id=j.table_id JOIN oceanbase.__all_database d ON d.database_id=t.database_id WHERE d.database_name='ai_column_edges' AND t.table_name='composite'")
 assert tasks==((0,),),tasks
 query('ROLLBACK')
 print('PASS composite/string key, NULL READY, PL write protection, identity update, rollback and own-transaction strict read',flush=True)
 query("CREATE TABLE large_payload(id INT PRIMARY KEY,body LONGTEXT,result LONGTEXT AI COLUMN(AI_COMPLETE('ai_column_edge_model',body)) WITH(max_retries=0))")
 large_text="汉\\'\n"*10000
 query('INSERT INTO large_payload(id,body) VALUES(1,%s)',(large_text,))
 wait_empty('large_payload')
 assert query('SELECT result FROM large_payload')==(('mock:'+large_text,),)
 print('PASS large Unicode LONGTEXT input and output',flush=True)
 query('CREATE TABLE backfill(id BIGINT PRIMARY KEY,content TEXT)')
 query("INSERT INTO backfill VALUES(1,'existing')")
 query("ALTER TABLE backfill ADD result VARCHAR(400) AI COLUMN(AI_COMPLETE('ai_column_edge_model',content)) WITH(max_retries=0)")
 wait_empty('backfill');assert query('SELECT result FROM backfill')==(('mock:existing',),)
 sql="SELECT COUNT(*) FROM oceanbase.__all_ai_column a JOIN oceanbase.__all_table t ON t.table_id=a.table_id JOIN oceanbase.__all_database d ON d.database_id=t.database_id WHERE d.database_name='ai_column_edges' AND t.table_name='backfill' AND JSON_EXTRACT(a.definition,'$.state_column_id')=a.state_column_id AND JSON_EXTRACT(a.definition,'$.request_column_id')=a.request_column_id AND JSON_EXTRACT(a.definition,'$.result_spec_column_id')=a.result_spec_column_id"
 assert query(sql)==((1,),)
 query("ALTER TABLE backfill MODIFY result VARCHAR(400) AI COLUMN(AI_COMPLETE('ai_column_edge_model',CONCAT('changed:',content))) WITH(max_retries=0)")
 wait_empty('backfill');assert query('SELECT result FROM backfill')==(('mock:changed:existing',),)
 assert len(query('SHOW AI COLUMNS FROM backfill'))==1
 print('PASS ADD/MODIFY existing-row backfill, persisted hidden IDs and SHOW AI COLUMNS',flush=True)
 query("CREATE TABLE heap(content TEXT,result VARCHAR(400) AI COLUMN(AI_COMPLETE('ai_column_edge_model',content)) WITH(max_retries=0))")
 query("INSERT INTO heap(content) VALUES('heap-row')")
 wait_empty('heap');assert query('SELECT result FROM heap')==(('mock:heap-row',),)
 print('PASS AI column on a table with an internal primary key',flush=True)
 query("CREATE TABLE pending(id BIGINT PRIMARY KEY,content TEXT,result VARCHAR(400) AI COLUMN(AI_COMPLETE('ai_column_edge_model',content)) WITH(max_retries=0))")
 query("INSERT INTO pending(id,content) VALUES(1,'slow-old')");time.sleep(.2)
 assert query('SELECT result FROM pending')==((None,),)
 connection.close();connection=None;process.kill();process.wait(timeout=10)
 process=start();connection=connect();query('USE ai_column_edges')
 wait_empty('pending',35);assert query('SELECT result FROM pending')==(('mock:slow-old',),)
 query('SET SESSION ai_column_strict_read=ON');assert query('SELECT result FROM backfill')==(('mock:changed:existing',),)
 print('PASS committed task recovery after process termination and READY schema/state persistence',flush=True)
 query('SET SESSION ai_column_strict_read=OFF')
 echo=ThreadingHTTPServer(('127.0.0.1',19125),Echo)
 echo_thread=threading.Thread(target=echo.serve_forever,daemon=True);echo_thread.start()
 try:
  for sql in ["CALL DBMS_AI_SERVICE.DROP_AI_MODEL_ENDPOINT('ai_column_cast_ep')","CALL DBMS_AI_SERVICE.DROP_AI_MODEL('ai_column_cast_model')"]:
   try:query(sql)
   except pymysql.MySQLError as error:assert error.args[0] in (11112,11118)
  query("CALL DBMS_AI_SERVICE.CREATE_AI_MODEL('ai_column_cast_model','{\"type\":\"completion\",\"model_name\":\"echo\"}')")
  query("CALL DBMS_AI_SERVICE.CREATE_AI_MODEL_ENDPOINT('ai_column_cast_ep','{\"ai_model_name\":\"ai_column_cast_model\",\"provider\":\"openai\",\"url\":\"http://127.0.0.1:19125/v1/chat/completions\",\"access_key\":\"test\"}')")
  query("CREATE TABLE cast_results(id INT PRIMARY KEY,body TEXT,bad FLOAT AI COLUMN(AI_COMPLETE('ai_column_cast_model',body)) WITH(max_retries=1),good VARCHAR(40) AI COLUMN(AI_COMPLETE('ai_column_cast_model',body)) WITH(max_retries=0))")
  query("INSERT INTO cast_results(id,body) VALUES(1,'1e100')")
  wait_empty('cast_results')
  assert query('SELECT bad,good FROM cast_results')==((None,'1e100'),)
  assert Echo.calls==3,Echo.calls
  query('SET SESSION ai_column_strict_read=ON')
  assert query('SELECT good FROM cast_results')==(('1e100',),)
  expect_error('SELECT bad FROM cast_results',(11120,))
  query('SET SESSION ai_column_strict_read=OFF')
  query("ALTER TABLE cast_results RETRY AI COLUMN bad WHERE status='FAILED'")
  wait_empty('cast_results')
  assert Echo.calls==5,Echo.calls
  assert query('SELECT bad,good FROM cast_results')==((None,'1e100'),)
  print('PASS native target-type conversion failure, bounded retry and grouped READY/FAILED publication',flush=True)
 finally:
  echo.shutdown();echo.server_close();echo_thread.join(timeout=5)
 query('DROP DATABASE ai_column_edges')
 query("CALL DBMS_AI_SERVICE.DROP_AI_MODEL_ENDPOINT('ai_column_cast_ep')")
 query("CALL DBMS_AI_SERVICE.DROP_AI_MODEL('ai_column_cast_model')")
 query("CALL DBMS_AI_SERVICE.DROP_AI_MODEL_ENDPOINT('ai_column_edge_ep')");query("CALL DBMS_AI_SERVICE.DROP_AI_MODEL('ai_column_edge_model')")
finally:
 if runner is not None:runner.close()
 if connection is not None:connection.close()
 if process.poll() is None:process.terminate();process.wait(timeout=10)
