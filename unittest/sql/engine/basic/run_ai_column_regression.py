"""Execute the supplied mysqltest SQL and mock assertions on native seekdb.

Implements only the control statements used by this supplied suite; original SQL,
fixtures and mock handler are loaded directly from the checkout.
"""
from pathlib import Path
import tempfile
import importlib.util,sys,subprocess,time,re,json,socket,io,contextlib,os
import pymysql
ROOT=Path(__file__).resolve().parents[4]
SUITE=ROOT/'tools/deploy/mysql_test/test_suite/ai_column'
STATE=Path(os.environ.setdefault('SEEKDB_AI_COLUMN_TEST_STATE',str(Path(tempfile.gettempdir())/('seekdb-ai-column-'+str(os.getpid())))))
STATE.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('contest_mock',SUITE/'include/ai_column_mock.py')
mock=importlib.util.module_from_spec(spec);spec.loader.exec_module(mock)
mock.state_dir=lambda port:str(STATE/str(port))

if len(sys.argv)>1 and sys.argv[1]=='serve':
 mock.serve(int(sys.argv[2]));raise SystemExit

def parse(source):
 result=[];i=0
 def space():
  nonlocal i
  while i<len(source):
   if source[i].isspace():i+=1
   elif source[i]=='#':
    p=source.find('\n',i);i=len(source) if p<0 else p+1
   else:break
 while i<len(source):
  space()
  if i>=len(source):break
  if source[i]=='}':i+=1;result.append(('end',''));continue
  if source[i]=='{':i+=1;result.append(('begin',''));continue
  if source.startswith('--',i):
   end=source.find('\n',i);end=len(source) if end<0 else end
   result.append(('directive',source[i+2:end].strip().rstrip(';')));i=end;continue
  match=re.match(r'(while|if)\s*\(([^)]*)\)',source[i:])
  if match:result.append((match[1],match[2]));i+=match.end();continue
  start=i;quote=None
  while i<len(source):
   c=source[i];i+=1
   if quote:
    if c=='\\' and i<len(source):i+=1
    elif c==quote:
     if i<len(source) and source[i]==quote:i+=1
     else:quote=None
   elif c in "'\"`":quote=c
   elif c==';':break
  result.append(('statement',source[start:i].strip().rstrip(';')))
 def blocks(index=0):
  nodes=[]
  while index<len(result):
   kind,text=result[index];index+=1
   if kind=='end':return nodes,index
   if kind in ('if','while'):
    assert result[index][0]=='begin',(kind,text,result[index]);body,index=blocks(index+1);nodes.append((kind,text,body))
   else:nodes.append((kind,text))
  return nodes,index
 return blocks()[0]

class Runner:
 def __init__(self,connection):self.connection=connection;self.variables={};self.allowed=[0];self.processes={};self.sets=[];self.errors=[];self.transcript=[];self.query_log=True;self.result_log=True
 def substitute(self,value):return re.sub(r'\$([a-zA-Z_][\w]*)',lambda m:str(self.variables.get(m[1],0)),value)
 def condition(self,value):
  text=self.substitute(value).replace('&&',' and ').replace('||',' or ')
  assert re.fullmatch(r'[\d\s<>=!().andor-]+',text),text
  return bool(eval(text,{'__builtins__':{}},{}))
 def sql(self,sql,visible=True):
  sql=self.substitute(sql);allowed=self.allowed;self.allowed=[0]
  if visible and self.query_log:self.transcript.append(sql+';')
  try:
   with self.connection.cursor() as q:
    q.execute(sql);rows=q.fetchall();headers=tuple(c[0] for c in q.description) if q.description else ()
  except pymysql.MySQLError as error:
   if error.args[0] not in allowed:print('FAILED SQL:',sql,flush=True);raise
   if visible:self.transcript.append('ERROR '+str(error.args[0])+': '+str(error.args[1]))
   if visible and 0 not in allowed:self.errors.append((sql,str(error.args[1])))
   return ()
  assert 0 in allowed,('expected SQL failure',allowed,sql)
  if visible and headers and self.result_log:
   self.sets.append((sql,headers,rows))
   self.transcript.append('\t'.join(headers))
   for row in rows:self.transcript.append('\t'.join('NULL' if v is None else str(v) for v in row))
   print('RESULT',headers,rows,flush=True)
  return rows
 def command(self,text):
  fields=text.split();assert 'ai_column_mock.py' in text,text
  at=next(i for i,v in enumerate(fields) if v.endswith('ai_column_mock.py'))
  mode,port=fields[at+1],int(fields[at+2]);directory=STATE/str(port);directory.mkdir(exist_ok=True)
  if mode=='start':
   self.command('python ai_column_mock.py stop '+str(port))
   (directory/'events.jsonl').write_text('',encoding='utf-8')
   log=(directory/'server.out').open('ab')
   process=subprocess.Popen([sys.executable,__file__,'serve',str(port)],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);log.close();self.processes[port]=process
   for _ in range(100):
    try:
     with socket.create_connection(('127.0.0.1',port),timeout=.1):return
    except OSError:time.sleep(.03)
   raise RuntimeError('mock startup failed')
  elif mode=='stop':
   if port in self.processes:
    process=self.processes.pop(port);process.terminate();process.wait(timeout=5)
  elif mode=='reset':(directory/'events.jsonl').write_text('',encoding='utf-8')
  elif mode=='assert':
   output=io.StringIO()
   with contextlib.redirect_stdout(output):mock.assert_events(port,fields[at+3])
   self.transcript.extend(output.getvalue().splitlines())
  else:raise ValueError(mode)
 def run(self,nodes):
  for node in nodes:
   kind,text=node[:2]
   if kind in ('if','while'):
    iterations=0
    while self.condition(text):
     self.run(node[2]);iterations+=1
     if kind=='if':break
     assert iterations<=1000,('loop did not terminate',text)
   elif kind=='directive':
    if text.startswith('error '):self.allowed=[int(x) for x in text[6:].split(',')]
    elif text.startswith('real_sleep '):time.sleep(float(text.split()[1]))
    elif text.startswith('exec '):self.command(text[5:])
    elif text=='disable_query_log':self.query_log=False
    elif text=='enable_query_log':self.query_log=True
    elif text=='disable_result_log':self.result_log=False
    elif text=='enable_result_log':self.result_log=True
    elif text.startswith(('disable_warnings','enable_warnings','replace_regex','replace_column')):pass
    else:raise ValueError(('unknown directive',text))
   elif kind=='statement':
    match=re.fullmatch(r'let\s+\$(\w+)\s*=\s*(.*)',text,re.S)
    if match:
     value=match[2].strip()
     if value.startswith('`') and value.endswith('`'):
      rows=self.sql(value[1:-1],False);value=rows[0][0] if rows else 0
     else:value=int(self.substitute(value))
     self.variables[match[1]]=0 if value is None else value
    elif text.startswith('dec $'):self.variables[text[5:].strip()]-=1
    elif text.startswith('inc $'):self.variables[text[5:].strip()]+=1
    elif text.startswith('connection '):pass
    elif text.startswith('eval '):self.sql(text[5:])
    else:self.sql(text)
 def close(self):
  for port in list(self.processes):self.command('python ai_column_mock.py stop '+str(port))

def main():
 connection=pymysql.connect(host='127.0.0.1',port=int(os.environ.get('SEEKDB_TEST_PORT','2885')),user='root',autocommit=True,read_timeout=40,charset='utf8mb4')
 names=sys.argv[1:] or ['ai_column_ddl','ai_column_t1_dml','ai_column_worker','ai_column_retry_strict','ai_column_vector_index']
 for name in names:
  runner=Runner(connection)
  try:
   runner.run(parse((SUITE/'t'/(name+'.test')).read_text(encoding='utf-8')))
   golden=(SUITE/'r/mysql'/(name+'.result')).read_text(encoding='utf-8')
   sections=[];query=[];body=[]
   starts=lambda line:bool(re.match(r'^(?:select|explain|create|alter|drop|call|insert|update|delete|set|start|commit|rollback)\b',line.strip(),re.I))
   for line in golden.splitlines():
    if not line.strip() or line.startswith(('#','mock_')):continue
    if starts(line) and not query:
     if sections:sections[-1][1].extend(body);body=[]
     query=[line.strip()]
    elif query:query.append(line.strip())
    else:body.append(line.strip())
    if query and line.rstrip().endswith(';'):
     sections.append(['\n'.join(query),[]]);query=[]
   if sections:sections[-1][1].extend(body)
   normalize=lambda sql:re.sub(r'\s+','',sql.rstrip(';')).lower()
   position=0
   for sql,headers,rows in runner.sets:
    while position<len(sections) and normalize(sections[position][0])!=normalize(sql):position+=1
    assert position<len(sections),('missing golden SQL',name,sql)
    expected=sections[position][1];position+=1
    actual=['\t'.join(headers)]+['\t'.join('NULL' if v is None else format(v,'.15g') if isinstance(v,float) else str(v) for v in row) for row in rows]
    actual='\n'.join(actual).splitlines()
    clean=lambda lines:[re.sub(r'__type_[0-9]+_[0-9]+','__type_0_0',re.sub(r'__vid_[0-9]+(?:_[0-9]+)?','__vid_0',line.strip())) for line in lines]
    assert clean(actual)==clean(expected),(name,sql,actual,expected)
   error_clean=lambda text:re.sub(r'table_id=[0-9]+, column_id=[0-9]+','table_id=<table_id>, column_id=<column_id>',text)
   for sql,message in runner.errors:
    expected=[line.split(': ',1)[1] for statement,lines in sections if normalize(statement)==normalize(sql) for line in lines if line.startswith('ERROR ') and ': ' in line]
    assert any(error_clean(message)==error_clean(value) for value in expected),(name,sql,message,expected)
   print('PASS',name,flush=True)
  finally:
   (STATE/(name+'.actual')).write_text('\n'.join(runner.transcript)+'\n',encoding='utf-8');runner.close()
 connection.close()

if __name__=='__main__':main()
