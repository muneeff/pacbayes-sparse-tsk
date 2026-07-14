from __future__ import annotations
from pathlib import Path
import hashlib,json,datetime

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def freeze(protocol_path, tracked_paths, output_path):
    protocol=Path(protocol_path); root=protocol.parent.parent
    rows=[]
    for rel in sorted(set(tracked_paths)):
        p=(root/rel).resolve()
        if not p.is_file(): raise FileNotFoundError(rel)
        rows.append({"path":rel,"sha256":sha256_file(p),"size_bytes":p.stat().st_size})
    payload={"schema_version":"3.0","created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"protocol":{"path":str(protocol.relative_to(root)),"sha256":sha256_file(protocol)},"tracked_files":rows,"single_authorized_run_only":True,"status":"FROZEN_NOT_RUN"}
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2),encoding='utf-8'); return payload
