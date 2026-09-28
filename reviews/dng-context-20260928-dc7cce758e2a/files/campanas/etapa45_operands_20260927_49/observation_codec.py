"""Lossless storage of repeated CSR operands; not quantization or resampling."""
from pathlib import Path
import json,hashlib,io
import numpy as np

def need(x,m):
 if not x:raise ValueError(m)
def digest(a):
 h=hashlib.sha256();h.update(a.dtype.str.encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def decode(path):
 with np.load(path,allow_pickle=False) as p:
  meta=json.loads(str(p['codec_metadata']))
  shape=tuple(meta['operand_shape']);x=np.bitwise_xor.accumulate(p['transmission_xor_transposed'].T,axis=0)
  o=np.empty(shape,np.float32);o[...,1]=x.view(np.float32).reshape(shape[:-1]);static=p['static_weight_cap_mask']
  for j,column in enumerate([0,2,3]):o[...,column]=static[:,j]
  data={k:(o if k=='operands' else p[k]) for k in meta['keys']}
  for k,a in data.items():need(digest(a)==meta['array_sha256'][k],'Decoded array differs '+k)
 return data,meta

def encode(source,destination):
 source,destination=Path(source),Path(destination);need(not destination.exists(),'Preserve packed data')
 with np.load(source,allow_pickle=False) as p:data={k:p[k] for k in p.files}
 o=data['operands'];static=o[0,0,:,[0,2,3]].T.copy()
 for j,c in enumerate([0,2,3]):need(np.array_equal(o[...,c],np.broadcast_to(static[:,j],o.shape[:-1])),'Dynamic field cannot use static codec')
 b=np.ascontiguousarray(o[...,1]).reshape(-1,o.shape[2]).view(np.uint32);delta=np.empty_like(b);delta[0]=b[0];delta[1:]=np.bitwise_xor(b[1:],b[:-1])
 meta=dict(schema='lossless_operand_xor49_v1',keys=list(data),operand_shape=list(o.shape),array_sha256={k:digest(a) for k,a in data.items()},original_npz_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
 packed={k:v for k,v in data.items() if k!='operands'};packed.update(static_weight_cap_mask=static,transmission_xor_transposed=delta.T.copy(),codec_metadata=np.array(json.dumps(meta)))
 np.savez_compressed(destination,**packed)
 decoded,_=decode(destination)
 for k in data:need(np.array_equal(decoded[k],data[k]),'Exact codec roundtrip '+k)
 return dict(source=str(source),destination=str(destination),original_bytes=source.stat().st_size,packed_bytes=destination.stat().st_size,array_bitexact=True,original_sha256=meta['original_npz_sha256'],packed_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();print(json.dumps(encode(a.source,a.destination)))
