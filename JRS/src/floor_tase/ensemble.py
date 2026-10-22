"""Identical frozen ensemble weights: NumPy CPU or batched CUDA matmul."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .surrogate import FEATURE_SCHEMA,FEATURE_NAMES


class ReturnEnsemble:
    def __init__(self,path,sha256,device='cpu'):
        path=Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha256:raise ValueError('Ensemble hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            self.metadata=json.loads(str(z['metadata']))
            self.weights={k:np.asarray(z[k],dtype=np.float32) for k in ('mean','scale','w1','b1','w2','b2','w3','b3')}
        if self.metadata.get('ensemble_schema')!='return-ensemble-v1' or self.metadata['feature_schema']!=FEATURE_SCHEMA:
            raise ValueError('Unsupported ensemble schema')
        if not all(np.isfinite(v).all() for v in self.weights.values()) or (self.weights['scale']<=0).any():
            raise ValueError('Invalid ensemble arrays')
        self.device=device;self.sha256=sha256;self.torch=None;self.info={'device':device,'model_sha256':sha256}
        if device!='cpu':
            if not device.startswith('cuda'):raise ValueError('Use cpu or cuda[:index]')
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA required but unavailable')
            self.torch=torch;self.tensors={k:torch.as_tensor(v,device=device) for k,v in self.weights.items()}
            self.info.update(torch=str(torch.__version__),cuda_runtime=torch.version.cuda,
                             gpu=torch.cuda.get_device_name(device))

    def runtime_info(self):return dict(self.info)

    def predict(self,features):
        x=np.asarray(features,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all():raise ValueError('Invalid public features')
        if self.torch is None:
            w=self.weights;x=(x[None,:,:]-w['mean'][:,None,:])/w['scale'][:,None,:]
            x=np.maximum(np.matmul(x,w['w1'])+w['b1'][:,None,:],0)
            x=np.maximum(np.matmul(x,w['w2'])+w['b2'][:,None,:],0)
            out=np.matmul(x,w['w3'])+w['b3'][:,None,:]
            out[:,:,0]=1/(1+np.exp(-np.clip(out[:,:,0],-40,40)))
            out[:,:,1]=np.maximum(0,out[:,:,1]*5.)
        else:
            t=self.torch;w=self.tensors
            with t.inference_mode():
                x=(t.as_tensor(x,device=self.device)[None,:,:]-w['mean'][:,None,:])/w['scale'][:,None,:]
                x=t.relu(t.bmm(x,w['w1'])+w['b1'][:,None,:])
                x=t.relu(t.bmm(x,w['w2'])+w['b2'][:,None,:])
                raw=t.bmm(x,w['w3'])+w['b3'][:,None,:]
                out=t.stack((t.sigmoid(raw[:,:,0]),t.clamp(raw[:,:,1]*5.,min=0)),2).cpu().numpy()
        if not np.isfinite(out).all():raise ValueError('Nonfinite model output')
        return out
