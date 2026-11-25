"""Observe an existing target cloud through an independently moving broad cloud."""
import hashlib
import numpy as np
from scipy.ndimage import gaussian_filter
def frames(p):return p["x"].reshape(-1,4,4,4,4,4).transpose(0,3,1,4,2,5).reshape(-1,4,16,16)
def contaminate(p):
    source=frames(p);axis=np.linspace(-1,1,16);xx,yy=np.meshgrid(axis,axis);tt=np.arange(4)[:,None,None]
    out=[]
    for im in source:
        seed=int.from_bytes(hashlib.sha256(im.tobytes()+b"broad-cloud-01").digest()[:8],"little")
        g=np.random.default_rng(seed);cx,cy=g.uniform(-.62,.62,2);vx,vy=g.uniform(-.16,.16,2);a=g.uniform(.55,1.6);wx,wy=g.uniform(.37,.66,2)
        cloud=a*np.exp(-.5*((xx-cx-vx*tt)**2/wx**2+(yy-cy-vy*tt)**2/wy**2))
        out.append(im+cloud)
    return np.array(out,np.float32)
def descriptors(images):
    cols=[]
    for s in (0.,.65,1.1):
        sm=gaussian_filter(images,(0,0,s,s)) if s else images
        for width in (1.,2.,4.,0.):
            view=sm-gaussian_filter(sm,(0,0,width,width)) if width else sm
            avg=(view[:,1:]+view[:,:-1])/2
            dt=np.diff(view,axis=1);gy,gx=np.gradient(avg,axis=(2,3))
            # Match the centered derivative's scale to distances in the [-1,1] field.
            deriv=np.stack([gx,gy],-1)[:,:,2:-2,2:-2,:].reshape(len(view),-1,2)
            rhs=-dt[:,:,2:-2,2:-2].reshape(len(view),-1,1)
            normal=deriv.transpose(0,2,1)@deriv
            v=np.linalg.solve(normal+np.eye(2)*1.e-7,deriv.transpose(0,2,1)@rhs)*2/15
            err=(deriv@(v*7.5)-rhs).reshape(len(view),-1)
            cols.append(np.column_stack([np.linalg.norm(v,axis=(1,2)),v[:,:,0],np.sqrt((err**2).mean(1)),np.trace(normal,axis1=1,axis2=2)]))
    return np.stack(cols,1).astype("float32")
