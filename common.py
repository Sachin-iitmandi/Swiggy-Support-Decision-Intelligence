import csv, os
import numpy as np

OUT="outputs"
os.makedirs(OUT, exist_ok=True)

def load():
    a=np.genfromtxt("data/records.csv",delimiter=",",skip_header=1,dtype=str)
    x=a[:,1:8].astype(float); y=a[:,8].astype(int); value=a[:,9].astype(float)
    return a,x,y,value

def write_txt(name, d):
    with open(f"{OUT}/{name}","w") as f:
        for k,v in d.items():
            if isinstance(v,(str,bool)): s=str(v).lower() if isinstance(v,bool) else v
            elif isinstance(v,(int,np.integer)): s=str(int(v))
            else: s=f"{float(v):.4f}"
            f.write(f"{k} = {s}\n")

def write_csv(name, header, rows):
    with open(f"{OUT}/{name}","w",newline="") as f:
        w=csv.writer(f); w.writerow(header)
        for row in rows:
            w.writerow([f"{v:.4f}" if isinstance(v,(float,np.floating)) else v for v in row])

def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p).astype(int)
    tp=int(np.sum((y==1)&(p==1))); fp=int(np.sum((y==0)&(p==1)))
    tn=int(np.sum((y==0)&(p==0))); fn=int(np.sum((y==1)&(p==0)))
    div=lambda a,b:a/b if b else 0.0
    pr=div(tp,tp+fp); re=div(tp,tp+fn)
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,accuracy=div(tp+tn,len(y)),precision=pr,recall=re,
                f1=div(2*pr*re,pr+re),specificity=div(tn,tn+fp),tpr=re,fpr=div(fp,fp+tn))

def split(y):
    i0=np.where(y==0)[0]; i1=np.where(y==1)[0]
    tr=np.sort(np.r_[i0[:7*len(i0)//10],i1[:7*len(i1)//10]])
    te=np.setdiff1d(np.arange(len(y)),tr,assume_unique=True)
    return tr,te

def sigmoid(z):
    z=np.clip(z,-500,500)
    return 1/(1+np.exp(-z))

def logistic_fit(X,y,eta=0.1,steps=1000,track=False):
    w=np.zeros(X.shape[1]); b=0.0; losses={}
    wanted={0,1,10,100,200,1000}
    def loss(xx,yy):
        p=np.clip(sigmoid(xx@w+b),1e-15,1-1e-15)
        return float(-np.mean(yy*np.log(p)+(1-yy)*np.log(1-p)))
    if track: losses[0]=loss(X,y)
    for ep in range(1,steps+1):
        p=sigmoid(X@w+b); e=p-y
        w-=eta*(X.T@e)/len(y); b-=eta*np.mean(e)
        if track and ep in wanted: losses[ep]=loss(X,y)
    return w,b,losses

def knn_predict(Xref,yref,Xq,k,prob=False):
    out=[]
    for q in Xq:
        dist=np.sqrt(np.sum((Xref-q)**2,axis=1))
        idx=np.argsort(dist,kind="stable")[:k]
        rate=float(np.mean(yref[idx]))
        out.append(rate if prob else int(rate>=0.5))
    return np.asarray(out)
