import numpy as np
from common import load,write_txt,write_csv
_,x,y,v=load(); X0=x[:,:5]; mu=X0.mean(axis=0); sd=X0.std(axis=0,ddof=1); Z=(X0-mu)/sd
d={}
def kmeans(k):
    step=len(Z)//k; cent=Z[np.arange(k)*step].copy(); labels=np.full(len(Z),-1)
    while True:
        dist=np.sum((Z[:,None,:]-cent[None,:,:])**2,axis=2)
        new=np.argmin(dist,axis=1)
        if np.array_equal(new,labels): break
        labels=new
        for j in range(k):
            if np.any(labels==j): cent[j]=Z[labels==j].mean(axis=0)
    wcss=float(np.sum((Z-cent[labels])**2))
    return labels,cent,wcss
results={}
for k in range(2,7):
    lab,cen,wc=kmeans(k); results[k]=(lab,cen,wc); d[f"wcss_k{k}"]=wc
elbow=5
for k in (3,4,5):
    if d[f"wcss_k{k}"]-d[f"wcss_k{k+1}"] < .5*(d[f"wcss_k{k-1}"]-d[f"wcss_k{k}"]):
        elbow=k; break
d["elbow_k"]=elbow
lab,cen,wc=results[3]
groups=[]
for j in range(3):
    ix=np.where(lab==j)[0]; groups.append((float(X0[ix,2].mean()),ix))
groups.sort(key=lambda z:z[0]); segment_rows=[]; segidx={}
for name,(m3,ix) in zip("ABC",groups):
    segidx[name]=ix
    segment_rows.append([name,len(ix),*X0[ix].mean(axis=0),float(y[ix].mean()),float(v[ix].mean())])
    d[f"size_segment_{name}"]=len(ix)
write_csv("part3_segments.csv",["segment","size","mean_X1","mean_X2","mean_X3","mean_X4","mean_X5","outcome_rate","mean_value"],segment_rows)
D=np.array([[0,7.9,7.4,2.9,2.9,6.1],[7.9,0,1.4,5.9,5.3,2.0],[7.4,1.4,0,5.8,4.6,1.3],[2.9,5.9,5.8,0,2.8,4.5],[2.9,5.3,4.6,2.8,0,3.4],[6.1,2.0,1.3,4.5,3.4,0]],float)
hrows=[]
for method in ("single","complete","average"):
    clusters=[{i} for i in range(6)]; heights=[]
    for step in range(5):
        pairs=[]
        for a in range(len(clusters)):
            for b in range(a+1,len(clusters)):
                vals=[D[i,j] for i in clusters[a] for j in clusters[b]]
                dist=min(vals) if method=="single" else max(vals) if method=="complete" else float(np.mean(vals))
                pairs.append((dist,a,b))
        height,a,b=min(pairs)
        ca,cb=clusters[a],clusters[b]
        if min(ca)>min(cb): ca,cb=cb,ca
        hrows.append([method,step+1,"+".join(f"E{i+1}" for i in sorted(ca)),"+".join(f"E{i+1}" for i in sorted(cb)),height])
        heights.append(height); merged=clusters[a]|clusters[b]
        clusters=[c for q,c in enumerate(clusters) if q not in (a,b)]+[merged]
        clusters.sort(key=lambda c:min(c))
    gaps=np.diff(heights); cut=int(np.argmax(gaps))+1 # number of merges below largest gap
    d[f"{method}_clusters_at_cut"]=6-cut
write_csv("part3_hierarchical.csv",["linkage","merge_no","cluster_a","cluster_b","height"],hrows)
eig,vec=np.linalg.eigh((Z-Z.mean(axis=0)).T@(Z-Z.mean(axis=0))/(len(Z)-1))
order=np.argsort(eig)[::-1]; eig=eig[order]; vec=vec[:,order]
for j in range(5):
    ix=np.argmax(np.abs(vec[:,j]))
    if vec[ix,j]<0: vec[:,j]*=-1
total=float(eig.sum()); ratios=eig/total; cum=np.cumsum(ratios); proj=Z@vec[:,:2]
d["eigenvalue_sum"]=total
for i in range(5):
    d[f"eigenvalue_{i+1}"]=eig[i]; d[f"explained_ratio_{i+1}"]=ratios[i]; d[f"cumulative_ratio_{i+1}"]=cum[i]
d["components_for_80pct"]=int(np.argmax(cum>=.8)+1)
for j in range(5):
    d[f"pc1_loading_X{j+1}"]=vec[j,0]; d[f"pc2_loading_X{j+1}"]=vec[j,1]
for i in range(3):
    d[f"row{i+1}_pc1"]=proj[i,0]; d[f"row{i+1}_pc2"]=proj[i,1]
for name,ix in segidx.items(): d[f"mean_pc1_segment_{name}"]=float(proj[ix,0].mean())
write_txt("part3_results.txt",d)
