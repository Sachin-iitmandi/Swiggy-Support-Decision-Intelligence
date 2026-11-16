import numpy as np
from common import load,write_txt,write_csv,split,metrics,sigmoid,knn_predict,logistic_fit
_,x,y,v=load(); tr,te=split(y); mu=x[tr].mean(axis=0); sd=x[tr].std(axis=0,ddof=1); X=(x-mu)/sd
xt=X[tr]; xv=X[te]; yt=y[tr]; yv=y[te]
w,b,_=logistic_fit(xt,yt,.1,1000,False); lp=sigmoid(xv@w+b)
# Cross-validation to select K, same fixed training standardisation.
fold=np.arange(len(tr))%5; cv=[]
for k in range(1,22,2):
    scores=[]
    for f in range(5):
        va=np.where(fold==f)[0]; ref=np.where(fold!=f)[0]
        scores.append(metrics(yt[va],knn_predict(xt[ref],yt[ref],xt[va],k))["f1"])
    cv.append([k,float(np.mean(scores))])
best_k=sorted(cv,key=lambda r:(-r[1],r[0]))[0][0]
kp=knn_predict(xt,yt,xv,best_k,True); kpred=(kp>=.5).astype(int)
# entropy and Gini
def entropy(yy):
    if len(yy)==0:return 0.
    p=np.mean(yy)
    return 0. if p in (0,1) else -p*np.log2(p)-(1-p)*np.log2(1-p)
def gini(yy):
    if len(yy)==0:return 0.
    p=np.mean(yy); return 1-p*p-(1-p)*(1-p)
root_e=entropy(yt); root_g=gini(yt); splits=[]
conds=[xt[:,5]==1,xt[:,6]==1,x[tr,2]>.5,x[tr,0]<=24]
for name,c in zip(("S1","S2","S3","S4"),conds):
    a=yt[c]; z=yt[~c]; ie=root_e-(len(a)*entropy(a)+len(z)*entropy(z))/len(yt)
    gg=root_g-(len(a)*gini(a)+len(z)*gini(z))/len(yt)
    splits.append([name,len(a),int(a.sum()),len(z),int(z.sum()),ie,gg])
best=splits[int(np.argmax(np.array(splits)[:,5].astype(float)))][0]
chosen=conds[("S1","S2","S3","S4").index(best)]
yes_rate=float(yt[chosen].mean()); no_rate=float(yt[~chosen].mean())
sp=np.where((x[te,5]==1) if best=="S1" else (x[te,6]==1) if best=="S2" else (x[te,2]>.5) if best=="S3" else (x[te,0]<=24),yes_rate,no_rate)
preds={"logistic":(lp>=.3).astype(int),"knn":kpred,"stump":(sp>=.3).astype(int)}
preds["hard_vote"]=(sum(preds.values())>=2).astype(int)
preds["soft_vote"]=((lp+kp+sp)/3>=.3).astype(int)
vrows=[]
for name,p in preds.items():
    m=metrics(yv,p); vrows.append([name,m["tp"],m["fp"],m["tn"],m["fn"],m["precision"],m["recall"],m["f1"]])
write_csv("part4_voting.csv",["model","tp","fp","tn","fn","precision","recall","f1"],vrows)
d={"root_entropy":root_e,"root_gini":root_g,"best_split_by_info_gain":best,
   "best_split_by_gini_gain":splits[int(np.argmax(np.array(splits)[:,6].astype(float)))][0],
   "stump_rate_yes":yes_rate,"stump_rate_no":no_rate}
write_csv("part4_splits.csv",["split","n_yes","positives_yes","n_no","positives_no","info_gain","gini_gain"],splits)
# AdaBoost
yy=np.array([1,1,1,-1,-1,-1,1,-1]); h1=np.array([1,1,-1,-1,-1,-1,-1,-1]); h2=np.array([-1,1,1,-1,1,-1,1,-1])
weights=np.ones(8)/8; adrows=[]; alphas=[]; errors=[]
for rnd,h in enumerate((h1,h2),1):
    before=weights.copy(); err=float(weights[h!=yy].sum()); alpha=.5*np.log((1-err)/err)
    un=before*np.exp(-alpha*yy*h); weights=un/un.sum()
    errors.append(err); alphas.append(alpha)
    for i in range(8): adrows.append([rnd,i+1,before[i],un[i],weights[i]])
ensemble=np.sign(alphas[0]*h1+alphas[1]*h2)
d.update(round1_error=errors[0],round1_alpha=alphas[0],round2_error=errors[1],round2_alpha=alphas[1],
         ensemble_accuracy=float(np.mean(ensemble==yy)),oob_fraction=(1-1/560)**560)
write_csv("part4_adaboost.csv",["round","point","weight_before","weight_unnormalised","weight_after"],adrows)
# Architecture parameters
arch=[]
def add(name,p): arch.append([name,int(p)]); return int(p)
mlp=[add("mlp_hidden1",784*16+16),add("mlp_hidden2",16*16+16),add("mlp_output",16*10+10)]
cnn1=add("cnn_conv1",5*5*1*6+6); cnn2=add("cnn_conv2",5*5*6*16+16)
# 28 -> conv1 28 -> pool 14 -> conv2 10 -> pool 5, depth 16 => 400
cnn3=add("cnn_dense1",400*32+32); cnn4=add("cnn_dense2",32*10+10)
d.update(mlp_total=sum(mlp),cnn_total=cnn1+cnn2+cnn3+cnn4,
         fc_layer_params=128*128*3*1024+1024,conv_layer_params=3*3*3*64+64,
         rnn_total=64*128+128*128+128*16+128+16)
write_csv("part4_architecture.csv",["layer","parameters"],arch)
write_txt("part4_results.txt",d)
