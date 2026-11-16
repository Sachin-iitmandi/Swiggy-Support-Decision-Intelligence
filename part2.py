import numpy as np
from common import load,write_txt,write_csv,split,sigmoid,logistic_fit,knn_predict,metrics
_,x,y,v=load(); tr,te=split(y); mu=np.mean(x[tr],axis=0); sd=np.std(x[tr],axis=0,ddof=1)
X=(x-mu)/sd; xt=X[tr]; xv=X[te]; yt=y[tr]; yv=y[te]
d=dict(train_rows=len(tr),train_positives=int(yt.sum()),test_rows=len(te),test_positives=int(yv.sum()))
w,b,losses=logistic_fit(xt,yt,0.1,1000,True)
for ep,val in losses.items(): d[f"loss_epoch_{ep}"]=val
d.update({f"weight_X{i+1}":z for i,z in enumerate(w)}); d["bias"]=b
p_train=sigmoid(xt@w+b); p_test=sigmoid(xv@w+b)
def bce(yy,pp):
    pp=np.clip(pp,1e-15,1-1e-15); return float(-np.mean(yy*np.log(pp)+(1-yy)*np.log(1-pp)))
d["test_bce"]=bce(yv,p_test)
for eta in (0.01,0.1,1.0):
    _,_,ls=logistic_fit(xt,yt,eta,200,True); d[f"loss_epoch_200_eta_{eta}"]=ls[200]
for i,z in enumerate(w): d[f"odds_ratio_X{i+1}"]=np.exp(z)
for tau in (0.5,0.3):
    mm=metrics(yv,p_test>=tau)
    for k,val in mm.items():
        if k not in ("tpr","fpr"): d[f"tau_{tau}_{k}"]=val
d["blind_accuracy"]=float(np.mean(yv==0))
rows=[]
for tau in np.arange(.1,1,.1):
    mm=metrics(yv,p_test>=tau); rows.append([round(tau,1),mm["tp"],mm["fp"],mm["tn"],mm["fn"],mm["tpr"],mm["fpr"]])
write_csv("part2_thresholds.csv",["tau","tp","fp","tn","fn","tpr","fpr"],rows)
fold=np.arange(len(tr))%5; cv=[]
for k in range(1,22,2):
    scores=[]
    for f in range(5):
        va=np.where(fold==f)[0]; ref=np.where(fold!=f)[0]
        pp=knn_predict(xt[ref],yt[ref],xt[va],k)
        scores.append(metrics(yt[va],pp)["f1"])
    cv.append([k,float(np.mean(scores))])
write_csv("part2_knn_cv.csv",["K","mean_cv_f1"],cv)
best_k=sorted(cv,key=lambda r:(-r[1],r[0]))[0][0]; d["knn_best_k"]=best_k
knn=knn_predict(xt,yt,xv,best_k)
km=metrics(yv,knn)
for k,val in km.items():
    if k not in ("tpr","fpr","specificity"): d[f"knn_{k}"]=val
costs=[]
for tau in np.arange(.05,1,.05):
    m=metrics(yt,p_train>=tau); costs.append((50*m["fp"]+400*m["fn"],tau))
cost_train,tau=min(costs,key=lambda q:(q[0],q[1]))
d["cost_tau"]=tau; d["cost_train_at_tau"]=cost_train
d["cost_test_at_tau"]=50*metrics(yv,p_test>=tau)["fp"]+400*metrics(yv,p_test>=tau)["fn"]
d["cost_test_at_0.5"]=50*metrics(yv,p_test>=.5)["fp"]+400*metrics(yv,p_test>=.5)["fn"]
d["cost_test_flag_nobody"]=400*int(yv.sum())
write_txt("part2_results.txt",d)
