import numpy as np
from common import load,write_txt
_,x,y,v=load(); n=len(v); x4=x[:,3]
d={}
mean=float(np.mean(v)); sd=float(np.std(v,ddof=1)); se=sd/np.sqrt(n)
d.update(n=n,mean_value=mean,sd_value=sd,se_value=se,ci95_low=mean-1.96*se,ci95_high=mean+1.96*se)
z=(mean-8)/se; d.update(z_stat=z,reject_h0="yes" if abs(z)>1.96 else "no")
d["n_required"]=int(np.ceil((1.96*sd/0.5)**2))
def reg(a,b):
    am=np.mean(a); bm=np.mean(b)
    slope=np.sum((a-am)*(b-bm))/np.sum((a-am)**2); intercept=bm-slope*am
    fit=intercept+slope*a; resid=b-fit
    tss=np.sum((b-bm)**2); ess=np.sum((fit-bm)**2); rss=np.sum(resid**2)
    return slope,intercept,fit,resid,tss,ess,rss
slope,inter,fit,res,tss,ess,rss=reg(x4,v)
d.update(reg_slope=slope,reg_intercept=inter,tss=tss,ess=ess,rss=rss,
 tss_equals_ess_plus_rss="yes" if abs(tss-ess-rss)<1e-6*tss else "no",
 r2=1-rss/tss,rmse=np.sqrt(rss/n),mae=np.mean(np.abs(res)))
zx=(x4-np.mean(x4))/np.std(x4,ddof=1); m=c=0.0
def cost(): return float(np.mean((v-(m*zx+c))**2))
d["gd_J0"]=cost()
for ep in range(1,501):
    e=v-(m*zx+c); gm=-2*np.mean(zx*e); gc=-2*np.mean(e)
    m-=0.1*gm; c-=0.1*gc
    if ep in (1,10,100,500): d[f"gd_J{ep}"]=cost()
gds=m/np.std(x4,ddof=1); gdi=c-m*np.mean(x4)/np.std(x4,ddof=1)
d.update(gd_slope=gds,gd_intercept=gdi,gd_slope_gap=abs(gds-slope))
order=np.argsort(fit,kind="stable")
d["resid_sd_lower_raw"]=np.std(res[order[:400]],ddof=1); d["resid_sd_upper_raw"]=np.std(res[order[400:]],ddof=1)
d["resid_sd_ratio_raw"]=d["resid_sd_upper_raw"]/d["resid_sd_lower_raw"]
lx=np.log(x4); lv=np.log(v)
ls,li,lfit,lres,ltss,less,lrss=reg(lx,lv)
d.update(loglog_intercept=li,loglog_slope=ls,loglog_r2=1-lrss/ltss,loglog_rmse=np.sqrt(lrss/n))
lo=np.argsort(lfit,kind="stable")
d["resid_sd_lower_log"]=np.std(lres[lo[:400]],ddof=1); d["resid_sd_upper_log"]=np.std(lres[lo[400:]],ddof=1)
d["resid_sd_ratio_log"]=d["resid_sd_upper_log"]/d["resid_sd_lower_log"]
X=np.column_stack([np.ones(n),np.log(x4),x[:,0],x[:,1],x[:,2],x[:,4],x[:,5],x[:,6]])
beta=np.linalg.solve(X.T@X,X.T@lv); pred=X@beta
mrss=np.sum((lv-pred)**2); mrtss=np.sum((lv-np.mean(lv))**2); mr2=1-mrss/mrtss
d.update({f"mr_beta_{i}":b for i,b in enumerate(beta)})
d["mr_r2"]=mr2; d["mr_adj_r2"]=1-(1-mr2)*(n-1)/(n-7-1)
d["loglog_adj_r2"]=1-(1-d["loglog_r2"])*(n-1)/(n-1-1)
write_txt("part1_results.txt",d)
