import numpy as np, json, os
OUT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/out'
lb=json.load(open(f'{OUT}/s1_lb_decode.json')); sg=json.load(open(f'{OUT}/s7_significance.json'))
ag=json.load(open(f'{OUT}/s12_agg.json'))
R0=1.6461597403364463; R1=1.646143314225527; RT=1.6446514942
S=lb['RMS_d_applied']; RR=R0
rho_obs=lb['implied_rho']; se=sg['SE_50k_empirical']
o={}
# ---------- F. remaining gap from the NEW best ----------
M1=R1**2; MT=RT**2
o['gap']=dict(from_score=R1,to_score=RT,req_dRMSLE=RT-R1,MSE_from=M1,MSE_to=MT,req_dMSE=MT-M1,
              req_rho_independent=float(np.sqrt(M1-MT)/R1))
# ---------- old vs new channel predictions of the EXP075 outcome ----------
def dR(rho,S=S,RR=RR): return (-2*rho*S*RR+S*S)/(2*RR)
def amp(rho,S=S,RR=RR): return rho*RR/S
PRED={'OLD clean-forward joint (EXP075 headline)':0.037939815873089,
      'OLD late-fold S_pred weighted (audit)':0.040508,
      'OLD late-fold S_wide weighted (audit)':0.029942,
      'OLD late-fold S_wide latest (audit)':0.032472,
      'NEW SBVC strong min-proj weighted':ag['rho_strong'],
      'NEW SBVC strong min-proj latest':ag['rho_strong_latest'],
      'NEW SBVC strong post-proj weighted':ag['rho_strong_postproj'],
      'NEW SBVC strong post-proj latest':ag['rho_strong_postproj_latest']}
tab=[]
for k,v in PRED.items():
    tab.append(dict(channel=k,rho_pred=v,dRMSLE_pred=dR(v),amp_pred=amp(v),
                    err_rho=v-rho_obs,err_dRMSLE=dR(v)-lb['delta_RMSLE'],err_amp=amp(v)-lb['implied_opt_amplitude'],
                    z_vs_observed=(v-rho_obs)/se))
o['prediction_test']=tab
o['observed']=dict(rho=rho_obs,dRMSLE=lb['delta_RMSLE'],amp=lb['implied_opt_amplitude'],SE_rho=se)
# ---------- 5. posterior on the true full-test amplitude ----------
for tau,lab in [(0.006,'wide'),(0.004,'tight')]:
    for mu0,l2 in [(ag['rho_strong_postproj'],'postproj'),(ag['rho_strong'],'minproj')]:
        s=sg['SE_rho_pub_minus_full']
        mu=(mu0/tau**2+rho_obs/s**2)/(1/tau**2+1/s**2); sd=(1/tau**2+1/s**2)**-0.5
        a_star=mu*RR/S
        be1=S/(2*RR)                                     # break-even rho at amplitude 1
        from math import erf,sqrt
        P=lambda x: 0.5*(1+erf((mu-x)/(sd*sqrt(2))))
        dm=lambda a: -2*a*mu*S*RR+a*a*S*S
        o[f'posterior_{lab}_{l2}']=dict(prior_mu=mu0,prior_sd=tau,obs=rho_obs,obs_sd=s,
            post_mu=mu,post_sd=sd,post_opt_amplitude=a_star,post_opt_amp_sd=sd*RR/S,
            P_rho_gt_breakeven_at_a1=P(be1),
            P_rho_gt_breakeven_at_a_opt=P(a_star*S/(2*RR)),
            E_dRMSLE_at_a1=dm(1.0)/(2*RR), E_dRMSLE_at_a_opt=dm(a_star)/(2*RR),
            E_dRMSLE_at_a_prior_only=dm(mu0*RR/S)/(2*RR),
            prior_only_opt_amplitude=mu0*RR/S)
print(json.dumps(o,indent=1)); json.dump(o,open(f'{OUT}/s18_decision.json','w'),indent=1)
