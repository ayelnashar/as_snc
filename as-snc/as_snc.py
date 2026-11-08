"""Real AS-SNC with cached rotations and reduced Newton restoration."""
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh, solve
from scipy.optimize import nnls

def quality(Q,L,nu,beta,w,lam=None):
    if w is None or not np.all(np.isfinite(w)):
        return dict(primal=np.inf,stationarity=np.inf,complementarity=np.inf,residual=np.inf)
    t=norm(w); slack=L.T@w-nu*t-beta
    scale=1+abs(beta)+abs(L.T@w)+nu*t
    G=L-np.outer(w/max(t,1e-300),nu); grad=2*Q@w
    if lam is None:
        A=np.flatnonzero(slack/scale<1e-5);lam=np.zeros(len(beta))
        if len(A):lam[A]=nnls(G[:,A],grad,maxiter=max(1000,30*len(A)))[0]
    primal=float(np.max(np.maximum(-slack,0)/scale,initial=0))
    stationarity=float(norm(grad-G@lam)/(1+norm(grad)+norm(G@lam)))
    comp=float(np.max(abs(lam*slack),initial=0)/(1+abs(w@Q@w)))
    return dict(primal=primal,stationarity=stationarity,complementarity=comp,
                residual=max(primal,stationarity,comp),objective=float(w@Q@w))

def _linear(A,b):
    try:return np.linalg.solve(A,b)
    except np.linalg.LinAlgError:return np.linalg.lstsq(A,b,rcond=1e-13)[0]

def _residual(s,B,nu,b,v,lam):
    t=norm(v)
    if t<1e-30:return np.inf
    stat=2*s*v-B@lam+(nu@lam)*v/t
    slack=B.T@v-nu*t-b
    return max(norm(stat)/(1+norm(2*s*v)+norm(B@lam)),
               np.max(abs(slack)/(1+abs(b)+abs(B.T@v)+nu*t),initial=0))

def _newton(s,B,nu,b,v,lam,tol):
    """Newton KKT correction via diagonal-plus-rank-one inverse and a x a solve."""
    for k in range(40):
        merit=_residual(s,B,nu,b,v,lam)
        if merit<tol:return v,lam,True,k
        t=norm(v);u=v/t;mu=(nu@lam)/t
        if mu < -1e-12:return v,lam,False,k
        mu=max(mu,0.);d=1/(2*s+mu);den=1-mu*np.dot(u*d,u)
        if den<=1e-14:return v,lam,False,k
        def hinv(x):
            if x.ndim==1:return d*x+mu/den*(d*u)*np.dot(d*u,x)
            return d[:,None]*x+mu/den*np.outer(d*u,(d*u)@x)
        G=B-np.outer(u,nu);stat=2*s*v-G@lam;slack=B.T@v-nu*t-b
        HG=hinv(G);hs=hinv(stat)
        dl=_linear(G.T@HG,-slack+G.T@hs);dv=HG@dl-hs
        step=1.
        for _ in range(25):
            vn=v+step*dv;ln=lam+step*dl
            if _residual(s,B,nu,b,vn,ln)<merit*(1-1e-4*step):break
            step*=.5
        else:return v,lam,False,k
        v,lam=vn,ln
    return v,lam,False,40

def as_snc(Q,L,nu,beta,max_outer=100,tol=1e-9,return_trace=False,return_info=False):
    """Preserves original tuple API; optional info includes multipliers/provenance.

    End-to-end cost includes eigendecomposition and final original-coordinate
    KKT check. Conic fallback is separately identified, never labelled fast AS-SNC.
    """
    if any(np.iscomplexobj(x) for x in (Q,L,nu,beta)):raise ValueError('Real inputs required')
    Q,L,nu,beta=[np.asarray(x,dtype=float) for x in (Q,L,nu,beta)]
    if L.ndim!=2 or Q.shape!=(L.shape[0],L.shape[0]) or nu.shape!=(L.shape[1],) or beta.shape!=nu.shape:raise ValueError('Input shapes')
    if not all(np.all(np.isfinite(x)) for x in (Q,L,nu,beta)) or np.any(nu<0) or tol<=0:raise ValueError('Invalid inputs')
    if not np.allclose(Q,Q.T,rtol=1e-12,atol=1e-14):raise ValueError('Q must be symmetric')
    s,U=eigh(Q,check_finite=False,driver='evd');scale=s[-1]
    if s[0]<=0:raise ValueError('Q must be positive definite')
    s=s/scale;B=U.T@L;M,N=L.shape;A=list(range(N));trace=[];newton_steps=0
    gate=max(tol,1e-10);inner=min(1e-8,max(tol*.05,1e-13))
    if np.all(beta<=0):
        result=(np.zeros(M),0,np.array([],int));info=dict(method='AS-SNC',newton_steps=0,lam=np.zeros(N),**quality(Q,L,nu,beta,result[0],np.zeros(N)))
        if return_trace:result+=([result[0].copy()],)
        return result+(info,) if return_info else result
    mu,t=1.,1.;visits={};reason='iteration limit'
    for outer in range(max_outer):
        if not A:A=[int(np.argmax(beta))]
        key=tuple(A);visits[key]=visits.get(key,0)+1
        if visits[key]>3:reason='repeated working set';break
        BA=B[:,A];na=nu[A];ba=beta[A];ok=False
        for k in range(20):
            D=1/(2*s+mu);S=(BA.T*D)@BA
            lam=_linear(S,na*t+ba);v=D*(BA@lam)
            if np.min(lam)<-gate:break
            tn=norm(v)
            if tn<1e-30:break
            mun=(na@lam)/tn
            err=max(abs(tn-t)/(1+tn),abs(mun-mu)/(1+abs(mun)))
            mu,t=max(mun,0.),tn
            if err<inner:ok=True;break
        if return_trace:trace.append(U@v)
        if np.min(lam)<-gate:
            A.pop(int(np.argmin(lam)));continue
        if not ok or _residual(s,BA,na,ba,v,lam)>inner:
            v,lam,ok,steps=_newton(s,BA,na,ba,v,lam,inner);newton_steps+=steps
            if return_trace and steps:trace.append(U@v)
        if np.min(lam)<-gate:
            A.pop(int(np.argmin(lam)));continue
        slack=B.T@v-nu*norm(v)-beta
        inactive=[i for i in range(N) if i not in A]
        if inactive:
            j=min(inactive,key=lambda i:slack[i]/(1+abs(beta[i])))
            if slack[j]<-gate*(1+abs(beta[j])):
                A.append(j);t=norm(v);mu=max((na@lam)/max(t,1e-30),0.);continue
        w=U@v;full=np.zeros(N);full[A]=np.maximum(lam,0)*scale
        cert=quality(Q,L,nu,beta,w,full)
        if cert['residual']<=gate:
            result=(w,outer+1,np.array(A,int));info=dict(method='AS-SNC',newton_steps=newton_steps,lam=full,**cert)
            if return_trace:result+=(trace,)
            return result+(info,) if return_info else result
        reason='final KKT check';break
    # Rare recovery route; maintain explicit provenance.
    import cvxpy as cp
    x=cp.Variable(M);tt=cp.Variable();cs=[cp.norm(x)<=tt,L.T@x-nu*tt>=beta]
    problem=cp.Problem(cp.Minimize(cp.quad_form(x,cp.psd_wrap(Q/scale))),cs)
    for backend in ['CLARABEL','SCS']:
        try:
            opts=dict(tol_gap_abs=1e-11,tol_gap_rel=1e-11,tol_feas=1e-11,max_iter=500) if backend=='CLARABEL' else dict(eps=1e-10,max_iters=200000)
            problem.solve(solver=backend,**opts)
            if x.value is None:continue
            lam=np.maximum(np.asarray(cs[1].dual_value),0)*scale
            cert=quality(Q,L,nu,beta,x.value,lam)
            if cert['residual']>max(gate,1e-8):continue
            w=np.asarray(x.value);slack=L.T@w-nu*norm(w)-beta;A=np.flatnonzero(abs(slack)<1e-6*(1+abs(beta)))
            result=(w,max_outer,A);info=dict(method='fallback:'+backend,reason=reason,newton_steps=newton_steps,lam=lam,**cert)
            if return_trace:result+=(trace+[w.copy()],)
            return result+(info,) if return_info else result
        except cp.error.SolverError:continue
    raise RuntimeError('No checked solution: '+reason)
