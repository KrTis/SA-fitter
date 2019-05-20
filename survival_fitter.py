import numpy                                 as np
from   uncertainties                         import ufloat
from   lifelines                             import KaplanMeierFitter
from   lifelines                             import WeibullFitter
from   scipy.special                         import gamma              as Γ
from   scipy.optimize                        import curve_fit
from   mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate                         import odeint
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import scipy.odr                             as odr
import emcee
from scipy.special import erf
from corner import corner
σ_interval      = [50, 15.87, 84.13] 

def weibull(x, λ, k):
    return  np.exp(-(x/λ)**k)
def weibullPDF(x, λ,k):
    return (k/λ)*(x/λ)**(k-1)*np.exp(-(x/λ)**k)*np.max(x,0)
def get_Λ(S):
    return - np.log(S)
def lnΛ_weibull(x, λ, k):
    return k*x-λ
def get_WB(bf):
    λ = bf[1]
    k = bf[0]
    μ = λ * Γ(1+1/k)
    m = λ * np.log(2)**(1./k)
    σ = λ*(Γ(1+2/k)-Γ(1+1/k)**2)**0.5
    return μ, σ, m
class modeling_helpers:
    @staticmethod
    def apply(f, x, flat=True):
        """
        Description
        -----------
        Applies the fitting function
        
        Parameters
        -----------
        f:      function
                function to be applied on the set x
        x:      array
                array of x-values
        flat:   bool
                if true, flattens the array after f
        
        Returns
        -----------
        array: numpy
               f(x)
        
        """
        if flat:
            return np.array([f(xx) for xx in x]).flatten()
        else:
            return np.array([f(xx) for xx in x])

    @staticmethod
    def σ(Σ):
        """
        Description
        -----------
        Calculates σ from the covariance matrix
        
        Parameters
        -----------
        Σ:      array
                covariance matrix
        
        Returns
        -----------
        array: numpy
               σ
        """
        return np.sqrt(np.diag(Σ))

    @staticmethod
    def bootstrap(size, N):
        """
        Description
        -----------
        Does bootstrap
        
        Parameters
        -----------
        size:   int
                size of the array to be indexed
        N:      int
                number of bootstraps
        
        Returns
        -----------
        array: numpy
               array of indices
        """
        c = []
        for i in range(N):
                c.append(np.random.randint(0, size, size=size))
        c = np.array(c)
        return c
class modeling:
    def wrap_θ(self,θ):
            if self.fix is None:
                return θ
            else:
                 return self.fix(θ)
    def check_limits(self,θ):
        truths = []
        if self.low is not None:
                truths.append( np.logical_or.reduce(θ<self.low))
                    
        if self.high is not None:
                truths.append(np.logical_or.reduce(θ>self.high))
        if len(truths)>0:
            return np.logical_or.reduce(truths)
    class Weibull_Λ:
        def __init__(self, θ, x, *args):
            self.θ = θ
            self.x = x
            self.N = np.size(θ)
            self.u = np.ones(np.size(x))

        def fit(self):
            self.f = self.θ[0]*(self.x-np.log(self.θ[1]))
            return self.f

        def jacobian(self):
            self.jac = np.array([(self.x-np.log(self.θ[1])),
                                 -self.θ[0]/self.θ[1]*self.u])
            return self.jac
class models(modeling_helpers):
    '''
    Description
    ===========
    Class used in model fitting
    '''
    ν_norm=3
    def __init__(self,
                 model=None,
                 init=None,
                 x=None,
                 y=None,
                 xerr=None,
                 yerr=None,
                 θ=None,
                 cov=None,
                 σ=None,
                 r=None,
                 δ=None,
                 add=(None,)):
        '''
        Description
        -----------
        Main fitting class
        
        Parameters
        -----------

        model:  string
                name of the model to be used
        init:   list
                initial parameters for the model
        x:      array
                x values
        y:      array
                y values
        xerr:   array
                x values' error (optional)
        yerr:   array
                y values' error (optional)
                
        θ:      array
                model parameters (optional)
        cov:    N^2 array
                covariance matrix
        σ:      array
                standard deviation
        r:      array
                full output of the ODR fit
        δ:      array
                survival analysis indices

        ────────────────────────────────────────────────────────────────────────
        '''

        self.model = model
        self.init  = init
        self.x     = x
        self.y     = y
        self.xerr  = xerr
        self.yerr  = yerr
        self.θ     = θ
        self.Σ     = cov
        self.σ     = σ
        self.r     = r
        self.δ     = δ
        self.add   = add

    def run(self, x):
        '''
        Description
        -----------
        Main fitting class Evaluation of the model
        
        Parameters
        -----------
        x:      array
                evaluets the model at values x
        
        Returns
        -----------
        array: numpy
               fitting model evaluated
        '''
        return modeling.__dict__[self.model](self.θ, x, *self.add).fit()
        # return eval('self.'+self.model+'_fit(self.θ,x, *self.add)')
    @staticmethod
    def domod(model, θ, x, *args, **kwargs):
        return modeling.__dict__[model](θ, x, *args, **kwargs).fit()
        # return eval('models.'+model+'_fit(θ, x, *args)')

    def fit(self):
        '''
        Description
        -----------
        Standard fit with or without errors 
        
        Parameters
        -----------
        
        Returns
        -----------
        θ:      array
                parameters
        σ:      array
                standard deviations
        Σ:      array
                covariance matrix
        '''


        # Fitting
        usemod = odr.Model(
                 lambda θ, x: modeling.__dict__[self.model](θ, x, *self.add).fit(),
                 lambda θ, x: modeling.__dict__[self.model](θ, x, *self.add).jacobian())
        #usemod = eval('odr.Model(self.'+self.model+'_fit, extra_args=self.add)')
        dats   = odr.RealData(self.x,
                              self.y,
                              sx=self.xerr,
                              sy=self.yerr)
        r      = odr.ODR(dats,
                         usemod,
                         self.init).run()

        # Outputting
        self.θ = r.beta
        self.σ = r.sd_beta
        self.Σ = r.cov_beta
        self.r = r
        return self.θ, self.σ, self.Σ

    def jacobian(self, x):
        '''
        Description
        -----------
        Gives the jacobian of the model
        
        Parameters
        -----------
        x:      array
                evalues the model at values x
        
        Returns
        -----------
        array: numpy
               fitting model jacobian evaluated
        '''
        return modeling.__dict__[self.model](self.θ, x, *self.add).jacobian()

    def error(self, x):
        '''
        Description
        -----------
        Gives the 1 σ error of the model
        
        Parameters
        -----------
        x:      array
                evalues the model at values x
        
        Returns
        -----------
        array: numpy
               fitting model 1σ error evaluated
        '''
        J = self.jacobian(x).flatten()
        return np.sqrt(np.einsum('i,ij,j', J, self.Σ, J) )

    def kσ(self, x, k):
        "  "
        '''
        Description
        -----------
        Gives the k σ confidence interval
        
        Parameters
        -----------
        x:      array
                evalues the model at values x
        k:      float
                k σ
        Returns
        -----------
        array: numpy
               mean + k σ
        '''
        return self.apply(lambda y:self.run(y)+k*self.error(y), x)
    
    def kinterval(self, x, k):
        return self.kσ(x, -k), self.kσ(x, k)
class S_fun:
    def __init__(self, x, M, T, S):
        self.M          = M
        self.x          = x
        self.S          = S
        self.T          = T
        self.Λ          = - np.log(S)
        self.lnΛ        = - np.log(self.Λ)
        self.CDF        = 1.-S
        self.Sfun       = None
        self.mean       = None
        self.mean_σ     = None
        self.median     = None
        self.median_σ   = None
        self.CI         = None
        self.σ          = None
        self.f_lnΛ      = None
        self.ax_lnΛ     = None
        
        
class Survival_data(S_fun):

    def __init__(self, X, N=50, fromdata=True, confidence_α=0.68, plot=False):
        
        self.plot = plot
        for prop in range(len(X)):
            if  not isinstance(X[prop], np.ndarray):
                X[prop] = np.array(X[prop])
        if np.shape(X)[0] == 3:
            self.x, self.δ, self.T = X
            
        elif np.shape(X)[0] == 2:
            self.x, self.δ = X
            if fromdata:
                self.T = X[0]
                self.N = np.size(self.T)
            else:
                self.T = self.timeline(N)
        else:
            dimensionError('input', '2 or 3')
        arg    = np.argsort(self.x)
        self.x = self.x[arg]
        self.δ = self.δ[arg]
        if fromdata or np.shape(X)[0] == 3:
            self.T = self.T[arg]
    
        self.M            = np.max(self.x)+1
        self.N            = N
        self.confidence_α = confidence_α
        
    def timeline(self, N=None):
        if N is None:
            N = self.N
        return np.linspace(np.min(self.x), np.max(self.x), N)
    
    def retimeline(self, N):
        self.T = self.timeline(N)
    
    def mirror(ob, stats=False):
         ob.x  = ob.M-ob.x
         ob.T  = ob.M-ob.T
         
         if stats:
             names = ['KM', 'WB']
             stats = ['mean', 'median']
             for name  in names:
                 if name in ob.__dict__:
                     for stat in stats:
                         ob.__dict__[name].__dict__[stat] =\
                             ob.M - ob.__dict__[name].__dict__[stat]
                         

    @staticmethod
    def percentile(Sfun, T, q):
        S = Sfun(T)
        if q >= np.min(S) and q <= np.max(S):
            inf   = 1-S >= q
            sup   = 1-S <= q
            if np.sum(inf)>0 and np.sum(sup)>0:
                x_inf = np.max(T[inf])
            
                x_sup = np.min(T[sup])
                return 0.5*(x_inf+x_sup)
            else:
                return 0
        else:
            return 0
    def KM_estimate(self):
        kmf = KaplanMeierFitter()
        T   = self.T
        kmf.fit(self.x,
                self.δ.astype(np.bool),
                alpha=self.confidence_α,
                timeline=T)
        Survival         = np.array(kmf.predict(T))

        self.KM          = S_fun(self.x, self.M, T, Survival)
        self.KM.kmf      = kmf
        self.KM.Sfun     = self.KM.kmf.predict
        self.KM.mean     = np.sum([
                                      (T[nn+1]-T[nn])*Survival[nn]
                                       for nn in range(len(Survival)-1)
                                     ]).astype(float)+T[0]
        
        self.KM.σ        = np.array(
                                self.KM.kmf.survival_function_.std()
                                )[0].astype(float)
        self.KM.mean_σ   = self.KM.σ
        self.KM.CI       = np.array(self.KM.kmf.confidence_interval_)
        percents         = np.array([self.percentile(self.KM.Sfun, T, q/100.)
                                     for q in σ_interval])
        self.KM.median   = self.KM.kmf.median_
        self.KM.median_σ = 0.5*np.diff(percents[1:])[0]
        self.current     = 'KM'
        
    def WB_estimate(self):
        T    = self.T
        lnΛ  = np.log(get_Λ(self.KM.S))
        cutΛ = ~np.isinf(lnΛ)
        lnΛ  = lnΛ[cutΛ]
        T    = T[cutΛ]
        lnT  = np.log(T)

        model = models('Weibull_Λ', [1,1], x=lnT, y=lnΛ)
        model.fit()
        mean, std, median = get_WB(model.r.beta)

        self.WB          = S_fun(self.x, self.M, T, np.exp(-np.exp(lnΛ)))
        self.WB.mean     = mean
        self.WB.median   = median
        self.WB.mean_σ   = std
        self.WB.median_σ = std
        self.WB.residual = lnΛ-modeling.Weibull_Λ(model.r.beta, lnT).fit()
        self.WB.lnT      = lnT
        if self.plot:
            self.WB.f_lnΛ, self.WB.ax_lnΛ = plt.subplots()
            
            self.WB.ax_lnΛ.plot(lnT, lnΛ, '.', color='C1')
            self.WB.ax_lnΛ.plot(lnT, 
                                modeling.Weibull_Λ(model.r.beta, lnT).fit(),
                                color='C0')
            self.WB.ax_lnΛ.set_ylabel('$\ln \Lambda$')
            self.WB.ax_lnΛ.set_xlabel('$\ln T$')
            #implotting.minor_ticks(self.WB.ax_lnΛ)
            insty = inset_axes(self.WB.ax_lnΛ,
                               width="25%",
                               height="25%",
                               loc=4,
                               borderpad=0.5,
                               )
            insty.plot(lnT, self.WB.residual, '.', color='C1')
            insty.axhline(y=0, color='C0')
            insty.axhline(y=np.median(self.WB.residual), ls='dashed', color='k')
            insty.set_xlabel('$\ln T$', fontsize=8)
            insty.set_ylabel('Residuals', fontsize=8)
            insty.tick_params(axis='both', which='major', labelsize=5)
            insty.xaxis.tick_top()
            insty.xaxis.set_label_position('top')
def give_errors(x, σ, latex=False):
    if np.size(σ) == 1:
        out = "{:0.2uL}".format(ufloat(x, σ))
    else:
        out = "%.2f_{-%.2f}^{+%.2f}" % (x, σ[0], σ[1])
    if not latex:
        return out
    if latex:
        return "$"+out+"$"
def do_SA(x, d, plot, N,KM=False):
        D = Survival_data([x,d],
                              fromdata = True,
                              plot = plot)
        D.mirror()
        D.KM_estimate()
        D.WB_estimate()
        D.mirror(stats=True)
        if plot:
            if KM:
                D.WB.ax_lnΛ.set_title(N+
                                      'median '+
                                      give_errors(
                                              D.KM.median,
                                              D.KM.median_σ,
                                              latex=True)+'\n mean '+
                                      give_errors(
                                              D.KM.mean,
                                              D.KM.mean_σ,
                                              latex=True))
            else:
                D.WB.ax_lnΛ.set_title(N+
                                      'median '+
                                      give_errors(
                                              D.WB.median,
                                              D.WB.median_σ,
                                              latex=True)+'\n mean '+
                                      give_errors(
                                              D.WB.mean,
                                              D.WB.mean_σ,
                                              latex=True))
        return D

class SED_fitting:
    def __init__(self,ν, F, Δ, model, constraints,σ=None):
        self.ν = ν
        self.Δ = Δ
        self.F = F
        self.m = model
        self.C = constraints
        self.σ = σ
    def constrain_σ(self,θ):
        return do_SA(self.F-self.m(self.ν,θ),self.Δ,False,'').WB.mean_σ
    @staticmethod
    def χ2(ν,F, Δ, σ,model,θ):
        if σ is None:
            σ     = do_SA(F-model(ν,θ),Δ,False,'').WB.mean_σ
        Z     = ((F-model(ν,θ))/σ).astype(np.float64)
        χ_detected    = np.sum(Z[Δ==1]**2,axis=0)
        χ_nondetected = -2*np.sum(np.log(np.sqrt(np.pi/2.)*σ)+np.log1p(erf(Z[Δ==0]/np.sqrt(2.))),axis=0)
        return χ_detected+χ_nondetected
    def lnL(self,θ):
        if np.logical_and.reduce(θ>self.C[:,0]) and np.logical_and.reduce(θ<self.C[:,1]):
            k = -.5*self.χ2(self.ν,self.F, self.Δ,self.σ,self.m,θ)
            if not np.isnan(k):
                return k
        return -np.inf
        
    def do_mcmc(self, ndim,  nwalkers,Nsteps,priors,crop=0):
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnL)
        self.sampler.run_mcmc(priors, Nsteps)
        self.D = self.sampler.chain[:,crop:,:].reshape((-1, ndim))
        self.D = self.D[np.logical_and.reduce(np.logical_and(self.D>self.C[:,0], self.D<self.C[:,1]).T)]