from sys import stderr
from sys import stdout
import numpy as np

import time
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mCols
import matplotlib.cm as cm
import matplotlib.mlab as mlab

from sklearn import cluster 
from sklearn import metrics
from sklearn import mixture

#from sets import Set

import math
import random
from scipy.stats import norm
from scipy.stats import fisher_exact

import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hclust

#from sets import Set
from scipy.stats.mstats import mode

def overlap(s1, e1, s2, e2):
    """
    if one starts wheere the other ends, they don't overlap 
    """
    if s1==e2 or s2 == e1:
        return False
    elif (s1<=e2 and s2<=e1):
        return True
    return False

def eval_G(G, x):

    u, v = G
    s = np.sqrt(v)
    sq2pi = np.power(2*np.pi,0.5)

    y = (1/(s*sq2pi)) * np.exp( -1*((x-u)*(x-u))/(2*s*s) )
    return y

def get_intersection(G1, G2, ws, tol=0.01):
    """
    Get the integral of the shared area between two gaussians
    the intersection is computed exactly, but the integral is computed numerically
    """
    tol = min(ws)/1000
    #sort so G1.mu < G2.mu
    #ui < uj
    oGs = [G1, G2] 
    ows = ws
    Gs, ws = [], []
    args = np.argsort([G1[0],G2[0]])
    
    for i in args:
        Gs.append(oGs[i])
        ws.append(ows[i])
    ui, vi = Gs[0]
    uj, vj = Gs[1]
    si, sj = np.sqrt(vi), np.sqrt(vj)
    al, be = ws
    
    if si == sj:
        x=(ui+uj)/2.0
    else:
        sq2pi = np.power(2*np.pi,0.5)
        c = (2*si*si*sj*sj) * ( np.log( al/(si*sq2pi) ) - np.log( be/(sj*sq2pi) ) )
        c = c  + (si*si*uj*uj)-(sj*sj*ui*ui)
        b = -((2*uj*si*si)-(2*ui*sj*sj))
        a = (si*si)-(sj*sj)
        
        q=(b**2 - 4*a*c)
        if q<0: 
            x=None
        else:
            x1 = (-b + np.sqrt(q)) / (2*a)
            x2 = (-b - np.sqrt(q)) / (2*a)
            
            x=x1
            if (x1 < ui and x1 < uj) or (x1 > ui and x1 > uj):
                x=x2
    
    if x==None:
        return None, None, 1, 1 

    y = al*eval_G(G1, x) 
    
    k=5
    mn = ui - k*si
    mx = uj + k*sj

    o_gran_mx = (mx-x)/tol
    o_gran_mn = (x-mn)/tol
    
    while np.absolute((mx-x)/tol)>1e7 and k>2:
        mn = ui - k*si
        mx = uj + k*sj
        k-=.2
    
    if o_gran_mx!=(mx-x)/tol:
        print("\tgaussian intercept granularity reduction:", o_gran_mx, (mx-x)/tol, k)
        print("\tgaussian intercept granularity reduction:", o_gran_mn, (x-mn)/tol, k)

    if np.absolute((mx-x)/tol)>1e7:
        return None, None, 1, 1 


    xis = np.arange(x,mx, tol)
    xjs = np.arange(mn,x, tol)

    i_integral = np.sum(mlab.normpdf(xis, ui, si)*al)*tol
    j_integral = np.sum(mlab.normpdf(xjs, uj, sj)*be)*tol
    overlap = i_integral+j_integral

    return x, y, overlap/al, overlap/be

def assess_GT_overlaps(gmm):

    overlaps = []

    l = gmm.means.shape[0] 
    us = []
    ss = []
    ws = []
    for i in range(l):
        u = gmm.means[i,0]
        #s = gmm.covars[i][0][0]**.5
        s = gmm.covars[i]**.5
        w = gmm.weights[i]
        if w==0: continue
        us.append(u)
        ss.append(s)
        ws.append(w)
    
    sort_mu_args = np.argsort(np.array(us))
    all_os = []
    for k in range(len(sort_mu_args)-1):
        i, j = sort_mu_args[k], sort_mu_args[k+1] 
        u_1, u_2 = us[i], us[j]
        s1, s2 = ss[i], ss[j]
        G1, G2 = [u_1,s1*s1], [u_2,s2*s2]
        w1, w2 = ws[i], ws[j]
        t = w1+w2
        w1, w2 = w1/t, w2/t
        x, y, o1, o2 = get_intersection(G1, G2, [w1,w2], tol=0.01)
        s_dist=[np.absolute(u_1-u_2)/s1,np.absolute(u_1-u_2)/s2]
        all_os+=[o1,o2]
        overlaps.append({"us":tuple([u_1,u_2]),"os":tuple([o1, o2]),"ss":tuple([s1,s2]), "ws":tuple([w1,w2]), "sdist":tuple(s_dist)})
    
    u_o, med_o = np.mean(all_os), np.median(all_os)
    if len(overlaps) == 0:
        overlaps = None
    return u_o, med_o, overlaps
    
        
def output(g, contig, s, e, filt, include_indivs=None, plot_dir="./plotting/test", plot=False, v=False):

    print("%s %d %d"%(contig, s, e))
    stdout.flush()

    if include_indivs!=None and len(include_indivs) == 1:
        return
    
    X, idx_s, idx_e = g.get_gt_matrix(contig, s, e)
    gX = g.GMM_genotype(X, include_indivs = include_indivs)
    u_o, med_o, overlaps = assess_GT_overlaps(gX.gmm)
    if gX.fail_filter(filt):
        print("***********FAILED************")
    if gX.n_clusts ==1:  
        print("***********1_CLUST************")
    
    mus = np.mean(X,1)

    if g.is_segdup(contig, s, e) or np.mean(mus)>=3 or np.amax(mus) >2.5:
        Xs, s_idx_s, s_idx_e = g.get_sunk_gt_matrix(contig, s, e)
        gXs = g.GMM_genotype(Xs)
        if gXs.n_clusts == 1:
            print("***********1_SD_CLUST************")
            return
            
    if gX.n_clusts == 1 or gX.fail_filter(filt):
        return

    g.output(gX, contig, s, e, v=v)

    
     
    if plot:
        print("plotting %s %d %d"%(contig, s, e))
        Xs, s_idx_s, s_idx_e = g.get_sunk_gt_matrix(contig, s, e)
        gXs = g.GMM_genotype(Xs, include_indivs = include_indivs)
        g.plot(gX, gXs, contig, s, e, idx_s, idx_e, s_idx_s, s_idx_e, overlaps, fn="%s/%s_%d_%d.png"%(plot_dir, contig, s, e))

def assess_GT_overlaps(gmm):

    overlaps = []

    l = gmm.means.shape[0] 
    us = []
    ss = []
    ws = []
    for i in range(l):
        u = gmm.means[i,0]
        #s = gmm.covars[i][0][0]**.5
        s = gmm.covars[i]**.5
        w = gmm.weights[i]
        if w==0: continue
        us.append(u)
        ss.append(s)
        ws.append(w)
    
    sort_mu_args = np.argsort(np.array(us))
    all_os = []
    for k in range(len(sort_mu_args)-1):
        i, j = sort_mu_args[k], sort_mu_args[k+1] 
        u_1, u_2 = us[i], us[j]
        s1, s2 = ss[i], ss[j]
        G1, G2 = [u_1,s1*s1], [u_2,s2*s2]
        w1, w2 = ws[i], ws[j]
        t = w1+w2
        w1, w2 = w1/t, w2/t
        x, y, o1, o2 = get_intersection(G1, G2, [w1,w2], tol=0.01)
        s_dist=[np.absolute(u_1-u_2)/s1,np.absolute(u_1-u_2)/s2]
        all_os+=[o1,o2]
        overlaps.append({"us":tuple([u_1,u_2]),"os":tuple([o1, o2]),"ss":tuple([s1,s2]), "ws":tuple([w1,w2]), "sdist":tuple(s_dist)})
    
    u_o, med_o = np.mean(all_os), np.median(all_os)
    if len(overlaps) == 0:
        overlaps = None
    return u_o, med_o, overlaps

class simple_genotyper(object):
 
    def simple_GMM_genotype(self, X, max_cp=12):
        if type(X) is list:
            X = np.array(X)
        if len(X.shape) > 1:
            mus = np.mean(X,1)
        else:
            mus = np.array(X)
        if np.amax(mus)>max_cp or any(np.isnan(mus)):
            mus = np.around(mus)
            gts_by_indiv = mus.tolist()
        else:
            gX = self.GMM_genotype(X)
            gts_by_indiv, gts_to_label, labels_to_gt = gX.get_gts_list()

        return gts_by_indiv
        
    def GMM_genotype(self, X, include_indivs = None, FOUT = None, overload_indivs = None):
        """
        GMM genotyping
        merge_overlap_thresh, if -1, don't ever merge, however, 
        otherwise, if overlap > merge_overlap_thresh, then merge and recalculate 
        
        include indivs: only genotype a subset of indivs in the gt object

        overload indivs: this is if the gt object has nothing to do with the X you are
                         passing to the function
        """
        if len(X.shape) > 1:
            mus = np.mean(X,1)
        else:
            mus = np.array(X)

        mus = np.reshape(mus, (mus.shape[0],1))
        dist_mat = dist.pdist(mus)
        #print "h_clustering..."
        t = time.time()
        Z = hclust.linkage(mus, method='centroid', metric='euclidean')
        #print "done %fs"%(time.time()-t)
        params, bics, gmms, all_labels = [], [], [], []
        
        print("assessing genotypes") 
        t = time.time()

        prev_grps = np.array([])
        for k in np.arange(.2, 1.5,  0.001):
            grps = hclust.fcluster(Z, k, criterion='distance')
            if np.all(grps == prev_grps): continue
            
            init_mus, init_vars, init_weights = self.initialize(mus, grps) 
            
            if len(init_mus)>30 and len(bics)>0: continue
            
            gmm, labels, ic = self.fit_GMM(mus, init_mus, init_vars, init_weights)

            params.append(len(init_mus))
            bics.append(ic)
            gmms.append(gmm)
            all_labels.append(labels)
            prev_grps = grps 
            
        print("done %fs"%(time.time()-t))
        idx = np.argmin(bics)
        #print params
        #print np.where(np.array(params)==3)
        #idx = np.where(np.array(params)==3)[0][0]
        gmm = gmms[idx]
        labels = all_labels[idx]
        bic = bics[idx]
        

        ####NOW, finally merge calls that are too close 
        n_labels = np.unique(labels).shape[0] 
        if n_labels>1:
            gmm, labels = self.final_call_merge(gmm, bic, labels, mus) 

        if overload_indivs != None:
            return simple_GMM_gt(X, gmm, labels, Z, params, bics, overload_indivs)
        else:
            return simple_GMM_gt(X, gmm, labels, Z, params, bics, include_indivs)

    def final_call_merge(self, gmm, original_ic, labels,mus, max_overlap=0.5, min_dist=0.55):
        """
        take the final min_bic call and merge calls that are too close   
        """
        #max_overlap=8

        u_o, med_o, overlaps = assess_GT_overlaps(gmm)
        max_overlap_stat = sorted(overlaps, key = lambda x: max(x['os']))[-1]
        n_labels = np.unique(labels).shape[0] 
        
        ic = original_ic
        while (max(max_overlap_stat['os']) > max_overlap) and n_labels>1:
            u1, u2 = max_overlap_stat['us'] 
            l1, l2 = np.where(gmm.means==u1)[0][0], np.where(gmm.means==u2)[0][0]
            labels[labels==l2] = l1
            
            init_mus, init_vars, init_weights = self.initialize(mus, labels) 
            gmm, labels, ic = self.fit_GMM(mus, init_mus, init_vars, init_weights, n_iter=1000)

            n_labels = np.unique(labels).shape[0] 
            if n_labels>1:
                u_o, med_o, overlaps = assess_GT_overlaps(gmm)
                max_overlap_stat = sorted(overlaps, key = lambda x: max(x['os']))[-1]

        n_labels = np.unique(labels).shape[0] 

        if n_labels==1:
            return gmm, labels
        
        u_o, med_o, overlaps = assess_GT_overlaps(gmm)
        max_overlap_stat = sorted(overlaps, key = lambda x: np.absolute(x['us'][0]-x['us'][1]))[0]
        d = np.absolute(max_overlap_stat['us'][0]- max_overlap_stat['us'][1])
        #note, this is the min_sd of the max_overlap... 
        #maybe this is always the min_sd? but, perhaps not??
        sd = min(max_overlap_stat['sdist'])
        
        curr_ic = ic

        while (d < min_dist ) and n_labels>1:
            u1, u2 = max_overlap_stat['us'] 
            l1, l2 = np.where(gmm.means==u1)[0][0], np.where(gmm.means==u2)[0][0]
            labels[labels==l2] = l1
            
            init_mus, init_vars, init_weights = self.initialize(mus, labels) 
            gmm, labels, ic = self.fit_GMM(mus, init_mus, init_vars, init_weights, n_iter=1000)
            
            #if t_ic<curr_ic:
            #    curr_ic = t_ic
            #    gmm, labels = t_gmm, t_labels
            #else:
            #    break

            u_o, med_o, overlaps = assess_GT_overlaps(gmm)
            n_labels = np.unique(labels).shape[0] 

            if n_labels>1:
                max_overlap_stat = sorted(overlaps, key = lambda x: np.absolute(x['us'][0]-x['us'][1]))[0]
                d = np.absolute(max_overlap_stat['us'][0]- max_overlap_stat['us'][1])
                sd = min(max_overlap_stat['sdist'])
        
        return gmm, labels

    def initialize(self, X, grps):

        uniqs = np.unique(grps)
        init_mus =  []
        init_weights =  []
        init_vars =  []
        
        l = X.shape[0] 
        for grp in uniqs:
            init_mus.append(np.mean(X[grps==grp]))
            init_vars.append(np.var(X[grps==grp]))
            init_weights.append(float(np.sum(grps==grp))/l)
        
        return init_mus, init_vars, init_weights

    def fit_GMM(self, X, init_means, init_vars, init_weights, n_iter=1000):
        #min_covar=1e-5
        min_covar=1e-4
        n_components = len(init_means)
        #gmm = mixture.GMM(n_components, 'spherical')
        #gmm = mixture.GMM(n_components, 'diag')
        gmm = mixture.GMM(n_components, 'spherical', min_covar=min_covar,n_iter=n_iter, init_params='')
        gmm.means = np.reshape(np.array(init_means),(len(init_means),1))
        gmm.weights = np.array(init_weights)
        
        #covars = np.array([v[0][0] for v in gmm.covars])
        #covars = np.array([np.reshape(np.array([max(v, min_covar)]),(1,1)) for v in init_vars])
        covars = np.array([max(v, min_covar) for v in init_vars])
        gmm.covars = covars
        
        #gmm.fit(X, n_iter=n_iter, init_params='c')
        #gmm.fit(X, n_iter=n_iter, init_params='cmw')
        #gmm.fit(X, n_iter=n_iter, init_params='c')
        #gmm.fit(X, n_iter=n_iter, init_params='')
        gmm.fit(X)
        
        labels = gmm.predict(X)
        
        bic = -2*gmm.score(X).sum() + (3*n_components)*np.log(X.shape[0])
        aic = -2*gmm.score(X).sum() + 2*(3*n_components)
        
        #print np.unique(labels)
        #for i,l in enumerate(list(np.unique(labels))):
        #    print init_means[i], init_vars[i], init_weights[i], gmm.score(X)[labels==l].sum() 
        #print bic

        return gmm, labels, aic 
 
class simple_GMM_gt(object):

    def __init__(self, X, gmm, labels, Z, params, bics, indivs):
        self.X = X
        self.mus = np.mean(self.X,1) if len(self.X.shape) > 1 else np.array(self.X)
        self.gmm = gmm
        self.labels = labels
        self.Z = Z
        self.params = params
        self.bics = bics
        self.min_bic = min(self.bics)
        self.weights = self.gmm.weights
        
        #self.indivs = list(indivs)
        self.n_clusts = np.unique(self.labels).shape[0]
        
        self.n_wnds = 0 if len(self.X.shape) == 1 else self.X.shape[1]

        self.f_correct = None 
        
        self.shaped_mus = np.reshape(self.mus, (self.mus.shape[0],1))

        #Deprecated in sklearn 0.14, removed in 0.16
        #self.l_probs, self.posterior_probs = gmm.eval(self.shaped_mus)
        
        self.l_probs, self.posterior_probs = gmm.score_samples(self.shaped_mus)
        
        #unique labels are the unique labels and uniq mus are the 
        #mean of the self.mus for each label
        
        self.label_to_mu = {}
        self.mu_to_labels = {}
        self.label_to_std = {}
        
        self.all_uniq_labels = []
        self.all_uniq_mus = []

        for l in np.unique(self.labels):
            self.label_to_mu[l] = np.mean(self.mus[self.labels==l])
            self.label_to_std[l] = np.std(self.mus[self.labels==l])
            self.mu_to_labels[np.mean(self.mus[self.labels==l])] = l
            self.all_uniq_labels.append(l)
            self.all_uniq_mus.append(np.mean(self.mus[self.labels==l]))

        self.all_uniq_labels = np.array(self.all_uniq_labels)
        self.all_uniq_mus = np.array(self.all_uniq_mus)

   
    def simple_plot(self, fn_out):

        cps = self.mus
        plt.rc('grid',color='0.75',linestyle='l',linewidth='0.1')
        fig, ax_arr = plt.subplots(1,3)
        fig.set_figwidth(9)
        fig.set_figheight(4)
        axescolor  = '#f6f6f6'
        print(ax_arr) 
        print(self.bics)
        print(self.params)

        ax_arr[1].plot(self.params, self.bics)

        n, bins, patches = ax_arr[0].hist(cps,alpha=.9,ec='none',normed=1,color='#8DABFC',bins=len(cps)/10)
        #self.addGMM(gX.gmm, axarr[1,1], cps, gX.labels, overlaps)
        
        G_x=np.arange(0,max(cps)+1,.1)
        l = self.gmm.means.shape[0]
        
        for i in range(l):
            c = cm.hsv(float(i)/l,1)
            mu = self.gmm.means[i,0]
            #var = self.gmm.covars[i][0][0]
            var = self.gmm.covars[i]

            G_y = mlab.normpdf(G_x, mu, var**.5)*self.gmm.weights[i]
            ax_arr[0].plot(G_x,G_y,color=c)
            ax_arr[0].plot(mu,-.001,"^",ms=10,alpha=.7,color=c)
        
        if np.amax(cps)<2:
            ax_arr[0].set_xlim(0,2)
        ylims = ax_arr[0].get_ylim()
        if ylims[1] > 10:
            ax_arr[0].set_ylim(0,10)

        fig.sca(ax_arr[2]) 
        dendro = hclust.dendrogram(self.Z, orientation='right')
        ylims = ax_arr[2].get_ylim()
        ax_arr[2].set_ylim(ylims[0]-1, ylims[1]+1)

        fig.savefig(fn_out)


    def fail_filter(self, filt):
        
        if self.n_clusts ==1 or self.gmm.means.shape[0] == 1:  
            return True

        u_o, med_o, overlaps = assess_GT_overlaps(self.gmm)
        max_overlap_stat = sorted(overlaps, key = lambda x: max(x['os']))[-1]
        
        """
        filter by the fractional overlap between adjacent clusters
        """
        if max(max_overlap_stat['os'])>=filt.max_overlap:
            return True

        mu_mu_d, min_mu_d, max_mu_d = self.get_mean_min_max_inter_mu_dist()
        
        """ 
        filter by the maximum distance between clusters, 
        it must be greater than min
        """
        if (max_mu_d < filt.min_max_mu_d):
            return True

        """
        filter out regions with really high copy
        """
        if np.mean(self.mus)>filt.max_mu_cp:
            return True
        
        """
        if flagged, then test for x-linked association and remove  
        """
        if filt.filter_X_linked and filt.is_X_linked(self.indivs, self.labels):
            return True
        
        """
        if singleton, impose strict theshold
        """
        if  ( self.n_clusts == 2 ) and ( min(np.sum(self.labels==l) for l in np.unique(self.labels)) == 1 ):
            min_z = self.get_min_z_dist()
            if min_z < filt.singleton_min_sigma:
                return True
        
        """
        if a dup, force stricter genotype filtering
        gts_by_indiv, gts_to_label, labels_to_gt = gX.get_gts_by_indiv()
        """
        if (np.amax(self.mus)>2.5):
            min_z = self.get_min_z_dist()
            if min_z < filt.dup_min_sigma:
                return True
            
        return False
        
        #mean_mu_delta = self.get_mean_inter_mu_dist()
        
        #if self.f_correct == None: 
        #    self.f_correct = self.correct_order_proportion()
        #if (self.f_correct <= frac_dir_min):
        #    return True


    def output_filter_data(self, g, info_ob, contig, s, e, labels_to_gt, gts_by_indiv):

        if self.f_correct == None:
            self.f_correct = self.correct_order_proportion()
        
        mu_mu_d, min_mu_d, max_mu_d = self.get_mean_min_max_inter_mu_dist()
        n_clusts = self.n_clusts 
        
        min_AC = min(np.sum(self.labels==l) for l in np.unique(self.labels))

        n_wnds = self.n_wnds
        min_z = self.get_min_z_dist()
        bic_delta = self.get_bic_delta() 
        Lscore = self.get_Lscore()

        min_inter_label_dist = self.get_min_inter_label_dist()

        mean_responsibility = self.get_mean_responsibility()
        min_responsibility = self.get_min_responsibility()
        ll_responsibility = self.get_ll_responsibility()
        
        labels_sorted_by_AC = sorted([[np.sum(self.labels==l),l] for l in np.unique(self.labels)], key=lambda x: x[0] )
        min_AC_label = labels_sorted_by_AC[0][1]
        max_AC_label = labels_sorted_by_AC[-1][1]
        min_AC_mean_responsibility = self.get_mean_responsibility_by_label(min_AC_label)
        min_AC_l_prob = self.get_l_prob_by_label(min_AC_label)
        is_singleton = (min_AC==1 and n_clusts==2)
        
        is_dup_singleton = is_singleton and labels_to_gt[min_AC_label]>labels_to_gt[max_AC_label] 
        is_del_singleton = is_singleton and labels_to_gt[min_AC_label]<labels_to_gt[max_AC_label]
        
        singleton_id = "None"
        if is_singleton:
            indiv_by_gt = {v:k for k, v in gts_by_indiv.items()}
            gt = labels_to_gt[min_AC_label]
            indiv = indiv_by_gt[gt]
            singleton_id = indiv
            singleton_cp_z, singleton_logR_z = self.get_singleton_cp_z(g, contig, s, e, indiv)
        else:
            singleton_cp_z, singleton_logR_z = -1,-1
        
        all_gts = set(np.unique(np.array(list(labels_to_gt.values()))))
        
        is_biallelic_del = False
        is_biallelic_dup = False

        if (all_gts == set([0,1,2])) or (all_gts == set([1,2])) or (all_gts == set([0,2])):
            is_biallelic_del = True

        if (all_gts == set([1,2,3])) or (all_gts == set([2,3])) or (all_gts == set([2,3,4])):
            is_biallelic_dup = True
        
        Tstring, Fstring = "TRUE", "FALSE"
        entry = info_ob.init_entry()
        info_ob.update_entry(entry,"contig", contig)
        info_ob.update_entry(entry,"start", s)
        info_ob.update_entry(entry,"end", e)
        info_ob.update_entry(entry,"GC", g.GC_inf.get_GC(contig, s, e))
        info_ob.update_entry(entry,"mu_mu_d", mu_mu_d)
        info_ob.update_entry(entry,"max_mu_d", max_mu_d)
        info_ob.update_entry(entry,"min_mu_d", min_mu_d)
        info_ob.update_entry(entry,"f_correct_direction", self.f_correct)
        info_ob.update_entry(entry,"min_z", min_z)
        info_ob.update_entry(entry,"wnd_size", n_wnds)
        info_ob.update_entry(entry,"bic_delta", bic_delta)
        info_ob.update_entry(entry,"n_clusts", n_clusts)
        info_ob.update_entry(entry,"min_allele_count", min_AC)
        info_ob.update_entry(entry,"Lscore", Lscore)
        info_ob.update_entry(entry,"min_inter_label_dist",min_inter_label_dist)
        info_ob.update_entry(entry,"mean_responsibility", mean_responsibility)
        info_ob.update_entry(entry,"min_responsibility", min_responsibility)
        info_ob.update_entry(entry,"ll_responsibility", ll_responsibility)
        info_ob.update_entry(entry,"singleton_cp_z", singleton_cp_z)
        info_ob.update_entry(entry,"singleton_logR_z", singleton_logR_z)
        info_ob.update_entry(entry,"is_singleton",is_singleton and Tstring or Fstring)
        info_ob.update_entry(entry,"is_dup_singleton",is_dup_singleton and Tstring or Fstring)
        info_ob.update_entry(entry,"is_del_singleton",is_del_singleton and Tstring or Fstring)
        info_ob.update_entry(entry,"singleton_id",singleton_id)
        info_ob.update_entry(entry,"is_biallelic_dup",is_biallelic_dup and Tstring or Fstring)
        info_ob.update_entry(entry,"is_biallelic_del",is_biallelic_del and Tstring or Fstring)
        info_ob.update_entry(entry,"min_AC_mean_responsibility", min_AC_mean_responsibility)
        info_ob.update_entry(entry,"min_AC_lprob",min_AC_l_prob)
        info_ob.output_entry(entry)
        
    """
    functions for getting different filtering info 
    """

    def get_singleton_cp_z(self, g, contig, s, e, indiv):
        
        n = 10
        pad = 3 
        
        i = g.indivs.index(indiv)
        X, idx_s, idx_e = g.get_gt_matrix(contig, s, e)
        idx_d = (idx_e - idx_s)
        
        idx_s_l, idx_e_l = max(0, idx_s-n*idx_d-pad), idx_s-pad
        idx_s_r, idx_e_r = idx_e+pad, min(idx_e+n*idx_d+pad, g.wnd_ends.shape[0])
        
        Xl = g.get_gt_matrix_by_idx(contig, idx_s_l, idx_e_l)
        Xr = g.get_gt_matrix_by_idx(contig, idx_s_r, idx_e_r)

        cps = np.r_[Xl[i,:], Xr[i,:]]
        csum = np.cumsum(cps)
        conv= (np.r_[csum[idx_d-1], csum[idx_d:]-csum[:-idx_d]])/float(idx_d)

        v = np.var(conv)
        mu = np.mean(conv)
        z = np.absolute((np.mean(X[i,:])-mu)/(v**.5))
        
        ind_ids = np.r_[np.arange(i),np.arange(i+1,X.shape[0])]
        all_cps = np.c_[Xl[ind_ids,:], Xr[ind_ids,:]]
        logR = np.log(cps/all_cps)/np.log(2.0)
        csum_logR = np.cumsum(logR,1)
        conv_logR= (np.c_[csum_logR[:,idx_d-1], csum_logR[:,idx_d:]-csum_logR[:,:-idx_d]])/float(idx_d)

        v_R = np.var(conv_logR)
        mu_R = np.mean(conv_logR)
        t_L = np.mean(np.log(X[i,:]/X[ind_ids,:])/np.log(2.0))
        z_R = np.absolute((t_L-mu_R)/(v_R**.5))

        return z, z_R

    def get_min_inter_label_dist(self):
        ###
        args = np.argsort(self.mus)
        sorted_mus = self.mus[args]
        sorted_labels = self.labels[args]

        min_d = 9e9
        for i in range(sorted_mus.shape[0]-1):
            d = np.absolute(sorted_mus[i+1]-sorted_mus[i])
            if sorted_labels[i]!=sorted_labels[i+1] and d<min_d:
                min_d = d

        return min_d

    def get_mean_responsibility(self):
        return np.mean(self.posterior_probs[np.arange(self.labels.shape[0]),self.labels])
    
    def get_min_responsibility(self):
        return np.min(self.posterior_probs[np.arange(self.labels.shape[0]),self.labels])
    
    def get_ll_responsibility(self):
        return np.sum(np.log(self.posterior_probs[np.arange(self.labels.shape[0]),self.labels]))
    
    def get_l_prob_by_label(self, label):
        w_min_label = np.where(self.labels == label)
        return np.sum(self.l_probs[w_min_label])

    def get_mean_responsibility_by_label(self, label):
        w_min_label = np.where(self.labels == label)
        resps = self.posterior_probs[np.arange(self.labels.shape[0]),self.labels][w_min_label]
        return  np.mean(resps)
    
    def get_ll_probs(self):
        return np.sum(self.l_probs)  

    def get_Lscore(self):
        return self.get_ll_probs() 
        
    def correct_order_proportion(self):
        #wnd_proportion_dir
        sorted_mus = np.sort(self.all_uniq_mus)
        labels = np.array(self.labels)
        if len(sorted_mus)==1: return 0.0
        
        t = 0 
        p = 0
        
        for i in range(sorted_mus.shape[0]-1):
            mu_0 = sorted_mus[i]
            l_0 = self.mu_to_labels[mu_0]
            
            mu_1 = sorted_mus[i+1]
            l_1 = self.mu_to_labels[mu_1]

            assert mu_0<mu_1
            
            wl_0 = np.where(labels==l_0)
            wl_1 = np.where(labels==l_1)
            t+=self.X.shape[1]
            s=np.sum(np.amax(self.X[wl_0,:], 1)<np.amin(self.X[wl_1,:], 1))
            s2=np.sum(np.median(self.X[wl_0,:], 1)<np.median(self.X[wl_1,:], 1))
            p+=s
        
        return float(p)/t
    
    def get_min_z_dist(self):

        min_z_dist = 9e9

        sorted_mus = np.sort(self.all_uniq_mus)
        labels = np.array(self.labels)
        
        delta = np.diff(sorted_mus)
        std_lefts = []
        std_rights = []

        for i in range(sorted_mus.shape[0]-1):
            mu_left = sorted_mus[i]
            mu_right = sorted_mus[i+1]
            
            l_left = self.mu_to_labels[mu_left]
            l_right = self.mu_to_labels[mu_right]
            
            std_left = max(1e-9, self.label_to_std[l_left])
            std_right = max(1e-9, self.label_to_std[l_right])
            
            std_lefts.append(std_left)
            std_rights.append(std_right)
        
        std_lefts = np.array(std_lefts)
        std_rights = np.array(std_rights)

        min_l = np.amin(np.absolute(delta/std_lefts)) 
        min_r = np.amin(np.absolute(delta/std_rights)) 
        
        return min(min_l, min_r)

    def get_mean_min_max_inter_mu_dist(self):
        s_mus = np.sort(self.all_uniq_mus)
        ds = np.diff(s_mus)
        if ds.shape[0] == 0:
            return 0, 0, 0

        return np.mean(ds), np.amin(ds), np.amax(ds) 

    def get_gt_lls_by_indiv(self, labels_to_gt):
        

        gt_lls_by_indiv = {}

        for i, indiv in enumerate(self.indivs):  
            lls = {gt:max(np.log(self.posterior_probs[i,label])/np.log(10),-1000) for label, gt in labels_to_gt.items() }
            gt_lls_by_indiv[indiv] = lls 

        return gt_lls_by_indiv


    def get_bic_delta(self):
        
        if self.n_clusts+1 in self.params:
            idx_bic_r = self.params.index(self.n_clusts+1) 
            bic_r = self.bics[idx_bic_r]
        else:
            bic_r = self.min_bic
    
        if self.n_clusts-1 in self.params:
            idx_bic_l = self.params.index(self.n_clusts-1) 
            bic_l = self.bics[idx_bic_l]
        else:
            bic_l = self.min_bic
        
        delta = bic_r-self.min_bic+bic_l-self.min_bic
        return delta
    
    def get_gts_list(self, correct_for_odd_major = True):
        """Get list of genotypes instead of as dict with indiv name"""

        cp_2_thresh=1.0
        mode_label = int(mode(self.labels)[0][0])
        
        indiv_labels = np.array(self.labels)
        
        mu_args = np.argsort(self.all_uniq_mus) 
          
        ordered_labels = self.all_uniq_labels[mu_args]
        ordered_mus = self.all_uniq_mus[mu_args] 
        d_from_2 = np.absolute(ordered_mus-2.0)
        
        labels_to_gt = {}
        """
        if there is something that looks like a 2, use it to callibrate others
        assign 2 to the closest 1, then assign the rest as +-1 in the order
        make sure that you aren't assigning -1 genotypes
        then finally, consolidate w/ the mus 
        """
        if np.amin(d_from_2)<cp_2_thresh:
            idx = np.argmin(d_from_2)
            idx_cp = 2
        else:
            idx = 0
            idx_cp = round(ordered_mus[0])

        for i,l in enumerate(ordered_labels): 
            labels_to_gt[l] = idx_cp-(idx-i)
        
        ## ensure no -1s
        while min(labels_to_gt.values())<0:
            print("<0's detected...")
            new_labels_to_gt = {}
            for l, gt in labels_to_gt.items():
                new_labels_to_gt[l] = gt+1
            labels_to_gt = new_labels_to_gt
       
        ##correct for odd major alleles out of HWE 
        if correct_for_odd_major and (labels_to_gt[mode_label] %2 == 1) and np.sum(indiv_labels==mode_label) >= .5*(indiv_labels.shape[0]):
            d=0
            if self.label_to_mu[mode_label]-labels_to_gt[mode_label]>0 or min(labels_to_gt.values())==0:
                d=1
            else:
                d=-1
            new_labels_to_gt = {}
            for l, gt in labels_to_gt.items():
                new_labels_to_gt[l] = gt+d
            labels_to_gt = new_labels_to_gt
        
        gts = [int(labels_to_gt[self.labels[i]]) for i in range(self.X.shape[0])]
        
        new_labels_to_gt = {k:int(v) for k,v in labels_to_gt.items()}
        labels_to_gt = new_labels_to_gt

        gt_to_labels = {v:k for k,v in labels_to_gt.items()} 

        return gts, gt_to_labels, labels_to_gt

    def get_gts_by_indiv(self, correct_for_odd_major = True):
        cp_2_thresh=1.0
        mode_label = int(mode(self.labels)[0][0])
        
        indiv_labels = np.array(self.labels)
        
        mu_args = np.argsort(self.all_uniq_mus) 
          
        ordered_labels = self.all_uniq_labels[mu_args]
        ordered_mus = self.all_uniq_mus[mu_args] 
        d_from_2 = np.absolute(ordered_mus-2.0)
        
        labels_to_gt = {}
        """
        if there is something that looks like a 2, use it to callibrate others
        assign 2 to the closest 1, then assign the rest as +-1 in the order
        make sure that you aren't assigning -1 genotypes
        then finally, consolidate w/ the mus 
        """
        if np.amin(d_from_2)<cp_2_thresh:
            idx = np.argmin(d_from_2)
            idx_cp = 2
        else:
            idx = 0
            idx_cp = round(ordered_mus[0])

        for i,l in enumerate(ordered_labels): 
            labels_to_gt[l] = idx_cp-(idx-i)
        
        ## ensure no -1s
        while min(labels_to_gt.values())<0:
            print("<0's detected...")
            new_labels_to_gt = {}
            for l, gt in labels_to_gt.items():
                new_labels_to_gt[l] = gt+1
            labels_to_gt = new_labels_to_gt
       
        ##correct for odd major alleles out of HWE 
        if correct_for_odd_major and (labels_to_gt[mode_label] %2 == 1) and np.sum(indiv_labels==mode_label) >= .5*(indiv_labels.shape[0]):
            d=0
            if self.label_to_mu[mode_label]-labels_to_gt[mode_label]>0 or min(labels_to_gt.values())==0:
                d=1
            else:
                d=-1
            new_labels_to_gt = {}
            for l, gt in labels_to_gt.items():
                new_labels_to_gt[l] = gt+d
            labels_to_gt = new_labels_to_gt
        
        gts_by_indiv = {}
        for i, indiv in enumerate(self.indivs):  
            gts_by_indiv[indiv] = int(labels_to_gt[self.labels[i]]) 
        
        new_labels_to_gt = {k:int(v) for k,v in labels_to_gt.items()}
        labels_to_gt = new_labels_to_gt

        gt_to_labels = {v:k for k,v in labels_to_gt.items()} 

        return gts_by_indiv, gt_to_labels, labels_to_gt
            

    def get_cp(self, indiv, g):
        
        idx = g.indivs.index(indiv_id)
        

    def is_var(self, indiv_id, g, force_not_mode = False):

        idx = g.indivs.index(indiv_id)
        
        if (not force_not_mode) and len(np.unique(self.labels))>2:
            return True
        else:
            idx = g.indivs.index(indiv_id)
            m = mode(self.labels)[0]
            if self.labels[idx] != m:
                return True
        
        return False
