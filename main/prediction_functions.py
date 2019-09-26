import numpy as np
import h5py
import os
import sys
import time
from scipy.signal import hann
from obspy.signal import trigger
import theano.ifelse
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt
from scipy.stats import kurtosis,skew,linregress

def read_hdf5(path,fb=0):
    f=h5py.File(path,'a')
    dataset=f['/TRACE_DATA/DEFAULT']
    channels=f['/TRACE_DATA/DEFAULT/CHANNEL/']
    offset=f['/TRACE_DATA/DEFAULT/OFFSET/'][:]
    cdp=f['/TRACE_DATA/DEFAULT/CDP/'][:]
    ftrace=f['/TRACE_DATA/DEFAULT/FTRACE/']
    data=f['/TRACE_DATA/DEFAULT/data_array/']
    gapsize=f['/TRACE_DATA/DEFAULT/GAP_SIZE/']
    shotids=dataset['SHOTID'][:]
    user=dataset['USER']
    sourcenum=dataset['SOURCENUM']
    recordnum=dataset['RECORDNUM']
    src_x=dataset['SOURCE_X']
    src_y=dataset['SOURCE_Y']
    cdp_x=dataset['CDP_X']
    cdp_y=dataset['CDP_Y']
    rec_x=dataset['REC_X']
    rec_y=dataset['REC_Y']
    if fb==1:
        pred_prefb=f['/pred_fb1']
        pred_postfb=f['/pred_fb2']
        pred_1samp=f['/pred_fb3']
        pred_avg=f['/pred_avg']
        dataset={'data':data,'shotids':shotids,'offset':offset,'cdp':cdp,'gapsize':gapsize,'user':user,'src_x':src_x,'src_y':src_y,'cdp_x':cdp_x,'cdp_y':cdp_y,'snum':sourcenum,'rnum':recordnum,'ftrc':ftrace,'rec_x':rec_x,'rec_y':rec_y,'chan':channels,'pred_prefb':pred_prefb,'pred_postfb':pred_postfb,'pred_1samp':pred_1samp,'pred_avg':pred_avg}
    else:
        dataset={'data':data,'shotids':shotids,'offset':offset,'cdp':cdp,'gapsize':gapsize,'user':user,'src_x':src_x,'src_y':src_y,'cdp_x':cdp_x,'cdp_y':cdp_y,'snum':sourcenum,'rnum':recordnum}
    print('Data loaded')
    return dataset

def add_hdrs(path):
    file=h5py.File(path,'a')
    data=file['/TRACE_DATA/DEFAULT/data_array/']
    try:
        file.create_dataset('/pred_fb1',(len(data),len(data[0])))
        file.create_dataset('/pred_fb2',(len(data),len(data[0])))
        file.create_dataset('/pred_fb3',(len(data),len(data[0])))
        file.create_dataset('/pred_avg',(len(data),len(data[0])))
        file['/pred_fb1'][:]=0
        file['/pred_fb2'][:]=0
        file['/pred_fb3'][:]=0
        file['/pred_avg'][:]=0
        file.close()
        print('Data arrays pred_fb1, pred_fb2, pred_fb3, pred_avg added to the data structure')
    except:
        file['/pred_fb1'][:]=0
        file['/pred_fb2'][:]=0
        file['/pred_fb3'][:]=0
        file['/pred_avg'][:]=0
        file.close()
        print('Data arrays pred_fb1, pred_fb2, pred_fb2, pred_avg already exist')
        
def plot(dataset,trcids,fsamp,lsamp):
    ntrcs=len(trcids)
    fig,axes=plt.subplots(ncols=4,nrows=ntrcs,sharey=True,sharex=True,figsize=(25,4*ntrcs))
    for i,j in enumerate(trcids):
        fb=dataset['gapsize'][j]/2
        axes[i,0].set_title('Single\nsample model',fontsize=15)
        axes[i,0].plot(dataset['pred_1samp'][j][fsamp:lsamp],c='red',linewidth=1,label='Probability\ndistribution')
        axes[i,0].plot(dataset['data'][j][fsamp:500]/np.amax(np.abs(dataset['data'][j][fsamp:lsamp])),c='black',linewidth=1,label='Seismic trace')
        axes[i,0].plot([fb,fb],[-1,0],'--',c='blue',label='Reference first-break sample')
        axes[i,0].set_ylim((-1,1))
        axes[i,1].set_ylim((-1,1))
        axes[i,2].set_ylim((-1,1))
        axes[i,3].set_ylim((-1,1))
        axes[i,0].set_xlim((fsamp,lsamp))
        axes[i,1].set_xlim((fsamp,lsamp))
        axes[i,2].set_xlim((fsamp,lsamp))
        axes[i,3].set_xlim((fsamp,lsamp))
        axes[i,0].set_ylabel('Probability',fontsize=15)
        axes[i,0].set_xlabel('Samples',fontsize=15)
        axes[i,0].legend()
        axes[i,2].set_title('50 sample\npost-FB model',fontsize=15)
        axes[i,2].plot(dataset['pred_postfb'][j][fsamp:lsamp],c='red',linewidth=1)
        axes[i,2].plot(dataset['data'][j][fsamp:lsamp]/np.amax(np.abs(dataset['data'][j][fsamp:lsamp])),c='black',linewidth=1)
        axes[i,2].plot([fb,fb],[-1,0],'--',c='blue',label='Reference first-break sample')
        axes[i,1].set_title('50 sample\npre-FB model',fontsize=15)
        axes[i,1].plot(dataset['pred_prefb'][j][fsamp:lsamp],c='red',linewidth=1)
        axes[i,1].plot(dataset['data'][j][fsamp:lsamp]/np.amax(np.abs(dataset['data'][j][fsamp:lsamp])),c='black',linewidth=1)
        axes[i,1].plot([fb,fb],[-1,0],'--',c='blue',label='Reference first-break sample')
        axes[i,3].set_title('Models average',fontsize=15)
        axes[i,3].plot(dataset['pred_avg'][j][fsamp:lsamp],c='red',linewidth=1)
        axes[i,3].plot(dataset['data'][j][fsamp:lsamp]/np.amax(np.abs(dataset['data'][j][fsamp:lsamp])),c='black',linewidth=1)
        axes[i,3].plot([fb,fb],[-1,0],'--',c='blue',label='Reference first-break sample')
    plt.tight_layout()

class fbpicker():
    """
    A class used to perfom automatic first-break picking on a given dataset.

    ...

    Attributes
    ----------
    dataset : dict
        the seismics stored and organized in a Hierarchical Data Format (HDF)
    models : ndarray
        the first-break models stacked together in a single container
        (1 sample model, pre-FB model, post-FB model)
    min_offset : int
        the smallest offset used for autopicking
    max_offset : int
        the largest offset used for autopicking
    scalar_offset : float
        the scalar values used for scaling the offsets
    first_sample : int
        the time sample where autopicker starts
    trc_slen : int
        the number of samples being analyzed
    dt : int
        the sampling interval
    features_prefb : list
        the features list obtained for the pre-FB model through a selection process
    features_postfb : list
        the features list obtained for the post-FB model through a selection process

    Methods
    -------
    entropy(trc,e_step)
        Calculates trace entropy
    fdm(trc,fd_step,lags,noise_scalar)
        Transforms trace into fractal dimension
    amp_spectrum(data,dt,single=1)
        Calculates amplitude spectrum
    norm(data)
        Performs simple normalization
    fq_win_sum(data,hwin,dt)
        Calculates summation of amplitude spectra of a data slice
    kurtosis_skewness(data,hwin)
        Calculates higher order statistics: kurtosis & skewness
    trc_fgen_prefb(trc,dt,nspad=200,hwin=150,vlen=51)
        Constructs feature matrices for the pre-FB model
    trc_fgen_postfb(trc,dt,hwin=150,vlen=51)
        Constructs feature matrices for the post-FB model
    trc_prep(m,ds,s,mode=2)
        Resizes a given matrix by down-sampling (Mode 1 - pre-FB, Mode 2 - post-FB)
    predict(self)
        Performs first-break prediction
    
    """
    
    def __init__(self,dataset,models,min_offset,max_offset,scalar_offset,first_sample,trc_slen,dt,features_prefb,features_postfb):
        """
        Parameters
        ----------
        dataset : dict
            the seismics stored and organized in a Hierarchical Data Format (HDF)
        models : ndarray
            the first-break models stacked together in a single container
            (1 sample model, pre-FB model, post-FB model)
        min_offset : int
            the smallest offset used for autopicking
        max_offset : int
            the largest offset used for autopicking
        scalar_offset : float
            the scalar values used for scaling the offsets
        first_sample : int
            the time sample where autopicking starts for a seismic trace
        trc_slen : int
            the number of samples being analyzed
        dt : int
            the sampling interval
        features_prefb : list
            the features list obtained for the pre-FB model through a selection process
        features_postfb : list
            the features list obtained for the post-FB model through a selection process
        """

        self.dataset = dataset
        self.models = models
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.scalar_offset = scalar_offset
        self.first_sample = first_sample
        self.trc_slen = trc_slen
        self.dt = dt
        self.features_prefb = features_prefb
        self.features_postfb = features_postfb
        
    def entropy(self,trc,e_step):
        """ Calculates trace entropy """
        t=len(trc)-1
        trc_out=trc.copy()
        trc_out[0:e_step]=0
        while t>e_step-1:
            trc_win=trc[t-e_step:t+1]
            t_win=e_step-1
            res=0
            while t_win>0:
                res+=np.abs(trc_win[t_win-1]-trc_win[t_win])
                t_win-=1
            res=np.log10(1/e_step*res)
            trc_out[t]=res
            t-=1
        return trc_out
        
    def fdm(self,trc,fd_step,lags,noise_scalar):
        """ Transforms trace into fractal dimension """
        ress=[]
        trc_out=trc/np.amax(np.abs(trc))
        noise=np.random.normal(0,1,len(trc_out))*(np.std(trc_out)/noise_scalar)
        trc_out=trc_out+noise
        for i,lag in enumerate(lags):
            trc_cp=trc_out.copy()
            t=len(trc)-1
            trc_cp[0:fd_step]=0
            while t>fd_step-1:
                trc_win=trc_out[t-fd_step:t+1]
                t_win=fd_step-1
                res=0
                while t_win>lag-1:
                    res+=np.square(trc_win[t_win-lag]-trc_win[t_win])
                    t_win-=1
                res=np.log10(1/(fd_step-lag)*res)
                trc_cp[t]=res
                t-=1
            if len(ress)==0:
                ress=np.reshape(trc_cp,(len(trc_cp),1))
            else:
                ress=np.concatenate((ress,np.reshape(trc_cp,(len(trc_cp),1))),axis=1)
        for i,j in enumerate(ress):
            slope = linregress(lags,ress[i,:])[0]
            trc_out[i]=slope
        
        return trc_out
    
    def amp_spectrum(self,data,dt,single=1):
        """ Calculates amplitude spectrum """
        if single==0:
            sp = np.average(np.fft.fftshift(np.fft.fft(data)),axis=1)
        else:
            sp=np.fft.fftshift(np.fft.fft(data))
        win=np.ones((1,len(data)))
        s_mag=np.abs(sp)*2/np.sum(win)
        s_dbfs=20*np.log10(s_mag/np.amax(s_mag))
        f = np.fft.fftshift(np.fft.fftfreq(len(data), dt))
        freq=f[np.int(len(data)/2)+1:]
        amps=s_dbfs[np.int(len(data)/2)+1:]
        return freq,amps
    
    def norm(self,data):
        """ Performs simple normalization """
        return data/np.amax(np.abs(data))
    
    def fq_win_sum(self,data,hwin,dt):
        """ Calculates summation of amplitude spectra of a data slice """
        data_cp=data.copy()
        for k,l in enumerate(data):
            if np.logical_and(k>hwin,k<len(data)):
                trc_slice=data[k-hwin:k+1]
                taper=hann(len(trc_slice))
                trc_slice_fft=self.amp_spectrum(trc_slice*taper,dt)
                data_cp[k]=np.sum(trc_slice_fft[1][0:])
        return data_cp
    
    def kurtosis_skewness(self,data,hwin):
        """ Calculates higher order statistics: kurtosis & skewness """
        data_cp1=data.copy()
        data_cp2=data.copy()
        for k,l in enumerate(data):
            if np.logical_and(k>hwin,k<len(data)):
                trc_slice=data[k-hwin:k+1]
                data_cp1[k]=kurtosis(trc_slice)
                data_cp2[k]=skew(trc_slice)
        return data_cp1,data_cp2
    
    def trc_fgen_prefb(self,trc,dt,nspad=200,hwin=150,vlen=51):
        """ Constructs feature matrices for the pre-FB model """
        output=np.zeros((len(trc),((11*(vlen))+1)))
        pad=np.random.rand(nspad)/100
        trc_norm=trc/np.amax(np.abs(trc))
        trc_norm_padded=np.hstack((pad,trc_norm))
        trc_entropy=self.entropy(trc_norm_padded,50)
        trc_fdm=self.fdm(trc_norm_padded,50,np.arange(1,4),15)
        trc_slta=trigger.classic_sta_lta(trc_norm_padded,2,100)
        trc_fq_win_sum=self.fq_win_sum(trc_norm_padded,hwin,dt)
        hwin2=50
        trc_kurtosis_skew=self.kurtosis_skewness(trc_norm_padded,hwin2)
        for i,j in enumerate(trc):
            ftrc=[]
            fb=i*dt
            ftrc=np.append(ftrc,trc_norm_padded[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
            ftrc=np.append(ftrc,self.norm(np.gradient(np.abs(trc_norm_padded)))[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
            ftrc=np.append(ftrc,self.norm(trc_entropy)[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
            ftrc=np.append(ftrc,self.norm(np.gradient(trc_entropy))[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])    
            ftrc=np.append(ftrc,self.norm(trc_fdm)[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
            ftrc=np.append(ftrc,self.norm(np.gradient(trc_fdm))[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])  
            ftrc=np.append(ftrc,self.norm(trc_slta)[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
            ftrc=np.append(ftrc,self.norm(trc_fq_win_sum)[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
            ftrc=np.append(ftrc,self.norm(np.gradient(trc_fq_win_sum))[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
            ftrc=np.append(ftrc,self.norm(trc_kurtosis_skew[0])[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
            ftrc=np.append(ftrc,self.norm(trc_kurtosis_skew[1])[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
            ftrc=np.append(ftrc,1)
            output[i,:]=ftrc
        return output
    
    def trc_fgen_postfb(self,trc,dt,hwin=150,vlen=51):
        """ Constructs a feature matrix for the post-FB model """
        output=np.zeros((len(trc),((11*(vlen))+1)))
        trc_norm=trc/np.amax(np.abs(trc))
        trc_entropy=self.entropy(trc_norm,50)
        trc_fdm=self.fdm(trc_norm,50,np.arange(1,4),15)
        trc_slta=trigger.classic_sta_lta(trc_norm,2,100)
        trc_fq_win_sum=self.fq_win_sum(trc_norm,hwin,dt)
        hwin2=50
        trc_kurtosis_skew=self.kurtosis_skewness(trc_norm,hwin2)
        for i,j in enumerate(trc):
            if i<len(trc)-vlen-1:
                ftrc=[]
                fb=i*dt
                ftrc=np.append(ftrc,trc_norm[np.int(fb/dt):np.int(fb/dt)+vlen])
                ftrc=np.append(ftrc,self.norm(np.gradient(np.abs(trc_norm)))[np.int(fb/dt):np.int(fb/dt)+vlen])
                ftrc=np.append(ftrc,self.norm(trc_entropy)[np.int(fb/dt):np.int(fb/dt)+vlen])
                ftrc=np.append(ftrc,self.norm(np.gradient(trc_entropy))[np.int(fb/dt):np.int(fb/dt)+vlen])    
                ftrc=np.append(ftrc,self.norm(trc_fdm)[np.int(fb/dt):np.int(fb/dt)+vlen])
                ftrc=np.append(ftrc,self.norm(np.gradient(trc_fdm))[np.int(fb/dt):np.int(fb/dt)+vlen])  
                ftrc=np.append(ftrc,self.norm(trc_slta)[np.int(fb/dt):np.int(fb/dt)+vlen])
                ftrc=np.append(ftrc,self.norm(trc_fq_win_sum)[np.int(fb/dt):np.int(fb/dt)+vlen])
                ftrc=np.append(ftrc,self.norm(np.gradient(trc_fq_win_sum))[np.int(fb/dt):np.int(fb/dt)+vlen])
                ftrc=np.append(ftrc,self.norm(trc_kurtosis_skew[0])[np.int(fb/dt):np.int(fb/dt)+vlen])
                ftrc=np.append(ftrc,self.norm(trc_kurtosis_skew[1])[np.int(fb/dt):np.int(fb/dt)+vlen])
                ftrc=np.append(ftrc,1)
                output[i,:]=ftrc
        return output
    
    def trc_prep(self,m,ds,s,mode=2):
        """ Resizes a given matrix by down-sampling (Mode 1 - pre-FB, Mode 2 - post-FB)"""
        m_out=np.zeros((m.shape[0],11*ds+1))
        if mode==1:
            for i,j in enumerate(m):
                m_out[i,:]=np.hstack((m[i,1*s-ds:1*s],m[i,2*s-ds:2*s],m[i,3*s-ds:3*s],m[i,4*s-ds:4*s],m[i,5*s-ds:5*s],m[i,6*s-ds:6*s],m[i,7*s-ds:7*s],m[i,8*s-ds:8*s],m[i,9*s-ds:9*s],m[i,10*s-ds:10*s],m[i,11*s-ds:11*s],m[i,11*s]))
        elif mode==2:
            for i,j in enumerate(m):
                m_out[i,:]=np.hstack((m[i,0*s:0*s+ds],m[i,1*s:1*s+ds],m[i,2*s:2*s+ds],m[i,3*s:3*s+ds],m[i,4*s:4*s+ds],m[i,5*s:5*s+ds],m[i,6*s:6*s+ds],m[i,7*s:7*s+ds],m[i,8*s:8*s+ds],m[i,9*s:9*s+ds],m[i,10*s:10*s+ds],m[i,11*s]))
        return m_out

    def predict(self):
        """ Performs first-break prediction """
        selected_trcs=np.where(np.logical_and(np.abs(self.dataset['offset'])/self.scalar_offset>=self.min_offset,np.abs(self.dataset['offset'])/self.scalar_offset<self.max_offset))[0]
        iprint=99
        start_time=time.time()
        for i,j in enumerate(selected_trcs):
            trc_data_prefb=self.trc_fgen_prefb(self.dataset['data'][j][self.first_sample:self.first_sample+self.trc_slen],dt=self.dt)
            trc_data_postfb=self.trc_fgen_postfb(self.dataset['data'][j][self.first_sample:self.first_sample+self.trc_slen],dt=self.dt)
            trc_predictions=np.zeros((trc_data_prefb.shape[0],3))
            for k,l in enumerate((1,51,51)):
                trc_dataslice_prefb=self.trc_prep(trc_data_prefb,l,51,mode=1)
                trc_dataslice_postfb=self.trc_prep(trc_data_postfb,l,51,mode=2)
                model=self.models[k]
                for m,n in enumerate(trc_dataslice_prefb):
                    if k==0:
                        trc_predictions[m,k]=model.predict(np.array([trc_dataslice_prefb[m,0:-1]]))
                    elif k==1:
                        trc_predictions[m,k]=model.predict(np.array([trc_dataslice_prefb[m,list(self.features_prefb)]]))
                    else:
                        trc_predictions[m,k]=model.predict(np.array([trc_dataslice_postfb[m,list(self.features_postfb)]]))
            self.dataset['pred_avg'][j,self.first_sample:self.first_sample+self.trc_slen]=np.reshape(np.average(trc_predictions,axis=1),(self.trc_slen,))
            self.dataset['pred_1samp'][j,self.first_sample:self.first_sample+self.trc_slen]=np.reshape(trc_predictions[:,0],(self.trc_slen,))
            self.dataset['pred_prefb'][j,self.first_sample:self.first_sample+self.trc_slen]=np.reshape(trc_predictions[:,1],(self.trc_slen,))
            self.dataset['pred_postfb'][j,self.first_sample:self.first_sample+self.trc_slen]=np.reshape(trc_predictions[:,2],(self.trc_slen,))
    
            if i==iprint:
                print('Trc_nums: ',i-98,'-',i+1,'out of: ',len(selected_trcs), 'completed in ', np.int((time.time()-start_time)), 's')
                iprint+=100
                start_time=time.time()
            sys.stdout.flush()
            
    def find_fb(self,q=99.9):
        """ Finds the first maximum probability value of first-break occurence based on a given percentile """
        fbs=np.zeros((self.dataset['pred_avg'].shape[0],1))
        for itrc in np.arange(0,self.dataset['pred_avg'].shape[0]):
            trc=d1['pred_avg'][itrc]
            nonzero=np.where(trc!=0)[0]
            perc=np.nanpercentile(trc[list(nonzero)],q)
            potential_fbs=np.where(trc[:]>=perc)[0]
            if len(potential_fbs)!=0:
                fbs[itrc]=np.int(potential_fbs[0])
            else:
                print('FB was not found for trace id:\t{}'.format(itrc))
        return fbs
    
    def find_approx_fb(self,min_offset,max_offset,min_cdp,max_cdp,offset_spacing,n_split=100):
        """ Finds the maximum value of a probability distribution that is averaged within selected offset and cdp range """
        fbs=np.zeros((self.dataset['pred_avg'].shape[0],1))
        min_offset=(min_offset+offset_spacing)*self.scalar_offset
        max_offset=(max_offset-offset_spacing)*self.scalar_offset
        offsets=np.arange(min_offset,max_offset,offset_spacing*self.scalar_offset)
        for i,coffset in enumerate(offsets):
                print('Working on central offset:\t{}'.format(coffset/self.scalar_offset))
                obin_trcs=np.where(np.logical_and(self.dataset['cdp'][:]<=max_cdp,np.logical_and(self.dataset['cdp'][:]>=min_cdp,np.logical_and(self.dataset['offset'][:]>=coffset-offset_spacing,self.dataset['offset'][:]<coffset+offset_spacing))))[0]
                tmp1=np.array_split(obin_trcs,n_split)   
                if len(obin_trcs)>10:
                    for k,l in enumerate(tmp1):
                        tmp0=d1['pred_avg'][list(tmp1[k]),:]
                        tmp2=np.sum(tmp0,axis=0)
                        tmp2=np.where(tmp2[:]==np.amax(tmp2))[0]
                        for m,n in enumerate(tmp1[k]):
                            fbs[n]=np.int(tmp2)
                else:
                    print('Not enough traces in a splitted offset bin')
                    
    def stats(self,reference_fbs,predicted_fbs,allowed_diff):
        """ Provides basic statistics on a first-break outcome """
        diff=np.abs(reference_fbs-predicted_fbs)
        diff_nonzero=np.where(diff!=0)[0]
        mispicked=np.where(diff>allowed_diff)[0]
        accuracy=1-len(mispicked)/len(predicted_fbs)
        print('Traces analyzed:\t{}\n\
              Allowed sample mismatch:\t{}\n\
              Traces mispicked:\t{}\n\
              Accuracy:\t{} %\n\
              Median sample mismatch:\t{}'.format(len(predicted_fbs),allowed_diff,len(mispicked),round(accuracy*100,1),np.int(np.median(diff[list(diff_nonzero)]))))