import scipy
import numpy as np
import h5py
import os
import sys
import copy,time
from scipy.fftpack import fft, ifft
import signal
from obspy.signal import filter
from obspy.signal import trigger
from obspy.signal import cpxtrace
from scipy import stats
from obspy.core.trace import Trace
import theano.ifelse
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt

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
        predictions1=f['/pred_fb1']
        pred_fb_avg=f['/pred_fb_avg']
        predictions2=f['/pred_fb2']
        predictions3=f['/pred_fb3']
        dataset={'data':data,'shotids':shotids,'offset':offset,'cdp':cdp,'predictions1':predictions1,'pred_fb_avg':pred_fb_avg,'gapsize':gapsize,'user':user,'src_x':src_x,'src_y':src_y,'cdp_x':cdp_x,'cdp_y':cdp_y,'snum':sourcenum,'rnum':recordnum,'ftrc':ftrace,'rec_x':rec_x,'rec_y':rec_y,'chan':channels,'predictions2':predictions2,'predictions3':predictions3}
    else:
        dataset={'data':data,'shotids':shotids,'offset':offset,'cdp':cdp,'gapsize':gapsize,'user':user,'src_x':src_x,'src_y':src_y,'cdp_x':cdp_x,'cdp_y':cdp_y,'snum':sourcenum,'rnum':recordnum}
    #pred_fb_avg_mp=f['/pred_fb_avg_mp']

    print('Data loaded')
    return dataset

def add_hdrs(path):
    file=h5py.File(path,'a')
    data=file['/TRACE_DATA/DEFAULT/data_array/']
    try:
        file.create_dataset('/pred_fb1',(len(data),len(data[0])))
        file.create_dataset('/pred_fb2',(len(data),len(data[0])))
        file.create_dataset('/pred_fb3',(len(data),len(data[0])))
        file.create_dataset('/pred_fb_avg',(len(data),len(data[0])))
        file['/pred_fb1'][:]=0
        file['/pred_fb2'][:]=0
        file['/pred_fb3'][:]=0
        file['/pred_fb_avg'][:]=0
        file.close()
        print('Headers pred_fb1 & pred_fb2 added to the data structure')
    except:
        file['/pred_fb1'][:]=0
        file['/pred_fb2'][:]=0
        file['/pred_fb3'][:]=0
        file['/pred_fb_avg'][:]=0
        file.close()
        print('Headers pred_fb1 & pred_fb2 already exist')
        
def entropy(trc,e_step):
    t=len(trc)-1
    trc_out=copy.deepcopy(trc)
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
    
def eps_filter(trc,eps_len):
    t=len(trc)-((eps_len-1)/2+1)
    trc_out=copy.deepcopy(trc)
    while t>0:
        tmp1=[]
        tmp2=[]
        tmp3=[]
        t_win=0
        while t_win<5:
            tmp1=np.append(tmp1,np.arange(t-eps_len-1,t+1))
            t_win+=1
        for i,j in enumerate(tmp1):
            tmp2=np.append(tmp2,trc[j])
            tmp3=np.append(tmp3,np.std(tmp2[-1]))
        tmp4=np.where(tmp3==np.amin(tmp3))[0]
        trc_out[t]=np.mean(tmp2[tmp4])
        t-=1
    return trc_out
    
def fdm(trc,fd_step,lags,noise_scalar):
    ress=[]
    trc_out=trc/np.amax(np.abs(trc))
    noise=np.random.normal(0,1,len(trc_out))*(np.std(trc_out)/noise_scalar)
    trc_out=trc_out+noise
    for i,lag in enumerate(lags):
        trc_cp=copy.deepcopy(trc_out)
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
        slope = stats.linregress(lags,ress[i,:])[0]
        trc_out[i]=slope
    
    return trc_out

def amp_spectrum(data,dt=0.004,single=1):
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

def despike(data,lperc,hperc):
    lamp=np.percentile(data,lperc)
    hamp=np.percentile(data,hperc)
    sample_to_kill=np.where(np.logical_or(data[:]<lamp,data[:]>hamp))[0]
    data[list(sample_to_kill)]=0
    return data

def norm(data):
    return data/np.amax(np.abs(data))

def fq_win_sum(data,hwin,dt):
    data_cp=data.copy()
    for k,l in enumerate(data):
        if np.logical_and(k>hwin,k<len(data)):
            trc_slice=data[k-hwin:k+1]
            taper=scipy.signal.hann(len(trc_slice))
            trc_slice_fft=amp_spectrum(trc_slice*taper,dt)
            data_cp[k]=np.sum(trc_slice_fft[1][0:])
    return data_cp

def prefgen_trc_vector(trc,dt=2,nspad=200,hwin=150,vlen=50):
    output=np.zeros((len(trc),((9*(vlen))+1)))
    pad=np.random.rand(nspad)/100
    trc_norm=trc/np.amax(np.abs(trc))
    trc_norm_padded=np.hstack((pad,trc_norm))
    trc_entropy=entropy(trc_norm_padded,50)
    trc_fdm=fdm(trc_norm_padded,50,np.arange(1,4),15)
    trc_slta=trigger.classic_sta_lta(trc_norm_padded,2,100)
    trc_fq_win_sum=fq_win_sum(trc_norm_padded,hwin,dt)
    for i,j in enumerate(trc):
        ftrc=[]
        fb=i*dt
        ftrc=np.append(ftrc,trc_norm_padded[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(np.gradient(np.abs(trc_norm_padded)))[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(trc_entropy)[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_entropy))[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])    
        ftrc=np.append(ftrc,norm(trc_fdm)[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_fdm))[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])  
        ftrc=np.append(ftrc,norm(trc_slta)[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(trc_fq_win_sum)[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_fq_win_sum))[np.int(nspad+fb/dt)-vlen+1:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,1)
        output[i,:]=ftrc
    return output

def postfgen_trc_vector(trc,dt=2,hwin=150,vlen=50):
    output=np.zeros((len(trc),((9*(vlen))+1)))
    trc_norm=trc/np.amax(np.abs(trc))
    trc_entropy=entropy(trc_norm,50)
    trc_fdm=fdm(trc_norm,50,np.arange(1,4),15)
    trc_slta=trigger.classic_sta_lta(trc_norm,2,100)
    trc_fq_win_sum=fq_win_sum(trc_norm,hwin,dt)
    for i,j in enumerate(trc):
        if i<len(trc)-vlen-1:
            ftrc=[]
            fb=i*dt
            ftrc=np.append(ftrc,trc_norm[np.int(fb/dt):np.int(fb/dt)+vlen])
            ftrc=np.append(ftrc,norm(np.gradient(np.abs(trc_norm)))[np.int(fb/dt):np.int(fb/dt)+vlen])
            ftrc=np.append(ftrc,norm(trc_entropy)[np.int(fb/dt):np.int(fb/dt)+vlen])
            ftrc=np.append(ftrc,norm(np.gradient(trc_entropy))[np.int(fb/dt):np.int(fb/dt)+vlen])    
            ftrc=np.append(ftrc,norm(trc_fdm)[np.int(fb/dt):np.int(fb/dt)+vlen])
            ftrc=np.append(ftrc,norm(np.gradient(trc_fdm))[np.int(fb/dt):np.int(fb/dt)+vlen])  
            ftrc=np.append(ftrc,norm(trc_slta)[np.int(fb/dt):np.int(fb/dt)+vlen])
            ftrc=np.append(ftrc,norm(trc_fq_win_sum)[np.int(fb/dt):np.int(fb/dt)+vlen])
            ftrc=np.append(ftrc,norm(np.gradient(trc_fq_win_sum))[np.int(fb/dt):np.int(fb/dt)+vlen])
            ftrc=np.append(ftrc,1)
            output[i,:]=ftrc
    return output

def dataset_prep(m,ds,s,mode=2):
    m_out=np.zeros((m.shape[0],9*ds+1))
    if mode==1:
        for i,j in enumerate(m):
            m_out[i,:]=np.hstack((m[i,1*s-ds:1*s],m[i,2*s-ds:2*s],m[i,3*s-ds:3*s],m[i,4*s-ds:4*s],m[i,5*s-ds:5*s],m[i,6*s-ds:6*s],m[i,7*s-ds:7*s],m[i,8*s-ds:8*s],m[i,9*s-ds:9*s],m[i,9*s]))
    elif mode==2:
        for i,j in enumerate(m):
            m_out[i,:]=np.hstack((m[i,0*s:0*s+ds],m[i,1*s:1*s+ds],m[i,2*s:2*s+ds],m[i,3*s:3*s+ds],m[i,4*s:4*s+ds],m[i,5*s:5*s+ds],m[i,6*s:6*s+ds],m[i,7*s:7*s+ds],m[i,8*s:8*s+ds],m[i,9*s]))
    return m_out

def prediction(dataset,models,min_offset,max_offset,first_sample,trc_slen):
    selected_trcs=np.where(np.logical_and(np.abs(dataset['offset'])/10>=min_offset,np.abs(dataset['offset'])/10<max_offset))[0]
    iprint=99
    start_time=time.time()
    for i,j in enumerate(selected_trcs):
        trc_out_before=prefgen_trc_vector(dataset['data'][j][first_sample:first_sample+trc_slen])
        trc_out_after=postfgen_trc_vector(dataset['data'][j][first_sample:first_sample+trc_slen])
        trc_out_probs=np.zeros((trc_out_before.shape[0],3))
        for k,l in enumerate((1,10,10)):
            data_before=dataset_prep(trc_out_before,l,50,mode=1)
            data_after=dataset_prep(trc_out_after,l,50,mode=2)
            model=models[k]
            for m,n in enumerate(data_before):
                if k<2:
                    trc_out_probs[m,k]=model.predict(np.array([data_before[m,0:-1]]))
                else:
                    trc_out_probs[m,k]=model.predict(np.array([data_after[m,0:-1]]))
        trc_out_pred_6fold=np.zeros((trc_out_probs.shape[0],1))
        for k,l in enumerate(trc_out_probs):
            trc_out_pred_6fold[k]=np.sum(trc_out_probs[k,:])/len(models)
        dataset['pred_fb_avg'][j,first_sample:first_sample+trc_slen]=np.reshape(trc_out_pred_6fold,(trc_slen,))
        dataset['predictions1'][j,first_sample:first_sample+trc_slen]=np.reshape(trc_out_probs[:,0],(trc_slen,))
        dataset['predictions2'][j,first_sample:first_sample+trc_slen]=np.reshape(trc_out_probs[:,1],(trc_slen,))
        dataset['predictions3'][j,first_sample:first_sample+trc_slen]=np.reshape(trc_out_probs[:,2],(trc_slen,))

    #    tmp4=np.where(trc_out_pred[:]>0.9)[0]
    #    if len(tmp4)>0:
    #        d1['gapsize'][j]=np.int(tmp4[-1])
        if i==iprint:
            print('Trc_nums: ',i-98,'-',i+1,'out of: ',len(selected_trcs), 'completed in ', np.int((time.time()-start_time)), 's')
            iprint+=100
            start_time=time.time()
        sys.stdout.flush()
        
def plot(dataset,trcids,fsamp,lsamp):
    ntrcs=len(trcids)
    fig,axes=plt.subplots(ncols=4,nrows=ntrcs,sharey=True,sharex=True,figsize=(25,4*ntrcs))
    for i,j in enumerate(trcids):
        fb=dataset['gapsize'][j]/2
        axes[i,0].set_title('Single\nsample model',fontsize=15)
        axes[i,0].plot(dataset['predictions1'][j][fsamp:lsamp],c='red',linewidth=1,label='Probability\ndistribution')
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
        axes[i,2].set_title('10 sample\npost-FB model',fontsize=15)
        axes[i,2].plot(dataset['predictions2'][j][fsamp:lsamp],c='red',linewidth=1)
        axes[i,2].plot(dataset['data'][j][fsamp:lsamp]/np.amax(np.abs(dataset['data'][j][fsamp:lsamp])),c='black',linewidth=1)
        axes[i,2].plot([fb,fb],[-1,0],'--',c='blue',label='Reference first-break sample')
        axes[i,1].set_title('10 sample\npre-FB model',fontsize=15)
        axes[i,1].plot(dataset['predictions3'][j][fsamp:lsamp],c='red',linewidth=1)
        axes[i,1].plot(dataset['data'][j][fsamp:lsamp]/np.amax(np.abs(dataset['data'][j][fsamp:lsamp])),c='black',linewidth=1)
        axes[i,1].plot([fb,fb],[-1,0],'--',c='blue',label='Reference first-break sample')
        axes[i,3].set_title('Models average',fontsize=15)
        axes[i,3].plot(dataset['pred_fb_avg'][j][fsamp:lsamp],c='red',linewidth=1)
        axes[i,3].plot(dataset['data'][j][fsamp:lsamp]/np.amax(np.abs(dataset['data'][j][fsamp:lsamp])),c='black',linewidth=1)
        axes[i,3].plot([fb,fb],[-1,0],'--',c='blue',label='Reference first-break sample')
    plt.tight_layout()