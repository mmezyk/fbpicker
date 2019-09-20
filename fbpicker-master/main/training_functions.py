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
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import plot_model
from sklearn.cross_validation import train_test_split
from scipy.stats import randint as sp_randint
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix

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

def gaussian_perturbartion(dataset,n):
    iprint=0
    output_data=np.zeros(((n+1)*len(dataset['data']),len(dataset['data'][0])))
    output_fbs=np.zeros(((n+1)*len(dataset['data']),1))
    for i,j in enumerate(np.arange(0,len(dataset['data']))):
        trc_norm=norm(dataset['data'][j])
        mean=np.average(np.abs(trc_norm))*0.5
        std=np.std(np.abs(trc_norm))*0.5
        k=0
        output_data[i*(n+1),:]=trc_norm
        output_fbs[i*(n+1):(i+1)*(n+1)]=dataset['gapsize'][j]
        while k<n/2:
            std_temp=np.random.rand(1)*std
            mean_temp=np.random.rand(1)*mean
            output_data[i+k*2+1+i*n,:]=trc_norm+np.random.normal(mean_temp,std_temp, trc_norm.shape)
            std_temp=np.random.rand(1)*std
            mean_temp=-np.random.rand(1)*mean
            output_data[i+k*2+2+i*n,:]=trc_norm+np.random.normal(mean_temp,std_temp, trc_norm.shape)
            #print('__',i+k*2+1+i*n,i+k*2+2+i*n)
            k+=1
        if i==iprint:
            print('Trace perturbation No. ',i,'completed')
            iprint+=100
    return output_data,output_fbs

def prefgen_vector(dataset,fbs,dt=2,nspad=200,hwin=150,vlen=10):
    iprint=0
    feature_matrix=np.zeros((len(dataset),(9*(vlen+1))+1))
    for trcid,trc in enumerate(dataset):
        ftrc=[]
        trc=dataset[trcid]
        fb=fbs[trcid]
        pad=np.random.rand(nspad)/100
        trc_norm=trc/np.amax(np.abs(trc))
        trc_norm_padded=np.hstack((pad,trc_norm))
        ftrc=np.append(ftrc,trc_norm_padded[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(np.gradient(np.abs(trc_norm_padded)))[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        trc_entropy=entropy(trc_norm_padded,50)
        ftrc=np.append(ftrc,norm(trc_entropy)[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_entropy))[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])    
        trc_fdm=fdm(trc_norm_padded,50,np.arange(1,4),15)
        ftrc=np.append(ftrc,norm(trc_fdm)[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_fdm))[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])  
        trc_slta=trigger.classic_sta_lta(trc_norm_padded,2,100)
        ftrc=np.append(ftrc,norm(trc_slta)[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        trc_fq_win_sum=fq_win_sum(trc_norm_padded,hwin,dt)
        ftrc=np.append(ftrc,norm(trc_fq_win_sum)[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_fq_win_sum))[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,1) 
        feature_matrix[trcid,:]=ftrc
        if trcid==iprint:
            print('Feature vector for trace No. ',trcid,'completed')
            iprint+=100
    return feature_matrix

def postfgen_vector(dataset,fbs,dt=2,hwin=150,vlen=10):
    iprint=0
    feature_matrix=np.zeros((len(dataset),(9*(vlen+1))+1))
    for trcid,trc in enumerate(dataset):
        ftrc=[]
        trc=dataset[trcid]
        fb=fbs[trcid]
        trc_norm=trc/np.amax(np.abs(trc))
        ftrc=np.append(ftrc,trc_norm[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        ftrc=np.append(ftrc,norm(np.gradient(np.abs(trc_norm)))[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        trc_entropy=entropy(trc_norm,50)
        ftrc=np.append(ftrc,norm(trc_entropy)[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_entropy))[np.int(fb/dt):np.int(fb/dt)+vlen+1])    
        trc_fdm=fdm(trc_norm,50,np.arange(1,4),15)
        ftrc=np.append(ftrc,norm(trc_fdm)[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_fdm))[np.int(fb/dt):np.int(fb/dt)+vlen+1])  
        trc_slta=trigger.classic_sta_lta(trc_norm,2,100)
        ftrc=np.append(ftrc,norm(trc_slta)[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        trc_fq_win_sum=fq_win_sum(trc_norm,hwin,dt)
        ftrc=np.append(ftrc,norm(trc_fq_win_sum)[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_fq_win_sum))[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        ftrc=np.append(ftrc,1) 
        feature_matrix[trcid,:]=ftrc
        if trcid==iprint:
            print('Feature vector for trace No. ',trcid,'completed')
            iprint+=100        
    return feature_matrix

def prefgen_false_vector(dataset,fbs,dt=2,nspad=200,hwin=150,fb_hzone=500,vlen=10):
    iprint=0
    feature_matrix=np.zeros((len(dataset),(9*(vlen+1))+1))
    for trcid,trc in enumerate(dataset):
        ftrc=[]
        trc=dataset[trcid]
        fb_true=fbs[trcid]
        fb=fb_true
        while np.logical_or(fb==fb_true,fb<0):    
            fb=np.int(np.random.uniform(fb_true-fb_hzone,fb_true+fb_hzone,1))
        pad=np.random.rand(nspad)/100
        trc_norm=trc/np.amax(np.abs(trc))
        trc_norm_padded=np.hstack((pad,trc_norm))
        ftrc=np.append(ftrc,trc_norm_padded[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(np.gradient(np.abs(trc_norm_padded)))[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        trc_entropy=entropy(trc_norm_padded,50)
        ftrc=np.append(ftrc,norm(trc_entropy)[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_entropy))[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])    
        trc_fdm=fdm(trc_norm_padded,50,np.arange(1,4),15)
        ftrc=np.append(ftrc,norm(trc_fdm)[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_fdm))[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])  
        trc_slta=trigger.classic_sta_lta(trc_norm_padded,2,100)
        ftrc=np.append(ftrc,norm(trc_slta)[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        trc_fq_win_sum=fq_win_sum(trc_norm_padded,hwin,dt)
        ftrc=np.append(ftrc,norm(trc_fq_win_sum)[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_fq_win_sum))[np.int(nspad+fb/dt)-vlen:np.int(nspad+fb/dt)+1])
        ftrc=np.append(ftrc,0) 
        feature_matrix[trcid,:]=ftrc
        if trcid==iprint:
            print('Feature vector for trace No. ',trcid,'completed')
            iprint+=100            
    return feature_matrix

def postfgen_false_vector(dataset,fbs,dt=2,hwin=150,fb_hzone=500,vlen=10):
    iprint=0
    feature_matrix=np.zeros((len(dataset),(9*(vlen+1))+1))
    for trcid,trc in enumerate(dataset):
        ftrc=[]
        trc=dataset[trcid]
        fb_true=fbs[trcid]
        fb=fb_true
        while np.logical_or(fb==fb_true,fb<0):    
            fb=np.int(np.random.uniform(fb_true-fb_hzone,fb_true+fb_hzone,1))
        trc_norm=trc/np.amax(np.abs(trc))
        ftrc=np.append(ftrc,trc_norm[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        ftrc=np.append(ftrc,norm(np.gradient(np.abs(trc_norm)))[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        trc_entropy=entropy(trc_norm,50)
        ftrc=np.append(ftrc,norm(trc_entropy)[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_entropy))[np.int(fb/dt):np.int(fb/dt)+vlen+1])    
        trc_fdm=fdm(trc_norm,50,np.arange(1,4),15)
        ftrc=np.append(ftrc,norm(trc_fdm)[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_fdm))[np.int(fb/dt):np.int(fb/dt)+vlen+1])  
        trc_slta=trigger.classic_sta_lta(trc_norm,2,100)
        ftrc=np.append(ftrc,norm(trc_slta)[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        trc_fq_win_sum=fq_win_sum(trc_norm,hwin,dt)
        ftrc=np.append(ftrc,norm(trc_fq_win_sum)[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        ftrc=np.append(ftrc,norm(np.gradient(trc_fq_win_sum))[np.int(fb/dt):np.int(fb/dt)+vlen+1])
        ftrc=np.append(ftrc,0) 
        feature_matrix[trcid,:]=ftrc
        if trcid==iprint:
            print('Feature vector for trace No. ',trcid,'completed')
            iprint+=100            
    return feature_matrix

def dataset_prep(m,ds,s,mode=2):
    m_out=np.zeros((m.shape[0],9*ds+1))
    if mode==1:
        for i,j in enumerate(m):
            m_out[i,:]=np.hstack((m[i,1*s-ds:1*s],m[i,2*s-ds:2*s],m[i,3*s-ds:3*s],m[i,4*s-ds:4*s],m[i,5*s-ds:5*s],m[i,6*s-ds:6*s],m[i,7*s-ds:7*s],m[i,8*s-ds:8*s],m[i,9*s-ds:9*s],m[i,9*s]))
    elif mode==2:
        for i,j in enumerate(m):
            m_out[i,:]=np.hstack((m[i,0*s:0*s+ds],m[i,1*s:1*s+ds],m[i,2*s:2*s+ds],m[i,3*s:3*s+ds],m[i,4*s:4*s+ds],m[i,5*s:5*s+ds],m[i,6*s:6*s+ds],m[i,7*s:7*s+ds],m[i,8*s:8*s+ds],m[i,9*s]))
    return m_out

def model_training(train_set,validation_set,hidden_layers=2,neurons=50,input_dim=90,batch_size=10,epochs=100):
    x_train = train_set[:,0:-1]
    y_train = train_set[:, -1]
    x_test = validation_set[:,0:-1]
    y_test = validation_set[:, -1]
    model = Sequential()
    if hidden_layers==1:
        model.add(Dense(output_dim = neurons, init = 'he_normal', activation = 'relu', input_dim = input_dim))
    elif hidden_layers==2:
        model.add(Dense(output_dim = neurons, init = 'he_normal', activation = 'relu', input_dim = input_dim))
        model.add(Dense(output_dim = neurons, init = 'he_normal', activation = 'relu'))
    elif hidden_layers==3:
        model.add(Dense(output_dim = neurons, init = 'he_normal', activation = 'relu', input_dim = input_dim))
        model.add(Dense(output_dim = neurons, init = 'he_normal', activation = 'relu'))
        model.add(Dense(output_dim = neurons, init = 'he_normal', activation = 'relu'))
    elif hidden_layers>=4:
        model.add(Dense(output_dim = neurons, init = 'he_normal', activation = 'relu', input_dim = input_dim))
        model.add(Dense(output_dim = neurons, init = 'he_normal', activation = 'relu'))
        model.add(Dense(output_dim = neurons, init = 'he_normal', activation = 'relu'))
        model.add(Dense(output_dim = neurons, init = 'he_normal', activation = 'relu'))
    model.add(Dense(output_dim = 1, init = 'he_normal', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model_log=model.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size = batch_size, nb_epoch = epochs)
    return model,model_log

def plot_training_log(log1,label1,log2,label2,log3,label3,lossmin,lossmax,valmin,valmax):
    fig,axes=plt.subplots(nrows=2,sharex=True,figsize=(10,5))
    axes[0].set_title('Validation test\nrun on unseen dataset',fontsize=20)
    axes[0].plot(log1.history['val_loss'],label=label1,c='red',linewidth=2)
    axes[0].plot(log2.history['val_loss'],label=label2,c='blue',linewidth=2)
    axes[0].plot(log3.history['val_loss'],label=label3,c='green',linewidth=2)
    axes[0].grid()
    axes[0].set_xlim((0,99))
    axes[0].set_ylim((lossmin,lossmax))
    axes[0].set_ylabel('Loss',fontsize=15)
    axes[1].plot(log1.history['val_acc'],label=label1,c='red',linewidth=2)
    axes[1].plot(log2.history['val_acc'],label=label2,c='blue',linewidth=2)
    axes[1].plot(log3.history['val_acc'],label=label3,c='green',linewidth=2)
    axes[1].legend(loc=4)
    axes[1].grid()
    axes[1].set_xlim((0,99))
    axes[1].set_ylim((valmin,valmax))
    axes[1].set_ylabel('Accuracy [%]',fontsize=15)
    axes[1].set_xlabel('Epoch',fontsize=15)
    plt.tight_layout()
    
def model_test(model,test_set,threshold):
    x_test = test_set[:,0:-1]
    y_test = test_set[:, -1]
    y_pred = model.predict(x_test)
    y_pred[y_pred>=threshold]=1
    y_pred[y_pred<threshold]=0
    cm = confusion_matrix(y_test, y_pred)
    print('Total predictions:',len(y_pred))
    print('True positive:',cm[0,0])
    print('True negative:',cm[1,1])
    print('False positive:',cm[0,1])
    print('False negative:',cm[1,0])
    print('Accuracy:',(cm[0,0]+cm[1,1])/len(y_pred))