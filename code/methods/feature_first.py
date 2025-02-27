import torch
import torch.nn as nn
import librosa
import numpy as np
from methods.utils.stft import (STFT, LogmelFilterBank, intensityvector,
                                spectrogram_STFTInput)
import math

def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

class LogmelIntensity_Extractor(nn.Module):
    #MAY NEED TO CHANGE THIS FOR BINAURAL
    def __init__(self, cfg):
        super().__init__()

        data = cfg['data']
        sample_rate, n_fft, hop_length, window, n_mels = \
            data['sample_rate'], data['nfft'], data['hoplen'], data['window'], data['n_mels']
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.n_mels=n_mels
        self.nfft=n_fft
        # STFT extractor
        self.stft_extractor = STFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, 
            window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)
        self.hopsize=hop_length
        fft_window = librosa.filters.get_window(window, n_fft, fftbins=True)
        
        self.window = torch.from_numpy(librosa.util.pad_center(fft_window, size=n_fft)).to(device="cuda")
        # Spectrogram extractor
        self.spectrogram_extractor = spectrogram_STFTInput
        
        # Logmel extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=n_mels, fmin=20, fmax=sample_rate/2, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        # Intensity vector extractor
        self.intensityVector_extractor = intensityvector
        self.melW = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
            fmin=50, fmax=14000).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW).to(dtype=torch.complex128))
        self.melW.requires_grad = False

    def forward(self, x):
        """
        input: 
            (batch_size, channels=4, data_length)
        output: 
            (batch_size, channels, time_steps, freq_bins) freq_bins->mel_bins
        """
        if x.ndim != 3:
            raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                            Now it is {}".format(x.shape))
        x_00=x[:,0,:]
        x_01=x[:,1,:]

        # mag,cos,sin,x_0,x_1 = self.stft_extractor(x)
        # x=(x_0,x_1)
        # raw_spec,logmel = self.logmel_extractor(self.spectrogram_extractor(x))
        value = 1e-20
        # ild=raw_spec[:,0,:,:]/(raw_spec[:,1,:,:]+value)
        # melcos,_=self.logmel_extractor(cos)
        # melsin,_=self.logmel_extractor(sin)
        # sinipd=melcos[:,1,:,:]*melsin[:,0,:,:]-melcos[:,0,:,:]*melsin[:,1,:,:]
        # cosipd=melcos[:,0,:,:]*melcos[:,1,:,:]+melsin[:,0,:,:]*melsin[:,1,:,:]
        # (a,b,c,d)=logmel.shape
        # ild=ild.view(a,1,c,d)
        # sinipd=sinipd.view(a,1,c,d)
        # cosipd=cosipd.view(a,1,c,d)
        dev=x_00.get_device()
        self.window=self.window.to(device=("cuda:"+str(dev)))
        Px = torch.stft(input=x_00,
                        n_fft=self.nfft,
                        hop_length=self.hopsize,
                        win_length=self.nfft,
                        window=self.window,
                        center=True,
                        pad_mode='reflect',
                        normalized=False, onesided=None, return_complex=True)
        dev=x_01.get_device()
        self.window=self.window.to(device=("cuda:"+str(dev)))
        Px_ref = torch.stft(input=x_01,
                            n_fft=self.nfft,
                            win_length=self.nfft,
                            hop_length=self.hopsize,
                            window=self.window,
                            center=True,
                            pad_mode='reflect',
                            normalized=False, onesided=None, return_complex=True)
        Px=torch.transpose(Px,1,2)
        Px_ref=torch.transpose(Px_ref,1,2)
        Px_mel=torch.matmul(Px, self.melW)
        Px_ref_mel=torch.matmul(Px_ref, self.melW)  
        (a,c,d)=Px_mel.shape
        Px_mag=torch.abs(Px_mel)
        Px_ref_mag=torch.abs(Px_ref_mel)
        ild_2=Px_mag/(Px_ref_mag+value)
        ild_2=ild_2.view(a,1,c,d)
        Px_angle=torch.angle(Px_mel)
        Px_ref_angle=torch.angle(Px_ref_mel)
        cosipd_2=torch.cos(Px_angle-Px_ref_angle)
        sinipd_2=torch.sin(Px_angle-Px_ref_angle)
        cosipd_2=cosipd_2.view(a,1,c,d)
        sinipd_2=sinipd_2.view(a,1,c,d)
        # Px_mag_log=10.0 * torch.log10(torch.clamp(Px_mag, min=1e-10, max=np.inf))
        # Px_mag_log -= 10.0 * np.log10(np.maximum(1e-10, 1.0))

        # Px_mag_log = torch.clamp(Px_mag_log, min=Px_mag_log.max().item() - 80.0, max=np.inf)
        # Px_ref_mag_log=10.0 * torch.log10(torch.clamp(Px_ref_mag, min=1e-10, max=np.inf))
        # Px_ref_mag_log -= 10.0 * np.log10(np.maximum(1e-10, 1.0))

        # Px_ref_mag_log = torch.clamp(Px_ref_mag_log, min=Px_ref_mag_log.max().item() - 80.0, max=np.inf)
        ddev=Px_mag.get_device()
        Px_mag_log=torch.Tensor(librosa.power_to_db(Px_mag.cpu().numpy())).to(device=("cuda:"+str(ddev)))
        ddev=Px_ref_mag.get_device()
        Px_ref_mag_log=torch.Tensor(librosa.power_to_db(Px_ref_mag.cpu().numpy())).to(device=("cuda:"+str(ddev)))
        Px_mag_log=Px_mag_log.view(a,1,c,d)
        Px_ref_mag_log=Px_ref_mag_log.view(a,1,c,d)
        R = Px*torch.conj(Px_ref)
        gcc = torch.fft.irfft(torch.exp(1.j*torch.angle(R)))
        gcc = torch.cat((gcc[:,:,-self.n_mels//2:],gcc[:,:,:self.n_mels//2]),dim=-1)
        gcc = gcc.view(a,1,c,d)   
        out = torch.cat((Px_mag_log,Px_ref_mag_log, ild_2,sinipd_2,cosipd_2,gcc), dim=1)
        out=out.detach()
        out=out.float()
   
        return out

class Logmel_Extractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        data = cfg['data']
        sample_rate, n_fft, hop_length, window, n_mels = \
            data['sample_rate'], data['nfft'], data['hoplen'], data['window'], data['n_mels']
        
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # STFT extractor
        self.stft_extractor = STFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, 
            window=window, center=center, pad_mode=pad_mode, 
            )
        
        # Spectrogram extractor
        self.spectrogram_extractor = spectrogram_STFTInput
        
        # Logmel extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=n_mels, fmin=20, fmax=sample_rate/2, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


    def forward(self, x):
        """
        input: 
            (batch_size, channels=4, data_length)
        output: 
            (batch_size, channels, time_steps, freq_bins) freq_bins->mel_bins
        """
        if x.ndim != 3:
            raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                            Now it is {}".format(x.shape))
        x = self.stft_extractor(x)
        logmel = self.logmel_extractor(self.spectrogram_extractor(x))
        out = logmel
        return out

class Features_Extractor_MIC():
    def __init__(self, cfg):
        self.cfg=cfg
        self.fs = cfg['data']['sample_rate']
        self.n_fft = cfg['data']['nfft']
        self.n_mels = cfg['data']['n_mels']
        self.hoplen = cfg['data']['hoplen']
        self.mel_bank = librosa.filters.mel(sr=self.fs, n_fft=self.n_fft, n_mels=self.n_mels).T
        if cfg['data']['audio_feature'] == 'salsalite':
            # Initialize the spatial feature constants
            c = 343
            self.fmin_doa = cfg['data']['salsalite']['fmin_doa']
            self.fmax_doa = cfg['data']['salsalite']['fmax_doa']
            self.fmax_spectra = cfg['data']['salsalite']['fmax_spectra']

            self.lower_bin = np.int(np.floor(self.fmin_doa * self.n_fft / np.float(self.fs)))
            self.lower_bin = np.max((self.lower_bin, 1))
            self.upper_bin = np.int(np.floor(self.fmax_spectra * self.n_fft / np.float(self.fs)))
            self.cutoff_bin = np.int(np.floor(self.fmax_spectra * self.n_fft / np.float(self.fs)))
            assert self.upper_bin <= self.cutoff_bin, 'Upper bin for doa feature is higher than cutoff bin for spectrogram {}!'
            #GCC Features
            # Normalization factor for salsalite
            self.delta = 2 * np.pi * self.fs / (self.n_fft * c)
            self.freq_vector = np.arange(self.n_fft // 2 + 1)
            self.freq_vector[0] = 1
            self.freq_vector = self.freq_vector[None, :, None]

    def _spectrogram(self, audio_input, _nb_frames):
        _nb_ch = audio_input.shape[1]
        spectra = []
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self.n_fft, hop_length=self.hoplen,
                                        win_length=self.n_fft, window=self.cfg['data']['window'])
            spectra.append(stft_ch[:, :_nb_frames])
        return np.array(spectra).T

    def _get_logmel_spectrogram(self, linear_spectra):
        logmel_feat = np.zeros((linear_spectra.shape[0], self.n_mels, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self.mel_bank)
            logmel_spectra = librosa.power_to_db(mel_spectra)
            logmel_feat[:, :, ch_cnt] = logmel_spectra
        return logmel_feat
    
    def _get_gcc(self, linear_spectra):
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self.n_mels, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc[:, -self.n_mels//2:], cc[:, :self.n_mels//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat
    
    
    def _get_salsalite(self, linear_spectra):
        # Adapted from the official SALSA repo- https://github.com/thomeou/SALSA
        # spatial features
        phase_vector = np.angle(linear_spectra[:, :, 1:] * np.conj(linear_spectra[:, :, 0, None]))
        phase_vector = phase_vector / (self.delta * self.freq_vector)
        phase_vector = phase_vector[:, self.lower_bin:self.cutoff_bin, :]
        phase_vector[:, self.upper_bin:, :] = 0
        phase_vector = phase_vector.transpose((2, 0, 1))

        # spectral features
        linear_spectra = np.abs(linear_spectra)**2
        for ch_cnt in range(linear_spectra.shape[-1]):
            linear_spectra[:, :, ch_cnt] = librosa.power_to_db(linear_spectra[:, :, ch_cnt], ref=1.0, amin=1e-10, top_db=None)
        linear_spectra = linear_spectra[:, self.lower_bin:self.cutoff_bin, :]
        linear_spectra = linear_spectra.transpose((2, 0, 1))
        
        return np.concatenate((linear_spectra, phase_vector), axis=0) 