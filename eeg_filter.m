function out = eeg_filter(y, fs, band)


%% notch filter
wo = 50/(fs/2); %notch frequency at 50Hz
q = 1; %quaity factor q=wo/bw
bw = wo/q;
[b,a]= iirnotch(wo,bw);
y_notch = filtfilt(b,a,y); %khong can chinh sua

%% band pass
order = 6;
Wn_low = band(1)/(fs/2);
Wn_high = band(2)/(fs/2);
Wn=[Wn_low Wn_high];
[z,p,k] = butter(order, Wn, 'bandpass');
sos = zp2sos(z,p,k);
y_bandpass = sosfilt(sos,y_notch);

out = y_bandpass;
