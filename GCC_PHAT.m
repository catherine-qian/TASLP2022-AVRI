function [features,tau] = GCC_PHAT(x,fs, wlen,hop, nfft)


v = 343;
c = 343;
chno=1:size(x,2); % Skippig the mic channels                获得麦克风的通道数
nChan=length(chno); %                                       麦克风通道数
nP= nChan*(nChan-1)/2; % Number of total combination        一共有多少对的组合
kp=zeros(nP,2);                                             % 初始化 
cnt=1;                                                      % 通道的初始索引        

for i=1:nChan-1                                             % 对偶
    for j=i+1:nChan
        kp(cnt,:)=[chno(i),chno(j)];                        % [1,2;1,3;l,4 .....]
        cnt=cnt+1;
    end
end

f = (1:ceil((nfft)/2))'*fs/nfft; 


tau = zeros(nP,1);
feats1=zeros(nP,21);                          % 对偶 pair * 时间延迟   就是  6 * 51 = 306 

for kk=1:nP
    kk1=kp(kk,1);     % 本次第一个通道
    kk2=kp(kk,2);     % 本次第二个通道                       % 计算通道1的短时傅里叶变换
    %%
    tau_grid = (-5:0.5:5)/fs;
    %tau_grid = linspace(-d(kk,1),d(kk,1),11)/v;   % 秒为单位
    %%
    X1 = stft_v3(x(:,kk1),wlen,nfft);
    X2 = stft_v3(x(:,kk2),wlen,nfft);
    X1 = X1(2:end,:);
    X2 = X2(2:end,:);
    
    %% different method
    
    spec = phat_spec(cat(3,X1,X2), f, tau_grid);
    spec(isnan(spec)) = 1;
    
    spec = squeeze(sum(sum(spec,1),2));
    %%
    
    feats1(kk,:)=spec;
    nsrc = 1;
    %[peaks, ind] = findpeaks(spec, 'minpeakdistance',1, 'sortstr','descend');
    
    [~,ind] = max(spec,[],1);
    tau(kk,1) = tau_grid(ind(1:min(length(ind),nsrc)));
end
% feats1 = feats1';
% features=feats1(:);
features = feats1;
end


function spec = phat_spec(X, f, tau_grid)
[nbin,nfram] = size(X(:,:,1));
ngrid = length(tau_grid);
X1 = X(:,:,1);
X2 = X(:,:,2);

spec = zeros(nbin,nfram,ngrid);
P = X1.*conj(X2);
P = P./abs(P);
for ind = 1:ngrid,
    EXP = repmat(exp(-2*1i*pi*tau_grid(ind)*f),1,nfram);
    spec(:,:,ind) = real(P.*EXP);
end
end


function spec = nonlin_spec(X, f, alpha, tau_grid)
[nbin,nfram] = size(X(:,:,1));
ngrid = length(tau_grid);
X1 = X(:,:,1);
X2 = X(:,:,2);

spec = zeros(nbin,nfram,ngrid);
P = X1.*conj(X2);
P = P./abs(P);
for ind = 1:ngrid,
    EXP = repmat(exp(-2*1i*pi*tau_grid(ind)*f),1,nfram);
    spec(:,:,ind) = ones(nbin,nfram) - tanh(alpha*real(sqrt(2-2*real(P.*EXP))));
end

end


function X=stft_v3(x,wlen,nfft)
    win=sin((.5:wlen-.5)/wlen*pi).';
    % Zero-padding
    x = x(:);
    nsampl = size(x,1);
    nfram=ceil(nsampl/wlen*2);
    x=[x;zeros((nfram*wlen/2-nsampl),1)];
    % Pre-processing for edges
    x=[zeros(wlen/4,1);x;zeros(wlen/4,1)];
    swin=zeros((nfram+1)*wlen/2,1);
    for t=0:nfram-1
        swin(t*wlen/2+1:t*wlen/2+wlen)=swin(t*wlen/2+1:t*wlen/2+wlen)+win.^2;
    end
    swin=sqrt(wlen*swin);
    nbin=nfft/2+1;
    X=zeros(nbin,nfram);
    
    for t=0:nfram-1
        % Framing
        frame=x(t*wlen/2+1:t*wlen/2+wlen).*win./swin(t*wlen/2+1:t*wlen/2+wlen);
        % FFT
        fframe=fft(frame,nfft);
        X(:,t+1)=fframe(1:nbin);
    end
end
