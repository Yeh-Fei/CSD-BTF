% Aurthor: Ye Fei
% E-mail: feiye@njust.edu.cn
% Create Date: May 29, 2026
% MATLAB R2021a

clear;  clc;
warning('off');
addpath(genpath(pwd));

if isempty(gcp('nocreate'))       
    parpool(maxNumCompThreads);    
end

SNRm = 40; SNRh = 30;
kernel_type = {'Gaussian', 'Uniform'};  
kernel_type = kernel_type{1};
sf = 4;
Tab=table;
Tab.name={'PSNR';'RMSE';'ERGAS';'SAM';'UIQI';'SSIM';'DD';'CC';'Time'};
rng(5,'twister');  
s=rng;

load('Lockwood.mat');
HSI=HSI/max(HSI(:));  MSI=MSI/max(MSI(:));
data.F=HSI;  data.M=MSI;  R=SRF;  
szF=size(data.F);
par.R = create_R(R);

%%  LR-HSI
shift=1;
if strcmp(kernel_type, 'Gaussian')
    sig=(1/(2*(2.7725887)/sf^2))^0.5;  kernel_length=9;  
    par.psf=fspecial('gaussian',kernel_length,sig);
    disp('Gaussian kernel generated.') ;
elseif strcmp(kernel_type, 'Uniform')
    par.psf=fspecial('average',sf);
    disp('Average kernel generated.') ;
else
    error('Wrong kernel type assigned !!!');
end
par.otf=psf2otf(par.psf, szF(1:2));
H=ConvC(data.F,par.otf);
data.H=H(1+shift:sf:end,1+shift:sf:end,:);
if SNRh ~= 0
    sigmah = sqrt(sum(data.H(:).^2)/(10^(SNRh/10))/numel(data.H));
else
    sigmah = 0;
end
data.H = data.H + sigmah * randn(size(data.H));
disp('LR-HSI generated.');
data.upH=imresize(data.H,sf);    

%%  HR-MSI
if SNRm ~= 0
    sigmam = sqrt(sum(data.M(:).^2)/(10^(SNRm/10))/numel(data.M));
else
    sigmam = 0;
end
rng(s);
data.M = data.M + sigmam * randn(size(data.M));
disp('HR-MSI generated.');
par.sf=sf;  par.shift=shift;  

%%  HSRSV
clearvars -except par data flag Tab SNRm SNRh kernel_type sigmah sigmam sf szF;
fprintf('==================================== HSRSV ==================================== \n');
par.K=20;         
par.iter=150;     
p={1/2,2/3};
SparseX={'1','mcp'};
SmoothA={'0','1'};
flag.Sum2One=0;  flag.SparseX=SparseX{2};  flag.SparseA=SmoothA{2};  flag.p=p{2};
Time.CSD_BTF=tic;
    [CSD_BTF, par] = HSRSV_Lockwood(data, par, flag);   % 
Time.CSD_BTF=toc(Time.CSD_BTF);

[Eval.PSNR, Eval.RMSE, Eval.ERGAS, Eval.SAM, Eval.UIQI, Eval.SSIM, ...
     Eval.DD, Eval.CC] = quality_assessment(double(im2uint8(data.F)), double(im2uint8(CSD_BTF.Fh)), 0, 1.0/par.sf);
expr=['Tab.', 'CSD_BTF',  ' =[Eval.PSNR; Eval.RMSE; Eval.ERGAS; Eval.SAM; Eval.UIQI; Eval.SSIM; Eval.DD; Eval.CC; Time.CSD_BTF];'];
eval(expr);
disp(Tab);

%%  Variable
F=data.F;  H=data.H;  M=data.M;  upH=data.upH;  R=par.R;  sf=par.sf;  psf=par.psf;  otf=par.otf;  shift=par.shift;
p=sqrt(diag(psf));
P1=conv2mat(p,szF(1:2));  P1=P1(1+shift:sf:end,:);   
P2=conv2mat(p',szF(1:2)); P2=P2(1+shift:sf:end,:);   

%%  BGS-GTF 
fprintf('============================================================= BGS-GTF ============================================================= \n');
BGS.Pr=cell(1,3);    BGS.Pr{1,1}=P1;    BGS.Pr{1,2}=P2;    BGS.Pr{1,3}=R;
BGS.rho=1e-3;   
BGS.nu=1.3;    BGS.lambda=1e0;    BGS.mu=1e0;    BGS.addatoms=[0,0];
BGS.turank=[80,100,50];    BGS.shape_re=[4,20,4,25,5,10];
BGS.MaxLoop=500;    BGS.IniLoop=100;
BGS.ConvTol=1e-3;
Time.BGS=tic;
    BGS.HSR=ReTF_playa(H, M, BGS.Pr, BGS.lambda, BGS.mu, BGS.rho, BGS.nu, BGS.turank, BGS.addatoms, BGS.shape_re, BGS.IniLoop, BGS.MaxLoop, BGS.ConvTol);
Time.BGS=toc(Time.BGS);

[Eval.PSNR, Eval.RMSE, Eval.ERGAS, Eval.SAM, Eval.UIQI, Eval.SSIM, ...
     Eval.DD, Eval.CC] = quality_assessment(double(im2uint8(F)), double(im2uint8(BGS.HSR)), 0, 1.0/par.sf);
expr=['Tab.', 'BGS_GTF',  ' =[Eval.PSNR; Eval.RMSE; Eval.ERGAS; Eval.SAM; Eval.UIQI; Eval.SSIM; Eval.DD; Eval.CC; Time.BGS];'];
eval(expr);
disp(Tab);

%%  FuVar 
fprintf('============================================================= FuVar ============================================================= \n');
p=20;  % 20
psfR=rot90(psf,2);
N1=size(H,1);  N2=size(H,2);  M1=size(M,1);  M2=size(M,2);  L=size(H,3); 
data_r = (reshape(H,size(H,1)*size(H,2),size(H,3))')';
M0 = vca_FuVar(data_r','Endmembers',p,'verbose','off');
A_FCLSU = FCLSU(data_r',M0)';
A_init = reshape(A_FCLSU',N1,N2,p);
A_init = imresize(A_init, sf);
A_init = reshape(A_init,M1*M2,p)';
FuVar.lambda_m = 1e-1;
FuVar.lambda_a = 1e-3;   
FuVar.lambda_1 = 1e-1;   
FuVar.lambda_2 = 1e0;
Psi_init = ones(L,p);
Time.FuVar=tic;
    [FuVarH,FuVarM,A,Psi]=FuVar_wrapper(H,M,A_init,M0,Psi_init,R,sf,psfR,FuVar.lambda_m,FuVar.lambda_a,FuVar.lambda_1,FuVar.lambda_2);
Time.FuVar=toc(Time.FuVar);
FuVarH = reshape(FuVarH',M1,M2,L);   FuVarM = reshape(FuVarM',M1,M2,L);

[Eval.PSNR, Eval.RMSE, Eval.ERGAS, Eval.SAM, Eval.UIQI, Eval.SSIM, ...
        Eval.DD, Eval.CC] = quality_assessment(double(im2uint8(F)), double(im2uint8(FuVarH)), 0, 1.0/sf);
expr=['Tab.', 'FuVar',  ' =[Eval.PSNR; Eval.RMSE; Eval.ERGAS; Eval.SAM; Eval.UIQI; Eval.SSIM; Eval.DD; Eval.CC; Time.FuVar];'];
eval(expr);
disp(Tab);

%%  GSFus 
fprintf('============================================================= GSFus ============================================================= \n');
GSFus.subspace=6;     
GSFus.lambda_1=1e-4;  
GSFus.lambda_2=1e-4; 
GSFus.mu=1e1;        
Time.GSFus=tic;
    GSFus.HSR = GSFus_main( H, M, R, otf, sf, F, GSFus.subspace, GSFus.lambda_1, GSFus.lambda_2, GSFus.mu);
Time.GSFus=toc(Time.GSFus);

[Eval.PSNR, Eval.RMSE, Eval.ERGAS, Eval.SAM, Eval.UIQI, Eval.SSIM, ...
     Eval.DD, Eval.CC] = quality_assessment(double(im2uint8(F)), double(im2uint8(GSFus.HSR)), 0, 1.0/sf);
expr=['Tab.', 'GSFus',  ' =[Eval.PSNR; Eval.RMSE; Eval.ERGAS; Eval.SAM; Eval.UIQI; Eval.SSIM; Eval.DD; Eval.CC; Time.GSFus];'];
eval(expr);
disp(Tab);

%%  BTD-Var 
fprintf('============================================================= BTD-Var ============================================================= \n');
BTD_Var.R=4;   
BTD_Var.L=14;   
BTD_Var.nIter=1e2;  
BTD_Var.lamda=1e2;
BTD_Var.r=BTD_Var.L*ones(1,BTD_Var.R);
Time.BTD_Var=tic;
    fprintf('Initializing ...\n');
    BTD_Var.C0=VCA(tens2mat(upH,3), 'Endmembers', BTD_Var.R, 'SNR', 0);
    A=BTD_Var.C0\tens2mat(upH,3);  A=mat2tens(A,[szF(1:2),BTD_Var.R],3);
    [BTD_Var.A0, BTD_Var.B0]=deal(zeros(0));
    for i=1:size(A,3)
        [u,s,v]=svd(A(:,:,i),'econ');
        u=u(:,1:BTD_Var.L);  s=s(1:BTD_Var.L,1:BTD_Var.L);  v=v(:,1:BTD_Var.L);
        BTD_Var.A0=[BTD_Var.A0,deal(u*s^(0.5))];  
        BTD_Var.B0=[BTD_Var.B0,deal(v*s^(0.5))]; 
    end
    [A_hat,B_hat,S,C_hat,C_tilde,cost,valid] = BTD_Var_main(F,H,M,P1,P2,R,BTD_Var.R,BTD_Var.B0,BTD_Var.C0,R*BTD_Var.C0,BTD_Var.nIter,BTD_Var.lamda);
    BTD_Var.HSR = ll1gen({A_hat,B_hat,C_hat},BTD_Var.r);
Time.BTD_Var=toc(Time.BTD_Var);

[Eval.PSNR, Eval.RMSE, Eval.ERGAS, Eval.SAM, Eval.UIQI, Eval.SSIM, ...
     Eval.DD, Eval.CC] = quality_assessment(double(im2uint8(F)), double(im2uint8(BTD_Var.HSR)), 0, 1.0/sf);
expr=['Tab.', 'BTD_Var',  ' =[Eval.PSNR; Eval.RMSE; Eval.ERGAS; Eval.SAM; Eval.UIQI; Eval.SSIM; Eval.DD; Eval.CC; Time.BTD_Var];'];
eval(expr);
disp(Tab);
