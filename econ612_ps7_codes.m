clc; clear; close all;
%% Problem 1

T = 1000; % Number of observations
rng(123); % Set seed for reproducibility
epsilon = randn(T,1); % White noise ~ N(0,1)

% Model (a): White noise
y_a = epsilon;

% Model (b): MA(1)
y_b = epsilon + 0.8 * [0; epsilon(1:end-1)];

% Model (c): MA(2)
y_c = epsilon - 0.6 * [0; epsilon(1:end-1)] + 0.4 * [0; 0; epsilon(1:end-2)];

% Model (d): AR(1)
y_d = zeros(T,1);
for t = 2:T
    y_d(t) = 0.5 * y_d(t-1) + epsilon(t);
end

q=40; % number of lags in the plot 

% Autocorrelation Functions for each series

acf_a = autocorr(y_a,'NumLags',q);  % autocorr at h=0,1,..,q
whh_a = 2 * cumsum(acf_a.* acf_a)-1;
ci_bart_a(:,2) = 1.96 * sqrt(whh_a/T);
ci_bart_a(:,1) = - 1.96 * sqrt(whh_a/T);

acf_b = autocorr(y_b,'NumLags',q); 
whh_b = 2 * cumsum(acf_b.* acf_b)-1;
ci_bart_b(:,2) = 1.96 * sqrt(whh_b/T);
ci_bart_b(:,1) = - 1.96 * sqrt(whh_b/T);

acf_c = autocorr(y_c,'NumLags',q); 
whh_c = 2 * cumsum(acf_c.* acf_c)-1;
ci_bart_c(:,2) = 1.96 * sqrt(whh_c/T);
ci_bart_c(:,1) = - 1.96 * sqrt(whh_c/T);

acf_d = autocorr(y_d,'NumLags',q); 
whh_d = 2 * cumsum(acf_d.* acf_d)-1;
ci_bart_d(:,2) = 1.96 * sqrt(whh_d/T);
ci_bart_d(:,1) = - 1.96 * sqrt(whh_d/T);


% Create figure for ACF plots
fig1=figure;

subplot(2,2,1);
plot(linspace(0,q,q+1),ci_bart_a,'k','LineWidth',3);
hold on;
autocorr(y_a, 'NumLags', q, 'NumSTD', 0);
title('ACF: White Noise (a)');
hold off;

subplot(2,2,2); 
plot(linspace(0,q,q+1),ci_bart_b,'k','LineWidth',3);
hold on;
autocorr(y_b, 'NumLags', q, 'NumSTD', 0);
title('ACF: MA(1) (b)');
hold off;

subplot(2,2,3); 
plot(linspace(0,q,q+1),ci_bart_c,'k','LineWidth',3);
hold on;
autocorr(y_c, 'NumLags', q, 'NumSTD', 0);
title('ACF: MA(2) (c)');
hold off;

subplot(2,2,4); 
plot(linspace(0,q,q+1),ci_bart_d,'k','LineWidth',3);
hold on;
autocorr(y_d, 'NumLags', q, 'NumSTD', 0);
title('ACF: AR(1) (d)');
hold off;

saveas(fig1,'1-1.png');

% Estimate MA(2) model
Mdl = arima(0,0,2);

disp('Estimated MA(2) for White Noise model:');
Est_a = estimate(Mdl, y_a);

disp('Estimated MA(2) for MA(1) model:');
Est_b = estimate(Mdl, y_b);

disp('Estimated MA(2) for MA(2) model:');
Est_c = estimate(Mdl, y_c);

disp('Estimated MA(2) for AR(1) model:');
Est_d = estimate(Mdl, y_d);



clc; clear; close all;
%% Problem 2
% Loading the dataset
filename = 'realgdpgrowth.xlsx';
datasp = readtable(filename);

% Extract Non Residential Investment and Defense series
pdi_nonresidential = datasp.pdi_nonresidential;
T=length(pdi_nonresidential);

% Number of lags to compute autocorrelation
numLags = 40;
acf_nonresidential=autocorr(pdi_nonresidential, 'NumLags', numLags);
whh_nonresidential = 2*cumsum(acf_nonresidential.*acf_nonresidential)-1;
ci_bart_nonresidential(:,2)=1.96*sqrt(whh_nonresidential/T);
ci_bart_nonresidential(:,1)=-1.96*sqrt(whh_nonresidential/T);

X = linspace(0,numLags,numLags+1);
fig2=figure;
% Plot of autocorrelation functions
plot(X,ci_bart_nonresidential,'k','LineWidth',3);
hold on;
autocorr( pdi_nonresidential, 'NumLags', numLags, 'NumSTD', 0);
title('Non-residential Investment');
hold off;

saveas(fig2,'2.png');


% Estimate MA(1) - MA(4) models
Mdl1 = arima(0,0,1); 
disp('Estimated MA(1):');
Est_1 = estimate(Mdl1, pdi_nonresidential);

Mdl2 = arima(0,0,2); 
disp('Estimated MA(2):');
Est_2 = estimate(Mdl2, pdi_nonresidential);

Mdl3 = arima(0,0,3); 
disp('Estimated MA(3):');
Est_3 = estimate(Mdl3, pdi_nonresidential);

Mdl4 = arima(0,0,4); 
disp('Estimated MA(4):');
Est_4 = estimate(Mdl4, pdi_nonresidential);

