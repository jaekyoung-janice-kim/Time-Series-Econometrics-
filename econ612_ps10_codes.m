clc; clear; close all;

cd('/Users/jaeky/Documents/MATLAB/econ612_matlab');

%% Problem 1
data1 = readtimetable("realgdpgrowth.xlsx");
pdi = data1.pdi;
date.pdi = data1.date;
est = (date.pdi >= datetime(1948,4,1));

X1=lagmatrix(pdi,1);
mdl1=fitlm(X1(est,:),pdi(est));
AIC(1,1)=mdl1.ModelCriterion.AIC;

X2=lagmatrix(pdi,1:2);
mdl2=fitlm(X2(est,:),pdi(est));
AIC(2,1)=mdl2.ModelCriterion.AIC;

X3=lagmatrix(pdi,1:3);
mdl3=fitlm(X3(est,:),pdi(est));
AIC(3,1)=mdl3.ModelCriterion.AIC;

X4=lagmatrix(pdi,1:4);
mdl4=fitlm(X4(est,:),pdi(est));
AIC(4,1)=mdl4.ModelCriterion.AIC;

display(AIC);
% Since AIC is smallest when lag=4, choose AR(4) model 

%% Problem 2
% 2(a)
data2 = readtimetable("s&p.csv");
date.adjClose = data2.date; 
adjClose = data2.adjClose; 
returns = 100 * (adjClose - lagmatrix(adjClose,1)) ./ lagmatrix(adjClose,1);

plot(date.adjClose(2:length(adjClose)), returns(2:length(adjClose))); 
xlabel('Time');
ylabel('adjClose Returns');
title('adjClose Returns plot');


% 2(b)
% Fit AR(2) by regressing r_t on r_{t-1} and r_{t-2}
mdl = fitlm(lagmatrix(returns,1:2), returns);

% Classical Wald test of H0: AR(1) and AR(2) coefficients = 0
% The model has 3 estimated coefficients (intercept, AR(1), AR(2)).
% R picks out the last two (the AR lags).
R = [0 1 0
     0 0 1];

[h_classical, pValue_classical, stat_classical, cValue_classical] = ...
    waldtest(mdl.Coefficients.Estimate(2:3,1), R, ...
             mdl.CoefficientCovariance)

% Heteroskedasticity-consistent (HAC) Wald test
[EstCov, se, coeff] = hac(mdl, 'display','off', 'Type','HC');
[h_hac, pValue_hac, stat_hac, cValue_hac]=waldtest(coeff(2:3,1), R, EstCov)

% Which one is appropriate? 
% In many cases—especially with financial return data—error variance can change over time (heteroskedasticity), 
% so the heteroskedasticity‐consistent (HAC) Wald test is generally more appropriate. 
% The classical Wald test assumes constant variance, which may not hold for returns, potentially leading to misleading inference.


%% Problem 3
exports = data1.exports;
mdl3 = fitlm(lagmatrix(exports,1:4), exports);

% Classical standard errors
mdl3
% Robust Standard Errors
[EstCov,se,coeff]=hac(mdl3,'display','off',Type="HC")

% Yes, there is a difference between classical and robust standard errors.
% e.g. for x1, classical se = 0.057361, but robust se = 0.1004. 
 