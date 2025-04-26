clc; clear; close all;

cd('/Users/jaeky/Documents/MATLAB/econ612_matlab');

%% Problem 1
data1 = readtimetable("realgdpgrowth.xlsx"); 

gdp = data1.gdp; 
t3month = data1.TB3MS;
t1year = data1.GS1;
t10year = data1.GS10;
BAA = data1.BAA;
AAA = data1.AAA;

% 1(a) 
spread1 = t1year - t3month;      % (i) Difference between 1-year and 3-month T-bill rates. 
                                 % Spread is measured by Long-Short, and negative spread predicts a future recession. 
spread2 = t10year - t3month;     % (ii) Difference between 10-year and 3-month T-bill rates. 
corporate = BAA - AAA;           % (iii) Difference between BAA and AAA corporate bond rates. 
                                 % Junk bond spread = Rate on Low-Grade – Rate on High-grade as an indicator of recession risk. 
dt3 = t3month - lagmatrix(t3month, 1); % (iv) First difference of 3-month T-bill rate
dt12 = t1year - lagmatrix(t1year, 1);  % (v) First difference of 1-year T-bill rate


% 	spread1: Measures the term spread between the 1-year and 3-month Treasury rates — a proxy for expectations about future short-term interest rates.
%	spread2: Measures the term spread between the 10-year and 3-month Treasury rates — often used as an indicator of the yield curve’s slope and potential economic outlook.
%	corporate: Reflects the credit spread between BAA-rated and AAA-rated corporate bonds — a measure of perceived credit risk in the economy.
%	dt3: Captures the quarterly change in the 3-month T-bill rate — indicating short-term interest rate dynamics.
%	dt12: Captures the quarterly change in the 1-year T-bill rate — showing medium-term interest rate adjustments.
% spread 1,2, corporate are all leading indicators for unemployment rate. 

% 1(b) 
date1 = data1.date;
date_filter = (date1 >= datetime(1954,4,1)) & (date1 <= datetime(2024,9,30));

% Variable list
x_list = {dt3, dt12, spread1, spread2, corporate};
x_names = {'dt3', 'dt12', 'spread1', 'spread2', 'corporate'};

% Filter GDP once
gdp_filtered = gdp(date_filter);

fprintf('Granger Causality Test (H0: variable does NOT Granger-cause GDP)\n');
fprintf('------------------------------------------------------------------\n');

for i = 1:length(x_list)
    x_filtered = x_list{i}(date_filter);
    
    % Fit model and get HAC standard errors
    OLSmdl = fitlm([lagmatrix(x_filtered, 1:3) lagmatrix(gdp_filtered, 1:3)], gdp_filtered);
    [EstCov, se, coeff] = hac(OLSmdl, 'display', 'off', Type="HC");

    R = zeros(3,1+3+3); R(1:3,2:3+1)=eye(3);
    
    % Perform Wald test
    EstCovsym = (EstCov + EstCov') / 2;
    [h,pValue,stat,cValue]=waldtest(coeff(2:3+1,1),R,EstCovsym);
    
    % Display results
    fprintf('%-10s → GDP | H0 rejected = %d | p-value = %.4f\n | stat = %.4f\n | cValue = %.2f\n', ...
        x_names{i}, h, pValue, stat, cValue);

end

% 1(c)
% To forecast GDP, use dt3, dt12, spread2, and corporate — they are statistically significant in Granger causality tests 
% and likely offer complementary signals about short-term interest rate changes, term spreads, and credit risk spreads, 
% which are all relevant macro predictors of economic activity.

% 1(d)
dt3_filtered = dt3(date_filter);
dt12_filtered = dt12(date_filter);
spread2_filtered = spread2(date_filter);
corporate_filtered = corporate(date_filter);

Tur = length(date1);
gdp_f=zeros(4,1); gdp_fi=zeros(4,2);

for h=1:4;
X=lagmatrix(gdp_filtered,h);
Xdt3=lagmatrix(dt3_filtered,h:h+2);
Xdt12=lagmatrix(dt12_filtered,h:h+2);
Xspread2=lagmatrix(spread2_filtered,h:h+2);
Xcorporate=lagmatrix(corporate_filtered,h:h+2);

mdl=fitlm([X Xdt3 Xdt12 Xspread2 Xcorporate],gdp_filtered);
[gdp_p,gdp_pi] = predict(mdl,[fliplr(gdp_filtered(Tur-3+1:Tur)') ...
    fliplr(dt3_filtered(Tur-3+2:Tur)') fliplr(dt12_filtered(Tur-3+2:Tur)') ...
    fliplr(spread2_filtered(Tur-3+2:Tur)') fliplr(corporate_filtered(Tur-3+2:Tur)')], ...
    'Prediction','observation','Alpha',0.1);
gdp_f(h,1)=gdp_p;
gdp_fi(h,:)=gdp_pi;
end









% Prepare filtered inputs
dt3_filtered = dt3(date_filter);
dt12_filtered = dt12(date_filter);
spread2_filtered = spread2(date_filter);
corporate_filtered = corporate(date_filter);
gdp_filtered = gdp(date_filter);

Tur = length(gdp_filtered);  % most recent time point
gdp_f = zeros(4,1);          % point forecasts
gdp_fi = zeros(4,2);         % prediction intervals

for h = 1:4  % forecast horizons 1 to 4
    % Use 3 lags of each variable
    X_gdp       = lagmatrix(gdp_filtered, 1:3);
    X_dt3       = lagmatrix(dt3_filtered, 1:3);
    X_dt12      = lagmatrix(dt12_filtered, 1:3);
    X_spread2   = lagmatrix(spread2_filtered, 1:3);
    X_corporate = lagmatrix(corporate_filtered, 1:3);

    % Combine into one matrix
    X_all = [X_gdp, X_dt3, X_dt12, X_spread2, X_corporate];

    % Fit model to aligned data (remove rows with NaNs)
    valid_rows = all(~isnan(X_all), 2) & ~isnan(gdp_filtered);
    mdl = fitlm(X_all(valid_rows, :), gdp_filtered(valid_rows));

    % Build input vector for forecast
    x_input = [
        fliplr(gdp_filtered(Tur-2:Tur)') ...
        fliplr(dt3_filtered(Tur-2:Tur)') ...
        fliplr(dt12_filtered(Tur-2:Tur)') ...
        fliplr(spread2_filtered(Tur-2:Tur)') ...
        fliplr(corporate_filtered(Tur-2:Tur)') ];

    % Direct forecast for horizon h
    [gdp_p, gdp_pi] = predict(mdl, x_input, ...
        'Prediction', 'observation', 'Alpha', 0.1);

    % Store forecast and 90% interval
    gdp_f(h) = gdp_p;
    gdp_fi(h, :) = gdp_pi;
end


figure(2);
plot(date1(date1 >= datetime(1954,4,1)),gdp_filtered);
hold on;
plot(date1(Tur)+calquarters(1:4),gdp_f,'LineWidth',1.5);
hold on;
plot(date1(Tur)+calquarters(1:4),gdp_fi,'LineWidth',1.5);
legend('GDP','Direct forecast','Direct lower forecast interval', ...
    'Direct upper forecast interval','Location','northwest');















clc; clear; close all;
%% Problem 2
url = "https://fred.stlouisfed.org/";
c = fred(url);
seriesA = 'AAA'; seriesB = 'BAA';
dataA = fetch(c,seriesA); dataB = fetch(c,seriesB); 
A_data = dataA.Data; B_data = dataB.Data;

date2 = A_data(:,1);
date2 = datetime(date2,'ConvertFrom','datenum'); % create time index 

high_returns = A_data(:,2);
low_returns = B_data(:,2);
spread = low_returns - high_returns; 

% Check whether returns on the highest rated (AAA) have unit roots 
[h,pValue,stat,cValue]=adftest(high_returns, Model="ARD", Alpha=[0.01 0.05 0.1], lags=12)
% p value is 0.4172 > 0.05, so we do not reject the hypothesis of unit root. 
% Higher rated bonds are non stationary.

% Check whether returns on the lower rated (BAA) bonds have unit roots 
[h,pValue,stat,cValue]=adftest(low_returns, Model="ARD", Alpha=[0.01 0.05 0.1], lags=12)
% p value is 0.1928 > 0.05, so we do not reject the hypothesis of unit root. 
% Lower rated bonds are non stationary.

% Check whether the high-yield (junk) bond spread has a unit root.
[h,pValue,stat,cValue]=adftest(spread, Model="ARD", Alpha=[0.01 0.05 0.1], lags=12)
% p value is 1.0e-03 * 1.000 < 0.05, so we do reject the hypothesis of unit root. 
% We find evidence that the spread is stationary.

clc; clear; close all;
%% Problem 4
% Set the number of time periods
T = 500;

% Generate random shocks: εt, ut, vt ~ i.i.d. N(0, 1)
epsilon = randn(T, 1); % εt
u = randn(T, 1);       % ut
v = randn(T, 1);       % vt

% Initialize xt and yt
x = zeros(T, 1);
y = zeros(T, 1);

% Simulate xt and yt
for t = 1:T
    x(t) = sum(epsilon(1:t)) + u(t); % xt = sum(εi from i=1 to t) + ut
    y(t) = 0.4 * sum(epsilon(1:t)) + v(t); % yt = 0.4 * sum(εi from i=1 to t) + vt
end

fitlm(y,x)
% It appears that x's coefficient is large and signfiicant. 