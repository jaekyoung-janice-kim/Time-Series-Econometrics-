clc; clear; close all;

% North Carolina State’s Total Number of Employees on nonfarm payrolls, seasonally adjusted[In thousands]. 

% cd('/Users/jaeky/Documents/MATLAB/econ612_matlab');
data = readtimetable("my_data2.xlsx");

nc = data.Value;
date = data.date;
T = length(date); 

% Plot the time series 
figure(1);
plot(date, nc);
title('NC Total (nonfarm payroll) Employee No. (Jan 1990-Feb 2025)');

% Plot autocorrealtion
acf = autocorr(nc,'NumLags',40);  % autocorr at h=0,1,..,q
whh = 2 * cumsum(acf.* acf)-1;
ci_bart(:,2) = 1.96 * sqrt(whh/T);
ci_bart(:,1) = - 1.96 * sqrt(whh/T);

figure(2); 
plot(linspace(0,40,40+1),ci_bart,'k','LineWidth',3);
hold on;
autocorr(nc, 'NumLags', 40, 'NumSTD', 0);
title('ACF of raw variable');

%% Check for Stationarity
% Conduct ADF test 

% Since time series is clearly trending upward, use 'TS' option, to show that it includes constant and linear trend
% p value is large for all possible lags of 0 to 12, so we fail to reject non stationarity 
% conclude nonstationarity, meaning differencing is necessary. 
[h, pValue] = adftest(nc, Model = 'TS', lags = 0:12)

% Apply first difference
y_diff = diff(nc);  
[h_diff, p_diff] = adftest(y_diff,  Model ='ARD')  % Use 'ARD' now (since there is intercept only, no trend)
[h_diff, p_diff] = adftest(y_diff, Model = "ARD", Lags = 0:12)
% h_diff=1 using y_diff. Thus, the differenced series is stationary.
% ARIMA(p,d,q) uses d=1. 

% Plot of difference variables 
figure(3);
plot(date(2:T), y_diff, date(2:T), zeros(T-1));
title('Differenced Employee No. (Feb 1990 - Feb 2025)');

%% Choose ARIMA model
% look at autocorrelation graph of differenced data(y_diff) first to look for AR/MA patterns
acf_diff = autocorr(y_diff,'NumLags',40);  % autocorr at h=0,1,..,q
whh_diff = 2 * cumsum(acf_diff.* acf_diff)-1;
ci_bart_diff(:,2) = 1.96 * sqrt(whh_diff/T);
ci_bart_diff(:,1) = - 1.96 * sqrt(whh_diff/T);

figure(4); 
plot(linspace(0,40,40+1),ci_bart_diff,'k','LineWidth',3);
hold on;
autocorr(y_diff, 'NumLags', 40, 'NumSTD', 0);
title('ACF of differenced variable');

% Check what AIC,BIC yields for p,q values 
results_AIC = [];  % rows: [AIC, p, q]
results_BIC = []; 

est = (date(2:T) >= datetime(1991,2,1));
size_check = 0; % to make sure we are using same sample size to calculate AIC/BIC

for p = 0:12
    for q = 0:12
        try
            model = arima(p, 0, q);  
            estModel = estimate(model, y_diff(est));  % use differenced y 

            if summarize(estModel).SampleSize ~= 409
                size_check = size_check + 1; % Increment the counter
            end

            AIC = summarize(estModel).AIC;
            BIC = summarize(estModel).BIC;
            
            results_AIC = [results_AIC; [AIC, p, q]];
            results_BIC = [results_BIC; [BIC, p, q]];

        catch
            % Estimation failed, skip this combination
            continue
        end
    end
end

size_check == 0

sorted_AIC = sortrows(results_AIC, 1);
sorted_BIC = sortrows(results_BIC, 1);

% top 5 performing AIC
for i = 1:5
    fprintf('AIC: %.4f, p: %d, q: %d\n', sorted_AIC(i, 1), sorted_AIC(i, 2), sorted_AIC(i, 3));
end

% AIC: 3951.9281, p: 0, q: 2
% AIC: 3952.1049, p: 2, q: 0
% AIC: 3953.7125, p: 0, q: 0
% AIC: 3953.9150, p: 1, q: 2
% AIC: 3953.9188, p: 0, q: 3

% top 5 performing BIC
for i = 1:5
    fprintf('BIC: %.4f, p: %d, q: %d\n', sorted_BIC(i, 1), sorted_BIC(i, 2), sorted_BIC(i, 3));
end

% BIC: 3961.7400, p: 0, q: 0
% BIC: 3967.3942, p: 0, q: 1
% BIC: 3967.4749, p: 1, q: 0
% BIC: 3967.9829, p: 0, q: 2
% BIC: 3968.1597, p: 2, q: 0

% Two possible cases: White Noise or ARMA(p,q)
% case 1. White Noise) Since most autocorrelations are not significantly different from and move around zero. 
% case 2. ARMA) Since looks like there is cutoff at q=2, it may be MA(2) model.  

% Check case 1. 
% It is very possible that y_diff is just white noise, and using an ARIMA may be overfitting. 

% y_diff is white noise 
OLSModel = fitlm(lagmatrix(y_diff, 1:12), y_diff);
[EstCov, se, coeff] = hac(OLSModel, 'display', 'off', Type = "HC");
EstCovsym = (EstCov + EstCov') / 2;
R = zeros(12,13); R(1:12, 2:13) = eye(12);
[h, pValue, stat, cValue] = waldtest(coeff(2:13,1), R, EstCovsym)

% y_diff^2 : squared differences are predictable 
y_diff2 = (y_diff-mean(y_diff)).*(y_diff-mean(y_diff));
OLSModel2 = fitlm(lagmatrix(y_diff2, 1:12), y_diff2);
[EstCov, se, coeff] = hac(OLSModel2, 'display', 'off', Type = "HC");
EstCovsym = (EstCov + EstCov') / 2;
R = zeros(12,13); R(1:12, 2:13) = eye(12);
[h, pValue, stat, cValue] = waldtest(coeff(2:13,1), R, EstCovsym)

% ACF graph of y_diff^2 
acf_diff2 = autocorr(y_diff2,'NumLags',40);  % autocorr at h=0,1,..,q
whh_diff2 = 2 * cumsum(acf_diff2.* acf_diff2)-1;
ci_bart_diff2(:,2) = 1.96 * sqrt(whh_diff2/T);
ci_bart_diff2(:,1) = - 1.96 * sqrt(whh_diff2/T);

figure(5); 
plot(linspace(0,40,40+1),ci_bart_diff2,'k','LineWidth',3);
hold on;
autocorr(y_diff2, 'NumLags', 40, 'NumSTD', 0);
title('ACF of differenced variable^2');


% Check case 2.  
% Use PLS to choose between AR(0,2), AR(2,0) 
% Data: y_diff (length T_diff)
T_diff = length(y_diff);
R = floor(0.05 * T_diff);  % Start-up period: first 5% of data
P = T_diff - R;            % Number of out-of-sample forecasts

% ---- ARMA(0,2) ----
e_t_tilda_02 = zeros(P,1);  % forecast errors
for t = R+1:T_diff
    model_02 = arima(0, 0, 2);
    try
        estModel_02 = estimate(model_02, y_diff(1:t-1), 'Display', 'off');
        y_t_hat_02 = forecast(estModel_02, 1, 'Y0', y_diff(1:t-1));
        e_t_tilda_02(t-R) = y_diff(t) - y_t_hat_02;
    catch
        e_t_tilda_02(t-R) = NaN;
    end
end
e_t_tilda_02 = e_t_tilda_02(~isnan(e_t_tilda_02));
P_02 = length(e_t_tilda_02);
pls_02 = sqrt(sum(e_t_tilda_02.^2)/P_02);

% ---- ARMA(2,0) ----
e_t_tilda_20 = zeros(P,1);  % forecast errors
for t = R+1:T_diff
    model_20 = arima(2, 0, 0);
    try
        estModel_20 = estimate(model_20, y_diff(1:t-1), 'Display', 'off');
        y_t_hat_20 = forecast(estModel_20, 1, 'Y0', y_diff(1:t-1));
        e_t_tilda_20(t-R) = y_diff(t) - y_t_hat_20;
    catch
        e_t_tilda_20(t-R) = NaN;
    end
end
e_t_tilda_20 = e_t_tilda_20(~isnan(e_t_tilda_20));
P_20 = length(e_t_tilda_20);
pls_20 = sqrt(sum(e_t_tilda_20.^2)/P_20);

% ---- Display results ----
fprintf('PLS for ARMA(0,2): %.4f\n', pls_02);
fprintf('PLS for ARMA(2,0): %.4f\n', pls_20);

%% Choose final model as (0,0,2) 

model = arima(0, 0, 2); 
estModel = estimate(model, y_diff);  % use original y (undifferenced)
summarize(estModel)

[res, v] = infer(estModel, y_diff);
y_diff_fitted = y_diff - res;

% Fitted vs actual data 
figure(6);
plot(1:(T-1), y_diff, 1:(T-1), y_diff_fitted);
title("Fitted vs. Actual y_diff values"); 

% actual nc values 
% Convert back to levels 
figure(7);
nc_fitted = nc(1) + [0; cumsum(y_diff_fitted)]
plot(1:T, nc, 1:T, (nc(1) + [0; cumsum(y_diff_fitted)]))
title("Fitted vs. Actual NC employee number values")


% Residuals Analysis 
% Are residuals white noise? 
% looks like white noise 
figure(8); 
plot(1:(T-1), res) 
title('Residuals of fitted values from ARMA(0,2) Model'); % COVID Outlier 
% use ACF plot to make sure they resemble white noise patterns 
acf_res = autocorr(res,'NumLags',40);  % autocorr at h=0,1,..,q
whh_res = 2 * cumsum(acf_res.* acf_res)-1;
ci_bart_res(:,2) = 1.96 * sqrt(whh_res/(T-1));
ci_bart_res(:,1) = - 1.96 * sqrt(whh_res/(T-1));
figure(9);
plot(linspace(0,40,40+1),ci_bart_res,'k','LineWidth',3);
hold on;
autocorr(res, 'NumLags', 40, 'NumSTD', 0);
title('ACF of residuals of fitted values from ARMA(0,2) Model');

% Residuals satisfy normality assumption? 
figure(10);
qqplot(res) % approximately follows normality for most of data, except for one outlier which corresponds to March 2020 (i.e. the signficant drop in the original data of April 2020 due to COVID)
title('QQplot of Residuals to check normality');

% Try the MA(2) on the train/test data and see if results match with actual forecasts
% Train MA(2) on all data except for the most recent last 3 observations. 
estModel_train = estimate(model, y_diff(1:T_diff-2));  
[YF_train,YMSE_train] = forecast(estModel_train,3,y_diff(1:T_diff-2));
% Comparison of results 
Trainresults = table( y_diff(T_diff-2:T_diff), YF_train(1:3), ...
    YF_train(1:3) - 1.96*sqrt(YMSE_train(1:3)), ...
    YF_train(1:3) + 1.96*sqrt(YMSE_train(1:3)), ...
    'VariableNames', {'Actual', 'PointForecast', 'Lower Bound', 'Upper Bound'});
disp(Trainresults)
% Not exactly right, but increase/decrease(positive/negative signs) are correct, and compared to the very large confidence interval, point forecasts are fairly close to the actual forecasts. 


%% FORECASTING 
% Using estModel (MA(2)), making 12 forecasts, using all y_diff data.  
[YF,YMSE] = forecast(estModel,12,y_diff); 

Forecasts_y_diff = table(YF, ...
    YF - 1.96*sqrt(YMSE), ...
    YF + 1.96*sqrt(YMSE), ...
    'VariableNames', {'PointForecast', 'Lower Bound', 'Upper Bound'});
disp(Forecasts_y_diff); 

figure(11); 
plot(1:T_diff, y_diff, ...
    T_diff+1: T_diff+12, YF, ...
    T_diff+1:T_diff+12, YF - 1.96*sqrt(YMSE), ...
    T_diff+1:T_diff+12, YF + 1.96*sqrt(YMSE))
legend('Actual Data', 'Point Forecast', '90% Lower CI', '90% Upper CI', 'Location', 'Best');
title("12 Period Forecast for Differenced Data"); 

figure(12); 
plot(T_diff-36:T_diff, y_diff(T_diff-36: T_diff), ...
    T_diff+1: T_diff+12, YF, ...
    T_diff+1:T_diff+12, YF - 1.96*sqrt(YMSE), ...
    T_diff+1:T_diff+12, YF + 1.96*sqrt(YMSE))
legend('Actual Data', 'Point Forecast', '90% Lower CI', '90% Upper CI', 'Location', 'Best');
title("12 Period Forecast for Differenced Data _ Magnified"); 

% Convert back to original data 

[YFD,YFDMSE] = forecast(estimate(arima(0,1,2), nc),12, nc)
nc_forecast_LB = YFD - 1.96*sqrt(YFDMSE); 
nc_forecast_UB = YFD + 1.96*sqrt(YFDMSE); 
disp(table((date(T)+calmonths(1:12))', YFD, nc_forecast_LB, nc_forecast_UB, ...
    'VariableNames', {'Forecast Month', 'PointForecast', 'Lower Bound', 'Upper Bound'}));

figure(13);
plot(date(1:T), nc, 'k', ... % Actual data
    date(T)+calmonths(1:12), YFD, ...
    date(T)+calmonths(1:12), nc_forecast_LB,  ...
    date(T)+calmonths(1:12), nc_forecast_UB);
legend('Actual Data', 'Point Forecast', '90% Lower CI', '90% Upper CI', 'Location', 'Best');
title("12 Period Forecast for No. of NC Employees");
xlabel('Date');
ylabel('No. of Employees');

figure(14);
plot(date(T-36:T), nc(T-36:T), 'k', ... % Actual data
    date(T)+calmonths(1:12), YFD, ...
    date(T)+calmonths(1:12), nc_forecast_LB,  ...
    date(T)+calmonths(1:12), nc_forecast_UB);
legend('Actual Data', 'Point Forecast', '90% Lower CI', '90% Upper CI', 'Location', 'Best');
title("12 Period Forecast for No. of NC Employees Magnified");
xlabel('Date');
ylabel('No. of Employees');






