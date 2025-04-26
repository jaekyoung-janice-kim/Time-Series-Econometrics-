clc; clear; close all;

cd('/Users/jaeky/Documents/MATLAB/econ612_matlab');
data = readtimetable("realgdpgrowth.xlsx");
%% Problem 1
y1 = data.pdi;
date = data.date;
T = length(date); 

mdl1 = fitlm(lagmatrix(y1,1:4), y1);
mdl1
[y1_p,y1_pi] = predict(mdl1,fliplr(y1(T-3:T)'),'Prediction','observation','Alpha',0.1);
y1_p
y1_pi

%% Problem 2
y2 = data.exports;

mdl2 = fitlm(lagmatrix(y2,1:4), y2);
mdl2
[y2_p,y2_pi] = predict(mdl2,fliplr(y2(T-3:T)'),'Prediction','observation','Alpha',0.1);
y2_p
y2_pi

%% Problem 3
y3 = data.pdi_residential;

y3_f = zeros(4,1);
y3_fi = zeros(4,2);

for i = 1:4;
    mdl = fitlm(lagmatrix(y3,i:i+3), y3);
    [y3_p, y3_pi] = predict(mdl,fliplr(y3(T-3:T)'),'Prediction','observation', 'Alpha', 0.1);
    y3_f(i,1) = y3_p;
    y3_fi(i,:) = y3_pi; 
end 

h = date(end, 1) + calquarters(1:4);
plot(date(1:T), y3(1:T), h, y3_f, h, y3_fi,'LineWidth',1.5);
legend('Exports','Point forecast', '5%', '95%','Location','northeast');
title('#3: PDI_Residentail, AR(4) model, 2024q4-2025q3 Predictions');

plot(date(T-15:T), y3(T-15:T), h, y3_f, h, y3_fi,'LineWidth',1.5);
legend('Exports','Point forecast', '5%', '95%','Location','north');
title('#3 (Detailed plot): PDI_Residentail, AR(4) model, 2024q4-2025q3 Predictions');


%% Problem 4

url = "https://fred.stlouisfed.org/";
c = fred(url);
series = 'MRTSSM45111USN'; % monthly retail sales of sporting goods stores 
dataS = fetch(c,series); 
S_data = dataS.Data;

date4 = S_data(:,1);
date4 = datetime(date4,'ConvertFrom','datenum'); % create time index 

y4 = S_data(:,2);

% (a)
T4=length(date4);

plot(date4(1:T4), y4(1:T4));
title('#4: Monthly retail sales of sporting goods stores, 1992m1-2025m1');

% (b)
% TREND 
% There seems to be an exponential trend so we log the y variable. 
ly4 = log(y4);
mdl_trend = fitlm([linspace(1,T4,T4)],ly4);
ly4_fitted = predict(mdl_trend,linspace(1,T4,T4)');

% The logged varaible seems to have a linear trend 
plot(date4(1:T4), ly4(1:T4), date4(1:T4), ly4_fitted);
xlabel('Time');
ylabel('Log(Sales)');
title('Log(Sales) Linear Trend');

% After accounting for linear trend,
% notice that residuals fluctuate around 0
% and there is sign of seasonality. 
res_trend=mdl_trend.Residuals.Raw;
plot(date4(1:T4), res_trend, date4(1:T4), zeros(T4));
ylabel('Residuals');
title('Residuals from Log(Sales) Linear Trend');

% SEASONALITY 
% Regressing residuals on 12 seasonal dummies
mdl_season = fitlm(dummyvar(month(date4)),res_trend, 'Intercept',false);
mdl_season; 
% shows that most coefficients are significant

% Residuals After Seasonal Regression
res_season=mdl_season.Residuals.Raw;
plot(date4(1:T4), res_season);
ylabel('Residuals');
title('Residuals After Seasonal Regression');
% Residuals do not look like white noise, and seems to be some persistence.

% FULL ESTIMATION
mdl4 = fitlm([ ...
    linspace(1,T4,T4)', ... % trend 
    dummyvar(month(date4)), ...% seasonality 
    lagmatrix(ly4, 1:12) ...% AR(12) cyclicality
    ], ...
    ly4, ...
    'Intercept', false ...% No intercept, all seasonal dummies are included
    );
mdl4; 

fitted_vals= predict( ...
    mdl4, ...
    [linspace(1, T4, T4)', dummyvar(month(date4)), lagmatrix(ly4, 1:12)] ...
    );

% Plot of Log(Sales) Fitted Values
plot(date4(1:T4), ly4, date4(1:T4), fitted_vals); 
legend('log(sales)','Fitted values','Location','northeast');
title('Log(Sales) Fitted Values');
% Seems to match well

% (c)
ly4_f = zeros(12, 1);
ly4_fi = zeros(12, 2);

month_dummy_forecast = dummyvar(month(date4(T4) + calmonths(1:12))); % 12 x 12

for i = 1:12
    % Fit model: Trend + Seasonal Dummies + Lags(i:i+11)
    mdl = fitlm(...
        [linspace(1, T4, T4)', dummyvar(month(date4)), lagmatrix(ly4, i:i+11)], ...
        ly4, ...
        'Intercept', false ... % No intercept (dummies cover it)
    );
    
    % Prepare forecast input:
    forecast_input = [...
        T4 + i, ... % Time trend 
        month_dummy_forecast(i, :), ... % Seasonal dummy
        fliplr(ly4(T4-11:T4)') ... % Cyclical 
    ]; 
    
    % Predict
    [ly4_p, ly4_pi] = predict(...
        mdl, ...
        forecast_input, ...
        'Prediction', 'observation', ...
        'Alpha', 0.1 ...
    );
    
    ly4_f(i, 1) = ly4_p;
    ly4_fi(i, :) = ly4_pi;
end

% (d)
ly4_f;
ly4_fi;

exp(ly4_f);
exp(ly4_fi);

h4 = date4(T4) + calmonths(1:12);

% Plot Log(Sales) 12 month forecast
plot(date4(T4-36:T4), ly4(T4-36:T4), h4, ly4_f, h4, ly4_fi,'LineWidth',1.5);
legend('log(Sales)','Direct forecast', 'Direct lower forecast interval', ...
    'Direct upper forecast interval','Location','northeast');
title('Log(Sales) 12-month Forecast');

% Plot Sales(Levels) 12 month forecast
plot(date4(T4-36:T4), exp(ly4(T4-36:T4)), h4, exp(ly4_f), h4, exp(ly4_fi),'LineWidth',1.5);
legend('Sales','Direct forecast', 'Direct lower forecast interval', ...
    'Direct upper forecast interval','Location','northeast');
title('Sales 12-month Forecast');

