% econ612 ps3
cd('/Users/jaeky/Documents/MATLAB/econ612_matlab');
data = readtimetable("realgdpgrowth.xlsx");

% 3(a)
% Plot actual data
plot(data.date, data.pce_nondurables,'b');
hold on;
% Fit model and calculate in-sample fitted values
y = data.pce_nondurables;
mdl = fitlm(ones(length(y), 1), y, "Intercept", false);
X = ones(length(y), 1);
y_fv = predict(mdl, X, "Alpha",0.1);

% Add in-sample fitted line
plot(data.date, y_fv);
hold on; 

xlabel('Date');
ylabel('pce_nondurables');
title('PCE Nondurables');
legend('Actual Data', 'Fitted Line', 'Location', 'Best');

% 3(b) 
% Generate out-of-sample predictions
h = data.date(end, 1) + calquarters(1:4);  % Add 4 quarters for forecast horizon
X2 = ones(length(h), 1);  % Design matrix for out-of-sample predictions
[y_pnt, y_ci] = predict(mdl, X2, 'Prediction', 'Observation','Alpha', 0.1);
disp(y_pnt);
disp(y_ci(:,1)); 
disp(y_ci(:,2));

% Add out-of-sample prediction line
plot(h, y_pnt,'LineWidth', 1);  % Point forecast 

% Add confidence intervals for out-of-sample predictions
plot(h, y_ci(:, 1), 'LineWidth', 1);  % Lower 90% CI 
plot(h, y_ci(:, 2), 'LineWidth', 1);  % Upper 90% CI 
legend('Actual Data', 'Fitted Values', 'Point Forecast', '90% Lower CI', '90% Upper CI', 'Location', 'Best');