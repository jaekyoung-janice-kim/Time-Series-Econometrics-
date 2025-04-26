clc; clear; close all;
%% Problem 1
url = 'https://fred.stlouisfed.org/';
c = fred(url);

data = fetch(c,'UNRATE'); % US unemployment rate
ur = data.Data(:,2);

date = data.Data(:,1);
date = datetime(date,'ConvertFrom','datenum');
T = length(date);

% Plug-In
mdl = fitlm(lagmatrix(ur,1),ur);
mdl

% Iterated
g = ur(T); 
p_iter = zeros(3,1);
for i = 1:3;
    p_iter(i,1) = predict(mdl,g);
    g = p_iter(i,1);
end

str = ["time", "forecast"];
[str; 
    string((date(T) + calmonths(1:3))') p_iter]

% Direct 
ur_f = zeros(3,1);
for i = 1:3;
    mdl = fitlm(lagmatrix(ur,i),ur)
    ur_p = predict(mdl,ur);
    ur_f(i,1) = ur_p(length(ur_p),1);
end 

ur_f



clc; clear; close all;
%% Problem 2
T = 240;
rng(123); % Set seed for reproducibility
epsilon = randn(T,1); % White noise ~ N(0,1)

% (a)
% (a)(ii) 
% Time series generation
y_d_a(1) = 1.33; % Set the initial value
for t = 2:T
    y_d_a(t) = 1 + 0.25 * y_d_a(t-1) + epsilon(t);
end
% Plot the time series
figure;
plot(1:T, y_d_a);
xlabel('Time');
ylabel('y');
title('(a) Time Series Plot of y');
% (a)(iii)
fitlm(lagmatrix(y_d_a,1), y_d_a)

% (b)
% (b)(ii) Time series generation
y_d_b(1) = 100; % Set the initial value
for t = 2:T
    y_d_b(t) = 10 + 0.9 * y_d_b(t-1) + epsilon(t);
end
% Plot the time series
figure;
plot(1:T, y_d_b);
xlabel('Time');
ylabel('y');
title('(b) Time Series Plot of y');
% (b)(iii)
fitlm(lagmatrix(y_d_b,1), y_d_b)

% (c)
% (c)(ii) Time series generation
y_d_c(1) = 0; % Set the initial value
for t = 2:T
    y_d_c(t) = - 0.5 * y_d_c(t-1) + epsilon(t);
end
% Plot the time series
figure;
plot(1:T, y_d_c);
xlabel('Time');
ylabel('y');
title('(c) Time Series Plot of y');
% (c)(iii)
fitlm(lagmatrix(y_d_c,1), y_d_c)

