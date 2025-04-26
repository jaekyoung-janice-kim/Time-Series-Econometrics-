% econ612_ps4_codes

% 1
url = "https://fred.stlouisfed.org/";
c = fred(url);
series = 'IMPGSC1'; % aggregate level of U.S. quarterly imports, seasonally adjusted, real 2017 chained dollars, 1947Q1 to 2024Q4
dataI = fetch(c,series); 
ur_Idata = dataI.Data;

date_I = ur_Idata(:,1);
date = datetime(date_I,'ConvertFrom','datenum'); % create time index 

ur_I = ur_Idata(:,2);

% (a)
d = datetime(2005,12,31);
tau=sum(date<=d); T=length(date);

figure(1);
plot(date(1:tau), ur_I(1:tau));
xlabel('Time');
ylabel('US Quarterly Imports');
title('US Quarterly Imports from 1947 to 2005Q4');

figure(2);
plot(date(1:tau), log(ur_I(1:tau)));
xlabel('Time');
ylabel('Log of US Quarterly Imports');
title('Log of US Quarterly Imports from 1947 to 2005Q4');

% (c)
mdl_I=fitlm([linspace(1,tau,tau)],log(ur_I(1:tau)))

% (d)
[uI_p,uI_pi] = predict(mdl_I,linspace(tau+1,T,T-tau)','Prediction','observation','Alpha',0.1); %predict for t=tau+1,...,T

figure(3)
plot(date(1:tau),log(ur_I(1:tau)), date(tau+1:T), uI_p, '--', date(tau+1:T),uI_pi,':','LineWidth',1.5);
legend('Log(US Quarterly Imports)','Point forecast', '5%', '95%','Location','southeast');
title('#1(d): Log(US Quarterly Imports) Prediction from 2006 to 2024Q4');

% (e)
figure(4)
plot(date,log(ur_I), date(tau+1:T), uI_p,'--', date(tau+1:T),uI_pi,':','LineWidth',1.5);
legend('Log(US Quarterly Imports)', 'Point forecast', '5%', '95%','Location','southwest');
title('#1(e): Log(US Quarterly Imports) Comparison of Actual vs Prediction from 2006 to 2024Q4');

% (f)
figure(5)
plot(date(1:tau),ur_I(1:tau), date(tau+1:T), exp(uI_p), '--', date(tau+1:T),exp(uI_pi),':','LineWidth',1.5);
legend('US Quarterly Imports','Point forecast', '5%', '95%','Location','southeast');
title('#1(f): Levels US Quarterly Imports Prediction from 2006 to 2024Q4');

% (g)
figure(6)
plot(date,ur_I, date(tau+1:T), exp(uI_p),'--', date(tau+1:T),exp(uI_pi),':','LineWidth',1.5);
legend('US Quarterly Imports', 'Point forecast', '5%', '95%','Location','southwest');
title('#1(g): Levels US Quarterly Imports Comparison of Actual vs Prediction from 2006 to 2024Q4');

% (h)

mdl_I_full=fitlm([linspace(1,T,T)],log(ur_I(1:T)));
disp(mdl_I_full);

date_forecast = datetime(2025,1,1):calmonths(3):datetime(2028,12,31); 
future_quarters = (T+1):(T+16);

[uI_p_full,uI_pi_full] = predict(mdl_I_full,future_quarters','Prediction','observation','Alpha',0.1); 

predict(mdl_I_full, (1:T)')

% level scale  
figure(7)
plot(date,ur_I, date, exp(predict(mdl_I_full, (1:T)')), '--', date_forecast, exp(uI_p_full),'--', ...
    date_forecast,exp(uI_pi_full),':','LineWidth',1.5);
legend('Historical US Quarterly Imports', 'Historic Imports Trend','Forecasted Imports', '5%', '95%','Location','northwest');
xlabel('Year');
ylabel('US Quarterly Imports');
title('Forecast of US Quarterly Imports for the Next 4 Years');


% (i)
% Graph visualization and insepcted pattern 
% Plot residuals to check for any patterns
res_I_full=mdl_I_full.Residuals.Raw;
figure(8)
plot(date(1:T),res_I_full);
ylabel('Residuals');
