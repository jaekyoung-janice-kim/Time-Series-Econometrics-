clear all; 
url = 'https://fred.stlouisfed.org/';
c = fred(url);

dataM = fetch(c,'LNU04000025'); %men's unemployment rate 
dataW = fetch(c,'LNU04000026'); %women's unemployment rate

data_wM = fetch(c,'LNU04000028'); %white men's unemployment rate
data_wW = fetch(c,'LNU04000029'); %white women's unemployment rate

data_bM = fetch(c,'LNU04000031'); %black men's unemployment rate
data_bW = fetch(c,'LNU04000032'); %black women's unemployment rate

ur_M = dataM.Data(:,2);
ur_W = dataW.Data(:,2); 

ur_wM = data_wM.Data(:,2);
ur_wW = data_wW.Data(:,2);

ur_bM = data_bM.Data(:,2);
ur_bW = data_bW.Data(:,2);

% 1(a) 
date1 = dataM.Data(:,1);
date1 = datetime(date1,'ConvertFrom','datenum');
A1=month(date1);
mo1=dummyvar(A1);

mdl_M = fitlm(mo1,ur_M, 'Intercept',false); disp(mdl_M);
mdl_W = fitlm(mo1,ur_W, 'Intercept',false); disp(mdl_W);

figure(1);
plot(1:12,mdl_M.Coefficients.Estimate, 1:12, mdl_W.Coefficients.Estimate);
legend('Men','Women','Location','northeast');
ylabel('Unemployment Rate - 20 Yrs. & over');
xlabel('Month');
title('#1(a): Unemployment Seasonality Fitted Values Men/Women Comparison');
grid off;

% 1(b) 
date2 = data_wM.Data(:,1);
date2 = datetime(date2, 'ConvertFrom', 'datenum'); 
A2=month(date2);
mo2=dummyvar(A2);

mdl_wM = fitlm(mo2,ur_wM, 'Intercept',false); 
mdl_wW = fitlm(mo2,ur_wW, 'Intercept',false); 

date3 = data_bM.Data(:,1);
date3 = datetime(date3, 'ConvertFrom', 'datenum'); 
A3=month(date3);
mo3=dummyvar(A3);

mdl_bM = fitlm(mo3,ur_bM, 'Intercept',false); 
mdl_bW = fitlm(mo3,ur_bW, 'Intercept',false); 

figure(2);
plot(1:12,mdl_wM.Coefficients.Estimate, 1:12, mdl_wW.Coefficients.Estimate, ...
    1:12,mdl_bM.Coefficients.Estimate, 1:12, mdl_bW.Coefficients.Estimate);
legend('White Men','White Women', 'Black Men', 'Black Women','Location','northeast');
ylabel('Unemployment Rate - 20 Yrs. & over');
xlabel('Month');
title('#1(b): Unemployment Seasonality Fitted Values White/Black Men/Women Comparison');
grid off;
