%1 
cd('/Users/jaeky/Documents/MATLAB/econ612_matlab');
data1 = readtimetable("realgdpgrowth.xlsx");

figure(1); 
subplot(1,2,1); 
autocorr(data1.pce_durables,'NumLags',40);
title('PCE Durables Autocorrelation');
subplot(1,2,2); 
autocorr(data1.pce_nondurables,'NumLags',40);
title('PCE Nondurables Autocorrelation');

%2
url = 'https://fred.stlouisfed.org/';
c = fred(url);

dataS = fetch(c,'HOUSTS'); % US south region housing starts, seasonally adjusted
dataSNSA = fetch(c, 'HOUSTSNSA'); % US south region housing starts, not easonally adjusted
dataMW = fetch(c, 'HOUSTMW'); % US midwest region housing starts, seasonally adjusted
dataMWNSA = fetch(c, 'HOUSTMWNSA'); % US midwest region housing starts, not seasonally adjusted

ur_S = dataS.Data(:,2);
ur_SNSA = dataSNSA.Data(:,2);
ur_MW = dataMW.Data(:,2);
ur_MWNSA= dataMWNSA.Data(:,2);

figure(2); 
subplot(2,2,1);
autocorr(ur_S,'NumLags',40);
title('South Housing Starts Seasonally Adjusted Autocorrelation');

subplot(2,2,2);
autocorr(ur_SNSA,'NumLags',40);
title('South Housing Starts Not Seasonally Adjusted Autocorrelation');

subplot(2,2,3);
autocorr(ur_MW,'NumLags',40);
title('Midwest Housing Starts Seasonally Adjusted Autocorrelation');

subplot(2,2,4);
autocorr(ur_MWNSA,'NumLags',40);
title('Midwest Housing Starts Not Seasonally Adjusted Autocorrelation');

%3 
data3 = readtimetable("s&p.csv");
date = datetime(data3.date, 'InputFormat', 'MM/dd/yyyy');
T=length(date);

mdl_V=fitlm([linspace(1,T,T)],log(data3.volume(1:T)));
res_V=mdl_V.Residuals.Raw;

figure(3); 
subplot(1,2,1); 
plot(date(1:T),res_V);
ylabel('Residuals');
title('Residuals against Time');

subplot(1,2,2);
autocorr(res_V,'NumLags',40);
title('Residuals Autocorrelation');