close all; clear; clc; format shortG;
%% INITIALISATION.
%stores the entire text file as a MATLAB table.
ds = tabularTextDatastore('set-a','FileExtensions','.txt');
outcomes = table2array(readtable('outcomes-a'));

%for testing purposes.
max_patients = 4000;
timelim = 2;

%'Parameters' is 1x42.
Parameters  = ["RecordID","Age","Gender","Height","ICUType","Weight","Albumin","ALP","ALT","AST","Bilirubin","BUN","Cholesterol","Creatinine","DiasABP","FiO2","GCS","Glucose","HCO3","HCT","HR","K","Lactate","Mg","MAP","MechVent","Na","NIDiasABP","NIMAP","NISysABP","PaCO2","PaO2","pH","Platelets","RespRate","SaO2","SysABP","Temp","TropI","TropT","Urine","WBC"];
Features = zeros(max_patients,length(Parameters));

%% BEHIND THE SCENES.
%cycles through patients ...
patient = 1;% pee1 = [NaN ; NaN]; pee2 = [NaN ; NaN];
while hasdata(ds)
    pData = read(ds); %read text file as MATLAB table.
    pDataNew = pData;
    
    %... time-limits the data ...
    for i=1:max(size(pData.Time))
        if str2double(pData.Time{i}(1:2)) <= timelim
            pDataNew(i,:) = pData(i,:);
        end
    end
        
    %... generates feature matrix ...
    %... for all general descriptors, adds it to the feature matrix ...
    %... for all other parameters, adds mean and variance to matrix.
    for i = 1:length(Parameters)
        d = pDataNew(strcmp(pDataNew.Parameter,Parameters(i)),:).Value;
        Features(patient,i) = mean(d);
    end
    
%     pee1 = [pee1 ; pData(strcmp(pData.Parameter,'Urine'),:).Value];
%     pee2 = [pee2 ; pDataNew(strcmp(pDataNew.Parameter,'Urine'),:).Value];

    patient = patient+1;
    if patient>max_patients
        break
    end
end

%% POST-PROCESSING.
%impute -1's and NaN's with column mean.
Features(Features == -1) = NaN;
M = nanmean(Features);
[r,c] = find(isnan(Features));
Features(isnan(Features)) = M(c);

outcomes(outcomes == -1) = NaN;

%normalise data appropriately.
% for i = 6:size(Features,2)
%     col = Features(:,i);
%     if quantile(col,0.99) > 100*(quantile(col,0.01)+1)
%         col = log(col);
%     end
%     
%     Features(:,i) = zscore(col);
% end

%clears any NaN-only columns.
Features = Features(:,any(~isnan(Features)));

%% STATISTICS ON CLUSTERS.
idx = kmeans(Features,5);
c1 = find(idx == 1); cluster1=Features(c1,:);
c2 = find(idx == 2); cluster2=Features(c2,:);
c3 = find(idx == 3); cluster3=Features(c3,:);
c4 = find(idx == 4); cluster4=Features(c4,:);
c5 = find(idx == 5); cluster5=Features(c5,:);

surv1 = outcomes(ismember(outcomes(:,1),cluster1(:,1)),5);
sum(isnan(surv1))/length(surv1)

surv2 = outcomes(ismember(outcomes(:,1),cluster2(:,1)),5);
sum(isnan(surv2))/length(surv2)

surv3 = outcomes(ismember(outcomes(:,1),cluster3(:,1)),5);
sum(isnan(surv3))/length(surv3)

surv4 = outcomes(ismember(outcomes(:,1),cluster4(:,1)),5);
sum(isnan(surv4))/length(surv4)

surv5 = outcomes(ismember(outcomes(:,1),cluster5(:,1)),5);
sum(isnan(surv5))/length(surv5)



%% VISUALISATION.
%the whole ass matrix.
size(Features)

%everyone's 'Urine' averages
% Features(:,find(Parameters == "Urine"))

%stacked histograms.
% figure; hold on; grid on;
% histogram(pee1,'BinWidth',30);
% histogram(pee2,'BinWidth',30); set(gca,'FontSize',14);
% lgd = legend('Unbound (48hr) Urine data','Bound (6hr) Urine data'); lgd.FontSize = 20;
% xlabel('Urine output (mL)','FontSize',20); ylabel('Frequency','FontSize',20);
% hold off;