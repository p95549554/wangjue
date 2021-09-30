clc;
clear;
Data = textread('D:\White_paper\test2.txt');
Data(:,4:6)=[];
gap = (6.2+6.2)/19;
tol = 0.030;
start_P = -6.2;
[m,n]=size(Data);
Z = [];
Data_z = [];
k = 1;
for l = 1:20
    for j = 1:20
        for i = 1:m
            if Data(i,1)>start_P+gap*(j-1)-tol && Data(i,1)<start_P+gap*(j-1)+tol && Data(i,2)>start_P+gap*(l-1)-tol && Data(i,2)<start_P+gap*(l-1)+tol
                Z(k) = Data(i,3);
                k = k+1;
            end
        end
        Data_z((l-1)*20+j) = mean(Z);
        Z = [];
        k = 1;
    end
end

%%
Data_z = (Data_z - Data_z(1)).*15;
Temp = xlsread(['D:\White_paper\python\Comsol_data_9x9\Terrain_2.csv']);
Temp = Temp(7:406,:);
Temp = Temp(:,4)-Temp(:,3);
Error = Data_z - Temp';
MSE = mse(Error)