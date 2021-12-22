%% Initialization
clc; clear all; close all;
dbstop if error
path = "D:\ML_Data\Dataset_3k\Testing_set";
img = [ "thermal_2019-06-26_1561563619_0_0.png"; "thermal_2019-06-27_1561637959_0_0.png"; ...
    "thermal_2019-06-28_1561717147_0_0.png"; "thermal_2019-06-29_1561770908_0_0.png"; ...
    "thermal_2019-06-29_1561838467_0_0.png"; "thermal_2019-07-05_1562282715_0_0.png"; ...
    "thermal_2019-07-07_1562456288_0_0.png"; "thermal_2019-08-02_1564733419_0_0.png"];

for i = 1:length(img)
    name_img = path + "\" + img(i);
    I = (imread(name_img)+1)/256-1; % 16 bit, goes from 0 to 65535, down to 8 bit
    figure
    surf(I)
    max(max(I))
end