%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sensor_data_path = fullfile("MobileSensorData", "sensorlog_20221225_173307.mat");
csv_data_path = fullfile("imu_csv_files");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% d = load(sensor_data_path);
% TT = readtimetable(sensor_data_path);
df=importdata(sensor_data_path);
acc = df.Acceleration;
ang = df.AngularVelocity;
[filepath,name,ext] = fileparts(sensor_data_path);
new_file_name = append(name,".csv");
new_file_path = fullfile(csv_data_path, new_file_name);
acc_ang = synchronize(acc, ang, 'first','linear');
acc_ang = acc_ang(3:end-4, :);
writetimetable(acc_ang, new_file_path);
disp("csv was created.");