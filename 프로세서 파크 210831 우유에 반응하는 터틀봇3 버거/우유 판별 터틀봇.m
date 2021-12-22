%% *Connect to the NVIDIA Hardware*
% 젯슨 보드 연결

hwobj = jetson('192.168.0.37','ros','jet1123');
%% *CUDA 코드*
% GPU coder에 변환할 m파일 함수를 먼저 만들고 실행.
%% *Generate CUDA Code for the Target Using GPU Coder*
% Coder 설정 변수들. GPU Coder를 사용하면 그냥 알아서 되는거 같기도...

cfg = coder.gpuConfig('exe');
cfg.Hardware = coder.hardware('NVIDIA Jetson');
cfg.Hardware.BuildDir = '~/remoteBuildDir';
cfg.GenerateExampleMain = 'GenerateCodeAndCompile';
cfg.BuildConfiguration = 'Faster Builds';
disp('타켓 확인')
%% generate CUDA code
% 딥러닝이 포함되어 있어서 그런지 오래걸림...

tic;
codegen('-config', cfg, 'MyNetTurtlebot', '-report')
t = toc/60;
disp([num2str(t),'분 경과'])
%% *Run the Sobel Edge Detection on the Target*
% *프로그램 실행*

exe = [hwobj.workspaceDir '/MyNetTurtlebot.elf'];
pid = hwobj.runExecutable(exe);
% 프로그램 종료
% 프로세스 번호를 킬

killApplication(hwobj,exe)
%% ROS2 실행
% |export TURTLEBOT3_MODEL=burger|
% 
% |ros2 launch turtlebot3_bringup robot.launch.py|
% 
% |ros2 run py_pubsub talker|

openShell(hwobj);
%% ROS2 노드 생성
% MATLAB 노드

node = ros2node('/matlab',1)
%% ROS2 토픽 리스트 확인

ros2("node","list","DomainID",1)
ros2("topic","list","DomainID",1) 
%% ROS2 토픽 선언
% subscriber는 받는 데이터, publisher은 보내는 데이터로 생각하면 좋다.

class_sub = ros2subscriber(node, '/class');

vel_pub = ros2publisher(node, '/cmd_vel');
vel_msg = ros2message(vel_pub);
%% 
% subscriber 테스트

class_topic = receive(class_sub);
disp(class_topic.data)
% 터틀봇 제어 코드

while 1
    class_topic = receive(class_sub);
    %disp(class_topic.data)    
    idxTop = str2double(class_topic.data);    
    %disp(['인덱스 : ', num2str(idxTop)])       
    
    if(idxTop == 1)         % 딸기우유
        disp(['딸기우유'])
        vel_msg.linear.x = 0.1;
        vel_msg.angular.z = 0;
        send(vel_pub,vel_msg)  
    elseif(idxTop == 2)     % 바나나우유
        disp(['바나나우유'])
        vel_msg.linear.x = 0;
        vel_msg.angular.z = 0.5;
        send(vel_pub,vel_msg)  
    elseif(idxTop == 3)     % 초코우유
        disp(['초코우유'])
        vel_msg.linear.x = 0;
        vel_msg.angular.z = -0.5;
        send(vel_pub,vel_msg)  
    elseif(idxTop == 4)     % 흰우유
        disp(['흰우유'])
        vel_msg.linear.x = -0.1;
        vel_msg.angular.z = 0;
        send(vel_pub,vel_msg)  
    else                    % 인식 못하면 정지
        disp(['정지'])
        vel_msg.linear.x = 0;
        vel_msg.angular.z = 0;
        send(vel_pub,vel_msg)  
    end
      
end

%%
vel_msg.linear.x = 0;
vel_msg.angular.z = 0;
send(vel_pub,vel_msg)