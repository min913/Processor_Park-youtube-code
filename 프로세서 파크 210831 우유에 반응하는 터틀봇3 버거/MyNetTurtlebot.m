function MyNetTurtlebot()   
    persistent net
    if isempty(net)
        net = coder.loadDeepLearningNetwork('googlenetTransfer.mat');        
    end
     
    hwobj = jetson; % To redirect to the code generatable functions.
    w = webcam(hwobj,1);
    d = imageDisplay(hwobj);   
    
    % Main loop    
    while 1 % for k = 1:1800        
        img = snapshot(w);
        img2 = imresize(img, [224, 224]);
        image(d, img);
        
        output = predict(net, img2);        
        [outputMax, classIdx] = max(output);
        
        if outputMax < 0.9
            classIdx = 0;
        end
        
        f = fopen('/home/ros/idx.txt','w+');
        fprintf(f, '%d\n', int16(classIdx));
        fclose(f);           
    end        
end