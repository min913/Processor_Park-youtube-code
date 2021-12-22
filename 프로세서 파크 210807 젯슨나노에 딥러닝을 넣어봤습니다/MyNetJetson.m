function MyNetJetson()
    % MYALEXNETGPU Accepts a 227x227x3 image to the deep neural network AlexNet
    % and returns the class index of the maximum confidence classification.
    % 
    % A list of all the classifications can be found in the file
    % classificationList.txt
    %
    % Copyright 2018 The MathWorks, Inc.
    % Since the function "alexnet" is not supported for generation we load it
    % from a MAT-file using coder.loadDeepLearningNetwork
    
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

        % Predict with AlexNet
        output = predict(net, img2);        
        
        % Determine the class index with the highest probability
        [~,classIdx] = max(output);      
        
        f = fopen('idx.bin','w+');
        fwrite(f, classIdx, 'uint8');
        fclose(f);
    end
        
end