%% AlexNet을 사용한 "우유 판별" 전이 학습(Transfer Learning)
%% 데이터 불러오기

clear, clc
%unzip('MerchData.zip');
imds = imageDatastore('우유','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');    
% 이미지 확인

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end
%% 사전 훈련된 신경망 불러오기

net = alexnet;
%analyzeNetwork(net)
% 신경망 layer 확인

net.Layers
inputSize = net.Layers(1).InputSize
% 마지막 계층 바꾸기

layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels))
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
% 신경망 훈련시키기
% 훈련 이미지 데이터 증대

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
% 검증 이미지 데이터 증대

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
% 네트워크 옵션

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
%% 신경망 훈련

netTransfer = trainNetwork(augimdsTrain,layers,options);
%% 
% 신경망 저장

save("netTransfer.mat","netTransfer")
%% 검증 영상 분류하기

[YPred,scores] = classify(netTransfer,augimdsValidation);
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)