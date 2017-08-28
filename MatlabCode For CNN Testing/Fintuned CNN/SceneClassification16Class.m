%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @author:  Muhammad Shahid (muhammad.shahid@kics.edu.pk)
%
% Scene Classification using CNN and SVM with 16 Classes
%
% "Deep Network Categorization"
% 
%   SCRIPT: SceneClassificationCNNSVM.m
%       - use pre-trained deep network to extact features
%       - then use these featurs to train an svm classifier which
%       discriminates betweeen 16 Scene categories
%       - finally, compare the deep-net features to SIFT features
%
%   Dependencies:
%       - Caffe package (C++ with Matlab interface)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% setup path to caffe director
%addpath('/tmp/caffe/matlab/')

% create our net
caffe.set_mode_gpu();
model = './Scene16Classes2k/places365-master/deploy_alexnet_scene16.prototxt';
weights = './Scene16Classes2k/places365-master/alexnet_scene16.caffemodel';
net = caffe.Net(model, weights, 'test')
%%
%%
net.layer_names
weights_cov1 = net.params('conv2',1).get_data();
w1 = mat2gray(weights_cov1);
%% w1 = imresize(w1,11);
  figure
%  montage(w1)
% title('First convolutional layer weights');

%%
% prepare oversampled input
% input_data is Height x Width x Channel x Num
%%
%im = imread('./places365-master/places365-master/docker/images/mountains.jpg'); % read image   Places365_val_00000021
%im = imread('./val_256/DSC_0657.JPG.jpg');
close all
im = imread('park1.png');
tic;
input_data = {prepare_image(im)};
toc;

% do forward pass to get scores
% scores are now Channels x Num, where Channels == 1000

tic;
% The net forward function. It takes in a cell array of N-D arrays
% (where N == 4 here) containing data of input blob(s) and outputs a cell
% array containing data from output blob(s)
scores = net.forward(input_data);
% %%   Result Image
% output_FC6pre = net.blobs('pool1').get_data();
% w1 = mat2gray(output_FC6pre(:,:,3:5,1:10));
%  w1 = imresize(w1,55);
%  figure
%  montage(w1)
output_FC7= net.blob_vec(1,12).get_data();
toc;
%%%%%%%%%%%%%%%%%%
output_FC6 = net.blobs('fc6').get_data();
output_FC8 = net.blobs('fc8-Scene16').get_data();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
scores = scores{1};
scores = mean(scores, 2);  % take average scores over 10 crops
%%
[~, maxlabel] = max(scores)

% call caffe.reset_all() to reset caffe
%caffe.reset_all();
B = A{1,1};
B(maxlabel,:)
%%%%
text = B(maxlabel,:);
box_color = {'red'};
position = [50,50];
RGB = insertText(im,position,text,'FontSize',55,'BoxColor',...
    box_color,'BoxOpacity',0.4,'TextColor','white');
%RGB = insertText(im,position,text)
figure, imshow(RGB);