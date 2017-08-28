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
clear all
close all
% create our net
caffe.set_mode_gpu();
model = './Scene16Classes2k/places365-master/deploy_alexnet_scene16.prototxt';
weights = './Scene16Classes2k/places365-master/alexnet_scene16c.caffemodel';
net = caffe.Net(model, weights, 'test')
% get our image mean
image_mean = caffe.io.read_mean('./Scene16Classes2k/places365-master/scene16_mean.binaryproto');
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% DATA EXTRACTION %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data is located at /tmp/data/.
% - There are 20 folders with animal names. Each folder has ~100 images of
% the animal specified.
% - Use matlab command imageSet and 'recursive' parameters to to obtain a
% list of folders and all images in them for easy processing
imset = imageSet('C:/DataSets/SCENE DATA/3500ImagesDataset', 'recursive');
% yields --> 
%imset =
%  1x20 imageSet array with properties:
%    Description
%    ImageLocation
%    Count
% Can access on)
% And then with:  imset(1).ImageLocation(1)
% - Folder name is its 'ground truth' label
% - For each image, must extact three freatures from the CNN and store them
% in a variable.
% - Later will train a linear SVM on __each__ of the features
% - To extract features from each image, use this function: 
%       * caffe.io.load_image('/path/to/image/to/load')
test_labels = [];
predicted_test_labels = [];
% create a unique ID number for each image, we'll use this to index into
% our feature sets and keep things organized
imNum = 1;
%categoryNum = 3;
for categoryNum = 1:size(imset,2)
    for imageNum = 2000:2500%size(imset(categoryNum).ImageLocation,2)
        feats = caffe.io.load_image(imset(categoryNum).ImageLocation{imageNum});
        label = imset(categoryNum).Description;
        Im = imread((imset(categoryNum).ImageLocation{imageNum}));
        categoryNum
        imageNum
        % - After loading, use imresize(227) --> height and width must BOTH be set
        % to 227px.
        % - After re-sizing, subtract the image_mean 
        input_data = {prepare_image(Im)};
        %%
        % - After re-sizing, subtract the image_mean 
        % resize_minus_mean = resize - image_mean;
        % - Run the image through the neural net using this command:
        %       * net.forward({image});
        scores = net.forward(input_data);
        % - Once image has been run through the Net, we are ready to extact features
        % from the network. Extract features from the 3 fully connected layers of
        % the network, 'fc8', 'fc7', and 'fc6'. To extract an image feature from
        % the network, use the command: 
        %       * net.blobs(feature_name).get_data()
        %           --> store each set of features some where for training the SVM
         data_labels(imNum) = categoryNum;
         scores = scores{1};
         scores = mean(scores, 2);  % take average scores over 10 crops
        %%
         [~, maxlabel] = max(scores);
         predicted_test_labels(imNum) = maxlabel;
        % increment the id number, each time
        imNum = imNum + 1;
    end
end

%%
CM = confusionmat(data_labels, predicted_test_labels);
 
confMat = confMatGet(data_labels, predicted_test_labels);
opt=confMatPlot('defaultOpt');
%opt.className=IntrctAnn2;
% === Example 1: Data count plot
opt.mode='dataCount';
figure; confMatPlot(confMat, opt);
% === Example 2: Percentage plot
opt.mode='percentage';
opt.format='8.2f';
figure; confMatPlot(confMat, opt);
% === Example 3: Plot of both data count and percentage
opt.mode='both';
figure; confMatPlot(confMat, opt);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%Confusion Matrix%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
mat = CM./500;% rand(16);           %# A 5-by-5 matrix of random values from 0 to 1
imagesc(mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

textStrings = num2str(mat(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:16);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick',1:16,...                         %# Change the axes tick marks
        'XTickLabel',{'Building','Cafeteria','Classroom','Conference Room','Downtown','Driveway','Highway Road','Hospital Room', 'Office', 'Park','Parking','Parking Underground', 'ShoppingMall', 'Street','Supermarket'},'XTickLabelRotation',45,...  %#   and tick labels
        'YTick',1:16,...
        'YTickLabel',{'Building','Cafeteria','Classroom','Conference Room','Downtown','Driveway','Highway Road','Hospital Room', 'Office', 'Park','Parking','Parking Underground', 'ShoppingMall', 'Street','Supermarket'},...
        'TickLength',[0 0]);
