close all; clear variables; clear global; clc;              % clean desk
doTraining              = false;
doHDF5_output           = true;                        % save training/validation data to HDF5 file for Caffe input 
doImageList_output      = false;                        % save training/validation data to list file for Caffe input

% load the file data for training the CNN
IMDS = imageDatastore('SceneDataset\','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames'); % use imageDatastore for loading the two image categories
example_image = readimage(IMDS,1);                      % read one example image
numChannels = size(example_image,3);                    % get color information
numImageCategories = size(categories(IMDS.Labels),1);   % get category labels
[trainingDS,validationDS] = splitEachLabel(IMDS,0.7,'randomize'); % generate training and validation set
LabelCnt = countEachLabel(IMDS);                        % load lable information
for cats=1:numImageCategories                           % print out how many images we have for each category
    fprintf('%s\t%d\n',LabelCnt.Label(cats),LabelCnt.Count(cats));
end



%% HDF5 output
if doHDF5_output
    fprintf('output training/validation data sets to HDF5\n');
    fprintf('training set to HDF5...\n');
    Img = zeros([size(example_image) 1 size(trainingDS.Files,1)]);
    Seg = zeros(1,size(trainingDS.Labels,1));
    for img_nr = 1:size(trainingDS.Files,1)
        Img(:,:,:,img_nr) = readimage(trainingDS,img_nr);
        if trainingDS.Labels(img_nr) == 'circles_sm'
            Seg(img_nr) = 0;
        else
            Seg(img_nr) = 1;
        end
    end
    hdf5write( 'training_set.h5','data', single(Img),'label', single(Seg));

    fprintf('validation set to HDF5...\n');
    Img = zeros([size(example_image) 1 size(validationDS.Files,1)]);
    Seg = zeros(1,size(validationDS.Labels,1));
    for img_nr = 1:size(validationDS.Files,1)
        Img(:,:,img_nr) = readimage(validationDS,img_nr);
        if validationDS.Labels(img_nr) == 'circles_sm'
            Seg(img_nr) = 0;
        else
            Seg(img_nr) = 1;
        end
    end
    hdf5write( 'validation_set.h5','data', single(Img),'label', single(Seg));
end