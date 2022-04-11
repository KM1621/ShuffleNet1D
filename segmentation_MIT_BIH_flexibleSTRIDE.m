clear all
close all
clc
load('ann_loc_all.mat')
load('ann_type_all.mat')
load('ecg_all.mat')
Index_seg = 1; %    Indexing segments
seg_SIZE = 720;
stride_SIZE = 720; % Used for flexible striding size to increase number of samples from the minority class
seg_samples = double.empty;
%ann_samples = {};
label_samples = double.empty;
all_subjects = length(ecg_all);
countOthers = 0;
countN = 0;
countF = 0;
countV = 0;
countS = 0;
% countQ = 0;

for subject_no=1: all_subjects
    tic
    fprintf('Working on Subject no %f \n', subject_no);  % subject_no
    
    all_samples = length(ecg_all{subject_no, 1});
    %     if (subject_no == 30 || subject_no == 34 || subject_no == 28 || subject_no == 32) % These are subjects whose records have arrhythmias from the under represented classes
    %         stride_SIZE = 720;
    %     else
    %         stride_SIZE = 720;
    %     end
    k = 1;
    N_loc = ann_loc_all{subject_no, 1}(ismember(ann_type_all{subject_no, 1}  , 'NLRej')); % Location of N only beats
    S_loc = ann_loc_all{subject_no, 1}(ismember(ann_type_all{subject_no, 1}  , 'AaJS')); % Location of S only beats
    V_loc = ann_loc_all{subject_no, 1}(ismember(ann_type_all{subject_no, 1}  , 'VE')); % Location of V only beats
    F_loc = ann_loc_all{subject_no, 1}(ismember(ann_type_all{subject_no, 1}  , 'F')); % Location of F only beats
    Q_loc = ann_loc_all{subject_no, 1}(ismember(ann_type_all{subject_no, 1}  , '/fQ')); % Location of F only beats
    while k < (all_samples + seg_SIZE)
        % for k = 1:(all_samples/stride_SIZE)
        
        
        if(sum(k < F_loc + 700 & k > F_loc - 700))
            stride_SIZE = 10;
            countF = countF + 1;
        elseif (sum(k < V_loc + 700 & k > V_loc - 700))
            stride_SIZE = 200;
            countV = countV + 1;
        elseif (sum(k < S_loc + 700 & k > S_loc - 700))
            stride_SIZE = 200;
            countS = countS + 1;
        elseif (sum(k < N_loc + 10 & k > N_loc - 10))
            stride_SIZE = 720;
            countN = countN + 1;
        end
        
        class = [0 0 0 0];    %  Corresponds to 4 classes i.e. excluding the class label Q
        new_range = k + stride_SIZE + 1:k + stride_SIZE + seg_SIZE;
        if(k + stride_SIZE + seg_SIZE <= all_samples && sum(ismember(Q_loc  , new_range)) < 1)   %   Ensures the segment is within range
            seg_samples(Index_seg,:) = ecg_all{subject_no, 1}(k + stride_SIZE + 1:k + stride_SIZE + seg_SIZE);
            logicalIndexes = ann_loc_all{subject_no, 1} < (k + stride_SIZE + seg_SIZE) & ann_loc_all{subject_no, 1} > k + stride_SIZE+1;
            linearIndexes = find(logicalIndexes);
            ann_type_seg = ann_type_all{subject_no, 1}(linearIndexes);
            % Check for N
            if (sum(ismember(ann_type_seg  ,'NLRej') >= 1))
                class(1) = 1;
            end
            % Check for S
            if (sum(ismember(ann_type_seg  ,'AaJS') >= 1))
                class(2) = 1;
            end
            
            % Check for V
            if (sum(ismember(ann_type_seg  ,'VE') >= 1))
                class(3) = 1;
            end
            
            % Check for F
            if (sum(ismember(ann_type_seg  ,'F') >= 1))
                class(4) = 1;
            end
            
            % Check for Q
            %             if (sum(ismember(ann_type_seg  ,'/fQ') >= 1))
            %                 class(5) = 1;
            %             end
            label_samples(Index_seg,:) = class;
            %ann_samples{Index_seg} = ann_type_seg;   % Added for debugging
            %to verify the label matrix for correctness
            Index_seg=Index_seg+1;
        end
        k = k + stride_SIZE;
    end
    
    
    
    toc
end


save('seg_samples_STRIDED_No_Q.mat','seg_samples','-v7.3')
save('label_samples_STRIDED_No_Q.mat','label_samples','-v7.3')