%% 0.30% error on MNIST with 155 Gabor filters

[pathstr,~,~] = fileparts(mfilename('fullpath'));
addpath(fullfile(pathstr,'../autocnn_unsup'))

opts = [];
opts.libsvm = '/home/boris/Project/3rd_party/libsvm/matlab';
opts.matconvnet = '/home/boris/Project/3rd_party/matconvnet'; % optional

% if memory is an issue (may get larger error)
% opts.pca_fast = true; % randomized PCA 
% opts.n_unlabeled = 10e3;
% opts.fix_unlabeled = false;

% if memory is not an issue
opts.fix_unlabeled = true; % use all training images to perform PCA
opts.pca_fast = false;

opts.PCA_dim = 90;

% Generate a large set of complex-valued Gabor filters
[filters, params] = generate_filters(4, 7, 18);

% Run heuristic selection (skipped here)

% 155 filters heuristically selected from the entire set
selected_filters = [12,18,31,44,52,69,71,75,83,102,116,131,161,166,175,187,196,197,199,214,220,225,234,241,253,255,257,262,265,288,289,290,301,308,311,317,...
321,324,328,334,336,337,341,342,344,346,348,354,355,356,361,362,365,368,375,376,384,385,388,399,405,408,425,427,428,434,440,442,446,456,...
464,468,473,498,520,525,530,549,558,565,569,574,577,578,579,584,585,586,588,589,592,595,597,603,604,605,610,613,614,615,616,618,619,620,...
621,626,628,631,634,637,638,639,641,643,648,649,651,665,691,704,718,726,736,737,739,740,746,751,788,795,803,804,805,828,835,838,845,849,...
850,851,857,861,862,863,865,871,880,885,892,904,915,916,919,935,942];

opts.filters = {cat(4,filters{selected_filters})};
close all
subplot(2,1,1),imsetshow(mat2gray(real(opts.filters{1})), 5), title('Real parts')
subplot(2,1,2),imsetshow(mat2gray(imag(opts.filters{1})), 5), title('Imaginary parts'), drawnow

% Forward pass and classification
opts.arch = sprintf('%dc%d-2p', size(opts.filters{1},4), size(opts.filters{1},1));
autocnn_mnist(opts)