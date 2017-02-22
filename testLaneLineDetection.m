%% Test lane line detection
% vidObj = VideoReader('solidYellowLeft.mp4');
% vidObj = VideoReader('solidWhieRight.mp4');
vidObj = VideoReader('challenge.mp4');
% Determine the height and width of the frames.

vidHeight = vidObj.Height;
vidWidth = vidObj.Width;
% Create a MATLAB® movie structure array, s.

s = struct('cdata',zeros(vidHeight,vidWidth,3,'uint8'),...
    'colormap',[]);
% Read one frame at a time using readFrame until the end of the file is reached. Append data from each video frame to the structure array.
k = 1;
while hasFrame(vidObj)
    frame = readFrame(vidObj);
    s(k).cdata = frame;
    k = k+1;
end

% pth = 'test_images\';
%
% % I = imread([pth, 'solidWhiteRight.jpg']);
% I = imread([pth, 'hole15.jpg']);
% I = imread([pth, 'curved_road2.jpg']);
% vidHeight = size(I,1);
% vidWidth = size(I,2);
%
% s = struct('cdata',zeros(vidHeight,vidWidth,3,'uint8'),...
%     'colormap',[]);
%
% s(1).cdata = I;
% I = imread([pth, 'whiteCarLaneSwitch.jpg']);


hist_size = 10;

clear history
history(hist_size) = struct('left',[],'right',[],'vanish',[],'count',[]);
nms = {'left','right','vanish','count'};

figure(1)
clf(1)
im = imshow(s(1).cdata);
hold on

h = imshow(cat(3,ones([vidHeight,vidWidth]),zeros([vidHeight,vidWidth]), zeros([vidHeight,vidWidth])));
h.AlphaData = zeros([vidHeight,vidWidth]);

lf_ln = line([0 0],[0 0],'color','r','linewidth',7);
lf_ln.Color(4) = 0.5;
rt_ln = line([0 0],[0 0],'color','r','linewidth',7);
rt_ln.Color(4) = 0.5;
vn_pt = line([0 0],[0 0],'color','g','linewidth',3,'marker','x','markersize',14);

title_h = title('Frame rate : ');
handle_array = [im, h, lf_ln, rt_ln, vn_pt, title_h];
data_array = cell(6,1);
data_array{1} = struct('CData',[]);
data_array{2} = struct('AlphaData', []);
data_array{3} = struct('XData', [], 'YData', []);
data_array{4} = struct('XData', [], 'YData', []);
data_array{5} = struct('XData', [], 'YData', []);
data_array{6} = struct('String', []);


[X,Y] = meshgrid(1:vidWidth,1:vidHeight);
poly_x = [0, round(0.45*vidWidth), round(0.55*vidWidth), vidWidth];
poly_y = [vidHeight, round(vidHeight*0.55), round(vidHeight*0.55), vidHeight];
roi = inpolygon(X,Y,poly_x,poly_y);

curRes = history(end);
% profile on
for k = 1:numel(s)
    
    tic
    
    [results, count] = findLaneLines(s(k).cdata,curRes,roi);
    
    history(1) = results;
    history = circshift(history,-1);
    curRes = results;
    
    for i = nms
        
%         tmp = cat(3,history(max(1,hist_size-k+1):hist_size).(i{1}));
%         w = linspace(0.3,1,size(tmp,3));
%         curRes.(i{1}) = sum(tmp.*permute(w,[1,3,2]),3) ./ sum(w);
        
        curRes.(i{1}) = mean(cat(3,history(max(1,hist_size-k+1):hist_size).(i{1})),3);
        
    end
    
    calcTime = toc;
    
    data_array{1} = struct('CData',s(k).cdata);
    data_array{2} = struct('AlphaData', []);%count*0.5);
    data_array{3} = struct('XData', history(end).left(:,1), 'YData', history(end).left(:,2));
    data_array{4} = struct('XData', history(end).right(:,1), 'YData', history(end).right(:,2));
    data_array{5} = struct('XData', history(end).vanish(:,1), 'YData', history(end).vanish(:,2));
    data_array{6} = struct('String', sprintf('Frame rate : %0.1f', 1/calcTime));

    asyncPlotUpdate(handle_array, data_array)    
end
% profile off
% profile viewer
