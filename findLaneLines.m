function [results, count] = findLaneLines(I,history,roi)


RESIZE_FACTOR = round(400/size(I,2),2);%0.4;

I = mean(I(:,:,1:2),3); % This converts I to double

I = imresize(I,RESIZE_FACTOR,'bilinear','Antialiasing',false);
roi = imresize(roi,RESIZE_FACTOR,'bilinear','Antialiasing',false)>0;

y_min = find(sum(roi,2),1,'first');


tmpI = I;
tmpI(~roi) = Inf;
minI = prctile(tmpI',30)';
% minI(any(roi,2)) = minI_mask(any(roi,2));
I = I - minI;
I(I<0) = 0;

tmpI = I;
tmpI(~roi) = 1;
maxI = prctile(tmpI',99)';

I = 255*(I./maxI);%prctile(I(roi),95)));
I(I>255) = 255;

I = imfilter(I,fspecial('gaussian',7,1),'replicate');



h = -fspecial('sobel'); % Align mask correctly along the x- and y- axes
Gx = imfilter(I,h','replicate');
if nargout > 1
    Gy = imfilter(I,h,'replicate');
end

% [Gx,Gy] = gradient(I);
G = sqrt(Gx.^2 + Gy.^2);


% Create the set of pixels that we will create lines on
mask = G > max(G(:))/2;

% mask = edge(I,'canny',[100,180]/255);
mask = logical(mask .* roi);

mask_mask = false(numel(mask),1);
mask_mask(1:3:end) = true;
mask_mask = reshape(mask_mask,size(mask));
mask = mask & mask_mask;



[i_y,i_x] = find(mask);

% At each pixel, compute the slope of the line.
Th = atan2(Gy(mask),Gx(mask)); % Angle of gradient
Th(Th<0) = Th(Th<0) + 2*pi;
Th = Th + pi/2; % The line is perpendicular to the gradient
m = tan(Th); % copmute the slope

% The the end points of each line bounded by the image.
x = [1,size(I,2)] .* ones(numel(i_x),1);
y = m.*x - m.*i_x + i_y;

y2 = [1, size(I,1)] .* ones(numel(i_x),1);
x2 = y2./m - i_y./m + i_x;

[x2,si] = sort(x2,2);
y2 = y2((si-1)*numel(i_x) + (1:numel(i_x))');

replace = y(:,1) < 1 | y(:,1) > size(I,1);
x(replace,1) = x2(replace,1);
y(replace,1) = y2(replace,1);

replace = y(:,2) > size(I,1) | y(:,2) < 1;
x(replace,2) = x2(replace,2);
y(replace,2) = y2(replace,2);

% Round end points to nearest integer and store as end points.
x = round(x);
y = round(y);

x1 = [y(:,1), x(:,1)];
x2 = [y(:,2), x(:,2)];

% Accumulate the lines into an image
count = accumLines(x1,x2,size(I),4);


% Smooth out the counts
count = imfilter(count,fspecial('gaussian',14,2));


% Normalize the counts using the mask
% mask2 = imdilate(mask,ones(3));
[~,vanish_idx] = max(count(:));%./(1+mask2(:)));
[y_max,x_max] = ind2sub(size(count),vanish_idx);


results = [];
results.count = count;

if ~isempty(history.count)
    count = mean(cat(3,count,history.count),3);
end

count = count .* roi;

count_mean = sum(count,1);
count = (count_mean > prctile(count_mean,25)) .* count ./ (count_mean);


% Threshold the image
count = count > prctile(count(:),97); %This should require approximately 10 lines to pass through each point.
% count = imclose(count,strel('disk',2)); % close the image to connect the two sides of the lane lines


% Grap the
% [H,T,R] = hough(bwmorph(count,'thin',Inf),'RhoResolution',1,'Theta',-90:2:89);
% P  = houghpeaks(H,5,'threshold',ceil(0.2*max(H(:))));
% lines = houghlines(count,T,R,P,'FillGap',5,'MinLength',round(max(size(I))/3));
% 
% % figure(2)
% % imshow(count + bwmorph(count,'thin',5),[0,2])
% 
% th = [lines.theta];
% thp = th + 360 * (th<0);
% [~,idx1] = min(thp); % left lane
% thn = th - 360 * (th>0);
% [~,idx2] = max(thn); % right lane
% 
% 
% right = [lines(idx2).point1; lines(idx2).point2];
% left = [lines(idx1).point1; lines(idx1).point2];
% 
% 
% 
% if ~isempty(right)
%     [~,right_ord] = sort(right(:,2),1,'descend');
%     right = right(right_ord,:);% + [left_bnd,top_bnd];
% end
% 
% if ~isempty(left)
%     [~,left_ord] = sort(left(:,2),1,'descend');
%     left = left(left_ord,:);% + [left_bnd,top_bnd];
% end



[y,x] = find(count);

x_mid = mean(x);

inrange = x > x_mid+10;

% inrange = (x < max(right(:,1))) & (x > min(right(:,1)));
right_param = [ones(sum(inrange),1), x(inrange)]\y(inrange);
right(:,2) = [size(I,1); y_min];
right(:,1) = round((right(:,2) - right_param(1))/right_param(2));


inrange = x < x_mid-10;
% inrange = (x < max(left(:,1))) & (x > min(left(:,1)));
left_param = [ones(sum(inrange),1), x(inrange)]\y(inrange);
left(:,2) = [size(I,1); y_min];
left(:,1) = round((left(:,2) - left_param(1))/left_param(2));


top = [x_max,y_max];

right = right / RESIZE_FACTOR;
left = left / RESIZE_FACTOR;
top = top / RESIZE_FACTOR;


results.left = left;
results.right = right;
results.vanish = top;

end
