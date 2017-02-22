function asyncPlotUpdate(handle_array, data_array)

for i = [1,3,4,6]%1:numel(handle_array)
    fns = fieldnames(data_array{i});
    for j = 1:numel(fns)
        handle_array(i).(fns{j}) = data_array{i}.(fns{j});
    end
end

drawnow;
end