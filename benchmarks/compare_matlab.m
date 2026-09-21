function compare_matlab(path, sfo_dir, repeats, warmups, epsilon)
% Generic SFO minimum-norm-point adapter. Run with MATLAB -singleCompThread.
addpath(genpath(sfo_dir));
assert(exist('sfo_min_norm_point','file') == 2, 'SFO toolbox not found');
fid = fopen(path,'r'); assert(fid >= 0, 'Cannot open instance');
cleanup = onCleanup(@() fclose(fid));
n = 0; kind = ''; offset = 0; unary = []; arcs = zeros(0,3);
features = {}; weights = []; concavityScale = 0; hasConcavityScale = false;
rightN = 0; matchingEdges = zeros(0,2); matchedLeft = [];
while true
    line = fgetl(fid); if ~ischar(line), break; end
    t = strsplit(strtrim(line));
    if isempty(t{1}) || strcmp(t{1},'c') || startsWith(t{1},'#'), continue; end
    switch t{1}
        case 'p'
            kind = t{2}; n = str2double(t{3}); unary = zeros(n,1);
            assert(any(strcmp(kind,{'cut','coverage','concave','matching'})),'Unsupported instance kind');
            if strcmp(kind,'matching')
                assert(numel(t)==4,'Matching header needs right-side size');
                rightN = str2double(t{4});
            else
                assert(numel(t)==3,'Unexpected header field');
            end
        case 'offset', offset = str2double(t{2});
        case 'u', i = str2double(t{2})+1; unary(i) = unary(i)+str2double(t{3});
        case 'a', arcs(end+1,:) = [str2double(t{2})+1,str2double(t{3})+1,str2double(t{4})]; %#ok<AGROW>
        case 'f', weights(end+1) = str2double(t{2}); features{end+1} = str2double(t(3:end))+1; %#ok<AGROW>
        case 'q'
            assert(strcmp(kind,'concave') && ~hasConcavityScale,'Invalid q row');
            concavityScale = str2double(t{2}); hasConcavityScale = true;
            assert(concavityScale>=0,'Negative concavity scale');
        case 'e'
            assert(strcmp(kind,'matching'),'Invalid matching edge row');
            matchingEdges(end+1,:) = [str2double(t{2})+1,str2double(t{3})+1]; %#ok<AGROW>
        case 'o' % exact independent verification is performed by Python
        otherwise, error('Unknown instance tag');
    end
end
assert(n>0,'Invalid ground set');
assert(~strcmp(kind,'concave') || hasConcavityScale,'Concave instance has no q row');
assert(~strcmp(kind,'matching') || (rightN>0 && all(matchingEdges(:,1)>=1) && ...
    all(matchingEdges(:,1)<=n) && all(matchingEdges(:,2)>=1) && all(matchingEdges(:,2)<=rightN)), ...
    'Invalid matching dimensions or edge');
opt = sfo_opt({'verbosity_level',0,'minnorm_stopping_thresh',epsilon,'minnorm_tolerance',1e-10});
calls = 0;
samples = repmat(struct('minimum',0,'selected',[],'elapsed_seconds',0,'oracle_calls',0,'reported_suboptimality',0),1,repeats);
for trial = (1-warmups):repeats
    calls = 0;
    tic;
    [selected, subopt] = sfo_min_norm_point(@objective,1:n,opt);
    actual = objective(selected)+offset;
    elapsed = toc;
    measured_calls = calls;
    if trial <= 0, continue; end
    samples(trial).minimum = actual;
    samples(trial).selected = reshape(selected-1,1,[]);
    samples(trial).elapsed_seconds = elapsed;
    samples(trial).oracle_calls = measured_calls;
    samples(trial).reported_suboptimality = subopt;
end
% Force JSON arrays for one sample and singleton/empty sets (MATLAB encodes
% scalar numeric arrays as numbers unless explicitly wrapped in cells).
output = struct('implementation','SFO sfo_min_norm_point','arithmetic','Float64', ...
    'matlab_version',version,'toolbox_function',which('sfo_min_norm_point'), ...
    'epsilon',epsilon,'numeric_tolerance',1e-10,'samples',[]);
for k=1:repeats, samples(k).selected = num2cell(samples(k).selected); end
output.samples = num2cell(samples);
fprintf('%s\n',jsonencode(output));

    function value = objective(A)
        calls = calls+1;
        selected_mask = false(n,1); selected_mask(A)=true;
        value = 0; % F(empty)=0; restore original offset after timing
        for i=1:n
            if selected_mask(i), value=value+unary(i); end
        end
        if strcmp(kind,'cut')
            for j=1:size(arcs,1)
                if selected_mask(arcs(j,1)) && ~selected_mask(arcs(j,2)), value=value+arcs(j,3); end
            end
        elseif strcmp(kind,'coverage')
            for j=1:numel(features)
                if any(selected_mask(features{j})), value=value+weights(j); end
            end
        elseif strcmp(kind,'concave')
            cardinality = nnz(selected_mask);
            value = value-concavityScale*cardinality*(cardinality-1)/2;
        elseif strcmp(kind,'matching')
            matchedLeft = zeros(rightN,1);
            for leftVertex=find(selected_mask)'
                seenRight = false(rightN,1);
                if augment(leftVertex,seenRight), value=value+1; end
            end
        end
    end

    function found = augment(leftVertex,seenRight)
        found = false;
        adjacent = matchingEdges(matchingEdges(:,1)==leftVertex,2)';
        for rightVertex=adjacent
            if seenRight(rightVertex), continue; end
            seenRight(rightVertex)=true;
            if matchedLeft(rightVertex)==0 || augment(matchedLeft(rightVertex),seenRight)
                matchedLeft(rightVertex)=leftVertex; found=true; return;
            end
        end
    end
end
