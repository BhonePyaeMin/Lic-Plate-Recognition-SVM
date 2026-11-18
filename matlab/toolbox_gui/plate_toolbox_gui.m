function plate_toolbox_gui()
%PLATE_TOOLBOX_GUI Image-processing toolbox GUI for the course project.
%   Load an image, pick an operation (Modules 1 to 3), adjust up to three
%   parameters, press Apply and compare the original with the result.
%   Run it from the repository's matlab folder or call setup_paths first.
%   The operations need no toolbox (see matlab/common). This GUI has not been run yet.
setup_paths();
ops = toolbox_operations();
S = struct('img', [], 'ref', [], 'result', []);   % shared state

fig = figure('Name', 'License Plate Image Toolbox', 'NumberTitle', 'off', ...
    'MenuBar', 'none', 'ToolBar', 'none', 'Position', [100 100 1100 640]);
moduleNames = {'Module 1: Enhancement', 'Module 2: Segmentation', 'Module 3: Geometry'};

% ---- left control column
panel = uipanel(fig, 'Units', 'pixels', 'Position', [10 10 300 620], 'Title', 'Controls');
uicontrol(panel, 'Style', 'pushbutton', 'String', 'Load image', 'Position', [10 560 135 30], 'Callback', @onLoad);
uicontrol(panel, 'Style', 'pushbutton', 'String', 'Load reference', 'Position', [150 560 135 30], 'Callback', @onLoadRef);
uicontrol(panel, 'Style', 'text', 'String', 'Operation', 'Position', [10 525 275 20], 'HorizontalAlignment', 'left');
labels = cell(1, numel(ops));
for k = 1:numel(ops)
    labels{k} = sprintf('%s | %s', moduleNames{ops(k).module}(1:8), ops(k).name);
end
popup = uicontrol(panel, 'Style', 'popupmenu', 'String', labels, 'Position', [10 500 275 25], 'Callback', @onOperation);

paramLabel = gobjects(1, 3);  paramEdit = gobjects(1, 3);
for k = 1:3
    y = 455 - (k - 1) * 55;
    paramLabel(k) = uicontrol(panel, 'Style', 'text', 'Position', [10 y + 22 275 20], 'HorizontalAlignment', 'left', 'String', '');
    paramEdit(k) = uicontrol(panel, 'Style', 'edit', 'Position', [10 y 275 25], 'String', '');
end
uicontrol(panel, 'Style', 'pushbutton', 'String', 'Apply', 'FontWeight', 'bold', 'Position', [10 270 135 35], 'Callback', @onApply);
uicontrol(panel, 'Style', 'pushbutton', 'String', 'Save result', 'Position', [150 270 135 35], 'Callback', @onSave);
status = uicontrol(panel, 'Style', 'text', 'Position', [10 20 275 230], 'HorizontalAlignment', 'left', ...
    'String', 'Load an image to begin.');

% ---- axes
axOrig = axes('Parent', fig, 'Units', 'pixels', 'Position', [330 330 750 280]);  title(axOrig, 'Original'); axis(axOrig, 'off');
axRes  = axes('Parent', fig, 'Units', 'pixels', 'Position', [330 20 750 280]);   title(axRes, 'Result');   axis(axRes, 'off');

onOperation();

    function onLoad(~, ~)
        [f, p] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif', 'Images'});
        if isequal(f, 0), return, end
        S.img = imread(fullfile(p, f));
        imshow(S.img, 'Parent', axOrig);  title(axOrig, 'Original');
        setStatus(sprintf('Loaded %s (%d x %d).', f, size(S.img, 2), size(S.img, 1)));
    end

    function onLoadRef(~, ~)
        [f, p] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif', 'Images'});
        if isequal(f, 0), return, end
        S.ref = imread(fullfile(p, f));
        setStatus(sprintf('Reference loaded: %s', f));
    end

    function onOperation(~, ~)
        op = ops(popup.Value);
        for k = 1:3
            if k <= numel(op.params)
                paramLabel(k).String = op.params(k).label;
                paramEdit(k).String = num2str(op.params(k).default);
                paramLabel(k).Visible = 'on';  paramEdit(k).Visible = 'on';
            else
                paramLabel(k).Visible = 'off'; paramEdit(k).Visible = 'off';
            end
        end
    end

    function onApply(~, ~)
        if isempty(S.img), setStatus('Load an image first.'); return, end
        op = ops(popup.Value);
        p = zeros(1, 3);
        for k = 1:numel(op.params)
            v = str2double(paramEdit(k).String);
            if isnan(v), setStatus(sprintf('"%s" must be a number.', op.params(k).label)); return, end
            p(k) = v;
        end
        try
            t0 = tic;
            res = op.run(S.img, p, S.ref);
            elapsed = toc(t0);
        catch err
            setStatus(['Operation failed: ' err.message]);
            return
        end
        S.result = res;
        imshow(res, 'Parent', axRes);  title(axRes, op.name);
        setStatus(sprintf('%s\nOutput %d x %d, %s.\nComputed in %.3f s.', op.name, size(res, 2), size(res, 1), class(res), elapsed));
    end

    function onSave(~, ~)
        if isempty(S.result), setStatus('Nothing to save yet.'); return, end
        [f, p] = uiputfile('result.png');
        if isequal(f, 0), return, end
        out = S.result;
        if islogical(out), out = uint8(out) * 255; end
        imwrite(out, fullfile(p, f));
        setStatus(['Saved ' f]);
    end

    function setStatus(msg)
        status.String = msg;
    end
end

function setup_paths()
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(here, fullfile(root, 'common'), fullfile(root, 'module1_enhancement'), fullfile(root, 'module2_segmentation'), fullfile(root, 'module3_transformations'));
end
