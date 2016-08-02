function [filters, params] = generate_filters(nScales, nTheta, sz)
filters = cell(1, nScales*nTheta);
p = 1;
coef = (28/17);
scales = linspace(0.3*pi/coef,2.5*pi/coef,nScales+2);
scales = scales(1:end-2);
thetas = linspace(0,pi,nTheta+1);
thetas = thetas(1:end-1);
if (length(sz) == 1)
    sz = repmat(sz,1,2);
end
for scale_x=scales
    for sig_L = [0.3, 0.4, 0.5, 0.6]
        for gamma=[0.4, 0.7, 0.8, 1, 1.2, 1.5]
            if (sig_L == 0.5 && gamma ~= 0.7 || sig_L ~= 0.5 && gamma == 0.7)
                continue;
            end
            phases = 0;
            scale_y = scale_x/gamma;
            for theta = thetas
                for phase = phases
                    for x0=0
                        
                        lambda = scale_x/sig_L;
                        if (abs(gamma - 1) < eps || scale_x == scales(end) ...
                                || (sig_L == 0.5 && gamma == 0.7))
                            beta_delta_v = 0;
                        else
                            beta_delta_v = [-pi/3:pi/3:pi/3]; % : 980
                        end
                        for beta_delta = beta_delta_v
                            beta = atan(tan(theta+beta_delta));
                            gaborModel = Gabor_2D(sz, struct('x0',x0,'y0',0,'lambda',lambda,'phi',phase,...
                                'theta',atan(tan(theta)),'stdx',scale_x,'stdy',scale_y,'beta',beta), []);
                            params{p} = [x0,0,lambda,phase,atan(tan(theta)),scale_x,scale_y,beta];
                            filters{p} = feature_scaling(gaborModel.complex,'stat',[]);
                            p = p + 1;
                        end
                    end
                end
            end
        end
    end
end

end

function gabor = Gabor_2D(sz, params, axis)
T = 1;
if (isempty(axis))
    [X_axis,Y_axis] = meshgrid(-sz(2)/2:T:sz(2)/2-1, -sz(1)/2:T:sz(1)/2-1);
    [x_modul, ~] = transform_axis(X_axis, Y_axis, params.theta, 0, 0, 1, 1, false);
    [x_gaus, y_gaus] = transform_axis(X_axis, Y_axis, params.beta, 0, 0, 1, 1, false);
elseif (numel(axis) == 2)
    [x_modul, ~] = transform_axis(axis{1}, axis{2}, params.theta, 0, 0, 1, 1, false);
    [x_gaus, y_gaus] = transform_axis(axis{1}, axis{2}, params.beta, 0, 0, 1, 1, false);
elseif (numel(axis) > 2)
    x_modul = axis{1};
    x_gaus = axis{3};
    y_gaus = axis{4};
end
scales_a = 1./(2*sqrt(pi).*params.stdx);
scales_b = 1./(2*sqrt(pi).*params.stdy);

[~,p] = transform_point([params.y0, params.x0], params.beta, 0, 0, 1, 1, false); % [r,c]
y0 = p(1);
x0 = p(2);
gaus = exp(-pi.*((x_gaus-x0).^2.*scales_a^2 + (y_gaus-y0).^2.*scales_b^2));
[~,p] = transform_point([params.y0, params.x0], params.theta, 0, 0, 1, 1, false); % [r,c]
x0 = p(2);
w0 = 1/params.lambda;
if (~isfield(params,'s'))
    params.s = -1;
end
modul = exp(params.s*1i*(2*pi*w0.*(x_modul-x0)+params.phi));
gaus = reshape(gaus, sz);
modul = reshape(modul, sz);

gabor = struct('complex', gaus.*modul, 'gaus', gaus, 'modul_1', modul);
end

function [X1, Y1] = transform_axis(X, Y, th, tx, ty, sx, sy, changeOrig) %[r,c]
T_transl = [1 0 tx; 0 1 ty; 0 0 1];
T_translM = [1 0 -tx; 0 1 -ty; 0 0 1];
% clockwise rotation (tan(th/2))
T_rot1 = [1 tan(th/2) 0; 0 1 0; 0 0 1];
T_rot2 = [1 0 0; -sin(th) 1 0; 0 0 1];
T_rot3 = T_rot1;

pts = [X(:), Y(:), ones(numel(X),1)];
if (changeOrig)
    p_real = T_translM*(pts');
else
   p_real = (pts');
end
p_real = T_transl*(T_rot3*(T_rot2*(T_rot1*p_real)));
X1 = reshape(p_real(1,:).*sx,size(X));
Y1 = reshape(p_real(2,:).*sy,size(X));
end

function [p_round, p_real] = transform_point(p, th, tx, ty, sx, sy, changeOrig) %[r,c]

T_transl = [1 0 tx; 0 1 ty; 0 0 1];
T_translM = [1 0 -tx; 0 1 -ty; 0 0 1];
% clockwise rotation (tan(th/2))
T_rot1 = [1 tan(th/2) 0; 0 1 0; 0 0 1];
T_rot2 = [1 0 0; -sin(th) 1 0; 0 0 1];
T_rot3 = T_rot1;

pts = [p(:,2) p(:,1) ones(size(p,1),1)]; % x,y,z
if (changeOrig)
    p_real = T_translM*(pts');
else
   p_real = (pts');
end
p_real = T_rot1*p_real;
p_round = round(p_real);

p_real = T_rot2*p_real;
p_round = round(T_rot2*p_round);

p_real = T_rot3*p_real;
p_real = T_transl*p_real;
p_real = [p_real(2,:)*sy; p_real(1,:)*sx]; % [r,c]

p_round = T_rot3*p_round;
p_round = T_transl*p_round;
p_round = [p_round(2,:)*sy, p_round(1,:)*sx];
p_round = round(p_round);
end