function [V_star, obj, error_cnt] = LPDiNMF_update(X, label, varargin)
% Input: X(m*n)

pnames = {'maxiter', 'tolfun', 'alpha', 'gamma'};
dflts  = {150, 1e-6, 0, 0};
[maxiter, tolfun, alpha, gamma] = internal.stats.parseArgs(pnames,dflts,varargin{:});

view_num = length(X); % ��ͼ����
class_num = length(unique(label)); % ������

U = cell(view_num, 1); % ������ͼ�Ļ�����
V = cell(view_num, 1); % ϵ������
obj = zeros(1, 1); %Ŀ�꺯��ֵ

% ���ݽ��й�һ��
for i = 1: view_num
    [X{i}, ~] = data_normalization(X{i}, [], 'std');
end

% ����ͼ
W = cell(view_num, 1); % Ȩ�ؾ���
W1 = cell(view_num, 1); % �Ⱦ���
L = cell(view_num, 1); % ������˹����
options = [];
options.WeightMode = 'Binary'; % 0-1Ȩ��
options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = 5; % K���ڵ��ھ���
% ��ÿ����ͼ����ͼ
for view_idx = 1: view_num
    W{view_idx} = constructW(X{view_idx}',options);
    W1{view_idx} = diag(sum(W{view_idx}, 1));
    L{view_idx} = W1{view_idx} - W{view_idx};
end

% ��ʼ��
for view_idx = 1: view_num
    [U{view_idx}, V{view_idx}] = KMeansdata(X{view_idx}, class_num); % ʹ��kmeans���г�ʼ��
    U{view_idx} = abs(U{view_idx});
    V{view_idx} = abs(V{view_idx});
end

% ���´��������
error_cnt = 0;

% ��������
for iter = 1: maxiter
    
   % update V^p(p = 1, ..., P)
   for view_idx = 1: view_num
       V{view_idx} = V{view_idx} .* (2*(U{view_idx}')*X{view_idx}+2*gamma*V{view_idx}*W{view_idx}) ./...
           max(2*(U{view_idx}')*U{view_idx}*V{view_idx}+alpha*calc_sum_V(V, view_idx)+2*alpha*V{view_idx}+2*gamma*V{view_idx}*W1{view_idx}, eps);
   end
   
   % update U^p(p = 1, ..., P)
   for view_idx = 1: view_num
          U{view_idx} = U{view_idx} .* (X{view_idx}*(V{view_idx}')) ./...
              max(U{view_idx}*V{view_idx}*(V{view_idx}'), eps);
   end
    
   % ��¼Ŀ�꺯��ֵ
   obj(iter) = calc_obj_value(X, U, V, L, alpha, gamma);
   fprintf('iter = %d, obj = %g\n', iter, obj(iter));
   
   % �������֮��Ŀ�꺯��ֵ���ӣ������������1
   if (iter>=2)&&(obj(iter)>obj(iter-1))
      error_cnt = error_cnt + 1; 
   end
   
   % �㷨ֹͣ����
    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<tolfun)|| iter==maxiter
        break;
    end
   
end

V_star = zeros(size(V{view_num})); % V*
for view_idx = 1: view_num
    V_star = V_star + V{view_idx};
end
V_star = V_star/view_num; % ��ƽ��

end

function [obj_value] = calc_obj_value(X, U, V, L, alpha, gamma)
view_num = length(X); % ��ͼ����
obj_value = 0;
for view_idx = 1: view_num
    obj_value = obj_value...
        + (norm(X{view_idx}-U{view_idx}*V{view_idx}, 'fro').^2)...
        + alpha*calc_sum_dive(V, view_idx)...
        + gamma*trace(V{view_idx}*L{view_idx}*(V{view_idx}'));
end
end

function [sum_V] = calc_sum_V(V, view_idx)
view_num = length(V); % ��ͼ����
sum_V = zeros(size(V{1}));
for i = 1: view_num
   if i ~=  view_idx
       sum_V = sum_V + V{i};
   end
end
end

function [sum_dive] = calc_sum_dive(V, view_idx)
view_num = length(V); % ��ͼ����
sum_dive = 0;
for i = 1: view_num
    sum_dive = sum_dive + trace(V{view_idx}*(V{i}'));
end
end

