% �Ӻ������ڰѾ���ת��ΪԪ������
function x1 = matToCell(x)
[~, m] = size(x);
x1 = cell(m, 1);
for i = 1 : m
    x1{i, 1} = x(:, i);
end

