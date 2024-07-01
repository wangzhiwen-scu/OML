function [] = cartesian_mask_generate_func_ablation1(index, ratio, subFolderPath)
%  mask_type, center_ratio
%% ratio
% total_rows = 240*ratio;
% center_rows = total_rows *30.56%;
% other_rows = total_rows -center_rows ;
% I want to you give me in the format "(center_rows +other_rows )/240"  at
% different ratios: 20, 30, 40, 50, 70, 80, 90%. center_rows, and
% other_rows  must be int. for chat GPT

if ratio == 20
    Num_fixed=15;   %������������ 1%=1+2; 5% = 8+5, 10% = 14+10, 15%=24+12, 20% = 
    Num_Ran=33;     %���������������
elseif ratio == 30
    Num_fixed=22;   %������������ 1%=1+2; 5% = 8+5, 10% = 14+10, 15%=24+12, 20% = 
    Num_Ran=50;     %���������������
elseif ratio == 40
    Num_fixed=29;   %������������ 1%=1+2; 5% = 8+5, 10% = 14+10, 15%=24+12, 20% = 
    Num_Ran=67;     %���������������
elseif ratio == 50
    Num_fixed=37;   %������������ 1%=1+2; 5% = 8+5, 10% = 14+10, 15%=24+12, 20% = 
    Num_Ran=83;     %���������������
elseif ratio == 60
    Num_fixed=44;   %������������ 1%=1+2; 5% = 8+5, 10% = 14+10, 15%=24+12, 20% = 
    Num_Ran=100;     %���������������
elseif ratio == 70
    Num_fixed=51;   %������������ 1%=1+2; 5% = 8+5, 10% = 14+10, 15%=24+12, 20% = 
    Num_Ran=117;     %���������������
elseif ratio == 80
    Num_fixed=59;   %������������ 1%=1+2; 5% = 8+5, 10% = 14+10, 15%=24+12, 20% = 
    Num_Ran=133;     %���������������
elseif ratio == 90
    Num_fixed=66;   
    Num_Ran=150;  
end
%%    
M=240; 
N=240;   %����ģ���С(M+2,N+2)

Mf=round(M/2);
Nf=round(N/2);

NH1=1:(Mf-Num_fixed/2);
NH2=(Mf+Num_fixed/2+1):M;
z1=length(NH1);
ranlist1= randperm(z1);  % randperm �������һ�����к��� y = randperm(n) y�ǰ�1��n��Щ��������ҵõ���һ����������
z2=length(NH2);
ranlist2= randperm(z2); 

o=zeros(M,N);
o((NH1(end)+1):(NH2(1)-1),:)=1;

ranrow1=ranlist1(1:Num_Ran/2);
o(ranrow1,:)=1;
ranrow2=ranlist2(1:Num_Ran/2)+length(NH1)+Num_fixed;
o(ranrow2,:)=1;

Umask=o;

o1=zeros(M+2,N+2);
o1(2:end-1,2:end-1)=o;
whos o1

disp('������������������')
disp(sum(sum(o1))/((M+2)*(N+2)))
  
% figure;imshow(o1)
% title('ƽ���߲���ģ��')
%  ����
fileName = strcat('cartesian_ablation1_240_240_',string(index));
filePath = fullfile(subFolderPath, fileName);
save(filePath, 'Umask');

end

