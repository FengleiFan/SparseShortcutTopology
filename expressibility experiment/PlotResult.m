K= [4, 5, 6, 7, 8, 9, 10];
Residual = [0.7708, 0.7463, 0.7307, 0.7196, 0.7074, 0.6975, 0.6902];
Proposed = [0.7514, 0.7261, 0.7034, 0.6846, 0.6687, 0.6567, 0.6512];


figure()

% h = bar(K, [Residual', Proposed']);
% 
% set(h(1), 'facecolor', 'r')
% set(h(2), 'facecolor', 'b')



% hold on
plot(K,Residual,'b-s','MarkerSize',3,'MarkerFaceColor','b','LineWidth',2)
hold on
plot(K,Proposed,'r-o','MarkerSize',3,'MarkerFaceColor','r','LineWidth',2)

set(gca,'XTick',[4 5 6 7 8 9 10], 'XTickLabel', {'4','5','6','7','8','9','10'}, 'fontsize', 12)
lgd = legend('Residual','Proposed')
lgd.FontSize = 14;
xlabel('K')
ylabel('Mean Squared Error')

