loadpredict=sim(net,x);
MAPE=((y-loadpredict).*100)./(y.*500)

n=1:499;
plot(n,loadpredict,'DisplayName','loadpredict','Color','m');
hold on
plot(n,y,'DisplayName','actualload','Color','b');
hold off
