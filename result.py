from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt


Y=[]
I=[]
Z=[]
for i in range(1,1000):
    I.append(i)
    #Z.append(i)
    log_dir = "logsppo"f"t{i}"
    x,y=ts2xy(load_results(log_dir), 'timesteps')
    print('y',y)
    print('len of y', len(y))
    Y.append(y)
plt.plot(I,Y)
plt.show()
print('Y',len(Y) )
#print('Y',Y )