# Boreholes
import numpy as np
import matplotlib.pyplot as plt

#facies
facii = np.load("facies_ammer_v3.npy")
k = np.load("facies_ammer_v3.npy")
facii[facii == 0] = 2
Z = np.load("facies_Z_v3.npy")

#randomly sample columns:
rows = np.random.choice(a=np.arange(facii.shape[1]),size=10)
cols = np.random.choice(a=np.arange(facii.shape[2]), size=10)

samples = facii[:,rows,cols]
z_samples = Z[:,0,0]
print(z_samples)
z_samples = np.squeeze(z_samples)
colors = {1:"#FCF2A8",2:"#cec47f",4:"#c9c186",5:"#7e7954",6:"#b0b468",7:"#c8b23d",8:"#323021",11:"#b7b7b7"}
hatches = {1:"o",2:".",4:"",5:".",6:"..",7:"..",8:"",11:"oo"}
labels = {1:"T1",2:"T2+T3",4:"T4",5:"T5",6:"T6",7:"T7",8:"T8+T9",11:"T11"}
#legend
fig = plt.figure(figsize=(2,6))
ax = fig.add_subplot(1,1,1)
legend_elements = [plt.Rectangle((0,0), width = 1, height=1,
                            fc=colors[i],hatch=hatches[i],alpha=0.9, label = labels[i]) for i in colors.keys()]
ax.legend(handles=legend_elements,loc="best", markerscale = 10.,)
fig.show()
fig.savefig("examples/ammer/legend_boreholes.png")
#plot samples:
i= 0
for row,col in zip(rows,cols):
    fig,axs = plt.subplots(1,2,figsize=(2,6))
    ax = axs[0]
    #row = rows[0]
    #col = cols[0]
    

    for zind in np.arange(z_samples.size):
        facies = facii[zind,row,col]
        color = colors[facies]
        hatch = hatches[facies]
        rec = plt.Rectangle((0,z_samples[zind]-0.1), width = 4, height=0.2,
                            fc=color,hatch=hatch,alpha=0.9) 
        pt = ax.add_patch(rec)
    axs[1].plot()


    ax.set_ylim(0,7)

    fig.show()
    fig.savefig(f"examples/ammer/borehole_v3_{i}.png",dpi = 400)
    i+=1
