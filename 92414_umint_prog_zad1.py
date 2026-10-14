#final verzia
#-imports main-----------------------------
import pandas as pd
import numpy as np
from sklearn.utils import resample
import seaborn as sns
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
#-imports cv2-----------------------
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
#-init--------------------------------------
df = pd.read_csv('zadanie1-data-virgin.csv', sep=';')
pd.set_option('future.no_silent_downcasting', True)
silent=True
#-constants---------------------------
NOTHING = 'nic'
EDA_MAPPINGS = {
    'age':            {},                #NUMERICAL
    'job':            {'unknown': NOTHING, 'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, #categ
                       'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8,
                       'technician': 9, 'unemployed': 10},
    'marital':        {'unknown': NOTHING, 'divorced': 0, 'married': 1, 'single': 2},                        #categ
    'education':      {'unknown': NOTHING, np.nan: NOTHING, 'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2,     #categ
                       'high.school': 3, 'illiterate': 4, 'professional.course': 5, 'university.degree': 6},
    'default':        {'unknown': NOTHING, 'no': 0, 'yes': 1}, #binary
    'housing':        {'unknown': NOTHING, 'no': 0, 'yes': 1}, #binary
    'loan':           {'unknown': NOTHING, 'no': 0, 'yes': 1}, #binary
    'contact':        {'cellular': 0, 'telephone': 1},         #binary
    'month':          {np.nan: NOTHING, 'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,          #categ
                       'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11},
    'day_of_week':    {np.nan: NOTHING, 'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4},                   #categ
    'duration':       {np.nan: NOTHING}, #NUMERICAL
    'campaign':       {np.nan: NOTHING}, #NUMERICAL
    'pdays':          {np.nan: NOTHING}, #NUMERICAL
    'previous':       {np.nan: NOTHING}, #NUMERICAL
    'poutcome':       {'failure': 0, 'nonexistent': 1, 'success': 2},                                        #categ
    'emp.var.rate':   {np.nan: NOTHING}, #NUMERICAL
    'cons.price.idx': {np.nan: NOTHING}, #NUMERICAL
    'cons.conf.idx':  {},                #NUMERICAL 
    'euribor3m':      {np.nan: NOTHING}, #NUMERICAL 
    'nr.employed':    {np.nan: NOTHING}, #NUMERICAL 
    'subscribed':     {'no': 0, 'yes': 1}                      #binary
}
NUMERIC_COLS = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
CATEG_COLS = ['job', 'marital', 'education', 'month', 'day_of_week', 'poutcome']
BINARY_COLS = ['default', 'housing', 'loan', 'contact', 'subscribed']
#-functions----------------------
def resolver():
    global df
    for col in df.columns:
        try:
            float(col.rsplit('_', 1)[-1])
            cap, ix=col.rsplit('_', 1)
            df.rename(columns={col:resolverlogic(cap,ix)}, inplace=True)
        except ValueError:
            continue
def resolverlogic(cap, ix):
    for k, v in EDA_MAPPINGS[cap].items():
        if v == float(ix):
            return k
def edamap():
    for col, mapping in EDA_MAPPINGS.items():
        df[col] = df[col].replace(mapping)
    if not silent:
        print('0.0 Eda mapped : unknown and np.nan->',NOTHING,', words->numbers')
def cullnull(null_toomany):
    global df
    for col in df.columns:
        count = (df[col] == NOTHING).sum() 
        if count >= null_toomany:
            df.drop(columns=[col], inplace=True)
            if not silent:
                print('1.2 Culled nulls under',col,': (ENTIRE COL DELETED) cuz has '+str(count)+'x',NOTHING,'rows')
        elif 0 < count < null_toomany:
            df = df[df[col] != NOTHING]
            if not silent:
                print('1.2 Culled nulls under',col,': '+str(count)+'x',NOTHING,'rows removed')
    df=df.astype(np.float64)
def culloutl(outl_low, outl_high):
    global df
    before = len(df)
    for col in NUMERIC_COLS:
        try:
            df[col]
        except:
            continue
        Q1 = df[col].quantile(outl_low)
        Q3 = df[col].quantile(outl_high)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    after = len(df)
    if not silent:
        print('1.1 Culled outliers with IQR(',outl_low,',',outl_high,'):',before-after,'x rows removed')
def dummify():
    global df
    df = pd.get_dummies(df, columns=CATEG_COLS).astype(np.float64)
    resolver()
    if not silent:
        print('1.3 Performed dummification:',CATEG_COLS,'dummified')
def upsamplebytarget():
    global df
    counts = df['subscribed'].value_counts()
    majority_class, minority_class = counts.idxmax(), counts.idxmin()
    df_majority = df[df['subscribed'] == majority_class]
    df_minority = df[df['subscribed'] == minority_class]
    df_minority_oversampled = resample(df_minority,replace=True,n_samples=len(df_majority),random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_oversampled])
    df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    if not silent:
        print('1.35 Upsampled by target:', counts.to_dict(), '->', df['subscribed'].value_counts().to_dict())    
def conf(y_true, y_pred, capa, capb, epochk, acc):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, annot_kws={'size': 20})
    ax.set_title(f'Konfúzna matica ({capa} set) ({capb}={epochk}, acc={acc:.3f})')
    ax.set_xlabel('Predikovaná trieda')
    ax.set_ylabel('Skutočná trieda')
    win = tk.Toplevel()
    win.title(f'Konfúzna matica ({capa} set) ({capb}={epochk}, acc={acc:.3f})')
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk.Button(win, text='Close', command=win.destroy).pack(pady=10)
    win.geometry('800x600')
def priebeh(y_a, y_b, capa):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(range(1, len(y_a) + 1), y_a, label=f'Train {capa} (epoch)')
    ax.plot(range(1, len(y_b) + 1), y_b, label=f'Val {capa} (epoch)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(capa)
    ax.set_title(f'Train vs Val {capa} (NN)')
    ax.legend()
    fig.tight_layout()
    win = tk.Toplevel()
    win.title(f'Train vs Val {capa} (NN)')
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    tk.Button(win, text='Close', command=win.destroy).pack(pady=10)
    win.geometry('800x600')
def simplemodel(x_train, x_val, x_test,
                y_train, y_val, y_test,
                max_k, patience):
    best_k = None
    best_val_acc = -1
    wait = 0
    for k in range(1, max_k+1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_val_pred = knn.predict(x_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f'k {k:02d} | val_acc={val_acc:.3f}')
        if val_acc > best_val_acc:
            best_k = k
            best_val_acc = val_acc
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped_k = k
                print(f'Early stopping po {stopped_k} k-ckach (patience={patience}). '
                      f'Najlepšia val_acc={best_val_acc:.4f}')
                break
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(x_train, y_train)
    y_train_pred = knn_final.predict(x_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    y_test_pred = knn_final.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f'Najlepšie k={best_k}')
    print(f'Train accuracy: {train_acc*100:.2f}%')
    print(f'Test accuracy: {test_acc*100:.2f}%')
    logs = {'best_k': best_k,
            'y_train': y_train,
            'y_train_pred': y_train_pred,
            'train_acc': train_acc,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'test_acc': test_acc,
    }
    return logs
def hardmodel(x_train, x_val, x_test,
              y_train, y_val, y_test,
              epochs, patience, h1, h2, lr):
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_val_t   = torch.tensor(x_val, dtype=torch.float32)
    x_test_t  = torch.tensor(x_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset   = TensorDataset(x_val_t, y_val_t)
    test_dataset  = TensorDataset(x_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64) # SHUFFLE FALSE LEBO VALIDACNE
    test_loader  = DataLoader(test_dataset, batch_size=64) # SHUFLLE FALSE LEBO TESTOVACIE

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(x.shape[1], h1),
                nn.ReLU(),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Linear(h2, 2)
            )
        def forward(self, x):
            return self.net(x)
    
    model = TinyNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    train_loss_per_epoch, val_loss_per_epoch = [], []
    train_acc_per_epoch,  val_acc_per_epoch  = [], []

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    stopped_epoch = None
    best_epoch=None
    best_val_acc = -1
    
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        for b_idx, (xb, yb) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()

            optimizer.step()

            preds = logits.argmax(dim=1)
            correct_train += int((preds == yb).sum())
            total_train += yb.numel()
            running_loss += float(loss.detach()) * xb.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc  = correct_train / total_train
        train_loss_per_epoch.append(epoch_train_loss)
        train_acc_per_epoch.append(epoch_train_acc)

        model.eval()
        val_running_loss = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running_loss += float(loss) * xb.size(0)
                preds = logits.argmax(dim=1)
                correct_val += int((preds == yb).sum())
                total_val += yb.numel()
                
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_acc  = correct_val / total_val
        val_loss_per_epoch.append(epoch_val_loss)
        val_acc_per_epoch.append(epoch_val_acc)

        print(f'Epoch {epoch:02d} | '
              f'train_loss={epoch_train_loss:.4f} | train_acc={epoch_train_acc:.3f} | '
              f'val_loss={epoch_val_loss:.4f} | val_acc={epoch_val_acc:.3f}')

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            best_epoch = epoch
        else:
            wait += 1
            if wait >= patience:
                stopped_epoch = epoch
                print(f'Early stopping po {stopped_epoch} epochách (patience={patience}). '
                      f'Najlepšia val_acc={best_val_acc:.4f}')
                break

    model.load_state_dict(best_state)

    model.eval()
    device = next(model.parameters()).device
    y_train_true, y_train_pred = [], []
    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_train_pred.extend(preds.tolist())
            y_train_true.extend(yb.numpy().tolist())
    train_acc = accuracy_score(y_train_true, y_train_pred)
    y_test_true, y_test_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_test_pred.extend(preds.tolist())
            y_test_true.extend(yb.numpy().tolist())
    test_acc = accuracy_score(y_test_true, y_test_pred)
    print(f'Najlepšia epocha={best_epoch}')
    print(f'Train accuracy: {train_acc*100:.2f}%')
    print(f'Test accuracy: {test_acc*100:.2f}%')
    logs = {
        'best_epoch': best_epoch,
        'y_train': y_train_true,
        'y_train_pred': y_train_pred,
        'train_acc': train_acc,
        'y_test': y_test_true,
        'y_test_pred': y_test_pred,
        'test_acc': test_acc,

        'train_loss_per_epoch': train_loss_per_epoch,
        'val_loss_per_epoch': val_loss_per_epoch,
        'train_acc_per_epoch': train_acc_per_epoch,
        'val_acc_per_epoch': val_acc_per_epoch,
    }
    return logs
#-runtime--------------------------------------------------
#0.0
edamap()
#1.2
cullnull(2000)
#1.1
culloutl(0.25,0.75)
#1.3
dummify()
#1.35
upsamplebytarget()
#1.4
x = copy.deepcopy(df.drop(columns=['duration','subscribed']).to_numpy())
y = copy.deepcopy(df['subscribed'].to_numpy())

tn, vd, tt = 8,1,1
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=(vd+tt)/(tn+vd+tt), random_state=42, stratify=y
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=(tt)/(vd+tt), random_state=42, stratify=y_temp
)
#1.5
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val   = scaler.transform(x_val)
x_test  = scaler.transform(x_test)
#1.6-8
logss = simplemodel(x_train, x_val, x_test, y_train, y_val, y_test,
                    1000, 30)
conf(logss['y_train'], logss['y_train_pred'], 'KNN train', 'k', logss['best_k'], logss['train_acc'])
conf(logss['y_test'], logss['y_test_pred'], 'KNN test', 'k', logss['best_k'], logss['test_acc'])
#3.1-5
logsh= hardmodel(x_train, x_val, x_test, y_train, y_val, y_test,
                 1000, 30, 128, 64, .005)
priebeh(logsh['train_loss_per_epoch'], logsh['val_loss_per_epoch'],
        'loss')
priebeh(logsh['train_acc_per_epoch'], logsh['val_acc_per_epoch'],
        'acc')
conf(logsh['y_train'], logsh['y_train_pred'], 'NN train', 'epoch', logsh['best_epoch'], logsh['train_acc'])
conf(logsh['y_test'], logsh['y_test_pred'], 'NN test', 'epoch', logsh['best_epoch'], logsh['test_acc'])
#-export-----------------
x=np.concatenate((x_train, x_val, x_test))
y=np.concatenate((y_train, y_val, y_test))
new_dfxfrag = pd.DataFrame(x, columns=df.drop(columns=['duration','subscribed']).columns)
new_dfyfrag = pd.Series(y, name='subscribed')
df = pd.concat([new_dfxfrag, new_dfyfrag], axis=1)
df.to_csv('out.csv', sep=';', index=False)
print('OK')



#avoid errors from culled cols
NUMERIC_COLS.remove('euribor3m')
BINARY_COLS.remove('default')
'''
#univar-mapa
axes=df.hist(bins=20)
for ax in axes.ravel():
    ax.grid(False)
plt.get_current_fig_manager().window.state('zoomed')
plt.subplots_adjust(left=0.03,right=.97,top=.97,bottom=0.03)
plt.show()
'''
'''
#bivar-mapa
#corr=pd.concat([df.drop(columns=NUMERIC_COLS+BINARY_COLS),df['subscribed']], axis=1).corr() #kateg-kateg
#corr=df[NUMERIC_COLS+BINARY_COLS].corr() #num-num
corr=pd.concat([df.drop(columns=['subscribed']),df['subscribed']],axis=1).corr() #kateg-num
ax = sns.heatmap(corr,annot=True,fmt='.2f',cmap='coolwarm',cbar=False,
                 xticklabels=corr.columns,yticklabels=corr.columns)
plt.get_current_fig_manager().window.state('zoomed')
plt.subplots_adjust(left=0.08,right=1,top=1,bottom=0.15)
plt.show()
'''
'''
#bivar-nalezisko
q1='month'
q2='subscribed'
sns.kdeplot(df[df[q2] == 0][q1], label='won\'t subscribe', fill=True)
sns.kdeplot(df[df[q2] == 1][q1], label='will subscribe', fill=True)
plt.xlabel(q1)
plt.ylabel(q2)
if q1 in CATEG_COLS or q1=='contact':
    ax = plt.gca()
    zero_at = list(EDA_MAPPINGS[q1].values()).index(0)
    ax.set_xticks(list(EDA_MAPPINGS[q1].values())[zero_at:])
    ax.set_xticklabels(list(EDA_MAPPINGS[q1].keys())[1:])
plt.legend()
plt.show()
'''
'''
#polyvar-nalezisko
q1 = 'emp.var.rate'
q2 = 'cons.price.idx'      
q3 = 'subscribed'
x0 = df[df[q3] == 0][q1]
y0 = df[df[q3] == 0][q2]
x1 = df[df[q3] == 1][q1]
y1 = df[df[q3] == 1][q2]
xmin, xmax = min(df[q1]), max(df[q1])
ymin, ymax = min(df[q2]), max(df[q2])
X, Y = np.mgrid[xmin:xmax:10j, ymin:ymax:10j]
kde0 = gaussian_kde(np.vstack([x0, y0]))
kde1 = gaussian_kde(np.vstack([x1, y1]))
Z0 = kde0(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
Z1 = kde1(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z0,  alpha=0.6, label='won\'t subscribe')
ax.plot_surface(X, Y, Z1, alpha=0.6, label='will subscribe')
ax.set_xlabel(q1)
ax.set_ylabel(q2)
ax.set_zlabel(q3)
if q1 in CATEG_COLS or q1=='contact':
    ax = plt.gca()
    zero_at = list(EDA_MAPPINGS[q1].values()).index(0)
    ax.set_xticks(list(EDA_MAPPINGS[q1].values())[zero_at:])
    ax.set_xticklabels(list(EDA_MAPPINGS[q1].keys())[zero_at:])
if q2 in CATEG_COLS or q2=='contact':
    ax = plt.gca()
    zero_at = list(EDA_MAPPINGS[q2].values()).index(0)
    ax.set_yticks(list(EDA_MAPPINGS[q2].values())[zero_at:])
    ax.set_yticklabels(list(EDA_MAPPINGS[q2].keys())[zero_at:])
plt.legend()
plt.show()
'''
