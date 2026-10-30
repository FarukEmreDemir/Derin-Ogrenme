import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, optimizers

# ==========================================
# 0. RANDOM SEED AYARI (OKUL NO SON 2 HANE: 30)
# ==========================================
SEED = 30

def set_seed(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# ==========================================
# 1. VERİ SETİ
# Giriş: -2 <= p <= 2
# Hedef: y = 1 + sin(pi * p / 4)
# ==========================================
p = np.linspace(-2.0, 2.0, 21).reshape(-1, 1)
y_target = 1.0 + np.sin(np.pi * p / 4.0)

os.makedirs("odev_grafikler", exist_ok=True)

# ==========================================
# 2. MODEL MİMARİSİ VE YARDIMCI FONKSİYONLAR
# ==========================================
def get_initializer(dist_type, seed=SEED):
    if dist_type == 'normal':
        return initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed)
    elif dist_type == 'uniform':
        return initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed)
    return 'glorot_uniform'

def get_optimizer(opt_name, lr):
    opt_name = opt_name.lower()
    if opt_name == 'sgd':
        return optimizers.SGD(learning_rate=lr)
    elif opt_name == 'rmsprop':
        return optimizers.RMSprop(learning_rate=lr)
    elif opt_name == 'adam':
        return optimizers.Adam(learning_rate=lr)
    raise ValueError(f"Bilinmeyen optimizer: {opt_name}")

def create_model(act='relu', num_layers=1, num_neurons=2, dist_type='normal', seed=SEED):
    init = get_initializer(dist_type, seed=seed)
    
    act_map = {
        'lineer': 'linear',
        'sigmoid': 'sigmoid',
        'hiperbolik tanjant': 'tanh',
        'relu': 'relu',
        'leakyrelu': layers.LeakyReLU(alpha=0.2),
        'elu': 'elu',
        'silu': 'silu'
    }
    keras_act = act_map[act.lower()]

    model = keras.Sequential()
    
    # Giriş ve ilk gizli katman
    if isinstance(keras_act, layers.Layer):
        model.add(layers.Dense(num_neurons, input_shape=(1,), kernel_initializer=init))
        model.add(keras_act)
    else:
        model.add(layers.Dense(num_neurons, activation=keras_act, input_shape=(1,), kernel_initializer=init))
    
    # Ek gizli katmanlar
    for _ in range(num_layers - 1):
        if isinstance(keras_act, layers.Layer):
            model.add(layers.Dense(num_neurons, kernel_initializer=init))
            model.add(keras_act)
        else:
            model.add(layers.Dense(num_neurons, activation=keras_act, kernel_initializer=init))
            
    # 1 lineer çıkış nöronu
    model.add(layers.Dense(1, activation='linear', kernel_initializer=init))
    return model

# ==========================================
# 3. METRİK VE EPOCH TAKİBİ (CALLBACK)
# ==========================================
class TrainingTracker(keras.callbacks.Callback):
    def __init__(self, x_data, y_true, target_epochs):
        super().__init__()
        self.x_data = x_data
        self.y_true = y_true
        self.target_epochs = target_epochs
        self.predictions = {}
        self.epoch_losses = []
        self.epoch_r2 = []

    def on_epoch_end(self, epoch, logs=None):
        loss_val = logs.get('loss')
        self.epoch_losses.append(loss_val)
        
        preds = self.model.predict(self.x_data, verbose=0)
        r2 = r2_score(self.y_true, preds)
        self.epoch_r2.append(r2)
        
        if epoch in self.target_epochs:
            self.predictions[epoch] = preds

# ==========================================
# 4. GRAFİK OLUŞTURMA
# ==========================================
def save_best_model_plots(table_name, best_label, tracker):
    epochs_range = range(1, len(tracker.epoch_losses) + 1)
    
    # 1. Epoch / MSE Kayıp Grafiği (Doğrusal ve Logaritmik)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_range, tracker.epoch_losses, color='#b22222', lw=2)
    ax1.set_title("Her epoch'taki kayıp (doğrusal eksen)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss = 1/2(t - a)^2")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, tracker.epoch_losses, color='#b22222', lw=2)
    ax2.set_yscale('log')
    ax2.set_title("Her epoch'taki kayıp (logaritmik eksen)")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss = 1/2(t - a)^2")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"{table_name} - En İyi Parametre: {best_label}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"odev_grafikler/{table_name}_kayip_grafigi.png", dpi=200)
    plt.close()

    # 2. Epoch / R2 Değişim Grafiği
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, tracker.epoch_r2, color='#1f77b4', lw=2)
    plt.title(f"{table_name} - Epoch / R2 Skoru ({best_label})")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Skoru")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"odev_grafikler/{table_name}_r2_grafigi.png", dpi=200)
    plt.close()

    # 3. Epoch / Hedef-Çıkış Uyum Grafiği
    saved_epochs = sorted(tracker.predictions.keys())
    n_plots = len(saved_epochs)
    cols = 4
    rows = int(np.ceil(n_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for idx, ep in enumerate(saved_epochs):
        ax = axes[idx]
        loss_val = tracker.epoch_losses[ep]
        pred = tracker.predictions[ep]
        
        ax.plot(p, y_target, color='#1c3d5a', lw=2, marker='.', label='Hedef' if idx == 0 else "")
        ax.plot(p, pred, color='#c0392b', lw=2, label='Ağ çıkışı' if idx == 0 else "")
        ax.set_title(f"Epoch {ep}  (loss = {loss_val:.4f})", fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx >= (rows - 1) * cols:
            ax.set_xlabel("p (giriş)")
        if idx % cols == 0:
            ax.set_ylabel("a^2 (çıkış)")

    for k in range(idx + 1, len(axes)):
        fig.delaxes(axes[k])

    fig.legend(loc='upper left', bbox_to_anchor=(0.08, 0.96))
    plt.suptitle(f"{table_name} - Epoch / Hedef-Çıkış Grafiği ({best_label})", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"odev_grafikler/{table_name}_hedef_cikis_grafigi.png", dpi=200)
    plt.close()

# ==========================================
# 5. DENEY KOŞUCUSU
# ==========================================
target_epochs_to_track = [0, 1, 2, 5, 10, 25, 50, 100, 150, 200, 250, 299]

def execute_experiment(table_name, param_key, param_values, default_settings):
    print(f"\n=======================================================")
    print(f"ÇALIŞIYOR: {table_name}")
    print(f"=======================================================")
    
    results = []
    trackers = {}

    for val in param_values:
        current_cfg = default_settings.copy()
        current_cfg[param_key] = val

        set_seed(SEED)
        model = create_model(
            act=current_cfg['act'],
            num_layers=current_cfg['num_layers'],
            num_neurons=current_cfg['num_neurons'],
            dist_type=current_cfg['dist'],
            seed=SEED
        )
        
        opt = get_optimizer(current_cfg['opt'], current_cfg['lr'])
        model.compile(optimizer=opt, loss='mean_squared_error')

        tracker = TrainingTracker(p, y_target, target_epochs_to_track)
        
        start_time = time.time()
        model.fit(
            p, y_target,
            epochs=300,
            batch_size=len(p),
            verbose=0,
            callbacks=[tracker]
        )
        duration = time.time() - start_time

        y_pred = model.predict(p, verbose=0)
        mse = mean_squared_error(y_target, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_target, y_pred)

        results.append({
            'Parametre': str(val),
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'Süre (sn)': duration
        })
        trackers[str(val)] = tracker

    # En iyi model: En yüksek R2 (veya en düşük MSE)
    best_item = max(results, key=lambda x: x['R2'])
    
    print(f"\nSONUÇ: {table_name}")
    print(f"{'Parametre':<22} | {'MSE':<12} | {'RMSE':<12} | {'R2':<12} | {'Süre (sn)':<10}")
    print("-" * 75)
    for res in results:
        is_best = (res['Parametre'] == best_item['Parametre'])
        marker = "** " if is_best else "   "
        print(f"{marker}{res['Parametre']:<19} | {res['MSE']:<12.6f} | {res['RMSE']:<12.6f} | {res['R2']:<12.6f} | {res['Süre (sn)']:<10.3f}")

    print(f"\nEn iyi parametre '{best_item['Parametre']}' için grafikler üretiliyor...")
    save_best_model_plots(table_name, best_item['Parametre'], trackers[best_item['Parametre']])
    return results

# ==========================================
# 6. DENEYLERİN İCRA EDİLMESİ
# ==========================================
if __name__ == "__main__":
    
    # Tablo 1: Aktivasyon Fonksiyonları (Öğrenme Oranı: 0.1, Yöntem: SGD, Dağılım: normal, Katman: 1, Nöron: 2)
    cfg1 = {'act': 'relu', 'lr': 0.1, 'opt': 'sgd', 'dist': 'normal', 'num_layers': 1, 'num_neurons': 2}
    acts = ['Lineer', 'Sigmoid', 'Hiperbolik Tanjant', 'ReLU', 'LeakyReLU', 'ELU', 'SiLU']
    execute_experiment("Tablo_1_Aktivasyon", 'act', acts, cfg1)

    # Tablo 2: Öğrenme Oranı (0.1, 0.01, 0.001)
    cfg2 = {'act': 'relu', 'lr': 0.1, 'opt': 'sgd', 'dist': 'normal', 'num_layers': 1, 'num_neurons': 2}
    lrs = [0.1, 0.01, 0.001]
    execute_experiment("Tablo_2_Ogrenme_Orani", 'lr', lrs, cfg2)

    # Tablo 3: Öğrenme Oranı Ayar Yöntemi (SGD, RMSProp, ADAM)
    cfg3 = {'act': 'relu', 'lr': 0.01, 'opt': 'sgd', 'dist': 'normal', 'num_layers': 1, 'num_neurons': 2}
    opts = ['SGD', 'RMSProp', 'ADAM']
    execute_experiment("Tablo_3_Optimizasyon_Yontemi", 'opt', opts, cfg3)

    # Tablo 4: Ağırlık Başlangıç Olasılık Dağılımı (normal, uniform)
    cfg4 = {'act': 'relu', 'lr': 0.01, 'opt': 'adam', 'dist': 'normal', 'num_layers': 1, 'num_neurons': 2}
    dists = ['normal', 'uniform']
    execute_experiment("Tablo_4_Baslangic_Dagilimi", 'dist', dists, cfg4)

    # Tablo 5: Gizli Katman Sayısı (1, 2, 5, 10, 20)
    cfg5 = {'act': 'relu', 'lr': 0.01, 'opt': 'adam', 'dist': 'normal', 'num_layers': 1, 'num_neurons': 2}
    layers_list = [1, 2, 5, 10, 20]
    execute_experiment("Tablo_5_Gizli_Katman_Sayisi", 'num_layers', layers_list, cfg5)

    # Tablo 6: Gizli Katmandaki Nöron Sayısı (2, 5, 10, 20, 50)
    cfg6 = {'act': 'relu', 'lr': 0.01, 'opt': 'adam', 'dist': 'normal', 'num_layers': 1, 'num_neurons': 2}
    neurons_list = [2, 5, 10, 20, 50]
    execute_experiment("Tablo_6_Neuron_Sayisi", 'num_neurons', neurons_list, cfg6)

    print("\nTüm deneyler ve grafik çizimleri tamamlandı.")